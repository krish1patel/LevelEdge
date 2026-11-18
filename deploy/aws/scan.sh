#!/usr/bin/env bash
# Set up (or update) the daily market scan on AWS. Safe to re-run.
#
#   bash deploy/aws/scan.sh            # create or update everything
#   bash deploy/aws/scan.sh --run-now  # ... and start a scan immediately
#
# Pipeline (all inside AWS free tiers at this volume):
#   EventBridge Scheduler, 5:45 PM ET Mon-Fri
#     -> leveledge-scan Lambda ("dispatch"): splits ~100 tickers into batches of 5
#     -> SQS queue leveledge-scan (dead-letter queue after 3 failed attempts)
#     -> leveledge-scan Lambda workers, at most 5 at once: forecast each ticker, write one item each
#     -> DynamoDB table leveledge-scan (items expire after 120 days)
#     -> /api/scan on the public API -> the Scanner page on the dashboard
# The worker uses the same container image as the API (deploy/aws/deploy.sh builds it) and runs
# app/worker.py instead of app/api.py. It has no public URL.
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
NAME="leveledge-scan"
API="${API_NAME:-leveledge-api}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
IMAGE="$(aws lambda get-function --region "$REGION" --function-name "$API" --query Code.ImageUri --output text)"
echo "Account $ACCOUNT, region $REGION, image ${IMAGE##*/}"

# ---- 1. DynamoDB table ----------------------------------------------------------------------
if ! aws dynamodb describe-table --region "$REGION" --table-name "$NAME" >/dev/null 2>&1; then
  aws dynamodb create-table --region "$REGION" --table-name "$NAME" --billing-mode PAY_PER_REQUEST \
    --attribute-definitions AttributeName=scan_date,AttributeType=S AttributeName=ticker,AttributeType=S \
    --key-schema AttributeName=scan_date,KeyType=HASH AttributeName=ticker,KeyType=RANGE >/dev/null
  aws dynamodb wait table-exists --region "$REGION" --table-name "$NAME"
  aws dynamodb update-time-to-live --region "$REGION" --table-name "$NAME" \
    --time-to-live-specification Enabled=true,AttributeName=expires >/dev/null
  echo "Created table $NAME"
fi
TABLE_ARN="$(aws dynamodb describe-table --region "$REGION" --table-name "$NAME" --query Table.TableArn --output text)"

# ---- 2. SQS queue and dead-letter queue ---------------------------------------------------------
queue() { aws sqs get-queue-url --region "$REGION" --queue-name "$1" --query QueueUrl --output text 2>/dev/null ||
          aws sqs create-queue --region "$REGION" --queue-name "$1" --query QueueUrl --output text; }
DLQ_URL="$(queue "$NAME-dlq")"
DLQ_ARN="$(aws sqs get-queue-attributes --region "$REGION" --queue-url "$DLQ_URL" --attribute-names QueueArn --query Attributes.QueueArn --output text)"
aws sqs set-queue-attributes --region "$REGION" --queue-url "$DLQ_URL" --attributes MessageRetentionPeriod=1209600
Q_URL="$(queue "$NAME")"
Q_ARN="$(aws sqs get-queue-attributes --region "$REGION" --queue-url "$Q_URL" --attribute-names QueueArn --query Attributes.QueueArn --output text)"
# visibility 6x the worker timeout, as AWS recommends for Lambda consumers; failed batches retry 3 times
aws sqs set-queue-attributes --region "$REGION" --queue-url "$Q_URL" --attributes "{\"VisibilityTimeout\":\"1800\",\"MessageRetentionPeriod\":\"86400\",\"RedrivePolicy\":\"{\\\"deadLetterTargetArn\\\":\\\"$DLQ_ARN\\\",\\\"maxReceiveCount\\\":\\\"3\\\"}\"}"

# ---- 3. Worker role: only this queue and this table ----------------------------------------------
ROLE="$NAME-role"
if ! ROLE_ARN="$(aws iam get-role --role-name "$ROLE" --query Role.Arn --output text 2>/dev/null)"; then
  ROLE_ARN="$(aws iam create-role --role-name "$ROLE" --query Role.Arn --output text --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}')"
  aws iam attach-role-policy --role-name "$ROLE" --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  CREATED_ROLE=1
fi
aws iam put-role-policy --role-name "$ROLE" --policy-name scan-access --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[
  {\"Effect\":\"Allow\",\"Action\":[\"sqs:SendMessage\",\"sqs:ReceiveMessage\",\"sqs:DeleteMessage\",\"sqs:GetQueueAttributes\",\"sqs:ChangeMessageVisibility\"],\"Resource\":\"$Q_ARN\"},
  {\"Effect\":\"Allow\",\"Action\":[\"dynamodb:GetItem\",\"dynamodb:PutItem\",\"dynamodb:Query\"],\"Resource\":\"$TABLE_ARN\"}]}"
[ "${CREATED_ROLE:-0}" = 1 ] && { echo "Created role $ROLE; waiting for it to propagate"; sleep 12; }

# ---- 4. Worker function (same image, runs app/worker.py) -----------------------------------------
aws logs create-log-group --region "$REGION" --log-group-name "/aws/lambda/$NAME" 2>/dev/null || true
aws logs put-retention-policy --region "$REGION" --log-group-name "/aws/lambda/$NAME" --retention-in-days 7
ENV="{\"Variables\":{\"SCAN_TABLE\":\"$NAME\",\"SCAN_QUEUE_URL\":\"$Q_URL\",\"AWS_LWA_READINESS_CHECK_PATH\":\"/healthz\",\"LEVELEDGE_DISABLE_LOGGING\":\"1\"}}"
CMD='{"Command":["uvicorn","app.worker:app","--host","0.0.0.0","--port","8080"]}'
if aws lambda get-function --region "$REGION" --function-name "$NAME" >/dev/null 2>&1; then
  aws lambda update-function-code --region "$REGION" --function-name "$NAME" --image-uri "$IMAGE" >/dev/null
  aws lambda wait function-updated --region "$REGION" --function-name "$NAME"
  aws lambda update-function-configuration --region "$REGION" --function-name "$NAME" --timeout 300 --memory-size 1536 \
    --environment "$ENV" --image-config "$CMD" >/dev/null
else
  aws lambda create-function --region "$REGION" --function-name "$NAME" --package-type Image --code "ImageUri=$IMAGE" \
    --role "$ROLE_ARN" --architectures arm64 --timeout 300 --memory-size 1536 --environment "$ENV" --image-config "$CMD" >/dev/null
fi
aws lambda wait function-updated --region "$REGION" --function-name "$NAME" 2>/dev/null || aws lambda wait function-active --region "$REGION" --function-name "$NAME"
aws lambda put-function-concurrency --region "$REGION" --function-name "$NAME" --reserved-concurrent-executions 6 >/dev/null
FN_ARN="$(aws lambda get-function --region "$REGION" --function-name "$NAME" --query Configuration.FunctionArn --output text)"

# ---- 5. Queue -> worker, at most 5 batches at once, partial-batch failures -------------------------
if [ -z "$(aws lambda list-event-source-mappings --region "$REGION" --function-name "$NAME" --event-source-arn "$Q_ARN" --query 'EventSourceMappings[0].UUID' --output text | grep -v None)" ]; then
  aws lambda create-event-source-mapping --region "$REGION" --function-name "$NAME" --event-source-arn "$Q_ARN" \
    --batch-size 1 --function-response-types ReportBatchItemFailures --scaling-config MaximumConcurrency=5 >/dev/null
  echo "Connected the queue to the worker"
fi

# ---- 6. The public API reads the table ---------------------------------------------------------------
API_ROLE="$(aws lambda get-function --region "$REGION" --function-name "$API" --query Configuration.Role --output text)"
aws iam put-role-policy --role-name "${API_ROLE##*/}" --policy-name scan-read --policy-document \
  "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"dynamodb:GetItem\",\"dynamodb:Query\"],\"Resource\":\"$TABLE_ARN\"}]}"
ENVFILE="$(mktemp)"; chmod 600 "$ENVFILE"; trap 'rm -f "$ENVFILE"' EXIT
aws lambda get-function-configuration --region "$REGION" --function-name "$API" --query Environment --output json \
  | python3 -c "import json,sys; e=json.load(sys.stdin) or {'Variables':{}}; e['Variables']['SCAN_TABLE']='$NAME'; print(json.dumps(e))" > "$ENVFILE"
aws lambda update-function-configuration --region "$REGION" --function-name "$API" --environment "file://$ENVFILE" >/dev/null

# ---- 7. Schedule: 5:45 PM ET, Monday to Friday ------------------------------------------------------
SROLE="$NAME-scheduler"
if ! SROLE_ARN="$(aws iam get-role --role-name "$SROLE" --query Role.Arn --output text 2>/dev/null)"; then
  SROLE_ARN="$(aws iam create-role --role-name "$SROLE" --query Role.Arn --output text --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"scheduler.amazonaws.com"},"Action":"sts:AssumeRole"}]}')"
  sleep 10
fi
aws iam put-role-policy --role-name "$SROLE" --policy-name invoke-scan --policy-document \
  "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"lambda:InvokeFunction\",\"Resource\":\"$FN_ARN\"}]}"
TARGET="{\"Arn\":\"$FN_ARN\",\"RoleArn\":\"$SROLE_ARN\",\"Input\":\"{\\\"action\\\":\\\"dispatch\\\"}\",\"RetryPolicy\":{\"MaximumRetryAttempts\":2}}"
SCHED=(--name "$NAME-daily" --schedule-expression "cron(45 17 ? * MON-FRI *)" --schedule-expression-timezone America/New_York
       --flexible-time-window Mode=OFF --target "$TARGET")
aws scheduler get-schedule --region "$REGION" --name "$NAME-daily" >/dev/null 2>&1 \
  && aws scheduler update-schedule --region "$REGION" "${SCHED[@]}" >/dev/null \
  || aws scheduler create-schedule --region "$REGION" "${SCHED[@]}" >/dev/null
echo "Scheduled: weekdays 5:45 PM ET"

if [ "${1:-}" = "--run-now" ]; then
  OUT="$(mktemp)"
  aws lambda invoke --region "$REGION" --function-name "$NAME" --cli-binary-format raw-in-base64-out \
    --payload '{"action":"dispatch"}' "$OUT" >/dev/null && echo "Dispatched: $(cat "$OUT")"
fi
echo "Done. Check progress: aws dynamodb query --table-name $NAME --key-condition-expression 'scan_date = :d' --expression-attribute-values '{\":d\":{\"S\":\"<date>\"}}' --select COUNT"
