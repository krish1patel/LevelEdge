#!/usr/bin/env bash
# Deploy the LevelEdge API to AWS Lambda (container image, arm64) behind a Function URL,
# then point the Vercel frontend at it. Safe to re-run: it updates what already exists.
#
#   bash deploy/aws/deploy.sh                 # needs: aws CLI logged in, docker
#   BUDGET_EMAIL=you@example.com bash deploy/aws/deploy.sh   # also create a $5/month budget alert
#   ROLE_ARN=arn:aws:iam::...:role/LabRole bash deploy/aws/deploy.sh   # AWS Academy: use the lab's role
#
# Cost guards (all inside the always-free Lambda tier at this traffic):
#   * Lambda: 1M requests and 400,000 GB-seconds a month free; 1.5 GB memory -> ~266,000 s free.
#   * At most 5 copies run at once (reserved concurrency), so a flood of requests can't run up a bill.
#   * ECR keeps only the 2 newest images; CloudWatch logs are kept 7 days.
#   * Optional AWS Budgets alert at $5 (two budgets are free).
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
NAME="${NAME:-leveledge-api}"
MEMORY="${MEMORY:-1536}"
TIMEOUT="${TIMEOUT:-120}"
CONCURRENCY="${CONCURRENCY:-5}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REPO="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$NAME"
echo "Account $ACCOUNT, region $REGION"

# ---- 1. image -> ECR ------------------------------------------------------------------
aws ecr describe-repositories --region "$REGION" --repository-names "$NAME" >/dev/null 2>&1 ||
  aws ecr create-repository --region "$REGION" --repository-name "$NAME" --image-scanning-configuration scanOnPush=false >/dev/null
aws ecr put-lifecycle-policy --region "$REGION" --repository-name "$NAME" --lifecycle-policy-text \
  '{"rules":[{"rulePriority":1,"description":"keep 2 newest","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":2},"action":{"type":"expire"}}]}' >/dev/null
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com" >/dev/null
TAG="$(git rev-parse --short HEAD)-$(date +%s)"
docker build --platform linux/arm64 --provenance=false -f deploy/aws/Dockerfile -t "$REPO:$TAG" .
docker push -q "$REPO:$TAG"

# ---- 2. role ------------------------------------------------------------------------------
if [ -z "${ROLE_ARN:-}" ]; then
  ROLE="$NAME-role"
  if ! ROLE_ARN="$(aws iam get-role --role-name "$ROLE" --query Role.Arn --output text 2>/dev/null)"; then
    ROLE_ARN="$(aws iam create-role --role-name "$ROLE" --query Role.Arn --output text --assume-role-policy-document \
      '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}')"
    aws iam attach-role-policy --role-name "$ROLE" --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
    echo "Created role $ROLE; waiting for it to propagate"; sleep 12
  fi
fi

# ---- 3. environment (secrets come from .env and are never printed) ------------------------
ENVFILE="$(mktemp)"; chmod 600 "$ENVFILE"; trap 'rm -f "$ENVFILE"' EXIT
# keep the market scan wired up if deploy/aws/scan.sh has created its table
aws dynamodb describe-table --region "$REGION" --table-name leveledge-scan >/dev/null 2>&1 && export SCAN_TABLE=leveledge-scan || true
python3 - "$ENVFILE" <<'PY'
import json, os, sys
keys = ["SUPABASE_URL", "SUPABASE_KEY", "ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "POLYGON_API_KEY"]
env = {"LEVELEDGE_DISABLE_LOGGING": "1"}
if os.environ.get("SCAN_TABLE"):
    env["SCAN_TABLE"] = os.environ["SCAN_TABLE"]
if os.path.exists(".env"):
    for line in open(".env"):
        k, _, v = line.strip().partition("=")
        if k in keys and v:
            env[k] = v.strip().strip('"').strip("'")
json.dump({"Variables": env}, open(sys.argv[1], "w"))
print("environment keys:", ", ".join(sorted(env)))
PY

# ---- 4. function ------------------------------------------------------------------------------
aws logs create-log-group --region "$REGION" --log-group-name "/aws/lambda/$NAME" 2>/dev/null || true
aws logs put-retention-policy --region "$REGION" --log-group-name "/aws/lambda/$NAME" --retention-in-days 7
if aws lambda get-function --region "$REGION" --function-name "$NAME" >/dev/null 2>&1; then
  aws lambda update-function-code --region "$REGION" --function-name "$NAME" --image-uri "$REPO:$TAG" >/dev/null
  aws lambda wait function-updated --region "$REGION" --function-name "$NAME"
  aws lambda update-function-configuration --region "$REGION" --function-name "$NAME" --memory-size "$MEMORY" \
    --timeout "$TIMEOUT" --environment "file://$ENVFILE" >/dev/null
else
  aws lambda create-function --region "$REGION" --function-name "$NAME" --package-type Image --code "ImageUri=$REPO:$TAG" \
    --role "$ROLE_ARN" --architectures arm64 --memory-size "$MEMORY" --timeout "$TIMEOUT" --environment "file://$ENVFILE" >/dev/null
fi
aws lambda wait function-updated --region "$REGION" --function-name "$NAME" 2>/dev/null || aws lambda wait function-active --region "$REGION" --function-name "$NAME"
aws lambda put-function-concurrency --region "$REGION" --function-name "$NAME" --reserved-concurrent-executions "$CONCURRENCY" >/dev/null ||
  echo "Note: couldn't reserve concurrency (new accounts have a low account limit); the function still works."

# the market-scan worker runs the same image (deploy/aws/scan.sh creates it)
if aws lambda get-function --region "$REGION" --function-name leveledge-scan >/dev/null 2>&1; then
  aws lambda update-function-code --region "$REGION" --function-name leveledge-scan --image-uri "$REPO:$TAG" >/dev/null
  echo "Updated the market-scan worker to the same image"
fi

# ---- 5. public URL ----------------------------------------------------------------------------
URL="$(aws lambda get-function-url-config --region "$REGION" --function-name "$NAME" --query FunctionUrl --output text 2>/dev/null ||
       aws lambda create-function-url-config --region "$REGION" --function-name "$NAME" --auth-type NONE --query FunctionUrl --output text)"
aws lambda add-permission --region "$REGION" --function-name "$NAME" --statement-id public-url --action lambda:InvokeFunctionUrl \
  --principal '*' --function-url-auth-type NONE >/dev/null 2>&1 || true
aws lambda add-permission --region "$REGION" --function-name "$NAME" --statement-id public-url-invoke --action lambda:InvokeFunction \
  --principal '*' --invoked-via-function-url >/dev/null 2>&1 || true
URL="${URL%/}"
echo "API: $URL"

# ---- 6. budget alert (optional) ----------------------------------------------------------------
if [ -n "${BUDGET_EMAIL:-}" ]; then
  aws budgets create-budget --account-id "$ACCOUNT" \
    --budget '{"BudgetName":"leveledge-5usd","BudgetLimit":{"Amount":"5","Unit":"USD"},"TimeUnit":"MONTHLY","BudgetType":"COST"}' \
    --notifications-with-subscribers "[{\"Notification\":{\"NotificationType\":\"ACTUAL\",\"ComparisonOperator\":\"GREATER_THAN\",\"Threshold\":50,\"ThresholdType\":\"PERCENTAGE\"},\"Subscribers\":[{\"SubscriptionType\":\"EMAIL\",\"Address\":\"$BUDGET_EMAIL\"}]}]" \
    2>/dev/null && echo "Budget alert: email at \$2.50 of a \$5/month budget" || echo "Budget already exists (or not permitted on this account)."
fi

# ---- 7. smoke test and Vercel rewrite ------------------------------------------------------------
for i in 1 2 3 4 5 6; do
  code="$(curl -s -o /dev/null -w '%{http_code}' -m 60 "$URL/api/health" || true)"
  [ "$code" = "200" ] && break; echo "waiting for the API (HTTP $code)"; sleep 10
done
[ "$code" = "200" ] || { echo "API did not answer /api/health" >&2; exit 1; }
python3 - "$URL" <<'PY'
import json, sys
json.dump({"$schema": "https://openapi.vercel.sh/vercel.json", "framework": "vite", "buildCommand": "npm run build",
           "outputDirectory": "dist",
           "rewrites": [{"source": "/api/:path*", "destination": sys.argv[1] + "/api/:path*"}]},
          open("web/vercel.json", "w"), indent=2)
PY
echo "web/vercel.json now sends /api/* to $URL. Deploy the frontend with:  cd web && vercel deploy --prod"
