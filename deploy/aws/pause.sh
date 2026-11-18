#!/usr/bin/env bash
# Pause or resume the LevelEdge API on AWS Lambda.
#
#   bash deploy/aws/pause.sh          # pause: the API refuses every request and the market scan stops, so neither can cost anything
#   bash deploy/aws/pause.sh resume   # resume: back to at most 5 copies at once
#
# Paused, the dashboard still loads from Vercel but its data panels show errors. Only the
# stored image (a few cents a month) is still billed; deleting the function and the ECR
# repository removes that too.
set -euo pipefail
REGION="${AWS_REGION:-us-east-1}"
NAME="${NAME:-leveledge-api}"
SCAN="leveledge-scan"
has_scan() { aws lambda get-function --region "$REGION" --function-name "$SCAN" >/dev/null 2>&1; }
if [ "${1:-pause}" = "resume" ]; then
  aws lambda put-function-concurrency --region "$REGION" --function-name "$NAME" --reserved-concurrent-executions "${CONCURRENCY:-5}" >/dev/null
  has_scan && aws lambda put-function-concurrency --region "$REGION" --function-name "$SCAN" --reserved-concurrent-executions 6 >/dev/null
  echo "Resumed: $NAME serves requests again${SCAN:+ and the market scan runs}."
else
  aws lambda put-function-concurrency --region "$REGION" --function-name "$NAME" --reserved-concurrent-executions 0 >/dev/null
  # the scan worker too: queued batches wait (up to a day) instead of running
  has_scan && aws lambda put-function-concurrency --region "$REGION" --function-name "$SCAN" --reserved-concurrent-executions 0 >/dev/null
  echo "Paused: $NAME refuses all requests and the market scan stops (resume with: bash deploy/aws/pause.sh resume)."
fi
