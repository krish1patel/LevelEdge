#!/usr/bin/env bash
# Pause or resume the LevelEdge API on AWS Lambda.
#
#   bash deploy/aws/pause.sh          # pause: the API refuses every request, so it can't cost anything
#   bash deploy/aws/pause.sh resume   # resume: back to at most 5 copies at once
#
# Paused, the dashboard still loads from Vercel but its data panels show errors. Only the
# stored image (a few cents a month) is still billed; deleting the function and the ECR
# repository removes that too.
set -euo pipefail
REGION="${AWS_REGION:-us-east-1}"
NAME="${NAME:-leveledge-api}"
if [ "${1:-pause}" = "resume" ]; then
  aws lambda put-function-concurrency --region "$REGION" --function-name "$NAME" --reserved-concurrent-executions "${CONCURRENCY:-5}" >/dev/null
  echo "Resumed: $NAME serves requests again."
else
  aws lambda put-function-concurrency --region "$REGION" --function-name "$NAME" --reserved-concurrent-executions 0 >/dev/null
  echo "Paused: $NAME refuses all requests (resume with: bash deploy/aws/pause.sh resume)."
fi
