#!/usr/bin/env bash
# Post a Markdown file as a comment on the "Paper book weekly report" issue
# (created on first use). Subscribe to that issue to get the reports and alerts.
#
#   scripts/post_report.sh <markdown-file>      needs GH_TOKEN with issues: write
set -euo pipefail

FILE="$1"
TITLE="Paper book weekly report"
[ -s "$FILE" ] || { echo "Nothing to post."; exit 0; }

num="$(gh issue list --state open --search "in:title \"$TITLE\"" --json number,title \
        --jq ".[] | select(.title == \"$TITLE\") | .number" | head -n1)"
if [ -z "$num" ]; then
  url="$(gh issue create --title "$TITLE" \
          --body "Weekly reports and failure alerts from the paper-trading job. Subscribe to this issue to be notified.")"
  num="${url##*/}"
fi
gh issue comment "$num" --body-file "$FILE"
echo "Posted to issue #$num."
