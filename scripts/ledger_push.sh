#!/usr/bin/env bash
# Append to the public `ledger` branch with optimistic concurrency.
#
#   bash scripts/ledger_push.sh "<commit message>" <ledger.py command and args...>
#
# Each attempt starts from the branch's latest commit, runs the command against that copy
# (appends are re-chained onto whatever the head is now, and are idempotent), then pushes.
# If another job pushed first, the push is rejected and the attempt repeats. History is
# only ever extended: no force pushes, no rewrites.
set -euo pipefail
msg="$1"; shift
root="$(git rev-parse --show-toplevel)"
wt="$root/.ledger-wt"

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

for attempt in 1 2 3 4 5; do
  rm -rf "$wt"; git worktree prune
  if git ls-remote --exit-code --heads origin ledger >/dev/null; then
    git fetch -q origin ledger
    git worktree add -q "$wt" origin/ledger
    git -C "$wt" checkout -q -B ledger
  else
    git worktree add -q --detach "$wt"
    git -C "$wt" checkout -q --orphan ledger
    git -C "$wt" rm -rfq .
    printf '# LevelEdge forecast ledger\n\nAppend-only, hash-chained forecasts in ledger.jsonl, committed before their outcomes and timestamped by freetsa.org.\n\nVerify: `python ledger.py verify https://raw.githubusercontent.com/krish1patel/LevelEdge/ledger/ledger.jsonl` with [ledger.py](https://github.com/krish1patel/LevelEdge/blob/main/src/leveledge/ledger.py).\n' > "$wt/README.md"
    python "$root/scripts/ledger.py" init --path "$wt/ledger.jsonl"
  fi
  python "$root/scripts/ledger.py" "$@" --path "$wt/ledger.jsonl"
  git -C "$wt" add -A
  if git -C "$wt" diff --cached --quiet; then
    echo "Nothing new for the ledger."; exit 0
  fi
  git -C "$wt" commit -qm "$msg"
  if git -C "$wt" push -q origin ledger; then
    echo "Ledger pushed (attempt $attempt)."; exit 0
  fi
  echo "Push rejected (another job appended first); retrying on the new head."
  sleep $((RANDOM % 10 + 5))
done
echo "Could not append to the ledger after 5 attempts." >&2
exit 1
