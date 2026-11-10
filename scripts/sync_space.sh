#!/usr/bin/env bash
# Push the current checkout to a HuggingFace Space as one snapshot commit.
#
#   scripts/sync_space.sh <space-git-url> [branch]
#
# The snapshot is the working tree's tracked files minus binaries the Space
# refuses, committed on top of the Space's existing branch (fast-forward).
set -euo pipefail

SPACE_URL="$1"
BRANCH="${2:-main}"
# binaries the Space rejects; none of them are needed to build or run the app
EXCLUDE_RE='\.(gz|zip|png|jpe?g|gif|webp|parquet|pkl|npy|pdf)$|(^|/)\.DS_Store$'

git remote remove space >/dev/null 2>&1 || true
git remote add space "$SPACE_URL"

parent_args=()
if git fetch --quiet --depth=1 space "$BRANCH" 2>/dev/null; then
  parent_args=(-p FETCH_HEAD)
fi

# build the snapshot tree in a throwaway index so the checkout is untouched
export GIT_INDEX_FILE="$(mktemp -d)/index"
git read-tree HEAD
excluded="$(git ls-files | grep -E "$EXCLUDE_RE" || true)"
if [ -n "$excluded" ]; then
  echo "Leaving out of the Space:"
  echo "$excluded" | sed 's/^/  /'
  echo "$excluded" | tr '\n' '\0' | xargs -0 git rm --cached --quiet --
fi
tree="$(git write-tree)"
unset GIT_INDEX_FILE

if [ "${#parent_args[@]}" -gt 0 ] && [ "$(git rev-parse FETCH_HEAD^{tree})" = "$tree" ]; then
  echo "Space already matches $(git rev-parse --short HEAD); nothing to push."
  exit 0
fi

export GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-LevelEdge sync}" GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-sync@leveledge.invalid}"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME" GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
commit="$(git commit-tree "$tree" "${parent_args[@]}" -m "Sync from GitHub $(git rev-parse --short HEAD)")"
git push space "$commit:refs/heads/$BRANCH"
echo "Pushed snapshot $commit of $(git rev-parse --short HEAD)."
