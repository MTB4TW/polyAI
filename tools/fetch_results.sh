#!/usr/bin/env bash
# Copy evaluation results and experiment records from Farmshare into this repo.
#
# Usage (from your Mac, anywhere in the repo):
#   FARMSHARE_USER=<sunetid> tools/fetch_results.sh PATTERN
#
# Copies every outputs/evaluations/*PATTERN*/ folder and outputs/experiments/*PATTERN*.json
# file. Model weights are never copied; get those with tools/wandb_checkpoint.py.
#
# Optional: FARMSHARE_HOST (default rice.stanford.edu),
#           FARMSHARE_REPO (repo path on Farmshare, relative to home; default polyAI).
set -euo pipefail

if [ $# -ne 1 ] || [ -z "$1" ]; then
    echo "Usage: FARMSHARE_USER=<sunetid> $0 PATTERN" >&2
    exit 2
fi
PATTERN=$1
: "${FARMSHARE_USER:?Set FARMSHARE_USER to your SUNet ID}"
HOST=${FARMSHARE_HOST:-rice.stanford.edu}
REMOTE_REPO=${FARMSHARE_REPO:-polyAI}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
mkdir -p outputs/evaluations outputs/experiments

rsync -av --include="*${PATTERN}*/***" --exclude='*' \
    "$FARMSHARE_USER@$HOST:$REMOTE_REPO/outputs/evaluations/" outputs/evaluations/
rsync -av --include="*${PATTERN}*.json" --exclude='*' \
    "$FARMSHARE_USER@$HOST:$REMOTE_REPO/outputs/experiments/" outputs/experiments/ || true

echo "Fetched results matching '$PATTERN'. Review with git status, then commit them."
