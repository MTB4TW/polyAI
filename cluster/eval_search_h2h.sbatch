#!/usr/bin/env bash
#SBATCH --job-name=search_h2h
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Head-to-head games of a search variant against a frozen checkpoint (tools/evaluate_search_h2h.py), CPU only.
#   PLAYER_MODEL=... MODE=order OPPONENT_MODEL=... EVAL_ID=... sbatch cluster/eval_search_h2h.sbatch
# Optional: REPEATS (default 2), MAX_MAPS.

set -euo pipefail
if [ ! -f tools/evaluate_search_h2h.py ]; then
    echo "Submit this job from the repository root." >&2
    exit 1
fi
mkdir -p logs
: "${PLAYER_MODEL:?}" "${MODE:?}" "${OPPONENT_MODEL:?}" "${EVAL_ID:?}"
echo "Job ID: ${SLURM_JOB_ID:-local} on $(hostname), started $(date)"
LOCAL_JDK_BIN=$(find -L "$PWD/.runtime" -maxdepth 3 -name "javac" -exec dirname {} \; 2>/dev/null | head -n 1 || true)
if [ -n "$LOCAL_JDK_BIN" ]; then
    export PATH="$LOCAL_JDK_BIN:$PATH"
    export JAVA_HOME="$(dirname "$LOCAL_JDK_BIN")"
fi
source .venv/bin/activate
export PYTHONUNBUFFERED="1"
ARGS=(--player-model "$PLAYER_MODEL" --mode "$MODE" --opponent-model "$OPPONENT_MODEL" --evaluation-id "$EVAL_ID"
      --repeats-per-map "${REPEATS:-2}")
if [ -n "${MAX_MAPS:-}" ]; then ARGS+=(--max-maps "$MAX_MAPS"); fi
python tools/evaluate_search_h2h.py "${ARGS[@]}"
echo "Finished at $(date)"
