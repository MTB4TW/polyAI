#!/usr/bin/env bash
#SBATCH --job-name=polytopia_combat_eval
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ==============================================================================
# Stanford Farmshare SLURM Job Script: canonical combat evaluation (CPU only)
#
# Runs tools/evaluate_combat.py on the 120-map eval_validation pool (3 policies, 5 replicates, seed 42) against
# SimpleAgent, so results are paired map by map with every other canonical combat evaluation
# (tools/compare_combat_evals.py). Rules and turn limit come from the checkpoint.
#
# Usage (from the repo root):
#   MODEL_PATH=runs/NAME/NAME.cleanrl_model \
#   EVAL_ID=20261114_NAME_combat_eval_validation_canonical \
#   sbatch cluster/eval_combat_farmshare.sbatch
#
# The held-out eval_test pool is refused here on purpose; run it by hand with --confirm-test, and only when a
# person asks. No partition is set, so SLURM uses the cluster default; override with -p.
# ==============================================================================

set -euo pipefail

if [ ! -f tools/evaluate_combat.py ]; then
    echo "Submit this job from the repository root." >&2
    exit 1
fi
mkdir -p logs

: "${MODEL_PATH:?Set MODEL_PATH to a combat .cleanrl_model checkpoint}"
if [ ! -f "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH.action_interface.json" ]; then
    echo "Checkpoint or sidecar not found: $MODEL_PATH (+ .action_interface.json)" >&2
    exit 1
fi

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Running on node: $(hostname)"
echo "Started at: $(date)"

# -L: .runtime may be a symlink in git worktrees.
LOCAL_JDK_BIN=$(find -L "$PWD/.runtime" -maxdepth 3 -name "javac" -exec dirname {} \; 2>/dev/null | head -n 1 || true)
if [ -n "$LOCAL_JDK_BIN" ]; then
    export PATH="$LOCAL_JDK_BIN:$PATH"
    export JAVA_HOME="$(dirname "$LOCAL_JDK_BIN")"
fi
source .venv/bin/activate
export PYTHONUNBUFFERED="1"

EVAL_ARGS=(--model-path "$MODEL_PATH" --pool eval_validation --repeats-per-map 5 --seed 42)
if [ -n "${EVAL_ID:-}" ]; then
    EVAL_ARGS+=(--evaluation-id "$EVAL_ID")
fi

python tools/evaluate_combat.py "${EVAL_ARGS[@]}"

echo "Evaluation finished at: $(date)"
