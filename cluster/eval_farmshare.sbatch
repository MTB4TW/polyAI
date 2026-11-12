#!/usr/bin/env bash
#SBATCH --job-name=polytopia_eval
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ==============================================================================
# Stanford Farmshare SLURM Job Script: canonical Phase 1 evaluation (CPU only)
#
# Runs the canonical 250-map suite (4 policies, 5 repeats, seed 42), so results
# are paired map-by-map with every other canonical validation evaluation.
#
# Usage (from the repo root):
#   MODEL_PATH=runs/experiments/NAME/NAME.cleanrl_model \
#   EVAL_ID=20261112_NAME_validation_canonical \
#   sbatch cluster/eval_farmshare.sbatch
#
# Optional: POOL=validation (default) or human_benchmark. The test pool is
# refused here on purpose; run it by hand with --confirm-test.
# No partition is set, so SLURM uses the cluster default; override with -p.
# ==============================================================================

set -euo pipefail

if [ ! -f tools/evaluate_phase1.py ]; then
    echo "Submit this job from the repository root." >&2
    exit 1
fi
mkdir -p logs

: "${MODEL_PATH:?Set MODEL_PATH to a .cleanrl_model checkpoint}"
POOL=${POOL:-validation}
if [ "$POOL" = "test" ]; then
    echo "Refusing to evaluate the test pool from a batch job; run it by hand with --confirm-test." >&2
    exit 2
fi
if [ ! -f "$MODEL_PATH" ]; then
    echo "Checkpoint not found: $MODEL_PATH" >&2
    exit 1
fi
if [ ! -f "$MODEL_PATH.action_interface.json" ]; then
    echo "Sidecar not found: $MODEL_PATH.action_interface.json (the evaluator requires it)" >&2
    exit 1
fi

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Running on node: $(hostname)"
echo "Started at: $(date)"

LOCAL_JDK_BIN=$(find "$PWD/.runtime" -maxdepth 3 -name "javac" -exec dirname {} \; 2>/dev/null | head -n 1 || true)
if [ -n "$LOCAL_JDK_BIN" ]; then
    export PATH="$LOCAL_JDK_BIN:$PATH"
    export JAVA_HOME="$(dirname "$LOCAL_JDK_BIN")"
fi
source .venv/bin/activate
export PYTHONUNBUFFERED="1"

EVAL_ARGS=(--model-path "$MODEL_PATH" --pool "$POOL" --suite full --repeats-per-map 5 --seed 42)
if [ -n "${EVAL_ID:-}" ]; then
    EVAL_ARGS+=(--evaluation-id "$EVAL_ID")
fi

python tools/evaluate_phase1.py "${EVAL_ARGS[@]}"

echo "Evaluation finished at: $(date)"
