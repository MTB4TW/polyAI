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
# Optional: WANDB_LOG_EVAL=1 copies the results to W&B afterwards (see the end of this file).
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

# Optional: copy the results to W&B (git stays the source of truth).
#   WANDB_LOG_EVAL=1       log this evaluation as a W&B run (needs EVAL_ID)
#   REFERENCE_EVAL=...     evaluation to compare against (default: current MLP reference)
#   CHECKPOINT_ARTIFACT=.. W&B checkpoint artifact to link, e.g. NAME or NAME:v0
if [ "${WANDB_LOG_EVAL:-0}" = "1" ]; then
    if [ -z "${EVAL_ID:-}" ]; then
        echo "warning: WANDB_LOG_EVAL=1 needs EVAL_ID; skipping the W&B log" >&2
    else
        LOG_ARGS=(--reference "${REFERENCE_EVAL:-20260827_phase1_v5_parity002_seed3_16m_terminal_spt_validation_canonical}")
        if [ -n "${CHECKPOINT_ARTIFACT:-}" ]; then
            LOG_ARGS+=(--checkpoint-artifact "$CHECKPOINT_ARTIFACT")
        fi
        python tools/wandb_log_eval.py "$EVAL_ID" "${LOG_ARGS[@]}" \
            || echo "warning: W&B evaluation log failed; results are still in outputs/evaluations/$EVAL_ID" >&2
    fi
fi
