#!/usr/bin/env bash
#SBATCH --job-name=polytopia_combat_h2h
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ==============================================================================
# Stanford Farmshare SLURM Job Script: head-to-head combat evaluation (CPU only)
#
# Runs tools/evaluate_combat_h2h.py: checkpoint A against checkpoint B on the 120-map eval_validation pool,
# 5 replicates in each seat order, seed 42, sampled actions. Results are from A's point of view.
#
# Usage (from the repo root):
#   MODEL_A=runs/NEW/NEW.cleanrl_model MODEL_B=runs/REF/REF.cleanrl_model \
#   EVAL_ID=20261117_NEW_vs_REF_combat_h2h_eval_validation \
#   sbatch cluster/eval_combat_h2h_farmshare.sbatch
#
# The held-out eval_test pool is refused here on purpose.
# ==============================================================================

set -euo pipefail

if [ ! -f tools/evaluate_combat_h2h.py ]; then
    echo "Submit this job from the repository root." >&2
    exit 1
fi
mkdir -p logs

: "${MODEL_A:?Set MODEL_A to a combat .cleanrl_model checkpoint}"
: "${MODEL_B:?Set MODEL_B to a combat .cleanrl_model checkpoint}"
for m in "$MODEL_A" "$MODEL_B"; do
    if [ ! -f "$m" ] || [ ! -f "$m.action_interface.json" ]; then
        echo "Checkpoint or sidecar not found: $m (+ .action_interface.json)" >&2
        exit 1
    fi
done

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Running on node: $(hostname)"
echo "Started at: $(date)"

LOCAL_JDK_BIN=$(find -L "$PWD/.runtime" -maxdepth 3 -name "javac" -exec dirname {} \; 2>/dev/null | head -n 1 || true)
if [ -n "$LOCAL_JDK_BIN" ]; then
    export PATH="$LOCAL_JDK_BIN:$PATH"
    export JAVA_HOME="$(dirname "$LOCAL_JDK_BIN")"
fi
source .venv/bin/activate
export PYTHONUNBUFFERED="1"

H2H_ARGS=(--model-a "$MODEL_A" --model-b "$MODEL_B" --pool eval_validation --repeats-per-map 5 --seed 42)
if [ -n "${EVAL_ID:-}" ]; then
    H2H_ARGS+=(--evaluation-id "$EVAL_ID")
fi

python tools/evaluate_combat_h2h.py "${H2H_ARGS[@]}"

echo "Head-to-head evaluation finished at: $(date)"
