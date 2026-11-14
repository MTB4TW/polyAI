#!/usr/bin/env bash
#SBATCH --job-name=polytopia_combat_ft
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# One stage of cluster/run_combat_finetune.sh. Resolves the checkpoint when the job starts (the previous stage's
# run folder is only known then), then runs cluster/train_combat_cpu.sbatch with the fine-tune settings.
#   INIT_FROM=path   or   INIT_FROM_EXP=<EXP_NAME of an earlier run in runs/>   (latest matching run)
#   SHAPING_SCALE, EXP_NAME, TOTAL_TIMESTEPS, SEED as for the training job.
set -euo pipefail
if [ -z "${INIT_FROM:-}" ]; then
    : "${INIT_FROM_EXP:?Set INIT_FROM or INIT_FROM_EXP}"
    INIT_FROM=$(ls -t runs/Tribes-v0__"${INIT_FROM_EXP}"__*/"${INIT_FROM_EXP}".cleanrl_model 2>/dev/null | head -n 1 || true)
    if [ -z "$INIT_FROM" ]; then
        echo "No final checkpoint for run ${INIT_FROM_EXP} under runs/" >&2
        exit 1
    fi
fi
export INIT_FROM
echo "Fine-tuning from $INIT_FROM with shaping scale ${SHAPING_SCALE:-0}"
source cluster/combat_outcome_finetune.env
exec bash cluster/train_combat_cpu.sbatch
