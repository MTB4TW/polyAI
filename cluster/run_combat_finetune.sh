#!/usr/bin/env bash
# Fade the shaping out of a trained combat checkpoint in two dependent Farmshare jobs (CHANGELOG 063):
#   stage 1: shaping x 0.5 for STAGE1_STEPS (default 500k), from CHECKPOINT;
#   stage 2: outcome only (shaping x 0) for STAGE2_STEPS (default 1.5M), from stage 1's final checkpoint,
#            keeping stage 1's critic (stage 1 resets it: its reward differs most from the original).
# Usage (repo root):  cluster/run_combat_finetune.sh runs/X/X.cleanrl_model NAME [SEED]
# Do not start it before the runs it fine-tunes have finished.
set -euo pipefail
CHECKPOINT=${1:?checkpoint path}
NAME=${2:?run name prefix}
SEED=${3:-1}
[ -f "$CHECKPOINT" ] && [ -f "$CHECKPOINT.action_interface.json" ] || { echo "Missing $CHECKPOINT or its sidecar" >&2; exit 1; }
mkdir -p logs
J1=$(INIT_FROM="$CHECKPOINT" SHAPING_SCALE=0.5 EXP_NAME="${NAME}_shape05" TOTAL_TIMESTEPS="${STAGE1_STEPS:-500000}" SEED="$SEED" \
     sbatch --parsable --export=ALL cluster/combat_finetune_stage.sbatch)
J2=$(INIT_FROM_EXP="${NAME}_shape05" SHAPING_SCALE=0 RESET_CRITIC=0 EXP_NAME="${NAME}_outcome" TOTAL_TIMESTEPS="${STAGE2_STEPS:-1500000}" SEED="$SEED" \
     sbatch --parsable --export=ALL --dependency=afterok:"$J1" cluster/combat_finetune_stage.sbatch)
echo "stage 1 (shaping x0.5): job $J1  -> run ${NAME}_shape05"
echo "stage 2 (outcome only): job $J2  -> run ${NAME}_outcome (starts after $J1 succeeds)"
