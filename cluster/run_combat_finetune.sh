#!/usr/bin/env bash
# Outcome-only fine-tuning of a trained combat checkpoint in one Farmshare job (CHANGELOG 063): every shaping term
# off (shaping x 0) for STEPS (default 2M), from CHECKPOINT, with a fresh critic (its values were in shaped-reward
# units) and learning rate 2e-4. The policy is trained on the game result only.
# Usage (repo root):  cluster/run_combat_finetune.sh runs/X/X.cleanrl_model NAME [SEED]
# Do not start it before the run it fine-tunes has finished.
set -euo pipefail
CHECKPOINT=${1:?checkpoint path}
NAME=${2:?run name}
SEED=${3:-1}
[ -f "$CHECKPOINT" ] && [ -f "$CHECKPOINT.action_interface.json" ] || { echo "Missing $CHECKPOINT or its sidecar" >&2; exit 1; }
mkdir -p logs
J=$(INIT_FROM="$CHECKPOINT" SHAPING_SCALE=0 RESET_CRITIC=1 EXP_NAME="$NAME" TOTAL_TIMESTEPS="${STEPS:-2000000}" SEED="$SEED" \
    sbatch --parsable --export=ALL cluster/combat_finetune_stage.sbatch)
echo "outcome-only fine-tune: job $J  -> run $NAME"
