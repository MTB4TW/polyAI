#!/usr/bin/env bash
# One segment of a long combat league run made of chained short runs (ppo.py has no resume).
#
# Segment K warm-starts with --init-from from segment K-1's final model (segment 1 from BASE_EXP's final model)
# and adds earlier models to the league as fixed opponents: BASE_EXP's final model and its
# model_checkpoint_${BASE_CHECKPOINT_STEP}, plus every earlier segment's final model and model_checkpoint_1000000.
# The LR is constant (--no-anneal-lr), so it does not saw-tooth across segments. Adam state, PFSP counts and own
# snapshots restart every segment; the privileged critic and its PopArt statistics carry over through --init-from.
#
# Usage (from the repo root on Farmshare; LEAGUE_ENV_FILE sets LEAGUE_GLOB, the swap and snapshot settings and
# EXTRA_PPO_ARGS, then sources cluster/combat_selfplay_v2.env, and must honour INIT_FROM / LEAGUE_GLOB_EXTRA):
#   J=$(sbatch --parsable --cpus-per-task=32 --mem=48G --time=36:00:00 cluster/combat_league_segment.sh 1)
#   for k in 2 3 4; do
#     J=$(sbatch --parsable --dependency=afterok:$J --cpus-per-task=32 --mem=48G --time=36:00:00 \
#         cluster/combat_league_segment.sh $k)
#   done
# Each segment still needs its canonical evaluation (cluster/eval_combat_farmshare.sbatch).
# Defaults reproduce combat_pcritic_t1_league8m_seg1..4 (docs/experiments.md, 2026-11-20).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
K=${1:?usage: combat_league_segment.sh K}
BASE_EXP=${BASE_EXP:-combat_pcritic_t1_league_swap_4m}
BASE_CHECKPOINT_STEP=${BASE_CHECKPOINT_STEP:-2000000}
SEGMENT_PREFIX=${SEGMENT_PREFIX:-combat_pcritic_t1_league8m_seg}
SEGMENT_STEPS=${SEGMENT_STEPS:-2000000}
LEAGUE_ENV_FILE=${LEAGUE_ENV_FILE:-logs/league_env.sh}
SEED=${SEED:-1}
name() { echo "${SEGMENT_PREFIX}$1"; }

RBASE=$(ls -d "$PWD"/runs/Tribes-v0__${BASE_EXP}__${SEED}__* | tail -1)
EXTRA="$RBASE/$BASE_EXP.cleanrl_model:$RBASE/model_checkpoint_${BASE_CHECKPOINT_STEP}.cleanrl_model"
if [ "$K" -eq 1 ]; then
  export INIT_FROM=$RBASE/$BASE_EXP.cleanrl_model
else
  PREV=$(name $((K-1)))
  export INIT_FROM=$(ls -d "$PWD"/runs/Tribes-v0__${PREV}__${SEED}__*/${PREV}.cleanrl_model | tail -1)
  for j in $(seq 1 $((K-1))); do
    N=$(name "$j")
    EXTRA="$EXTRA:$PWD/runs/Tribes-v0__${N}__${SEED}__*/${N}.cleanrl_model:$PWD/runs/Tribes-v0__${N}__${SEED}__*/model_checkpoint_1000000.cleanrl_model"
  done
fi
export LEAGUE_GLOB_EXTRA="$EXTRA"
export LEAGUE_SIMPLE_PROB=${LEAGUE_SIMPLE_PROB:-0}   # league games only (the 8M run); 0.5 mixes in SimpleAgent
source "$LEAGUE_ENV_FILE"
export EXTRA_PPO_ARGS="$EXTRA_PPO_ARGS --no-anneal-lr"
echo "[LEAGUE_SEGMENT] segment $K INIT_FROM=$INIT_FROM"
echo "[LEAGUE_SEGMENT] SIMPLE_PROB=$POLYVISION_COMBAT_LEAGUE_SIMPLE_PROB SWAP=$POLYVISION_COMBAT_LEAGUE_SWAP_PROB EXTRA_PPO_ARGS=$EXTRA_PPO_ARGS"
EXP_NAME=$(name "$K") SEED=$SEED TOTAL_TIMESTEPS=$SEGMENT_STEPS exec bash cluster/train_combat_cpu.sbatch
