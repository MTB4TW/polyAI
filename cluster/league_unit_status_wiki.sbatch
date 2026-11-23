#!/usr/bin/env bash
#SBATCH --job-name=league_us_wiki
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=47:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --signal=B:USR1@600

# League run with the unit-status inputs (CHANGELOG 087) on the wiki-parity map pool (CHANGELOG 086),
# warm-started from Devin's league8m segment 1 at 1.2M steps (5.2M steps of lineage from combat_pcritic_t1_league_swap_4m).
#
# The league is Devin's league8m setup: 9 fixed earlier opponents plus the 4M run and its continuation's snapshots,
# the learner's own snapshots every 200k (newest 20 kept), seat swap 0.5, no SimpleAgent games, no LR annealing.
# Old checkpoints have no unit-status inputs: they play through the without_unit_status adapter (combat_selfplay).
# Fixed opponents are copied into runs/league_us_opponents so the run does not depend on another user's files.
#
# One long run: on the time limit SIGUSR1 saves a resume bundle and AUTO_RESUME resubmits this script.
# Stop for good:  touch runs/Tribes-v0__${EXP_NAME}__1__*/STOP
# Usage (from the repo root): sbatch cluster/league_unit_status_wiki.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export EXP_NAME=${EXP_NAME:-league_us_wiki_8m}
export SEED=${SEED:-1}
export TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-8000000}
export NUM_ENVS=${NUM_ENVS:-16}
export TORCH_THREADS=${TORCH_THREADS:-8}
export POLYVISION_RULES=polytopia_2_15
export POLYVISION_UNIT_STATUS=1
export POLYVISION_COMBAT_POOL=phase2_pool_1v1_wiki
export NETWORK_ARCH=spatial_resnet_composed

OPP=$PWD/runs/league_us_opponents
INIT=${INIT_CHECKPOINT:-$PWD/runs/init_league8m_seg1_1200000/model_checkpoint_1200000.cleanrl_model}
if [ -z "${RESUME_FROM:-}" ]; then
    export INIT_FROM="$INIT"  # a resume restores everything, so --init-from is only for the first job
fi
export LEAGUE_GLOB="$OPP/*.cleanrl_model"
export LEAGUE_SIMPLE_PROB=0
export LEAGUE_SWAP_PROB=0.5
export LEAGUE_MAX_SELF=20
export LEAGUE_SNAPSHOT_EVERY=200000
export EXTRA_PPO_ARGS="--privileged-critic --critic-warmup-steps 0 --learning-rate 2.5e-4 --target-kl 0.02 --ent-coef 0.01"
source cluster/combat_selfplay_v2.env
export EXTRA_PPO_ARGS="$EXTRA_PPO_ARGS --no-anneal-lr"
# OVERRIDE_ENV="K=V K=V": settings applied after the shared env files, which set the reward weights
# unconditionally (e.g. OVERRIDE_ENV="POLYVISION_COMBAT_REWARD_ROAD=0.3 POLYVISION_COMBAT_REWARD_ROADS_TECH=1").
for kv in ${OVERRIDE_ENV:-}; do export "$kv"; echo "[LEAGUE_US] override: $kv"; done
# EXTRA_PPO_ARGS_APPEND="--learning-rate 1e-4": trainer arguments added last (the last value of a flag wins).
export EXTRA_PPO_ARGS="$EXTRA_PPO_ARGS ${EXTRA_PPO_ARGS_APPEND:-}"

export AUTO_RESUME=${AUTO_RESUME:-1}
export TRAIN_SCRIPT=cluster/league_unit_status_wiki.sbatch
echo "[LEAGUE_US] INIT_FROM=${INIT_FROM:-<resume>} RESUME_FROM=${RESUME_FROM:-} pool=$POLYVISION_COMBAT_POOL unit_status=$POLYVISION_UNIT_STATUS"
echo "[LEAGUE_US] opponents: $(ls $OPP/*.cleanrl_model | wc -l) fixed; EXTRA_PPO_ARGS=$EXTRA_PPO_ARGS"
exec bash cluster/train_combat_cpu.sbatch
