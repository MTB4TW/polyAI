#!/usr/bin/env bash
#SBATCH --job-name=polytopia_combat_cpu
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ==============================================================================
# Stanford Farmshare SLURM Job Script: short Phase 2 combat run on CPU only
#
# 1v1 vs SimpleAgent on the 11x11 Drylands pool, shaped combat reward (docs/rewards.md).
# CPU nodes start much sooner than the GPU queue. Defaults target a ~1M-step first look.
#
# Usage (from the repo root):
#   EXP_NAME=combat_cpu_1m sbatch cluster/train_combat_cpu.sbatch
# Optional overrides (environment variables):
#   TOTAL_TIMESTEPS=1000000  NUM_ENVS=8  NETWORK_ARCH=spatial_resnet  SEED=1
#   TORCH_THREADS=8          VALIDATION_STATES=10000  TRACK=1 (W&B curves)
#   EXTRA_PPO_ARGS="--learning-rate 3e-4"   appended last, so it overrides the defaults below
#   POLYVISION_COMBAT_REWARD_*               reward weights (see docs/rewards.md)
#   POLYVISION_RULES=polytopia_2_15          game rules (docs/rules-parity.md; default tribes_2020)
# ==============================================================================

set -euo pipefail
if [ ! -f py_rl/cleanrl/cleanrl/ppo.py ]; then
    echo "Submit this job from the repository root." >&2
    exit 1
fi
mkdir -p logs runs

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Running on node: $(hostname)"
echo "Started at: $(date)"

LOCAL_JDK_BIN=$(find "$PWD/.runtime" -maxdepth 3 -name "javac" -exec dirname {} \; 2>/dev/null | head -n 1 || true)
if [ -n "$LOCAL_JDK_BIN" ]; then
    export PATH="$LOCAL_JDK_BIN:$PATH"
    export JAVA_HOME="$(dirname "$LOCAL_JDK_BIN")"
fi
source .venv/bin/activate

# Combat environment
export POLYVISION_PHASE="2"
export POLYVISION_COMBAT_MODE="1"
export POLYVISION_SOLO_NO_OPPONENT_MODE="0"
export POLYVISION_LEVEL_POOL_GLOB="levels/phase2_pool_1v1/train/*.csv"
export POLYVISION_MAX_TURNS="${POLYVISION_MAX_TURNS:-20}"
export POLYVISION_INFO_MODE="fast"
export POLYVISION_RULES="${POLYVISION_RULES:-tribes_2020}"
echo "Game rules: $POLYVISION_RULES"
env | grep '^POLYVISION_COMBAT_REWARD_' || echo "Combat reward: defaults"
# Env worker processes stay single-threaded; the learner gets TORCH_THREADS via --torch-threads.
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export PYTHONUNBUFFERED="1"

NUM_ENVS=${NUM_ENVS:-8}                  # ~2 cores per env: each env is a Python worker plus a JVM
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-1000000}
EXP_NAME=${EXP_NAME:-"combat_cpu_1m"}
NETWORK_ARCH=${NETWORK_ARCH:-"spatial_resnet"}
SEED=${SEED:-1}
TORCH_THREADS=${TORCH_THREADS:-8}
VALIDATION_STATES=${VALIDATION_STATES:-10000}

TRACK_ARGS=()
if [ "${TRACK:-1}" = "1" ]; then
    TRACK_ARGS=(--track --wandb-project-name "${WANDB_PROJECT:-polyAI}")
fi
read -r -a EXTRA_ARGS <<< "${EXTRA_PPO_ARGS:-}"

echo "Starting Phase 2 combat training on CPU: $TOTAL_TIMESTEPS steps, $NUM_ENVS envs, $NETWORK_ARCH, seed $SEED, $TORCH_THREADS learner threads"

# PPO settings for combat (vs. the Phase 1 defaults):
#   --gamma-per-turn          discount once per turn, not per action (gamma 0.99 per turn ~ 0.82 over 20 turns)
#   --no-clip-vloss           the +/-0.2 value clip is far too tight for returns in the tens
#   --learning-rate 5e-4, --update-epochs 8
#                             the smoke run's updates were ~10x smaller than typical (approx_kl ~0.002)
#   --target-kl 0.03          stop an update early if the policy moves too far
python py_rl/cleanrl/cleanrl/ppo.py \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --num-envs "$NUM_ENVS" \
    --num-steps 128 \
    --seed "$SEED" \
    --gamma 0.99 \
    --gamma-per-turn \
    --no-clip-vloss \
    --learning-rate 5e-4 \
    --update-epochs 8 \
    --target-kl 0.03 \
    --ent-coef 0.02 \
    --torch-threads "$TORCH_THREADS" \
    --save-model \
    --save-frequency 250000 \
    --exp-name "$EXP_NAME" \
    --network-arch "$NETWORK_ARCH" \
    --validation-states "$VALIDATION_STATES" \
    ${TRACK_ARGS[@]+"${TRACK_ARGS[@]}"} \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "Combat training finished at: $(date)"
