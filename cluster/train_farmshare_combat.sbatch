#!/usr/bin/env bash
#SBATCH --job-name=polytopia_combat
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ==============================================================================
# Stanford Farmshare SLURM Job Script: Polytopia Phase 2 Combat Training
# 1v1 Matches against Bot Sparring Partner (SimpleAgent) on 11x11 Drylands
# ==============================================================================

set -euo pipefail
mkdir -p logs runs

echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Started at: $(date)"

# 1. Load Cluster Modules & Local Runtime
if command -v module &> /dev/null; then
    module load cuda 2>/dev/null || true
fi

# Detect local JDK in .runtime if present (use absolute path)
LOCAL_JDK_BIN=$(find "$PWD/.runtime" -maxdepth 3 -name "javac" -exec dirname {} \; 2>/dev/null | head -n 1 || true)
if [ -n "$LOCAL_JDK_BIN" ]; then
    export PATH="$LOCAL_JDK_BIN:$PATH"
    export JAVA_HOME="$(dirname "$LOCAL_JDK_BIN")"
fi

# 2. Activate Virtual Environment
source .venv/bin/activate

# 3. Report the device (GPU if this node has one, otherwise CPU)
DEVICE=$(python -c 'import torch; print("GPU" if torch.cuda.is_available() else "CPU")')
python -c '
import torch
print("PyTorch CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("Device count:", torch.cuda.device_count())
'

# 4. Phase 2 Combat Environment Variables
export POLYVISION_PHASE="2"
export POLYVISION_COMBAT_MODE="1"
export POLYVISION_SOLO_NO_OPPONENT_MODE="0"
export POLYVISION_LEVEL_POOL_GLOB="levels/phase2_pool_1v1/train/*.csv"
export POLYVISION_MAX_TURNS="${POLYVISION_MAX_TURNS:-20}"
export POLYVISION_INFO_MODE="fast"
# Combat reward weights (see docs/rewards.md); set any of these before sbatch to override:
#   POLYVISION_COMBAT_REWARD_WIN=10 POLYVISION_COMBAT_REWARD_LOSS=-10 POLYVISION_COMBAT_REWARD_CITY=1.0
#   POLYVISION_COMBAT_REWARD_SCORE_PER_1000=1.0 POLYVISION_COMBAT_REWARD_KILL=0.25
env | grep '^POLYVISION_COMBAT_REWARD_' || echo "Combat reward: defaults"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export PYTHONUNBUFFERED="1"

# 5. Training Parameters
NUM_ENVS=${NUM_ENVS:-16}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-2000000}  # Initial budget; validate learning before scaling
EXP_NAME=${EXP_NAME:-"farmshare_combat_resnet_2m"}
NETWORK_ARCH=${NETWORK_ARCH:-"spatial_resnet"}

# W&B curves are on by default for combat runs (TRACK=0 to turn off). Run `wandb login` once
# on this account; WANDB_ENTITY (e.g. polyai-stanford) picks the team.
TRACK_ARGS=()
if [ "${TRACK:-1}" = "1" ]; then
    TRACK_ARGS=(--track --wandb-project-name "${WANDB_PROJECT:-polyAI}")
fi

echo "Starting Phase 2 Combat training: $TOTAL_TIMESTEPS steps with $NUM_ENVS parallel environments on $DEVICE ($NETWORK_ARCH)..."

python py_rl/cleanrl/cleanrl/ppo.py \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --num-envs "$NUM_ENVS" \
    --num-steps 128 \
    --ent-coef 0.02 \
    --cuda \
    --save-model \
    --save-frequency 500000 \
    --exp-name "$EXP_NAME" \
    --network-arch "$NETWORK_ARCH" \
    --validation-states 10000 \
    ${TRACK_ARGS[@]+"${TRACK_ARGS[@]}"}

echo "Combat Training finished at: $(date)"
