#!/usr/bin/env bash
#SBATCH --job-name=polytopia_ai
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ==============================================================================
# Stanford Farmshare SLURM Job Script: Polytopia AI Training
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

# 4. Environment Variables for Maximum Simulation Throughput
export POLYVISION_LEVEL_POOL_GLOB="levels/phase1_pool_bardur_real/train/*.csv"
export POLYVISION_SOLO_NO_OPPONENT_MODE="1"
export POLYVISION_PHASE="1"
export POLYVISION_COMBAT_MODE="0"
export POLYVISION_INFO_MODE="fast"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export PYTHONUNBUFFERED="1"

# 5. Training Parameters
# 16 CPU cores allocated -> 16 parallel game environments running simultaneously
NUM_ENVS=${NUM_ENVS:-16}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-5000000}  # 5 Million steps
EXP_NAME=${EXP_NAME:-"farmshare_resnet_5m"}
NETWORK_ARCH=${NETWORK_ARCH:-"spatial_resnet"}
SEED=${SEED:-1}

# Optional (set by cluster/run_experiment.sh):
#   MODEL_PATH     final model location (default: runs/<run_name>/<EXP_NAME>.cleanrl_model)
#   TRACK=1        log training curves to W&B (project WANDB_PROJECT, default polyAI)
#   WANDB_UPLOAD=1 upload the final model + sidecar as a W&B artifact named EXP_NAME
#   EXTRA_PPO_ARGS="--norm-obs --learning-rate 3e-4"
#                  extra ppo.py flags, split on whitespace and appended last (they override the defaults below)
EXTRA_ARGS=()
if [ -n "${MODEL_PATH:-}" ]; then
    mkdir -p "$(dirname "$MODEL_PATH")"
    EXTRA_ARGS+=(--model-path "$MODEL_PATH")
fi
if [ "${TRACK:-0}" = "1" ]; then
    EXTRA_ARGS+=(--track --wandb-project-name "${WANDB_PROJECT:-polyAI}")
fi

read -r -a PPO_EXTRA_ARGS <<< "${EXTRA_PPO_ARGS:-}"
[ -z "${EXTRA_PPO_ARGS:-}" ] || echo "Extra PPO args: ${EXTRA_PPO_ARGS}"

echo "Starting training: $TOTAL_TIMESTEPS steps with $NUM_ENVS parallel environments on $DEVICE ($NETWORK_ARCH, seed $SEED)..."

python py_rl/cleanrl/cleanrl/ppo.py \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --num-envs "$NUM_ENVS" \
    --num-steps 128 \
    --cuda \
    --save-model \
    --save-frequency 500000 \
    --exp-name "$EXP_NAME" \
    --seed "$SEED" \
    --network-arch "$NETWORK_ARCH" \
    --validation-states 10000 \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    ${PPO_EXTRA_ARGS[@]+"${PPO_EXTRA_ARGS[@]}"}

echo "Training finished at: $(date)"

if [ "${WANDB_UPLOAD:-0}" = "1" ] && [ -n "${MODEL_PATH:-}" ]; then
    python tools/wandb_checkpoint.py upload "$MODEL_PATH" --name "$EXP_NAME" \
        || echo "warning: W&B upload failed; the checkpoint is still at $MODEL_PATH" >&2
fi
