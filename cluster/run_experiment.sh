#!/usr/bin/env bash
# Submit one Phase 1 experiment on Farmshare: train, then (only if training
# succeeds) run the canonical 250-map validation evaluation.
#
# Usage (on Farmshare, from anywhere in the repo):
#   cluster/run_experiment.sh NAME ARCH STEPS [SEED]
#
#   NAME   letters, digits, . _ - only; e.g. resnet-local-16m-seed1
#   ARCH   mlp | spatial_resnet | spatial_resnet_local | spatial_resnet_composed
#   STEPS  total environment steps, e.g. 16000000
#   SEED   training seed (default 1)
#
# Optional environment variables:
#   MAIL_USER=you@stanford.edu  email when each job ends or fails
#   TRACK=1                     log training curves to W&B
#   WANDB_UPLOAD=1              upload the final checkpoint as a W&B artifact named NAME
#   WANDB_LOG_EVAL=1            copy the evaluation results to W&B (linked to the checkpoint
#                               artifact when WANDB_UPLOAD=1)
#   WANDB_PROJECT=polyAI        W&B project for all of the above
#   NUM_ENVS=16                 parallel environments for training
#   EXTRA_PPO_ARGS="--norm-obs --learning-rate 3e-4"
#                               extra ppo.py flags, appended last (so they override the job's
#                               defaults) and recorded in outputs/experiments/NAME.json. Split on
#                               whitespace; only letters, digits, spaces and . _ = + : , / - allowed.
#
# Outputs:
#   runs/experiments/NAME/NAME.cleanrl_model (+ .action_interface.json sidecar)
#   outputs/evaluations/<date>_NAME_validation_canonical/
#   outputs/experiments/NAME.json   (what was submitted; commit it with the results)
set -euo pipefail

usage() {
    sed -n '5,11p' "$0" >&2
    exit 2
}

[ $# -ge 3 ] && [ $# -le 4 ] || usage
NAME=$1
ARCH=$2
STEPS=$3
SEED=${4:-1}

[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid NAME: $NAME" >&2; usage; }
case "$ARCH" in
    mlp|spatial_resnet|spatial_resnet_local|spatial_resnet_composed) ;;
    *) echo "Invalid ARCH: $ARCH" >&2; usage ;;
esac
[[ "$STEPS" =~ ^[0-9]+$ ]] || { echo "Invalid STEPS: $STEPS" >&2; usage; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Invalid SEED: $SEED" >&2; usage; }
EXTRA_PPO_ARGS=${EXTRA_PPO_ARGS:-}
if ! [[ "$EXTRA_PPO_ARGS" =~ ^[A-Za-z0-9\ ._=+:,/-]*$ ]]; then
    echo "EXTRA_PPO_ARGS may only contain letters, digits, spaces and . _ = + : , / - (got: $EXTRA_PPO_ARGS)" >&2
    exit 2
fi
for word in $EXTRA_PPO_ARGS; do
    case "${word%%=*}" in
        --model-path|--exp-name|--seed|--network-arch|--total-timesteps|--save-model|--no-save-model)
            echo "EXTRA_PPO_ARGS must not set ${word%%=*}: run_experiment.sh sets it (use NAME ARCH STEPS SEED)" >&2
            exit 2 ;;
    esac
done
for flag in TRACK WANDB_UPLOAD WANDB_LOG_EVAL; do
    value=${!flag:-0}
    [ "$value" = 0 ] || [ "$value" = 1 ] || { echo "$flag must be 0 or 1, got: $value" >&2; exit 2; }
done

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

EXP_DIR="runs/experiments/$NAME"
RECORD="outputs/experiments/$NAME.json"
if [ -e "$EXP_DIR" ] || [ -e "$RECORD" ]; then
    echo "Experiment '$NAME' already exists ($EXP_DIR or $RECORD); pick a new NAME." >&2
    exit 1
fi
mkdir -p "$EXP_DIR" outputs/experiments logs

MODEL_PATH="$EXP_DIR/$NAME.cleanrl_model"
EVAL_ID="$(date -u +%Y%m%d)_${NAME}_validation_canonical"

MAIL_ARGS=()
if [ -n "${MAIL_USER:-}" ]; then
    MAIL_ARGS=(--mail-type=END,FAIL "--mail-user=$MAIL_USER")
fi

TRAIN_OUT=$(EXP_NAME="$NAME" NETWORK_ARCH="$ARCH" TOTAL_TIMESTEPS="$STEPS" SEED="$SEED" \
    MODEL_PATH="$MODEL_PATH" TRACK="${TRACK:-0}" WANDB_UPLOAD="${WANDB_UPLOAD:-0}" \
    EXTRA_PPO_ARGS="$EXTRA_PPO_ARGS" \
    sbatch --parsable --job-name="train_$NAME" ${MAIL_ARGS[@]+"${MAIL_ARGS[@]}"} \
    cluster/train_farmshare.sbatch)
TRAIN_JOB=${TRAIN_OUT%%;*}

CHECKPOINT_ARTIFACT=""
if [ "${WANDB_UPLOAD:-0}" = "1" ]; then
    CHECKPOINT_ARTIFACT="$NAME"
fi
EVAL_OUT=$(MODEL_PATH="$MODEL_PATH" EVAL_ID="$EVAL_ID" \
    WANDB_LOG_EVAL="${WANDB_LOG_EVAL:-0}" CHECKPOINT_ARTIFACT="$CHECKPOINT_ARTIFACT" \
    sbatch --parsable --job-name="eval_$NAME" --dependency="afterok:$TRAIN_JOB" --kill-on-invalid-dep=yes \
    ${MAIL_ARGS[@]+"${MAIL_ARGS[@]}"} cluster/eval_farmshare.sbatch)
EVAL_JOB=${EVAL_OUT%%;*}

GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)
if [ "$GIT_COMMIT" != unknown ] && [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    GIT_DIRTY=true
else
    GIT_DIRTY=false
fi

cat > "$RECORD" <<EOF
{
  "name": "$NAME",
  "network_arch": "$ARCH",
  "total_timesteps": $STEPS,
  "seed": $SEED,
  "extra_ppo_args": "$EXTRA_PPO_ARGS",
  "submitted_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "submitted_by": "$(whoami)",
  "git_commit": "$GIT_COMMIT",
  "git_dirty": $GIT_DIRTY,
  "model_path": "$MODEL_PATH",
  "evaluation_id": "$EVAL_ID",
  "train_job": "$TRAIN_JOB",
  "eval_job": "$EVAL_JOB",
  "wandb_track": ${TRACK:-0},
  "wandb_upload": ${WANDB_UPLOAD:-0},
  "wandb_log_eval": ${WANDB_LOG_EVAL:-0}
}
EOF

echo "Submitted experiment '$NAME'"
echo "  train job $TRAIN_JOB  -> $MODEL_PATH"
[ -z "$EXTRA_PPO_ARGS" ] || echo "  extra PPO args: $EXTRA_PPO_ARGS"
echo "  eval job  $EVAL_JOB (starts after training succeeds) -> outputs/evaluations/$EVAL_ID"
echo "  record    $RECORD"
echo "Check progress with: squeue -u \$USER"
