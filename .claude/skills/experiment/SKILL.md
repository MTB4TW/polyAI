---
name: experiment
description: Submit a PolyAI training run on Farmshare that automatically runs the canonical validation evaluation afterwards. Use when asked to train a model, start an experiment, or launch a run. Arguments - NAME ARCH STEPS [SEED].
---

# Launch an experiment

Arguments: `NAME ARCH STEPS [SEED]`, where ARCH is `mlp`, `spatial_resnet`, `spatial_resnet_local` or `spatial_resnet_composed`.

1. Confirm you are on Farmshare (`hostname` contains `rice` or `wheat`) inside the repo. If you are on a Mac,
   tell the user the command to run there instead of submitting anything:
   `ssh <sunetid>@rice.stanford.edu`, then `cd ~/polyAI && cluster/run_experiment.sh ...`
   (Devin's older Farmshare copy is at `~/polytopia-ai`).
2. Check `docs/experiments.md` and `outputs/experiments/` so NAME is new and descriptive
   (e.g. `resnet-local-16m-seed1`).
3. Submit:
   `MAIL_USER=<their stanford email, if they gave one> TRACK=1 WANDB_UPLOAD=1 WANDB_LOG_EVAL=1 cluster/run_experiment.sh NAME ARCH STEPS SEED`
   Leave out `TRACK`/`WANDB_UPLOAD`/`WANDB_LOG_EVAL` if `wandb login` has not been done on this account.
   For a PPO variant, add `EXTRA_PPO_ARGS="--norm-obs --norm-returns"` (any `ppo.py` flags except the ones the
   script sets itself: model path, exp name, seed, arch, steps). It is recorded in `outputs/experiments/NAME.json`;
   put the variant in NAME too (e.g. `resnet-local-normobs-16m-seed1`).
4. Report the two job IDs, the model path, and the evaluation folder name the script printed.
5. Do not wait or poll. When the user says the jobs finished, use the `compare` skill.
