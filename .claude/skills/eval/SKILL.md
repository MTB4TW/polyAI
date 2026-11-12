---
name: eval
description: Run the canonical 250-map Phase 1 validation evaluation on an existing PolyAI checkpoint. Use when asked to evaluate a model or checkpoint. Argument - path to a .cleanrl_model file or a W&B artifact name.
---

# Evaluate a checkpoint

1. Locate the checkpoint. If given a W&B artifact name, download it first:
   `python tools/wandb_checkpoint.py download NAME:latest --dest runs/experiments/NAME`
2. Check that both `X.cleanrl_model` and `X.cleanrl_model.action_interface.json` exist.
3. Pick an evaluation ID: `<YYYYMMDD>_<name>_validation_canonical`, and check it is not already in
   `outputs/evaluations/`.
4. On Farmshare, submit:
   `MODEL_PATH=<path> EVAL_ID=<id> sbatch cluster/eval_farmshare.sbatch`
   On a Mac with the project set up, run it in the background instead:
   `.venv/bin/python tools/evaluate_phase1.py --model-path <path> --pool validation --suite full --repeats-per-map 5 --seed 42 --evaluation-id <id>`
5. Never use `--pool test` unless the user explicitly asked for a test evaluation in this session.
6. When it finishes, check `outputs/evaluations/<id>/config.json` has `"canonical": true`, then use the
   `compare` skill.
