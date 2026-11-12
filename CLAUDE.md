# PolyAI: notes for AI agents

PolyAI is a fork of PolyVision (NeoLorenzo/PolyVision, pinned in `local/upstream-commit.txt`): PPO agents for
*The Battle of Polytopia* on a Java engine (`pol_env/Tribes/`) driven from Python through Py4J. Two people
work on it (Kevin and Devin), each with their own agent, sharing the GitHub repo `MTB4TW/polyAI`.

## Where things are

- Trainer: `py_rl/cleanrl/cleanrl/ppo.py` (`--network-arch mlp | spatial_resnet | spatial_resnet_local`)
- Networks: MLP `Agent` in `ppo.py`, ResNet in `py_rl/cleanrl/cleanrl/spatial_agent.py`, selected by `agent_factory.py`
- Environment wrapper and contract: `pol_env/Tribes/py/register_env.py`, `pol_env/Tribes/py/environment_contract.py`
- Canonical evaluator: `tools/evaluate_phase1.py`; paired comparison: `tools/compare_evals.py`
- Cluster jobs: `cluster/` (see `cluster/FARMSHARE-GUIDE.md`); experiment record: `docs/experiments.md`

## Running things

- Mac: `.venv/bin/python local/playground.py build|check|smoke` (see `local/START-HERE.md`).
- Farmshare: `cluster/run_experiment.sh NAME ARCH STEPS [SEED]` trains, then evaluates automatically.
  Evaluate an existing checkpoint with `MODEL_PATH=... EVAL_ID=... sbatch cluster/eval_farmshare.sbatch`.
- W&B (team `polyai-stanford`, project `polyAI`) holds checkpoints and copies of results: `WANDB_UPLOAD=1` uploads
  the checkpoint, `WANDB_LOG_EVAL=1` logs the evaluation (`tools/wandb_log_eval.py`). Git remains the source of
  truth for results; W&B is a read-only copy.
- Tests: `python tools/run_core_ci_checks.py` (what CI runs), or pytest on `pol_env/Tribes/py/tests` and `tools/tests`.
- Training and full evaluations take minutes to hours: run them in the background or as SLURM jobs, never in the foreground.

## Rules

1. **Never evaluate the test pool** (`--pool test --confirm-test`) unless a human explicitly asks in this session.
   Development decisions use the validation pool only.
2. **A result only counts if it is canonical**: 250 validation maps, all 4 policies, `--repeats-per-map 5`,
   `--seed 42`. Anything smaller is a diagnostic and must be labelled as such wherever it is reported.
3. **Every evaluation folder under `outputs/evaluations/` gets committed**, plus `outputs/experiments/NAME.json`.
   Never commit `runs/`, model weights, `wandb/`, or `.venv`. Share checkpoints with `tools/wandb_checkpoint.py`.
4. **Record every trained model in `docs/experiments.md`** (newest first) with who ran it, the evaluation folder,
   and the paired difference against the current reference from `tools/compare_evals.py`.
5. **Add a `CHANGELOG.md` entry** for any change to the environment, reward, observation, action space,
   network, or training loop. If the change alters what a checkpoint sees or does, bump the relevant contract
   version so old checkpoints are rejected instead of silently mis-evaluated.
6. **Checkpoints are two files**: `X.cleanrl_model` and `X.cleanrl_model.action_interface.json`. Keep them
   together and never rename one without the other.
7. **Keep Phase 1 behaviour unchanged** when working on combat (Phase 2): combat code paths must stay behind
   `POLYVISION_COMBAT_MODE=1` / `POLYVISION_PHASE=2`.
8. **Git**: work on a branch (`feat/...`, `fix/...`, `exp/...`, `docs/...`), open a pull request, and let the
   other person review it. Do not commit to `main` directly. Do not push without being asked.
9. Attribute upstream work correctly: results from PolyVision runs are upstream results, not ours.
