# Experiments

One row per trained model. Scores are mean final Turn-10 SPT for deterministic PPO (`ppo_argmax`).
"Canonical" means the full 250-map suite from `tools/evaluate_phase1.py` (4 policies, 5 repeats, seed 42),
so canonical validation rows are paired map-by-map and can be compared with `tools/compare_evals.py`.

How to add a row:

1. Run the experiment with `cluster/run_experiment.sh NAME ARCH STEPS [SEED]` (or evaluate an existing
   checkpoint with `cluster/eval_farmshare.sbatch`).
2. Copy the results back with `tools/fetch_results.sh NAME` and commit the evaluation folder and
   `outputs/experiments/NAME.json`.
3. Compare against the current reference:
   `python tools/compare_evals.py 20260827_phase1_v5_parity002_seed3_16m_terminal_spt_validation_canonical <new eval folder>`
4. Add the row below, newest first. Do not fill in a test score unless a test evaluation was explicitly requested.

Current reference: **v5 PARITY002 MLP, 16M, seed 3** (validation 20.52, test 20.17).

| Date | Name | Arch | Steps | Seed | Ran by | Validation SPT | Δ vs reference (95% CI) | Test SPT | Evaluation | Notes |
|---|---|---|---:|---:|---|---:|---|---:|---|---|
| 2026-10-23 | farmshare_resnet_5m | spatial_resnet | 5M | 1 | Devin | 21.5 (2 maps only) | n/a (not canonical) | — | `training_fixes_20261023_diagnostic.json` | Diagnostic on 2 maps; MLP reference scores 21.0 on the same 2 maps. Full 250-map evaluation pending. |
| 2026-08-27 | v5 PARITY002 Terminal-SPT | mlp | 16M | 3 | PolyVision upstream | 20.52 | reference | 20.17 | `20260827_phase1_v5_parity002_seed3_16m_terminal_spt_validation_canonical` | Current reference. |
| 2026-08-26 | v4 PARITY001 Terminal-SPT | mlp | 16M | 3 | PolyVision upstream | 19.78 | — | 19.78 | `20260826_phase1_v4_parity001_seed3_16m_terminal_spt_validation_canonical` | Older observation contract; not pairable with v5 checkpoints. |
| 2026-08-25 | v3 Terminal-SPT | mlp | 16M | 3 | PolyVision upstream | 19.87 | — | 19.34 | `20260825_phase1_v3_seed3_16m_terminal_spt_validation_canonical` | Added the end-of-Turn-10 SPT bonus. |
| 2026-08-24 | v3 | mlp | 16M | 3 | PolyVision upstream | 17.18 | — | 16.88 | `20260824_phase1_v3_seed3_16m_validation_canonical` | 4M checkpoint scored 17.09 (validation). |
| 2026-08-23 | Seed 2 | mlp | — | 2 | PolyVision upstream | 15.34 | — | 15.28 | `20260823T161131Z_validation_canonical` | Corrected two-unit opening. |
| 2026-08-14 | Seed 1 | mlp | — | 1 | PolyVision upstream | 14.58 | — | — | `20260814T110912Z_validation_canonical` | Older mixed opening; shelved. |

Baselines on the same 250 validation maps: visible greedy 7.93, random legal 6.67.
