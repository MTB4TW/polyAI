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
| 2026-11-12 | farmshare_resnet_5m | spatial_resnet | 5M | 1 | Devin (training, 2026-10-20), Kevin (evaluation) | 19.84 [19.46, 20.22] | −0.67 [−1.21, −0.13], 102 W / 21 T / 127 L | — | `20261112_farmshare_resnet_5m_validation_canonical` | Checkpoint `da9894e0…`. Below the 16M MLP reference with under a third of its steps and a different seed; sampled PPO ties it (−0.05 [−0.37, +0.24]). The earlier 2-map diagnostic (21.5, `training_fixes_20261023_diagnostic.json`) reproduces exactly on those maps, which were favourable. Evaluated from an rsynced copy without `.git`, so no commit is recorded. |
| 2026-08-27 | v5 PARITY002 Terminal-SPT | mlp | 16M | 3 | PolyVision upstream | 20.52 | reference | 20.17 | `20260827_phase1_v5_parity002_seed3_16m_terminal_spt_validation_canonical` | Current reference. |
| 2026-08-26 | v4 PARITY001 Terminal-SPT | mlp | 16M | 3 | PolyVision upstream | 19.78 | — | 19.78 | `20260826_phase1_v4_parity001_seed3_16m_terminal_spt_validation_canonical` | Older observation contract; not pairable with v5 checkpoints. |
| 2026-08-25 | v3 Terminal-SPT | mlp | 16M | 3 | PolyVision upstream | 19.87 | — | 19.34 | `20260825_phase1_v3_seed3_16m_terminal_spt_validation_canonical` | Added the end-of-Turn-10 SPT bonus. |
| 2026-08-24 | v3 | mlp | 16M | 3 | PolyVision upstream | 17.18 | — | 16.88 | `20260824_phase1_v3_seed3_16m_validation_canonical` | 4M checkpoint scored 17.09 (validation). |
| 2026-08-23 | Seed 2 | mlp | — | 2 | PolyVision upstream | 15.34 | — | 15.28 | `20260823T161131Z_validation_canonical` | Corrected two-unit opening. |
| 2026-08-14 | Seed 1 | mlp | — | 1 | PolyVision upstream | 14.58 | — | — | `20260814T110912Z_validation_canonical` | Older mixed opening; shelved. |

Baselines on the same 250 validation maps: visible greedy 7.93, random legal 6.67.

## Combat (Phase 2)

Combat is judged by win rate against SimpleAgent (11×11, 1v1, CAPITALS mode, 20 turns). There is no canonical
combat evaluation yet, so these are **diagnostics from training games** (rate over the last 500 games, averaged over
each run's final 250k steps; about 9,000 games per run), not canonical results.

| Date | Name | Steps | Seeds | Ran by | Win (mean, per seed) | Loss | Timeout | Notes |
|---|---|---:|---|---|---|---:|---:|---|
| 2026-11-13 | combat_cap_all_2m | 2M | 1, 2, 3 | Kevin | **47.1%** (46.2, 48.3, 46.9) | 2.1% | 48.9% | All shaping, tech and capital terms. Best; still rising at 2M. W&B `vis7ipn7`, `lodsg4x4`, `4qzq4ajx`. |
| 2026-11-13 | combat_cap_good_2m | 2M | 1, 2, 3 | Kevin | 38.9% (41.7, 37.3, 37.8) | 1.5% | 59.5% | As `cap_all` without regional attack and early-win taper. Flat since 1M. W&B `ggq4vz9n`, `vse7p4e2`, `5l9u6h3f`. |
| 2026-11-13 | combat_cap_only_2m | 2M | 1, 2, 3 | Kevin | 36.8% (35.1, 37.9, 37.3) | 2.7% | 57.6% | Default reward plus `capital_hold=1`, `capital_region_kill=0.5`. W&B `liv3ko74`, `medxoi7u`, `p5li4rd7`. |
| 2026-11-13 | combat_none_2m | 2M | 1, 2, 3 | Kevin | 33.9% (42.9, 29.3, 29.5) | 1.9% | 65.0% | Default combat reward (win 30, loss −10, timeout 0). W&B `ydaqpk19`, `f688v065`, `3dhods8m`. |

All four: commit `64c8095`, `spatial_resnet_local`, 16 envs, seeded-random maps from `phase2_pool_1v1/train`.
In `cap_all`, 47% of timeouts end with one of our units within 1 tile of the enemy capital; the army is mostly
warriors (12.6 of 15.9 units), and the tech terms cut Philosophy/Meditation from 64%/80% (`none`) to about 1%.
Tech-researched rates from these runs undercount by the loss rate (fixed in CHANGELOG 060).

