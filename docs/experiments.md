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
| 2026-11-13 | normobs-local-1m-cpu-seed1 | spatial_resnet_local | 1M | 1 | Devin (Claude agent) | 21.74 | +1.23 [+0.64, +1.82], 130 W / 27 T / 93 L | — | `20261113_normobs-local-1m-cpu-seed1_validation_canonical` | 1M CPU matched-budget sweep (`--norm-obs`; all runs `--torch-threads 6`, 8 envs, JVM limits). vs baseline seed 1: +5.35 [+4.82, +5.88]. Commit `fe641b7` (exp/phase1-cpu-matched). Budget is 1/16 of the reference's. |
| 2026-11-13 | normobs-local-1m-cpu-seed2 | spatial_resnet_local | 1M | 2 | Devin (Claude agent) | 21.60 | +1.08 [+0.49, +1.68], 122 W / 35 T / 93 L | — | `20261113_normobs-local-1m-cpu-seed2_validation_canonical` | 1M CPU matched-budget sweep (`--norm-obs`; all runs `--torch-threads 6`, 8 envs, JVM limits). vs baseline seed 2: +5.24 [+4.66, +5.78]. Commit `fe641b7` (exp/phase1-cpu-matched). Budget is 1/16 of the reference's. |
| 2026-11-13 | normobs-local-1m-cpu-seed3 | spatial_resnet_local | 1M | 3 | Devin (Claude agent) | 21.93 | +1.41 [+0.81, +2.01], 140 W / 28 T / 82 L | — | `20261113_normobs-local-1m-cpu-seed3_validation_canonical` | 1M CPU matched-budget sweep (`--norm-obs`; all runs `--torch-threads 6`, 8 envs, JVM limits). vs baseline seed 3: +4.63 [+4.07, +5.17]. Commit `fe641b7` (exp/phase1-cpu-matched). Budget is 1/16 of the reference's. |
| 2026-11-13 | noclipv-normret-local-1m-cpu-seed1 | spatial_resnet_local | 1M | 1 | Devin (Claude agent) | 23.95 | +3.44 [+2.85, +4.04], 174 W / 18 T / 58 L | — | `20261113_noclipv-normret-local-1m-cpu-seed1_validation_canonical` | 1M CPU matched-budget sweep (`--no-clip-vloss --norm-returns`; all runs `--torch-threads 6`, 8 envs, JVM limits). vs baseline seed 1: +7.56 [+6.96, +8.18]. Commit `fe641b7` (exp/phase1-cpu-matched). Budget is 1/16 of the reference's. |
| 2026-11-13 | noclipv-normret-local-1m-cpu-seed2 | spatial_resnet_local | 1M | 2 | Devin (Claude agent) | 22.21 | +1.70 [+1.12, +2.29], 141 W / 20 T / 89 L | — | `20261113_noclipv-normret-local-1m-cpu-seed2_validation_canonical` | 1M CPU matched-budget sweep (`--no-clip-vloss --norm-returns`; all runs `--torch-threads 6`, 8 envs, JVM limits). vs baseline seed 2: +5.85 [+5.30, +6.39]. Commit `fe641b7` (exp/phase1-cpu-matched). Budget is 1/16 of the reference's. |
| 2026-11-13 | noclipv-normret-local-1m-cpu-seed3 | spatial_resnet_local | 1M | 3 | Devin (Claude agent) | 17.16 | −3.36 [−3.97, −2.76], 50 W / 17 T / 183 L | — | `20261113_noclipv-normret-local-1m-cpu-seed3_validation_canonical` | 1M CPU matched-budget sweep (`--no-clip-vloss --norm-returns`; all runs `--torch-threads 6`, 8 envs, JVM limits). vs baseline seed 3: −0.14 [−0.70, +0.42]. Commit `fe641b7` (exp/phase1-cpu-matched). Budget is 1/16 of the reference's. |
| 2026-11-13 | gammaturn-local-1m-cpu-seed1 | spatial_resnet_local | 1M | 1 | Devin (Claude agent) | 18.19 | −2.33 [−2.98, −1.71], 73 W / 22 T / 155 L | — | `20261113_gammaturn-local-1m-cpu-seed1_validation_canonical` | 1M CPU matched-budget sweep (`--gamma-per-turn`; all runs `--torch-threads 6`, 8 envs, JVM limits). vs baseline seed 1: +1.80 [+1.24, +2.36]. Commit `fe641b7` (exp/phase1-cpu-matched). Budget is 1/16 of the reference's. |
| 2026-11-13 | gammaturn-local-1m-cpu-seed2 | spatial_resnet_local | 1M | 2 | Devin (Claude agent) | 20.53 | +0.02 [−0.62, +0.63], 116 W / 17 T / 117 L | — | `20261113_gammaturn-local-1m-cpu-seed2_validation_canonical` | 1M CPU matched-budget sweep (`--gamma-per-turn`; all runs `--torch-threads 6`, 8 envs, JVM limits). vs baseline seed 2: +4.17 [+3.60, +4.73]. Commit `fe641b7` (exp/phase1-cpu-matched). Budget is 1/16 of the reference's. |
| 2026-11-13 | gammaturn-local-1m-cpu-seed3 | spatial_resnet_local | 1M | 3 | Devin (Claude agent) | 19.86 | −0.66 [−1.25, −0.08], 96 W / 16 T / 138 L | — | `20261113_gammaturn-local-1m-cpu-seed3_validation_canonical` | 1M CPU matched-budget sweep (`--gamma-per-turn`; all runs `--torch-threads 6`, 8 envs, JVM limits). vs baseline seed 3: +2.56 [+2.02, +3.08]. Commit `fe641b7` (exp/phase1-cpu-matched). Budget is 1/16 of the reference's. |
| 2026-11-13 | composed-1m-cpu-seed1 | spatial_resnet_composed | 1M | 1 | Devin (Claude agent) | 19.07 | −1.44 [−1.97, −0.94], 89 W / 22 T / 139 L | — | `20261113_composed-1m-cpu-seed1_validation_canonical` | 1M CPU matched-budget sweep (composed action representation (327k params); all runs `--torch-threads 6`, 8 envs, JVM limits). vs baseline seed 1: +2.68 [+2.23, +3.13]. Commit `fe641b7` (exp/phase1-cpu-matched). Budget is 1/16 of the reference's. |
| 2026-11-13 | composed-1m-cpu-seed2 | spatial_resnet_composed | 1M | 2 | Devin (Claude agent) | 17.53 | −2.99 [−3.58, −2.38], 59 W / 21 T / 170 L | — | `20261113_composed-1m-cpu-seed2_validation_canonical` | 1M CPU matched-budget sweep (composed action representation (327k params); all runs `--torch-threads 6`, 8 envs, JVM limits). vs baseline seed 2: +1.16 [+0.61, +1.72]. Commit `fe641b7` (exp/phase1-cpu-matched). Budget is 1/16 of the reference's. |
| 2026-11-13 | composed-1m-cpu-seed3 | spatial_resnet_composed | 1M | 3 | Devin (Claude agent) | 17.84 | −2.68 [−3.25, −2.13], 65 W / 22 T / 163 L | — | `20261113_composed-1m-cpu-seed3_validation_canonical` | 1M CPU matched-budget sweep (composed action representation (327k params); all runs `--torch-threads 6`, 8 envs, JVM limits). vs baseline seed 3: +0.54 [+0.03, +1.02]. Commit `fe641b7` (exp/phase1-cpu-matched). Budget is 1/16 of the reference's. |
| 2026-11-13 | bigupdate-local-1m-cpu-seed1 | spatial_resnet_local | 1M | 1 | Devin (Claude agent) | 11.98 | −8.54 [−9.13, −7.94], 8 W / 7 T / 235 L | — | `20261113_bigupdate-local-1m-cpu-seed1_validation_canonical` | 1M CPU matched-budget sweep (`--learning-rate 5e-4 --update-epochs 8 --target-kl 0.03`; all runs `--torch-threads 6`, 8 envs, JVM limits). vs baseline seed 1: −4.42 [−4.92, −3.89]. Commit `fe641b7` (exp/phase1-cpu-matched). Budget is 1/16 of the reference's. |
| 2026-11-13 | bigupdate-local-1m-cpu-seed2 | spatial_resnet_local | 1M | 2 | Devin (Claude agent) | 15.49 | −5.02 [−5.62, −4.46], 28 W / 13 T / 209 L | — | `20261113_bigupdate-local-1m-cpu-seed2_validation_canonical` | 1M CPU matched-budget sweep (`--learning-rate 5e-4 --update-epochs 8 --target-kl 0.03`; all runs `--torch-threads 6`, 8 envs, JVM limits). vs baseline seed 2: −0.87 [−1.40, −0.36]. Commit `fe641b7` (exp/phase1-cpu-matched). Budget is 1/16 of the reference's. |
| 2026-11-13 | bigupdate-local-1m-cpu-seed3 | spatial_resnet_local | 1M | 3 | Devin (Claude agent) | 15.33 | −5.19 [−5.80, −4.59], 37 W / 20 T / 193 L | — | `20261113_bigupdate-local-1m-cpu-seed3_validation_canonical` | 1M CPU matched-budget sweep (`--learning-rate 5e-4 --update-epochs 8 --target-kl 0.03`; all runs `--torch-threads 6`, 8 envs, JVM limits). vs baseline seed 3: −1.97 [−2.52, −1.44]. Commit `fe641b7` (exp/phase1-cpu-matched). Budget is 1/16 of the reference's. |
| 2026-11-13 | baseline-local-1m-cpu-seed1 | spatial_resnet_local | 1M | 1 | Devin (Claude agent) | 16.39 | −4.12 [−4.66, −3.59], 28 W / 34 T / 188 L | — | `20261113_baseline-local-1m-cpu-seed1_validation_canonical` | 1M CPU matched-budget sweep (defaults; all runs `--torch-threads 6`, 8 envs, JVM limits). Commit `fe641b7` (exp/phase1-cpu-matched). Budget is 1/16 of the reference's. |
| 2026-11-13 | baseline-local-1m-cpu-seed2 | spatial_resnet_local | 1M | 2 | Devin (Claude agent) | 16.36 | −4.15 [−4.74, −3.56], 42 W / 12 T / 196 L | — | `20261113_baseline-local-1m-cpu-seed2_validation_canonical` | 1M CPU matched-budget sweep (defaults; all runs `--torch-threads 6`, 8 envs, JVM limits). Commit `fe641b7` (exp/phase1-cpu-matched). Budget is 1/16 of the reference's. |
| 2026-11-13 | baseline-local-1m-cpu-seed3 | spatial_resnet_local | 1M | 3 | Devin (Claude agent) | 17.30 | −3.22 [−3.78, −2.66], 45 W / 10 T / 195 L | — | `20261113_baseline-local-1m-cpu-seed3_validation_canonical` | 1M CPU matched-budget sweep (defaults; all runs `--torch-threads 6`, 8 envs, JVM limits). Commit `fe641b7` (exp/phase1-cpu-matched). Budget is 1/16 of the reference's. |
| 2026-11-12 | farmshare_resnet_5m | spatial_resnet | 5M | 1 | Devin (training, 2026-10-20), Kevin (evaluation) | 19.84 [19.46, 20.22] | −0.67 [−1.21, −0.13], 102 W / 21 T / 127 L | — | `20261112_farmshare_resnet_5m_validation_canonical` | Checkpoint `da9894e0…`. Below the 16M MLP reference with under a third of its steps and a different seed; sampled PPO ties it (−0.05 [−0.37, +0.24]). The earlier 2-map diagnostic (21.5, `training_fixes_20261023_diagnostic.json`) reproduces exactly on those maps, which were favourable. Evaluated from an rsynced copy without `.git`, so no commit is recorded. |
| 2026-08-27 | v5 PARITY002 Terminal-SPT | mlp | 16M | 3 | PolyVision upstream | 20.52 | reference | 20.17 | `20260827_phase1_v5_parity002_seed3_16m_terminal_spt_validation_canonical` | Current reference. |
| 2026-08-26 | v4 PARITY001 Terminal-SPT | mlp | 16M | 3 | PolyVision upstream | 19.78 | — | 19.78 | `20260826_phase1_v4_parity001_seed3_16m_terminal_spt_validation_canonical` | Older observation contract; not pairable with v5 checkpoints. |
| 2026-08-25 | v3 Terminal-SPT | mlp | 16M | 3 | PolyVision upstream | 19.87 | — | 19.34 | `20260825_phase1_v3_seed3_16m_terminal_spt_validation_canonical` | Added the end-of-Turn-10 SPT bonus. |
| 2026-08-24 | v3 | mlp | 16M | 3 | PolyVision upstream | 17.18 | — | 16.88 | `20260824_phase1_v3_seed3_16m_validation_canonical` | 4M checkpoint scored 17.09 (validation). |
| 2026-08-23 | Seed 2 | mlp | — | 2 | PolyVision upstream | 15.34 | — | 15.28 | `20260823T161131Z_validation_canonical` | Corrected two-unit opening. |
| 2026-08-14 | Seed 1 | mlp | — | 1 | PolyVision upstream | 14.58 | — | — | `20260814T110912Z_validation_canonical` | Older mixed opening; shelved. |


### 2026-11-13: 1M-step CPU PPO sweep (3 seeds each, diagnostic budget)

Seed-pooled paired difference vs the 1M baseline (`spatial_resnet_local`, defaults); per-map SPT averaged
over 3 seeds, 95% bootstrap CI over the 250 maps (does not include seed-to-seed variance):

| Config | Mean SPT (argmax) | Δ vs baseline | W/T/L maps | Seeds better than baseline |
|---|---:|---|---|---|
| `--norm-obs` | 21.76 | +5.07 [+4.74, +5.41] | 243/1/6 | 3/3 |
| `--no-clip-vloss --norm-returns` | 21.11 | +4.42 [+4.08, +4.77] | 233/5/12 | 2/3 (seed 3 ties) |
| `--gamma-per-turn` | 19.53 | +2.84 [+2.50, +3.18] | 208/6/36 | 3/3 |
| `spatial_resnet_composed` | 18.15 | +1.46 [+1.15, +1.77] | 180/6/64 | 3/3 |
| `--learning-rate 5e-4 --update-epochs 8 --target-kl 0.03` | 14.27 | −2.42 [−2.76, −2.08] | 39/11/200 | 0/3 |
| baseline | 16.68 | — | — | — |

All three `--norm-obs` seeds also beat the 16M reference at 1M steps. Per-seed comparisons are in
`outputs/comparisons/phase1_1m_cpu/`. Next: combine the winners and confirm at a longer budget.

Baselines on the same 250 validation maps: visible greedy 7.93, random legal 6.67.

## Combat (Phase 2)

Combat is judged by win rate against SimpleAgent (11×11, 1v1, CAPITALS mode, 20 turns). The canonical combat
evaluation (`tools/evaluate_combat.py`, docs/evaluation.md, CHANGELOG 062) was added on 2026-11-14; the rows below
predate it and are **diagnostics from training games** (rate over the last 500 games, averaged over each run's final
250k steps; about 9,000 games per run), not canonical results.

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

