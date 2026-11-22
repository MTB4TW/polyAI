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

### Canonical combat results (eval_validation, 120 maps × 5 replicates, seed 42, vs SimpleAgent)

| Date | Name | Steps | Seeds | Ran by | Evaluation | argmax win (per seed) | Loss | Timeout | sampled win (per seed) | Notes |
|---|---|---:|---|---|---|---|---:|---:|---|---|
| 2026-11-15 | combat_v2b_pcritic_2m | 2M | 1, 2, 3 | Devin (Claude agent) | `outputs/evaluations/20261115_combat_v2b_pcritic_2m_seed{1,2,3}_combat_eval_validation_canonical` | **95.1%** (93.7 [90.7, 96.2], 96.3 [94.3, 98.0], 95.2 [93.2, 97.0]) | 1.9% | 3.0% | 94.3% (93.8, 95.5, 93.5) | **Current combat reference.** As v2b plus `--privileged-critic` (CHANGELOG 068: advantages from a full-information critic during training; the policy still plays from the fogged observation, and its checkpoint is unchanged). Paired vs v2b, same seed: win +9.2 [+4.2, +14.5], +22.8 [+17.3, +28.7], +25.0 [+19.3, +30.7] points; seeds pooled (per-map mean over 3 seeds each): win +19.0 [+15.4, +22.8] (89 maps better / 23 equal / 8 worse), loss −3.3 [−5.1, −1.7], timeout −15.7 [−18.5, −12.9]. Median win turn 9 (v2b seed 1: 10). Training-game diagnostic (second half of training): 93 / 95 / 91% win. Commit `11bcd1c`, jobs 1986031–1986033 (~16 h each), W&B `trmzxmtv`, `pth1oqsc`, `n41c36ge`. |
| 2026-11-15 | combat_v2b_s1_pcritic_warm_2m | 2M | 1, 2 (3 pending) | Devin (Claude agent) | `outputs/evaluations/20261115_combat_v2b_s1_pcritic_warm_2m_seed{1,2}_combat_eval_validation_canonical` | 84.3% (80.0 [74.7, 85.0], 88.5 [84.0, 92.5]) | 3.5% | 12.3% | 84.3% (81.0, 87.7) | Warm start from v2b seed 1 with `--privileged-critic --critic-warmup-steps 150000 --learning-rate 1e-4 --target-kl 0.015 --ent-coef 0.005`. Paired vs v2b seed 1: win −4.5 [−10.5, +1.3] and +4.0 [−1.3, +9.3] points, so no measurable change; the warm start keeps the policy near its source. Seed 3 trained (job 1986036) but its evaluation (job 1987832) had not run when this was recorded. Commit `11bcd1c`, W&B `ewyc9la3`, `ywdiu3qi` (seed 3 `4cll3jee`). |
| 2026-11-14 | combat_cap_all_v2b_2m | 2M | 1, 2, 3 | Devin | `outputs/evaluations/20261114_combat_cap_all_v2b_2m_seed{1,2,3}_combat_eval_validation_canonical` | **76.1%** (84.5 [79.5, 89.3], 73.5 [67.7, 79.2], 70.2 [64.3, 75.7]) | 5.3% | 18.7% | 48.7% (65.3, 40.0, 40.8) | 2.15 rules, `spatial_resnet_composed`, `cluster/combat_cap_all_v2.env` (win 60, loss −15, timeout −10, early win 60, rider potential). First canonical combat result; no earlier checkpoint exists to pair against (Kevin's cap_all checkpoints are not shared; the 2.15 cap_all runs saved none). Paired vs `random_legal` (0.3%): +84.2, +73.2, +69.8 points, no map worse. Training-game diagnostic (last 500 games, final 250k steps): 60.2 / 37.0 / 37.2% win. W&B `eeocxup2`, `97zognmo`, `5m7758zh`. |

### Final league checkpoints (2026-11-21, Kevin's Claude agent; `phase2_pool_1v1_wiki`)

Against SimpleAgent (eval_validation of the wiki pool, 120 maps × 5, seed 42; canonical for that pool):

| Checkpoint | argmax W / T / L | sampled W / T / L | Folder |
|---|---|---|---|
| rules on, best (2.0M) | 99.7 / 0.0 / 0.3% | 98.7 / 0.8 / 0.5% | `outputs/evaluations/20261121_ruleson_best_2m_combat_eval_validation_wiki_canonical` |
| no rules, final (8M) | 96.0 / 2.7 / 1.3% | 94.0 / 5.5 / 0.5% | `outputs/evaluations/20261121_norules_final_8m_combat_eval_validation_wiki_canonical` |
| riders branch, final (8M total) | 99.3 / 0.7 / 0.0% | 98.3 / 1.7 / 0.0% | `outputs/evaluations/20261121_riders_final_8m_combat_eval_validation_wiki_canonical` |
| Devin's league, 12M (rules on, old maps) | 98.5 / 1.3 / 0.2% | 95.0 / 4.8 / 0.2% | `outputs/evaluations/20261121_devin_league_12m_combat_eval_validation_wiki_canonical` |

Head to head (480 games, sampled; diagnostic):

| A vs B | A win | Timeout | A loss | A score | Margin | Folder |
|---|---:|---:|---:|---:|---|---|
| riders final vs Devin's 12M | 61.3% | 26.5% | 12.3% | 0.745 | [+0.42, +0.55] | `outputs/evaluations/20261121_riders_final_vs_devin_league_12m_wiki_h2h_diagnostic` |
| riders final vs no-rules final | 27.3% | 49.2% | 23.5% | 0.519 | [−0.02, +0.09] | `outputs/evaluations/20261121_riders_final_vs_norules_final_wiki_h2h_diagnostic` |

- SimpleAgent is at the ceiling for all four checkpoints.
- The riders final beats Devin's newest checkpoint clearly. B plays under A's settings, with no wrapper rules, which costs a rules-trained model about 8 points (above).
- Against the no-rules final it is even. The riders branch was ahead at equal training (5.8M), and the no-rules run caught up by 8M.
- W&B: clean runs uploaded from the TensorBoard logs (group `league-2026-11`): `lwfxh78t` (rules on), `g69cysyd` (no rules), `2nm9j5r5` (riders).
- Checkpoint artifacts: `league_us_wiki_8m-best-2m`, `league_us_wiki_norules_8m-final`, `league_us_wiki_norules_riders_4m-final`.

### League runs in progress: head to head against their starting checkpoint (diagnostic, 2026-11-19, Kevin's Claude agent)

- Starting checkpoint (B): Devin's league8m segment 1 at 1.2M. The pool is `phase2_pool_1v1_wiki` eval_validation:
  120 maps × 2 replicates × both seat orders = 480 games per row, both sides sampling.
- Margin = A's wins − B's wins, with a bootstrap 95% CI over maps.
- In the no-rules rows, B plays under A's rules, with every wrapper rule off. That handicaps B by about 8 points
  (see the next section).
- Run on `feat/combat-optional-rules`, with the head-to-head fix of CHANGELOG 090.

| A (step of its run) | Run | A win | A loss | Timeout | Score | Margin | Folder |
|---|---|---:|---:|---:|---:|---|---|
| 1.0M | `league_us_wiki_8m` (rules on) | 52.9% | 24.4% | 22.7% | 0.643 | +0.29 [+0.21, +0.35] | `outputs/evaluations/20261119_progress_ruleson_1000000_vs_seg1_1200000_wiki_h2h_diagnostic` |
| 2.0M | rules on | 61.9% | 16.7% | 21.5% | 0.726 | +0.45 [+0.39, +0.52] | `outputs/evaluations/20261119_progress_ruleson_2000000_vs_seg1_1200000_wiki_h2h_diagnostic` |
| 3.4M | rules on | 55.8% | 20.8% | 23.3% | 0.675 | +0.35 [+0.28, +0.43] | `outputs/evaluations/20261119_progress_ruleson_3400000_vs_seg1_1200000_wiki_h2h_diagnostic` |
| 1.0M | `league_us_wiki_norules_8m` | 52.1% | 21.9% | 26.0% | 0.651 | +0.30 [+0.23, +0.38] | `outputs/evaluations/20261119_progress_norules_1000000_vs_seg1_1200000_wiki_h2h_diagnostic` |
| 2.2M | no rules | 42.9% | 19.0% | 38.1% | 0.620 | +0.24 [+0.18, +0.30] | `outputs/evaluations/20261119_progress_norules_2200000_vs_seg1_1200000_wiki_h2h_diagnostic` |

Top-move replays (`--mode argmax`, otherwise identical; 474 and 471 of 480 games distinct):

| A | Moves | A win | A loss | Timeout | Margin | Folder |
|---|---|---:|---:|---:|---|---|
| rules on 3.4M | sampled | 55.8% | 20.8% | 23.3% | [+0.28, +0.43] | (above) |
| rules on 3.4M | top move | 53.7% | 28.3% | 17.9% | [+0.18, +0.33] | `outputs/evaluations/20261119_progress_ruleson_3400000_vs_seg1_1200000_wiki_h2h_argmax_diagnostic` |
| no rules 2.2M | sampled | 42.9% | 19.0% | 38.1% | [+0.18, +0.30] | (above) |
| no rules 2.2M | top move | 37.7% | 22.9% | 39.4% | [+0.07, +0.23] | `outputs/evaluations/20261119_progress_norules_2200000_vs_seg1_1200000_wiki_h2h_argmax_diagnostic` |

Timeouts barely change without sampling (23 → 18%, 38 → 39%), so sampling noise is not why games time out. Decisive
games end around turn 14–15 either way.

Later checkpoints against earlier ones of the same run (480 games each, sampled, same pool and seeds):

| A vs B | A win | B win | Timeout | A score | Margin | Folder |
|---|---:|---:|---:|---:|---|---|
| rules on 4.8M vs rules on 2.0M | 31.7% | 33.1% | 35.2% | 0.493 | [−0.08, +0.05] | `outputs/evaluations/20261119_progress_ruleson_4800000_vs_ruleson_2000000_wiki_h2h_diagnostic` |
| no rules 3.2M vs no rules 1.0M | 41.5% | 25.4% | 33.1% | 0.580 | [+0.09, +0.23] | `outputs/evaluations/20261119_progress_norules_3200000_vs_norules_1000000_wiki_h2h_diagnostic` |

- The rules-on run has not improved since 2.0M: 2.8M more steps gave no measurable gain.
- The no-rules run (road reward from 2.38M) is still improving.

Riders branch against the plain no-rules run at equal training (both from the no-rules run's 4.0M; 480 games):

| A vs B | A win | B win | Timeout | A score | Margin | Folder |
|---|---:|---:|---:|---:|---|---|
| riders 1.8M (5.8M total) vs no rules 5.8M | 31.7% | 14.6% | 53.7% | 0.585 | [+0.12, +0.23] | `outputs/evaluations/20261120_riders_1800000_vs_norules_5800000_wiki_h2h_diagnostic` |

- The Riding / Rider / Roads rewards (CHANGELOG 091) made the branch stronger.
- More than half of these games time out.

### Wrapper rules switched off at evaluation time (diagnostic, 2026-11-18, Kevin's Claude agent)

This switches off the optional wrapper rules (CHANGELOG 090) for a checkpoint trained with them on. The checkpoint is
`combat_pcritic_t1_league_swap_4m` (Devin), and the code is `feat/combat-optional-rules` (PR #34).

**Against SimpleAgent** (eval_validation, 120 maps × 5, seed 42; paired by map with `tools/compare_combat_evals.py`):

| Rules | Evaluation | argmax win | sampled win | random_legal loss | Paired argmax win vs rules on |
|---|---|---:|---:|---:|---|
| all on (as trained; canonical) | `20261118_league_swap_4m_rules_on_eval_validation` | 100.0% | 99.0% | 18.2% | — |
| village rules off | `20261118_league_swap_4m_rules_village_off_eval_validation_diagnostic` | 99.7% | 99.2% | 24.2% | −0.3 [−0.8, +0.0] |
| all off | `20261118_league_swap_4m_rules_all_off_eval_validation_diagnostic` | 99.2% | 98.2% | 24.8% | −0.8 [−1.7, −0.2] |

SimpleAgent is at the ceiling for this checkpoint, so the comparison cannot show a gain. Without the rules, a random
player loses 6.7 points more often ([+2.5, +11.2]): the rules mainly protect weak play.

**Against itself** (`20261118_league_swap_4m_rules_off_vs_on_selfplay_diagnostic`):
- Setup: the same checkpoint plays both sides, sampling. One side has every rule off, the other every rule on, and
  each side plays both seats and both code paths. 800 games; win 1, timeout 0.5, loss 0.
- The rules-off side scores **0.419 [0.388, 0.450]**: 0.62 moving first and 0.22 moving second. The same matchup with
  the same rules on both sides scores 0.685 and 0.320.
- So a policy trained with the rules is about 8 points weaker without them. It learned to rely on them.
- An example: with fewer than 2 cities it never saw a village CAPTURE it could choose, because the rule made that
  capture itself.
- Whether training without the rules ends up stronger needs a run trained with them off, compared head to head.

### Training-game diagnostics (before the canonical evaluation)

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

