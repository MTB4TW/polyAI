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
| 2026-11-13 | combined-ep2-local-1m-cpu-seed1 | spatial_resnet_local | 1M | 1 | Devin (Claude agent) | 25.05 | +4.54 [+3.96, +5.14], 205 W / 11 T / 34 L | — | `20261113_combined-ep2-local-1m-cpu-seed1_validation_canonical` | Combined winners `--norm-obs --no-clip-vloss --norm-returns --gamma-per-turn` `--update-epochs 2`; `--torch-threads 6`, 8 envs, CPU. Commit `fe641b7`. Budget 1/16 of the reference's. |
| 2026-11-13 | combined-ep2-local-1m-cpu-seed2 | spatial_resnet_local | 1M | 2 | Devin (Claude agent) | 24.84 | +4.32 [+3.71, +4.90], 187 W / 18 T / 45 L | — | `20261113_combined-ep2-local-1m-cpu-seed2_validation_canonical` | Combined winners `--norm-obs --no-clip-vloss --norm-returns --gamma-per-turn` `--update-epochs 2`; `--torch-threads 6`, 8 envs, CPU. Commit `fe641b7`. Budget 1/16 of the reference's. |
| 2026-11-13 | combined-ep2-local-1m-cpu-seed3 | spatial_resnet_local | 1M | 3 | Devin (Claude agent) | 24.94 | +4.43 [+3.84, +5.00], 198 W / 11 T / 41 L | — | `20261113_combined-ep2-local-1m-cpu-seed3_validation_canonical` | Combined winners `--norm-obs --no-clip-vloss --norm-returns --gamma-per-turn` `--update-epochs 2`; `--torch-threads 6`, 8 envs, CPU. Commit `fe641b7`. Budget 1/16 of the reference's. |
| 2026-11-13 | combined-mb2-local-1m-cpu-seed1 | spatial_resnet_local | 1M | 1 | Devin (Claude agent) | 24.72 | +4.21 [+3.63, +4.78], 189 W / 15 T / 46 L | — | `20261113_combined-mb2-local-1m-cpu-seed1_validation_canonical` | Combined winners `--norm-obs --no-clip-vloss --norm-returns --gamma-per-turn` `--num-minibatches 2`; `--torch-threads 6`, 8 envs, CPU. Commit `fe641b7`. Budget 1/16 of the reference's. |
| 2026-11-13 | combined-mb2-local-1m-cpu-seed2 | spatial_resnet_local | 1M | 2 | Devin (Claude agent) | 25.11 | +4.60 [+3.99, +5.20], 201 W / 12 T / 37 L | — | `20261113_combined-mb2-local-1m-cpu-seed2_validation_canonical` | Combined winners `--norm-obs --no-clip-vloss --norm-returns --gamma-per-turn` `--num-minibatches 2`; `--torch-threads 6`, 8 envs, CPU. Commit `fe641b7`. Budget 1/16 of the reference's. |
| 2026-11-13 | combined-mb2-local-1m-cpu-seed3 | spatial_resnet_local | 1M | 3 | Devin (Claude agent) | 23.67 | +3.15 [+2.57, +3.74], 180 W / 17 T / 53 L | — | `20261113_combined-mb2-local-1m-cpu-seed3_validation_canonical` | Combined winners `--norm-obs --no-clip-vloss --norm-returns --gamma-per-turn` `--num-minibatches 2`; `--torch-threads 6`, 8 envs, CPU. Commit `fe641b7`. Budget 1/16 of the reference's. |
| 2026-11-13 | combined-local-1m-cpu-seed1 | spatial_resnet_local | 1M | 1 | Devin (Claude agent) | 24.76 | +4.25 [+3.63, +4.86], 195 W / 8 T / 47 L | — | `20261113_combined-local-1m-cpu-seed1_validation_canonical` | Combined winners `--norm-obs --no-clip-vloss --norm-returns --gamma-per-turn`; `--torch-threads 6`, 8 envs, CPU. Commit `fe641b7`. Budget 1/16 of the reference's. |
| 2026-11-13 | combined-local-1m-cpu-seed2 | spatial_resnet_local | 1M | 2 | Devin (Claude agent) | 24.33 | +3.81 [+3.22, +4.41], 183 W / 18 T / 49 L | — | `20261113_combined-local-1m-cpu-seed2_validation_canonical` | Combined winners `--norm-obs --no-clip-vloss --norm-returns --gamma-per-turn`; `--torch-threads 6`, 8 envs, CPU. Commit `fe641b7`. Budget 1/16 of the reference's. |
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

### 2026-11-13: combined winners at 1M steps, plus PPO update size

Combined `--norm-obs --no-clip-vloss --norm-returns --gamma-per-turn` on `spatial_resnet_local`, 1M steps, CPU.
Seed-pooled paired differences (per-map SPT averaged over seeds; 95% bootstrap CI over maps):

| Config | Seeds | Mean SPT | Δ vs 1M baseline (16.68) | Δ vs combined |
|---|---:|---:|---|---|
| combined (4 minibatches × 4 epochs) | 2 | 24.55 | +7.86 [+7.43, +8.28], 247/1/2 maps | — |
| combined, `--update-epochs 2` | 3 | 24.94 | +8.26 [+7.85, +8.67] | +0.40 [−0.01, +0.81] |
| combined, `--num-minibatches 2` | 3 | 24.50 | +7.82 [+7.44, +8.19] | −0.04 [−0.41, +0.32] |

Every finished combined run beats the 16M reference (+3.15 to +4.60, all CIs above zero). The gains stack:
combined beats the best single change (`--norm-obs`, +5.07). Two epochs match four and trained ~9% faster.

`combined-local-1m-cpu-seed3` crashed at 43 min with `legal_action_count=261 exceeded max_legal_actions=256`
(no evaluation; its `outputs/experiments` record is kept). Stronger policies reach positions with more than
256 legal actions; use `--max-legal-actions 512` for new runs.

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
| 2026-11-16 | combat_pcritic_teacher_2m | 2M | 1, 2, 3 | Devin (Claude agent) | `outputs/evaluations/20261115_combat_pcritic_teacher_2m_seed{1,2,3}_combat_eval_validation_canonical` | **98.2%** (97.3 [95.5, 98.8], 99.0 [98.2, 99.7], 98.3 [97.2, 99.3]) | 0.8% | 1.0% | 96.6% (97.2, 96.3, 96.2) | From scratch: `combat_v2b_pcritic_2m` plus the tactical-search teacher (`cluster/combat_teacher_v2.env`: `--teacher tactical_search --teacher-coef 1.0 --teacher-decay-frac 0.3`; coefficient reached 0 at 602k steps). Paired vs `combat_v2b_pcritic_2m`, same seed: win +3.7 [+0.5, +7.0], +2.7 [+0.8, +4.8], +3.2 [+1.2, +5.3] points; seeds pooled (per-map mean over 3 seeds each): win +3.2 [+1.7, +4.8] (42 maps better / 67 equal / 11 worse), loss −1.2 [−2.2, −0.3], timeout −2.0 [−3.3, −0.8]. Better on every seed; sampled play pooled +2.3 [+0.8, +3.8]. Median win turn 9. Not slower: 78 / 74 / 45 mean steps/s (the 45 is node-dependent; reference 40–42), 8.5–14.6 h. Training-game diagnostic (last 500 games): 96 / 98 / 96% win. Commit `11bcd1c`, jobs 1989754/1989756/1989758, W&B `a83f75kt`, `xcxe4wf6`, `kbe2gtqy`. |
| 2026-11-16 | combat_pcritic_fogmem_2m | 2M | 1, 2, 3 | Devin (Claude agent) | `outputs/evaluations/20261115_combat_pcritic_fogmem_2m_seed{1,2,3}_combat_eval_validation_canonical` | 91.5% (89.3 [85.3, 93.0], 97.8 [96.2, 99.2], 87.3 [83.2, 91.0]) | 1.3% | 7.2% | 82.7% (92.3, 95.8, 59.8) | From scratch: `combat_v2b_pcritic_2m` plus fog memory (`cluster/combat_fog_memory_v2.env`, CHANGELOG 076; privileged observation checked to keep the same shape, with its memory block zero). Paired vs `combat_v2b_pcritic_2m`, same seed: win −4.3 [−8.3, −0.3], +1.5 [−0.8, +3.8], −7.8 [−11.8, −4.0] points; seeds pooled: win −3.6 [−5.8, −1.4] (22 / 58 / 40), loss −0.6 [−1.7, +0.4], timeout +4.2 [+2.4, +6.0]. **Seed 3's privileged critic collapsed to a constant** (`losses/value_std` ≈ 0, explained variance 0) from about 61k steps and never recovered, so that seed trained without a useful baseline; seeds 1–2 only: win −1.4 [−3.9, +1.1] (27 / 67 / 26), so no measurable gain from fog memory. Commit `11bcd1c`, jobs 1989748/1989750/1989752, W&B `hy7tx9zf`, `t99syeep`, `r9x3itdh`. |
| 2026-11-16 | combat_pcritic_outcome_2m | 2M | 1, 2, 3 | Devin (Claude agent) | `outputs/evaluations/20261115_combat_pcritic_outcome_2m_seed{1,2,3}_combat_eval_validation_canonical` | 62.8% (84.7 [80.7, 88.2], 89.8 [85.7, 93.5], 14.0 [10.5, 17.8]) | 5.1% | 32.1% | 51.9% (57.8, 88.3, 9.5) | From scratch: `combat_v2b_pcritic_2m` with the outcome-only reward of `cluster/combat_outcome_finetune.env` (shaping × 0, terminal potential 0; that file needs INIT_FROM, so set by hand, learning rate kept at 5e-4). Paired vs `combat_v2b_pcritic_2m`, same seed: win −9.0 [−12.8, −5.3], −6.5 [−11.0, −2.5], −81.2 [−85.0, −77.0] points; seeds pooled: win −32.2 [−34.5, −29.8] (1 / 2 / 117), loss +3.1 [+1.7, +4.4], timeout +29.1 [+26.9, +31.4]. **Seed 3's privileged critic collapsed to a constant** from about 131k steps (training win rate stayed below 10%); seed 1 learned slowly (13% training wins at 1M steps, against 87–92% for the reference). Seeds 1–2 only: win −7.8 [−10.5, −5.2] (12 / 53 / 55): win-only reward is worse from scratch even without the collapse. Commit `11bcd1c`, jobs 1989742/1989744/1989746, W&B `d7xy5ds4`, `robeqffm`, `omomgom8`. |
| 2026-11-15 | combat_v2b_pcritic_2m | 2M | 1, 2, 3 | Devin (Claude agent) | `outputs/evaluations/20261115_combat_v2b_pcritic_2m_seed{1,2,3}_combat_eval_validation_canonical` | **95.1%** (93.7 [90.7, 96.2], 96.3 [94.3, 98.0], 95.2 [93.2, 97.0]) | 1.9% | 3.0% | 94.3% (93.8, 95.5, 93.5) | **Current combat reference.** As v2b plus `--privileged-critic` (CHANGELOG 079: advantages from a full-information critic during training; the policy still plays from the fogged observation, and its checkpoint is unchanged). Paired vs v2b, same seed: win +9.2 [+4.2, +14.5], +22.8 [+17.3, +28.7], +25.0 [+19.3, +30.7] points; seeds pooled (per-map mean over 3 seeds each): win +19.0 [+15.4, +22.8] (89 maps better / 23 equal / 8 worse), loss −3.3 [−5.1, −1.7], timeout −15.7 [−18.5, −12.9]. Median win turn 9 (v2b seed 1: 10). Training-game diagnostic (second half of training): 93 / 95 / 91% win. Commit `11bcd1c`, jobs 1986031–1986033 (~16 h each), W&B `trmzxmtv`, `pth1oqsc`, `n41c36ge`. |
| 2026-11-15 | combat_v2b_s1_pcritic_warm_2m | 2M | 1, 2 (3 pending) | Devin (Claude agent) | `outputs/evaluations/20261115_combat_v2b_s1_pcritic_warm_2m_seed{1,2}_combat_eval_validation_canonical` | 84.3% (80.0 [74.7, 85.0], 88.5 [84.0, 92.5]) | 3.5% | 12.3% | 84.3% (81.0, 87.7) | Warm start from v2b seed 1 with `--privileged-critic --critic-warmup-steps 150000 --learning-rate 1e-4 --target-kl 0.015 --ent-coef 0.005`. Paired vs v2b seed 1: win −4.5 [−10.5, +1.3] and +4.0 [−1.3, +9.3] points, so no measurable change; the warm start keeps the policy near its source. Seed 3 trained (job 1986036) but its evaluation (job 1987832) had not run when this was recorded. Commit `11bcd1c`, W&B `ewyc9la3`, `ywdiu3qi` (seed 3 `4cll3jee`). |
| 2026-11-14 | combat_cap_all_v2b_2m | 2M | 1, 2, 3 | Devin | `outputs/evaluations/20261114_combat_cap_all_v2b_2m_seed{1,2,3}_combat_eval_validation_canonical` | **76.1%** (84.5 [79.5, 89.3], 73.5 [67.7, 79.2], 70.2 [64.3, 75.7]) | 5.3% | 18.7% | 48.7% (65.3, 40.0, 40.8) | 2.15 rules, `spatial_resnet_composed`, `cluster/combat_cap_all_v2.env` (win 60, loss −15, timeout −10, early win 60, rider potential). First canonical combat result; no earlier checkpoint exists to pair against (Kevin's cap_all checkpoints are not shared; the 2.15 cap_all runs saved none). Paired vs `random_legal` (0.3%): +84.2, +73.2, +69.8 points, no map worse. Training-game diagnostic (last 500 games, final 250k steps): 60.2 / 37.0 / 37.2% win. W&B `eeocxup2`, `97zognmo`, `5m7758zh`. |

### Play-time tactical search (`ppo_search`), diagnostic

Protocol: the canonical one (eval_validation, 120 maps × 5, seed 42, vs SimpleAgent, agent in seat 0), with
`ppo_search` added to the three canonical policies. The evaluator labels any run with a fourth policy
non-canonical, so these folders are diagnostics. The teacher's `ppo_argmax` games are identical, game for game
(600/600), to its canonical run `20261115_combat_pcritic_teacher_2m_seed2_combat_eval_validation_canonical`. Run by
Kevin (Claude agent), 2026-11-17, jobs 1994801/1994802, commit `5ea9532`.

| Checkpoint | Evaluation | argmax win | `ppo_search` win | search − argmax (paired, 95% CI) | maps better / equal / worse |
|---|---|---:|---:|---|---|
| `combat_pcritic_teacher_2m` seed 2 (final) | `outputs/evaluations/20261117_combat_pcritic_teacher_2m_seed2_search_combat_eval_validation_with_search_diagnostic` | 99.0% | 97.3% | **−1.7 [−3.0, −0.5]** points | 2 / 107 / 11 |
| `combat_pcritic_t1_league_swap_4m` (final, Devin's league run) | `outputs/evaluations/20261117_combat_pcritic_t1_league_swap_4m_search_combat_eval_validation_with_search_diagnostic` | **100.0%** (600/600) | 99.7% | −0.3 [−0.8, 0.0] points | 0 / 118 / 2 |

- Against SimpleAgent, play-time search does not help, and it costs 1.7 points on the teacher model. At
  99–100% there is almost nothing left to gain, and the search's depth-3 lines sometimes replace a sounder
  policy move.
- The same search found the winning turn-10 line against a human that the policy missed
  (`docs/human-games.md`). Its value shows against opponents who defend, not against SimpleAgent.
- The 4M league model wins all 600 games from seat 0 at argmax. Its sampled play reaches 99.0%. This is in
  line with SimpleAgent being saturated as an opponent.

### Play-time search variants, head to head against frozen checkpoints (diagnostic)

Player: `combat_pcritic_t1_league_swap_4m` (final) under each `tools/search_variants.py` mode. Opponents: the same final
model, and its 3M and 2M snapshots, all with the policy alone (argmax). Each pairing covers 120 eval_validation maps ×
2 canonical seeds × both seats = 480 games. Score: win 1, timeout 0.5, loss 0. "vs none" is paired game by game with
the policy-only player against the same opponent (95% bootstrap CI over maps). Run by Kevin (Claude agent),
2026-11-18, `tools/evaluate_search_h2h.py`; folders `outputs/evaluations/20261118_search_h2h_league4m_<mode>_vs_<opponent>`.

| Mode (share of moves decided by search) | vs itself | vs 3M snapshot | vs 2M snapshot |
|---|---|---|---|
| `none` (score) | 0.500 | 0.530 | 0.529 |
| `search` (17%) | −1.9 [−3.9, +0.0] | **−5.6 [−8.5, −2.9]** | +0.5 [−2.8, +4.0] |
| `order` (6.5%) | +0.7 [−0.7, +2.2] | +0.9 [−1.1, +3.1] | +1.7 [−0.7, +4.2] |
| `decisive` (4%) | **+1.3 [+0.2, +2.5]** | +1.1 [−0.3, +2.7] | +1.4 [−0.8, +3.4] |
| `decisive_kills` (12–13%) | −0.2 [−2.1, +1.8] | −2.3 [−4.8, +0.3] | +0.2 [−2.4, +2.8] |
| `plan` (7%) | +0.2 [−0.8, +1.3] | +0.0 [−2.5, +2.5] | +2.7 [−0.3, +5.6] |

- **The seat decides most games.** Between these models the first mover scores about 0.94 and the second 0.04–0.17,
  so every effect here comes from a small number of contested games.
- **Plain `ppo_search` hurts**, as it did against SimpleAgent.
- **Adding kills to `decisive` brings back the attrition trades:** 12–13% of moves are overridden, and it scores −2.3
  against 3M.
- **`decisive`** (overrides only for wins, captures, sieges and saving the capital) is positive against all three
  opponents, and its interval excludes zero against itself. It is the safest play-time option.
- **`order`** is positive but not significant.
- **All gains are small (≤ 3 points).** The bigger fix is giving the policy the information itself (CHANGELOG 087,
  unit status).
- **4M vs its own 3M and 2M snapshots:** 0.53. The extra training shows mainly when moving second.

### Training-game diagnostics (before the canonical evaluation)

Combat is judged by win rate against SimpleAgent (11×11, 1v1, CAPITALS mode, 20 turns). The canonical combat
evaluation (`tools/evaluate_combat.py`, docs/evaluation.md, CHANGELOG 072) was added on 2026-11-14; the rows below
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
Tech-researched rates from these runs undercount by the loss rate (fixed in CHANGELOG 069).

