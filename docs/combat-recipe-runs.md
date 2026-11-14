# Combat runs with the Phase 1 recipe and the speed fixes

Instructions for the next combat (Phase 2) experiments. They cover two things:

- **Training recipe:** the Phase 1 training fixes, applied to combat.
- **Speed:** double-buffered rollout, and GPU jobs when the queue allows.

Everything here is opt-in: with no extra flags, every script behaves as before.

## Why

In Phase 1 the same changes took PPO from 16.8 to 27.0 stars per turn at 4M steps (+10.2 [+9.8, +10.7], better on
all 250 validation maps). The biggest single change was observation normalisation (+5.1).

Combat already uses two of the fixes, `--gamma-per-turn` and `--no-clip-vloss`. It doesn't yet use observation
and return normalisation.

The first warm-start test (`combat-warm-1m`, from the old 5M Phase 1 ResNet, without the new recipe) did **not**
beat training from scratch. Mean win rate over the second half of training, 3 seeds each, 1M steps:

| Arm | Win rate |
|---|---:|
| scratch | 20.8% |
| warm | 16.4% |
| scratch, gentle | 14.4% |
| warm, gentle | 9.5% |

"Gentle" means `--learning-rate 2.5e-4 --update-epochs 4`. It hurts combat, so keep `--learning-rate 5e-4
--update-epochs 8`.

## 1. Get the code

`exp/combat-recipe` is `feat/combat-capital-win` (#8) plus:

- `--init-from` (#19);
- vectorised features and compact IPC (#12);
- `--overlap-rollout` (#17);
- `--norm-obs` / `--norm-returns` / `EXTRA_PPO_ARGS` (#14);
- `spatial_resnet_composed` (#15).

The merge conflicts are already resolved. The branch also adds `SEED` and `EXTRA_PPO_ARGS` to
`cluster/train_farmshare_combat.sbatch`. Use it until those PRs land on `main`.

On Farmshare, in a separate worktree so `~/polyAI` stays on `main`:

```bash
cd ~/polyAI && git fetch origin
git worktree add ~/polyAI-wt/combat-recipe origin/exp/combat-recipe
cd ~/polyAI-wt/combat-recipe
ln -s ~/polyAI/.venv .venv && ln -s ~/polyAI/.runtime .runtime && mkdir -p logs
J=$(find $PWD/.runtime -maxdepth 3 -name javac | head -1)
(cd pol_env/Tribes && $J -nowarn -cp lib/json.jar -d out -sourcepath src $(find src -name '*.java'))
```

## 2. Recipe runs (CPU, 1M steps, 3 seeds each)

`cluster/train_combat_cpu.sbatch` already passes `--gamma-per-turn --no-clip-vloss --learning-rate 5e-4
--update-epochs 8 --target-kl 0.03`. `EXTRA_PPO_ARGS` adds the rest.

`--overlap-rollout` only changes how actions are batched, not the algorithm. It gave +13% steps/s on Farmshare,
so it is on in every new arm.

```bash
cd ~/polyAI-wt/combat-recipe
R="--norm-obs --norm-returns --overlap-rollout"
P1=$HOME/polyAI-wt/exp/runs/experiments/combined-composed-4m-cpu-seed2/combined-composed-4m-cpu-seed2.cleanrl_model
for s in 1 2 3; do
  # Recipe on the current network
  SEED=$s TOTAL_TIMESTEPS=1000000 EXP_NAME=combat-recipe-1m-seed$s \
    EXTRA_PPO_ARGS="$R" sbatch --job-name=combat-recipe-1m-seed$s cluster/train_combat_cpu.sbatch
  # Recipe on the composed network (13x fewer parameters; tied the ResNet in Phase 1)
  SEED=$s TOTAL_TIMESTEPS=1000000 NETWORK_ARCH=spatial_resnet_composed EXP_NAME=combat-recipe-composed-1m-seed$s \
    EXTRA_PPO_ARGS="$R" sbatch --job-name=combat-recipe-composed-1m-seed$s cluster/train_combat_cpu.sbatch
  # Recipe + composed, warm-started from the 28.6 SPT Phase 1 model (composed, --norm-obs, 4M steps)
  SEED=$s TOTAL_TIMESTEPS=1000000 NETWORK_ARCH=spatial_resnet_composed EXP_NAME=combat-recipe-composed-warm-1m-seed$s \
    EXTRA_PPO_ARGS="$R --init-from $P1" sbatch --job-name=combat-recipe-composed-warm-1m-seed$s cluster/train_combat_cpu.sbatch
done
```

Notes:

- **Baseline.** Reuse the `combat-scratch-1m` runs (seeds 1–3; same defaults, same #8 code) instead of running
  the baseline again.
- **Warm-start source.** It must match the run: same network (`spatial_resnet_composed`) and both using
  `--norm-obs`. `--init-from` refuses anything else.
  - The 63,913 Phase 1 action ids keep their meaning in combat (ATTACK and RECOVER are appended), and the critic
    is re-initialised automatically.
  - Combat's legacy-scalar scaling is folded into the copied normalisation statistics.
- **What a warm start can't transfer.** The Phase 1 model never saw an enemy. Its weights for the enemy unit and
  city inputs carry no information, so warm start may still not help.
- **Run time.** About 3.5 h at ~80 steps/s per 1M steps on a busy node; the job limit is 8 h.

## 3. Compare

When the runs finish (training-time win rates against the training opponent; there is no combat evaluator yet):

```bash
python tools/compare_combat_runs.py combat-scratch-1m combat-recipe-1m combat-recipe-composed-1m combat-recipe-composed-warm-1m
```

It reports final rolling-100 win/loss/timeout rates, the second-half mean win rate and end-of-game potential per
seed and on average. Compare arms at the same step when runs finish at different times.

## 4. Speed

- **Overlap.** `--overlap-rollout` is already in the commands above.
- **GPU.** When the GPU queue is short, the same recipe runs on `cluster/train_farmshare_combat.sbatch`. That
  script now accepts `SEED` and `EXTRA_PPO_ARGS`, but it doesn't set the CPU script's PPO options, so pass them
  explicitly:

  ```bash
  G="--gamma-per-turn --no-clip-vloss --learning-rate 5e-4 --update-epochs 8 --target-kl 0.03"
  SEED=1 TOTAL_TIMESTEPS=4000000 NETWORK_ARCH=spatial_resnet_composed EXP_NAME=combat-recipe-composed-gpu-4m-seed1 \
    EXTRA_PPO_ARGS="$G --norm-obs --norm-returns --overlap-rollout" sbatch cluster/train_farmshare_combat.sbatch
  ```

  On a GPU the PPO update stops being the bottleneck (it was ~68% of CPU wall time in Phase 1). Expect roughly
  2× the CPU job's steps/s with 16 envs.
- **Before running longer.** Check the job log for `[INIT_FROM]` lines on warm-start runs, and for `Extra PPO
  args:`.
