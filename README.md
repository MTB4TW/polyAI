# PolyAI

PolyAI trains PPO agents for *The Battle of Polytopia*. It runs the game on the Java Tribes engine
(`pol_env/Tribes/`), drives it from Python through Py4J and Gymnasium, and trains with an adapted CleanRL PPO
(`py_rl/cleanrl/cleanrl/ppo.py`).

PolyAI is a fork of [PolyVision](https://github.com/NeoLorenzo/PolyVision) (pinned commit in
`local/upstream-commit.txt`), which in turn grew out of
[ClaireBookworm/polytopia_rl](https://github.com/ClaireBookworm/polytopia_rl). The engine is the Tribes research
framework (Perez-Liebana et al., 2020). PolyVision built the Phase 1 task, the map corpus, the stable action
interface, the canonical Phase 1 evaluator and the reference model; see [Attribution and licensing](#attribution-and-licensing).
PolyAI is developed by Kevin and Devin.

The project works in two phases:

- **Phase 1 (economy, from PolyVision).** Single player, Bardur on genuine 11×11 Drylands maps, no attacks,
  ends after Turn 10. Scored by final Turn-10 stars per turn (SPT).
- **Phase 2 (combat, ours).** 1v1 on 11×11 maps under fog: our agent plays Bardur as tribe 0 against Imperius,
  Xin-xi or Oumaji, in CAPITALS mode with a 20-turn limit. A game is won by taking the enemy capital. Judged by
  win rate, not shaped return. Combat code is opt-in behind `POLYVISION_COMBAT_MODE=1` / `POLYVISION_PHASE=2`, so
  Phase 1 behaviour is unchanged.

## What we've built

Everything below is merged into `main`. [CHANGELOG.md](CHANGELOG.md) has the details for each item.

**Training and networks**

- Network choice with `--network-arch`: the PolyVision MLP (`mlp`), a spatial ResNet (`spatial_resnet`), a ResNet
  that scores actions from features at their source and target tiles (`spatial_resnet_local`), and a composed action
  representation without a per-action-ID embedding table (`spatial_resnet_composed`).
  Code: `spatial_agent.py`, selected by `agent_factory.py`.
- Opt-in PPO options: `--norm-obs`, `--norm-returns`, `--gamma-per-turn`, `--no-clip-vloss`, update size flags,
  `--init-from` warm starts (with input-layer remapping) and `--critic-warmup-steps`.
- Throughput: vectorised legal-action features with compact IPC, cached board scans, overlapped rollouts
  (`--overlap-rollout`), MPS learner device on Mac.
- Phase 1 search teacher: an SPT heuristic, one-step lookahead and within-turn MCTS
  (`Phase1SptHeuristic.java`, `Phase1TurnMCTS.java`), usable for kickstarting with `--teacher lookahead_visible`.

**Combat (Phase 2)**

- Combat environment with potential-based reward shaping and optional terms (capital, tech, Rider preference,
  early-win bonus); the current reward is `cluster/combat_cap_all_v2.env`.
- Polytopia 2.15 rules behind `POLYVISION_RULES=polytopia_2_15` (default: the original Tribes rules). See
  [Rules parity](docs/rules-parity.md).
- Full-information (privileged) critic, `--privileged-critic`: advantages come from a critic that sees the unfogged
  state, while the policy still plays from the fogged observation and its checkpoint is unchanged. It has PopArt
  value normalisation and a guard that reinitialises the critic if its output goes flat.
- Tactical search teacher, `players/heuristics/CombatTactics.java`: a within-turn beam search over attacks, captures
  and moves on a fogged copy of the state, used for kickstarting (`--teacher tactical_search`) and as a play-time
  policy (`ppo_search`).
- Fog memory, `core/game/FogMemory.java`: remembers enemy units seen going into the clouds and estimates unseen ones,
  as extra observation channels.
- Self-play league: frozen snapshots and fixed opponents chosen by prioritised fictitious self-play (PFSP), with seat
  swap so the learner also plays tribe 1 (`pol_env/Tribes/py/combat_selfplay.py`, `cluster/combat_league_segment.sh`).
- Optional star-hoarding penalty (`cluster/combat_star_hoard_v2.env`).

**Evaluation and tools**

- Canonical Phase 1 evaluator (`tools/evaluate_phase1.py`, from PolyVision) and paired comparison
  (`tools/compare_evals.py`).
- Canonical combat evaluator against SimpleAgent (`tools/evaluate_combat.py`) and paired comparison
  (`tools/compare_combat_evals.py`).
- Head-to-head evaluator, checkpoint A against B in both seats (`tools/evaluate_combat_h2h.py`; on this branch tribe 1
  still has the village-capture bug fixed in PR [#33](https://github.com/MTB4TW/polyAI/pull/33)).
- Combat replay viewer: a turn-by-turn HTML replay showing the true and the fogged board (`tools/export_combat_replay.py`).
- Farmshare SLURM jobs that train and then evaluate automatically (`cluster/`), and W&B checkpoint sharing
  (`tools/wandb_checkpoint.py`) and evaluation logging (`tools/wandb_log_eval.py`).

## Results

Every number here comes from [docs/experiments.md](docs/experiments.md), which lists every trained model and its
evaluation folder under `outputs/evaluations/`. Paired differences are map by map with 95% bootstrap CIs over maps.

### Combat (Phase 2)

Canonical protocol: `tools/evaluate_combat.py`, all 120 `eval_validation` maps × 5 repeats, seed 42, against
SimpleAgent. Win rates are for deterministic play (`ppo_argmax`).

| Model | Steps | Seeds | Win | Paired win difference |
|---|---:|---:|---:|---|
| `combat_cap_all_v2b_2m` (2.15 rules, composed network) | 2M | 3 | 76.1% | first canonical result |
| `combat_v2b_pcritic_2m` (+ privileged critic) | 2M | 3 | 95.1% | +19.0 [+15.4, +22.8] vs v2b, seeds pooled |
| `combat_pcritic_teacher_2m` (+ tactical-search teacher) | 2M | 3 | 98.2% | +3.2 [+1.7, +4.8] vs privileged critic, seeds pooled |
| `combat_pcritic_t1_league_swap_4m` (+ seat-swap league) | 4M | 1 | 100.0% (600/600) | +2.7 [+1.2, +4.5] vs its init (teacher seed 1) |

- The privileged critic gained mostly by turning timeouts into wins (timeout −15.7 points).
  `combat_v2b_pcritic_2m` is marked as the current combat reference in the experiment record.
- The league model is one seed.
- An 8M continuation of the league model (four chained 2M segments) brought no further gain against SimpleAgent:
  −0.2 to −1.7 points per segment against the 4M model. SimpleAgent is at its ceiling for these models.
  Its head-to-head strength against the 4M model is **unknown**: the head-to-head runs so far (seg4 vs 4M margin
  −6.3, 4M vs itself −1.0) had a handicapped tribe 1 and are invalid as strength measures (see below).
- **League training to date was against handicapped opponents.** Kevin found that league snapshot opponents, and
  tribe 1 in `tools/evaluate_combat_h2h.py`, could never capture a village while they had one city (fix in PR
  [#33](https://github.com/MTB4TW/polyAI/pull/33), CHANGELOG 089). League win rates from training are inflated, and
  the ~90% first-mover win rate seen in head-to-head was mostly this bug: with the fix the first mover between
  equal models scores about 68% (sampled). Results against SimpleAgent are unaffected, because Java plays SimpleAgent.
- From-scratch ablations with the privileged critic: fog memory 91.5% and outcome-only reward 62.8%, both below the
  privileged-critic run; one seed of each had a collapsed critic, which led to the PopArt and flat-critic fixes.
  The star-hoarding penalty (96.0%) is a **one-seed diagnostic**.
- Combat runs before 2026-11-14 (Kevin's `combat_cap_*` runs, 34–47% wins) are **training-game diagnostics**, not
  canonical evaluations.

### Phase 1 (economy)

Canonical protocol: `tools/evaluate_phase1.py`, all 250 validation maps, all 4 policies, `--repeats-per-map 5`,
`--seed 42`. Scores are mean final Turn-10 SPT of deterministic PPO.

- **Reference (upstream PolyVision result):** v5 PARITY002 MLP, 16M steps, seed 3: 20.52 validation, 20.17 test.
  On the same validation maps, visible greedy scores 7.93 and random legal 6.67.
- **Ours, 1M-step CPU runs of `spatial_resnet_local`** (budget 1/16 of the reference's): with
  `--norm-obs --no-clip-vloss --norm-returns --gamma-per-turn`, every finished run scores 23.67–25.11, above the
  reference by +3.15 to +4.60 (all CIs above zero; 8 runs over 3 seeds and three update-size variants). Default
  settings at the same budget score 16.36–17.30. Single-change effects are in the sweep table in
  `docs/experiments.md`.
- **Ours, `farmshare_resnet_5m`** (`spatial_resnet`, 5M steps, seed 1): 19.84, −0.67 [−1.21, −0.13] against the
  reference.
- **Search policies (diagnostics):** visible-information lookahead 21.84 on 250 validation maps; within-turn MCTS
  22.32 on the first 50 validation maps; a privileged (full-information) lookahead oracle 27.51.

We have not evaluated any of our models on the test pool.

## Quick start

**Mac.** See [local/START-HERE.md](local/START-HERE.md) for setup. The engine needs JDK 11.

```sh
.venv/bin/python local/playground.py build
.venv/bin/python local/playground.py check
.venv/bin/python local/playground.py smoke
```

**Farmshare.** See [cluster/FARMSHARE-GUIDE.md](cluster/FARMSHARE-GUIDE.md) for setup (`bash cluster/setup_farmshare.sh`).
Train a Phase 1 model and evaluate it automatically:

```bash
cluster/run_experiment.sh NAME ARCH STEPS [SEED]
```

Evaluate an existing Phase 1 checkpoint:

```bash
MODEL_PATH=... EVAL_ID=... sbatch cluster/eval_farmshare.sbatch
```

**Combat.** A short CPU run (starts fast), with the current reward:

```bash
source cluster/combat_cap_all_v2.env && EXP_NAME=combat_cap_all_v2_2m sbatch cluster/train_combat_cpu.sbatch
```

`cluster/train_farmshare_combat.sbatch` is for long GPU runs. Add the privileged critic with
`EXTRA_PPO_ARGS="--privileged-critic"`; the teacher, fog memory and league settings are in `cluster/*.env` and
[docs/combat-next-steps.md](docs/combat-next-steps.md). Evaluate and compare:

```bash
MODEL_PATH=runs/NAME/NAME.cleanrl_model EVAL_ID=20261114_NAME_combat_eval_validation_canonical \
    sbatch cluster/eval_combat_farmshare.sbatch
python tools/compare_combat_evals.py BASELINE_DIR CANDIDATE_DIR
python tools/evaluate_combat_h2h.py --model-a runs/.../A.cleanrl_model --model-b runs/.../B.cleanrl_model
```

Replay one evaluation game as HTML (map and episode seed from the evaluation's `episodes.jsonl`):

```bash
python tools/export_combat_replay.py --model-path runs/.../NAME.cleanrl_model \
    --map map_2p_030048.csv --episode-seed 1028470471 --output-html replays/game.html
```

**Tests.** CI (`.github/workflows/core-contract-ci.yml`) runs:

```bash
python tools/run_core_ci_checks.py
```

or run pytest on `pol_env/Tribes/py/tests` and `tools/tests`. Training and full evaluations take minutes to hours;
run them in the background or as SLURM jobs.

## Repository layout

| Path | Contents |
|---|---|
| `pol_env/Tribes/src/` | Java engine, including `CombatTactics`, `FogMemory` and the Phase 1 search |
| `pol_env/Tribes/py/` | Gymnasium wrapper (`register_env.py`), environment contract, combat shaping and self-play |
| `py_rl/cleanrl/cleanrl/` | PPO trainer (`ppo.py`), networks, privileged critic, teacher kickstarting |
| `tools/` | Evaluators, comparisons, replay export, map tools, W&B helpers, tests (`tools/tests`) |
| `cluster/` | Farmshare SLURM jobs and combat settings (`*.env`) |
| `local/` | Mac playground |
| `outputs/` | Committed evaluation folders, experiment records and comparisons |
| `docs/` | Documentation; [experiments.md](docs/experiments.md) is the experiment record |

## Contributing

Read [CLAUDE.md](CLAUDE.md) first. In short:

- Make development decisions on the validation pool only; the test pool is evaluated only when a person asks.
- A result counts only if it is canonical; label anything smaller as a diagnostic.
- Commit every evaluation folder and `outputs/experiments/NAME.json`, and record every trained model in
  `docs/experiments.md` with its paired difference against the reference.
- Add a `CHANGELOG.md` entry for any change to the environment, reward, observation, action space, network or
  training loop, and bump the contract version if old checkpoints would see or do something different.
- Work on a branch and open a pull request for the other person to review.

## In progress (not merged)

- **Other evaluation opponents:** a polyfish-style greedy opponent, and the engine's MCTS and RHEA agents (local
  branches, not pushed).
- **Resumable training** (PR [#30](https://github.com/MTB4TW/polyAI/pull/30)), **play against the agent in the
  browser** ([#29](https://github.com/MTB4TW/polyAI/pull/29)), **saved-game engine**
  ([#28](https://github.com/MTB4TW/polyAI/pull/28)), **map generation matched to the Polytopia wiki**
  ([#31](https://github.com/MTB4TW/polyAI/pull/31)).
- **Unit status** (HP, veteran, has-acted) in the combat observation
  ([#32](https://github.com/MTB4TW/polyAI/pull/32)), a league run with it
  ([#33](https://github.com/MTB4TW/polyAI/pull/33)), and optional wrapper rules plus missing engine actions
  ([#34](https://github.com/MTB4TW/polyAI/pull/34)).
- **An exact-rules engine:** exploring polybots-engine as an alternative to the Tribes engine.

## Documentation

- [Experiments](docs/experiments.md), [Evaluation](docs/evaluation.md), [Combat next steps](docs/combat-next-steps.md),
  [Combat recipe runs](docs/combat-recipe-runs.md), [Rules parity](docs/rules-parity.md)
- [Getting started](docs/getting-started.md), [Architecture](docs/architecture.md), [Environment](docs/environment.md),
  [Actions](docs/actions.md), [Observations](docs/observations.md), [Rewards](docs/rewards.md),
  [Training](docs/training.md), [Maps](docs/maps.md), [Reproducibility](docs/reproducibility.md),
  [Troubleshooting](docs/troubleshooting.md)
- Upstream PolyVision results and audits: [Phase 1 v5 reference run](docs/results/Phase1_V5_PARITY002_Seed3_16M_TerminalSPT_Reference_Run.md),
  [Human benchmark](docs/human-benchmark.md), [Human–AI information parity audit](docs/Human_Information_Parity_Audit.md)

## Attribution and licensing

PolyAI is a fork of [NeoLorenzo/PolyVision](https://github.com/NeoLorenzo/PolyVision). PolyVision originated as a
fork of [ClaireBookworm/polytopia_rl](https://github.com/ClaireBookworm/polytopia_rl), and the game engine is the
Tribes framework. Results marked "PolyVision upstream" in `docs/experiments.md` are PolyVision's, not ours. Git
history is retained to preserve authorship and provenance.

### License status

PolyAI, like PolyVision, does **not currently have a repository-wide open-source license**.

The upstream `ClaireBookworm/polytopia_rl` repository does not provide a root license covering the project as a
whole, and this repository contains substantial code derived from it, so the codebase cannot be relicensed as a
whole without first resolving the licensing status of the inherited work.

Some components are separately licensed:

- RL code under [`py_rl/cleanrl`](py_rl/cleanrl/) contains code derived from [CleanRL](https://github.com/vwxyzjn/cleanrl)
  and other projects. The applicable notices and licenses are preserved in [`py_rl/cleanrl/LICENSE`](py_rl/cleanrl/LICENSE).
- Vendored dependencies and third-party components retain their own license and attribution notices where applicable.
- Third-party game assets, trademarks, and other intellectual property are not relicensed.

Unless a file or component explicitly states otherwise, the presence of source code in this repository should not be
interpreted as granting additional rights beyond those provided by the applicable copyright holders and licenses.

### Polytopia

*The Battle of Polytopia* and related names, assets, and intellectual property belong to their respective rights
holders. PolyAI is an independent research project and is not affiliated with or endorsed by Midjiwan AB.
