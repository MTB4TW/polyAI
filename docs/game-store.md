# Saved-game engine (`--game-store`)

PPO uses each game once and throws it away. The saved-game engine keeps finished games and trains
on them again between PPO updates. It has three parts, and each one can be swapped without
touching the others:

| Part | Question it answers | Where | Built-in choices |
|---|---|---|---|
| Selector | Which games are worth keeping, and how much? | `game_store/selectors.py` | `all`, `win`, `outcome`, `top_return`, `surprise`, `upset`, `reservoir`, `short_win`, `field`, combined with `\|` (any) and `&` (all) |
| Store | How many, in how much memory? | `game_store/store.py` | Byte budget, game cap, priority and age eviction, saving to disk |
| Learner | What is learnt from them? | `game_store/learners.py` | `sil`, `awr`, `bc`, `value`, `replay_ppo`, `vtrace` |

The code is in `py_rl/cleanrl/cleanrl/game_store/` and the tests in `tools/tests/test_game_store.py`.
It is off by default. Without `--game-store`, the trainer records nothing and runs exactly as before.

## Quick start

```bash
# Combat: keep every win and every game the critic got badly wrong; self-imitation on them.
# (no quotes inside EXTRA_PPO_ARGS: the job script splits it on spaces only)
EXTRA_PPO_ARGS="--game-store --game-store-selector surprise:min_error=0.5|win --game-store-learner sil" \
  EXP_NAME=combat_sil sbatch cluster/train_combat_cpu.sbatch

# Off-policy actor-critic on recent games, uniformly sampled.
--game-store --game-store-learner vtrace --game-store-alpha 0 --game-store-half-life 2000

# Keep only replayable action lists, for later analysis or reanalyse (no replay updates).
--game-store --game-store-payload actions --game-store-max-games 1000000
```

Every option is a `--game-store-*` flag on `ppo.py` (`python ppo.py --help`). Selectors and learners
take options as `name:key=value,key=value`, and list values are joined with `+`, as in
`outcome:keep=win+timeout`. The `game_store/...` W&B and TensorBoard curves show store size, memory,
the fraction of wins, the replay losses and the time spent.

## What a stored game costs

Each game is saved at one of two levels:

1. **Action list (a few hundred bytes).** The map, the episode seed, the outcome, the end-of-game
   statistics and the global action id of every step. The engine is deterministic given
   (map, seed, actions), so `game_store/replay.py` can rebuild every tensor by replaying the game,
   and `tools/verify_game_store_replay.py` checks this. A store can hold millions of games at this level.
2. **Training payload.** This is what a learner reads without touching the engine:
   - Observations.
   - The *valid* legal actions and their features, never the padded 512-row block.
   - The chosen slot, the behaviour log-probability, the value and the reward.

   Observations and features are compressed with XOR-delta + zlib. That works well because
   consecutive observations within a turn differ in only a few entries. Measured in a real combat
   run (spatial ResNet, 512 legal slots, float32 storage), it came to **637 bytes per step**, against
   about 128 KB per step for the padded rollout buffer. At that rate 1 GB holds about 1.6M steps,
   or roughly 15,000 combat games of about 115 steps each. float16 storage (the default) is smaller still.

When the payload budget (`--game-store-budget-mb`) is full, the lowest-priority games lose their
payload but stay as action lists. When the game cap is reached, the lowest-priority records are
dropped. With `--game-store-half-life N`, priorities halve every N games, so the store prefers
recent play but can still hold on to a rare, high-priority game.

Overhead, measured in the same run (2 envs × 512 steps, CPU learner):
- Recording took 0.05 s per iteration, against an iteration of about 32 s.
- The four default replay updates took 1.4 s on the CPU, about one PPO minibatch each; less on a GPU.
- Discarded games are never compressed.
- `game_store/time_ingest_s` and `game_store/time_update_s` report both costs in every run.

## Choosing a selector

- **`surprise`** keeps games where the Monte Carlo return was far from the critic's prediction.
  These have the most left to teach (the idea behind prioritised replay), and the priority is the
  size of the error.
- **`win` / `outcome` / `short_win`** keep results worth imitating. A short win is a cleaner plan.
- **`top_return`** keeps the top fraction of recent games by return, or by any meta field (`key=`).
  The threshold moves as the policy improves.
- **`upset`** keeps games whose result contradicted the critic's opening estimate, such as a win
  from a position it rated as losing.
- **`reservoir`** keeps a uniform random sample, which gives diverse, unbiased value targets.
- **`field`** keeps games by any end-of-game statistic, for example Phase 1 games above a score.

To add one, subclass `Selector`, implement `score(meta, game) -> priority or None`, and register it
with `@register_selector("name")`.

## Choosing a learner

| Learner | Loss | When to use it |
|---|---|---|
| `sil` | −log π(a\|s)·max(R−V,0) + value term (Oh et al., 2018) | Default. Imitates only past actions that did better than the critic expects, so bad games cost nothing. |
| `awr` | −log π(a\|s)·exp((R−V)/β) (Peng et al., 2019) | Smoother than `sil`: it also down-weights the worse actions and fits the critic to every return. |
| `bc` | −log π(a\|s) | Games that should be copied outright: search or teacher games, strong-opponent wins, human games. |
| `value` | ½(V−R)² | More critic data at no rollout cost. Use it when explained variance is low or the critic collapses. |
| `replay_ppo` | Clipped PPO with ratio π/μ against the stored behaviour policy | Re-uses recent games for the full PPO objective. Drops steps whose ratio has drifted too far. |
| `vtrace` | IMPALA V-trace actor-critic on whole games | The principled off-policy correction: it bootstraps from the critic and stays correct as games age. |

The step learners use the Monte Carlo return R from each step to the end of its game. Rewards are
stored with the `--norm-returns` scale of the moment and rescaled to the current scale when
sampled. To add a learner, subclass `ReplayLearner`, implement
`loss(agent, batch, ctx) -> (loss, metrics)`, and register it. Set `kind = "games"` to receive whole games.

## Reanalyse: action-only games plus a newer network

`replay.replay_game(env, record, agent=agent)` replays an action-only game and scores every step
with the *current* network. `replay.materialize` then turns the result into a training record.
That gives fresh targets for old games without playing new ones (MuZero Reanalyse). It is where
search targets belong: re-running the tactical search on stored positions and training on the
improved move distribution (expert iteration). It needs a single env with the Java engine, so it
runs as a separate process or tool, not inside the training loop.

## Caveats

- A stored game is tied to the environment contract it was played under (observation layout,
  action catalog, rules). Don't load a store into a run with a different contract.
  `verify_game_store_replay.py` catches most mismatches.
- With `--game-store-obs-dtype float16`, the replayed log-probability of a stored action differs
  slightly from the logged one. That matters only for `replay_ppo` and `vtrace` ratios, and is far
  below their clipping.
- The store lives in the learner process. `--game-store-dir` (by default `<run dir>/game_store`
  with `--save-model`) is pickled shards under `runs/`, which is never committed. Load only stores
  you wrote yourself.
