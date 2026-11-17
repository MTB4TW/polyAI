# Roadmap: one map size, one tribe, beat every bot, then humans

Written 2026-11-17 from a review of main after merging PRs #8–#27, and of the one unmerged combat branch
(`claude/combat-critic-init-96179a`).
"Target" throughout means **Bardur on 11×11 (Polytopia's Tiny map), 1v1**.

## Where we are

| Area | State | Source |
|---|---|---|
| Phase 1 (10-turn economy, solo) | Reference 20.52 SPT (16M MLP). Recipe runs reach 24.5–24.9 at 1M CPU steps (norm-obs, no value clip, per-turn γ). Visible lookahead 21.8, full-information lookahead 27.5 (oracle). | `experiments.md`, #16, CHANGELOG 058 |
| Combat (1v1 vs SimpleAgent, 20 turns, CAPITALS) | 76.1% (v2b) → **95.1%** with the privileged critic → **98.2%** with the tactical-search teacher. Canonical: 120 maps × 5. | #22, #23, #25 |
| Opponents reachable from Python | `simple`, `random`, `donothing`, plus `external` (a Python seat) on the combat branch. MCTS, RHEA, OEP, pMCTS, OSLA and MC exist in Java but **are not wired to the Python env**. | `PythonEnv.setOpponentAgent` |
| Self-play | A league exists (#22), but snapshots always played the seat they never trained in. The fix (seat swap, CHANGELOG 072) is on an unmerged branch and has not been run. | `claude/combat-critic-init-96179a` |
| Evaluation | Seat 0 only (Bardur always moves first). SimpleAgent is the only opponent, at 95–98% it no longer separates models, and the held-out `eval_test` pool has never been run. A head-to-head evaluator exists on an unmerged branch. | #22, CHANGELOG 070 |
| Humans | Phase 1 solo terminal UI only (17 maps; 1 completed attempt: human 27 SPT vs PPO 20). **No way for a human to play our agent in combat.** No imported human games. | `human-benchmark.md` |
| Throughput | Combat ~35 steps/s on Farmshare with the privileged critic (2M steps ≈ 16 h). Phase 1 ~577 SPS historically. | #23 |
| Rules | Default engine rules are ~2020. `polytopia_2_15` covers most of the current game except the 2.8 naval rework. The combat maps are all land (land fraction 1.0), so naval rules are irrelevant there. | `rules-parity.md` |

The honest summary: we have a very good **20-turn, seat-0 capital-rush agent against one scripted
bot that cheats** (SimpleAgent reads the unfogged live state). That is a strong base, but not yet
evidence of strength against search bots, against itself from the other seat, or in a full game.

## The gaps that matter most

1. **Measurement is saturated and one-sided.** At 98% against SimpleAgent, a better model and a
   lucky seed look the same. We never test seat 1, and real 1v1 games alternate who starts.
2. **No strong opponents.** The Tribes framework's search bots (MCTS, RHEA, OEP, portfolio MCTS)
   are the "pre-existing models" a paper would compare against, and we cannot play them yet.
3. **The task is not the real game.**
   - 20 turns and capital-only wins.
   - The economy is learnt separately in a 10-turn solo task, and combat starts from scratch.
   - Humans play Domination or Perfection over 30+ turns, so a capital rush that fails on turn 12
     has no plan B.
4. **Self-play has not really run.** The league bug meant the learner beat its snapshots 95–98%
   of the time, so PFSP had nothing to weight.
5. **Compute.** At 35 SPS each experiment costs a day, and three seeds cost three GPU-days. Every
   item below gets cheaper if this improves.
6. **No human loop.** We cannot collect a single human game against the agent, so "beat humans"
   is unmeasurable today.

## Plan

Ordered by dependency. Each step has an exit test that decides whether we move on.

### Stage 0: land and stabilise what exists (days)

- The PR stack is merged (CHANGELOG 052–081). Still to land: the seat-swap league fix, the privileged-critic
  warm start and the head-to-head evaluator from `claude/combat-critic-init-96179a`, which has no PR yet.
- Fix the three leftovers from the #22 review:
  - `compare_combat_evals.py` silently pairs a subset when the map lists differ;
  - the league RNG is seeded from the process id, so league runs are not reproducible;
  - value normalisation is on by default, so reproducing the 95.1% reference needs
    `--no-privileged-critic-value-norm`.
- Validate the PopArt critic fix (#24) canonically. 3 of 17 runs collapsed without it, and both
  of the best results (95.1%, 98.2%) used the old critic.
- Make the teacher + privileged-critic + PopArt recipe the combat reference, with 3 seeds.
- **Exit:** a reference of at least 97% on canonical validation, with no collapsed seeds.

### Stage 1: measure properly (about a week; do this before chasing strength)

1. **Wire the Java search bots into the env.**
   - Extend `PythonEnv.setOpponentAgent` with `osla`, `mc`, `mcts`, `pmcts`, `rhea` and `oep`,
     and expose their budgets: forward-model calls or milliseconds per move.
   - Give them the **fogged** `gs.copy(tribe)`, like `Game.java` does, so that every opponent
     plays under human information rules.
   - Keep a `simple_full_info` variant so old numbers stay reproducible.
2. **A rating ladder instead of a single win rate.**
   - Play head-to-head round robins on `eval_validation` from **both seats**.
   - Opponent pool: random, SimpleAgent, each search bot at two budgets, our milestone
     checkpoints, and PolyVision's.
   - Fit Bradley–Terry / Elo with bootstrap CIs over maps, anchored with random = 0.
   - Build it on the head-to-head evaluator (CHANGELOG 070). Report "Elo vs anchor pool" as the
     headline from now on, with SimpleAgent win rate as a sanity check.
3. **Seat and opponent coverage.**
   - Add a seat-1 copy of the eval maps, where Bardur moves second.
   - Keep the three opponent tribes and report per-tribe results.
4. Run `eval_test` **once** per milestone, only when a human asks, to confirm we have not
   overfitted validation.
- **Exit:**
  - Every model has an Elo with a CI.
  - We know where the current reference sits against MCTS and RHEA.
  - We know its seat-1 win rate.

### Stage 2: become strong against bots (2–4 weeks)

1. **Throughput first.** Three changes, smallest first:
   - (a) Profile the 35 SPS. The privileged critic and fog memory added most of the cost (10.8 h
     → 16 h), so batch the critic's privileged observation and drop unused info keys.
   - (b) Run many CPU-node actor jobs instead of one GPU job.
   - (c) **Decoupled actors with the game store.** Actor processes play games and write them to
     a shared store directory. The learner trains on them with `vtrace` (IMPALA), which corrects
     for actors running a slightly older policy. This turns the cheap CPU queue into compute.
     The pieces exist now: action-list records, persistence and V-trace. What is missing is the
     actor loop and checkpoint broadcast.
   - Target: at least 300 SPS aggregate.
2. **Real self-play league.** Run it with the seat-swap fix.
   - PFSP over past snapshots.
   - Search bots and SimpleAgent stay in the pool at a fixed share, about 25%, to prevent forgetting.
   - Add a **main exploiter** (AlphaStar): an agent trained only against the current main policy,
     whose snapshots join the pool. It finds blind spots cheaply.
   - Save every league game with `--game-store-selector upset|win` so wins against hard opponents
     and surprising losses are replayed (`sil` or `awr`).
3. **Search distillation as the main loop.**
   - The tactical-search teacher was the best single change (+3.2 points, better on every seed).
   - Make it continuous instead of a decaying warm-up: run the search on stored positions with the
     current network (game-store reanalyse) and train the policy towards the search's move
     choice (expert iteration).
   - Search can also run at play time (`ppo_search`), and policy + search is what a human will face.
4. **Start from the critical positions of lost games.**
   - Replays are deterministic, so a lost game can be reset to its prefix just before the losing
     turn (`replay.py` already reaches any step), and training can continue from there.
   - This aims practice at exactly the positions the agent gets wrong (Go-Explore style).
   - It needs one small env addition: reset-to-prefix.
- **Exit:**
  - Elo above every Java bot at its stronger budget, from both seats.
  - At least 60% head-to-head against the Stage 0 reference.

### Stage 3: play the whole game (3–6 weeks, overlapping Stage 2)

1. **One environment for the full game.**
   - Economy and combat together, 1v1 Domination (`SCORE` mode with elimination, which is how the
     engine ends), 30 turns, then 50.
   - No Phase 1 action restrictions and no scripted opening.
   - Warm-start from the best Phase 1 policy (`--init-from`, #19), so the economy is not relearnt.
   - Keep Phase 1 itself untouched, as CLAUDE.md requires: this is a new phase flag.
2. **Rules.** Switch the full-game env to `polytopia_2_15`, because current humans play current
   rules.
   - Close the open `rules-parity.md` items that affect land-only Tiny maps: the Explorer step
     count, second-player stars, and the Mountain Temple tech.
   - Decide whether combat maps should be real harvested Drylands maps (as in Phase 1) rather than
     generated all-land maps. Humans will play on real ones.
3. **Long-horizon credit assignment.**
   - Keep per-turn γ, plus a value head on game outcome (PopArt-normalised).
   - Over 30–50 turns, raise γ per turn to ≈0.995 and rely on the privileged critic.
4. **Memory.** The policy is feed-forward with hand-built fog memory (`FogMemory.java`).
   - Before adding an RNN, measure what fog memory buys over 30+ turns. In the ablation it cost
     3.6 points over 20 turns.
   - A small GRU over per-turn summaries is the next step only if fogged-information errors show
     up in replays.
- **Exit:**
  - The full-game agent beats the Stage 2 combat agent and all Java bots in the full-game setting,
    from both seats.
  - Phase 1 SPT inside the full game is at least the reference.

### Stage 4: humans (start the tooling in Stage 1, play in Stage 3)

1. **A human-vs-agent interface.** This is the critical path, so start it early.
   - The cheapest route is the existing Java Swing GUI (`HumanAgent`), with the PPO policy as an
     `external` player served from Python.
   - Alternatively, extend the combat replay viewer (#27) into a turn-based web UI over the
     `external` seat.
   - The human must see exactly the fogged information a Polytopia player sees.
2. **A human protocol.**
   - Fixed maps from `eval_test`, both seats, recorded with the game store (action lists replay
     exactly).
   - Volunteers with a known Polytopia multiplayer rating where possible.
   - Report wins with a CI, and never tune on these maps.
3. **Learn from human games.** Store them with a `bc` or `awr` learner:
   - human wins against the agent are the most valuable games we can collect;
   - the agent's losses show up as `upset` games.
4. **Later:** extend `polytopia_state_converter` past the map header (units, cities, techs) so the
   agent can advise in a real app game. This is the path to playing ranked humans.
- **Exit:** a positive win rate with a CI against human players of a stated level, on held-out maps.

## What the saved-game engine is for

`docs/game-store.md` describes the mechanics. It is the shared substrate for most of the plan:

| Plan item | Selector | Learner / use |
|---|---|---|
| Replay wins against hard opponents | `win`, `upset` | `sil`, `awr` |
| Critic stability (the collapse problem) | `reservoir` | `value` |
| Decoupled CPU actors (throughput) | `all`, with a half-life | `vtrace` |
| Search distillation on old positions | `surprise` | reanalyse + `bc` towards search targets |
| Critical-position restarts | `outcome:keep=loss` | replay to a prefix, then train from it |
| Human games | everything | `bc` / `awr` |

## Risks and known issues to fix along the way

- **SimpleAgent reads the unfogged state.** All our combat numbers are against a cheating
  opponent. That is fine as an opponent, but note it, and make the search bots fog-correct.
- **The canonical eval plays seat 0 only**, which inflates numbers. Fix in Stage 1.
- **Fog memory follows watched enemy moves into fog by default** (SEE_MOVES). Check that a human
  really gets that information in the real game before counting it as parity.
- **Legal-action overflow:** Phase 1 runs crashed at 261–263 actions with a limit of 256, so use
  `--max-legal-actions 512` everywhere.
- **The old-logprob check** (tolerance 1e-5) fails on a loaded Mac (1.05e-5 seen). Loosen it to
  about 1e-4.
- **Unrecorded results:** two Phase 1 figures from #21 (27.0 SPT at 4M, 28.6 composed) exist only
  in `docs/combat-recipe-runs.md`. Add them to `experiments.md` with evaluation folders, or drop them.
- **Numbers that are not comparable:** #8's training-time win rates (34–47%) are not canonical
  evaluations. Never put them next to canonical figures.
