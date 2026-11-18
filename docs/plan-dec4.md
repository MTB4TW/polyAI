# Plan to December 4

Written 2026-11-18 (Kevin, with Claude). Due date: **Thursday, December 4, 2026**, 16 days away.

## The goal, stated once

**North star:** an agent that beats any human.

**Primary goal by Dec 4:** beat strong humans at **1v1 Might (capture every capital) on a Tiny 11×11 map, as Bardur,
from both seats, in our engine.**
- **Reach goals, in order and only after the primary goal:** more map sizes, more tribes, other game modes.
- **Decide by Nov 20:**
  - the turn limit: currently 20; proposal 30. Humans play Might without a limit, and a human win came on turn 17;
  - the map type: our all-land generated maps, or real Drylands-style maps with a little water;
  - the fixed human protocol: maps, seats, and how results are reported.

## Compute we can get

| Source | Hard limit | Realistic over 16 days | Use |
|---|---|---|---|
| Farmshare CPU (`normal`, 20 nodes, 2,560 cores) | 512 cores per user; 48 h per job | **25k–65k core-hours.** November so far: 24k in 17 days, 60 cores on average, peaks of 384. The queue can stall for a day when the cluster is full (fair share 0.008 on Nov 18). | Training, evaluation |
| Farmshare GPU (6 × 4 L40S) | 1 GPU per user | Available, little benefit: training is limited by the Java engine on the CPU | Only a future screen-reading model or batched inference |
| Laptops (2 Macs) | ~10 cores each | ~5%; they never queue | Human games, UI, smoke tests |
| AWS ($100) | | ~3k–6k core-hours on spot | **Public hosting of the play server**, plus a queue backstop |
| Azure ($100 per student, CPU only) | | ~3k–6k core-hours each | Queue backstop |
| Modal ($30 a month each) | | ~600 core-hours a month each | Marginal |
| Together AI ($50–100), Colab, Kaggle | | not useful | Language-model API and small notebooks; can't run the Java engine at scale |

**The binding limit is wall-clock time per run, not total compute.**
- One run makes about 50 game steps per second on 32 cores, about 4M steps a day.
- A run started Nov 19 can reach about 70M steps by Dec 4; one started Nov 24 only about 35M.
- So **start the long run now** (chained in 48 h pieces with `AUTO_RESUME=1`), and spend the spare cores on
  parallel seeds, exploiters and experiments.

## Allocation (assuming ~40k core-hours plus credits)

| Share | For |
|---:|---|
| 45% | Main league lineage: 2 seeds × 32 cores, continuously from about Nov 19 |
| 15% | Exploiters: agents trained only against the current main policy, whose snapshots join its league |
| 20% | Short experiments, ~500 core-hours each (2M steps): unit status inputs, horizon, map types, league mix, saved-game learners |
| 10% | Measurement: frozen-model head-to-head ladder, canonical evaluations, search variants |
| 10% | Buffer: re-runs, queue losses |

## Strategies

- **A. Sharpen the current game:** status inputs, exploiters, play-time search, then train all-out.
  - Upside: highest chance of a convincing human result by Dec 4.
  - Downside: the claim only covers our version of the game.
- **B. Realism first:** real map types, a long or no turn limit, both seats, in week 1.
  - Upside: a stronger claim.
  - Downside: the agent restarts below its current level, the engine's naval rules are incomplete, and only about 7
    days of all-out training remain.
- **C. Search-heavy:** fix tactical blunders with play-time search (decisive or plan variants) rather than more
  training.
  - Upside: fast.
  - Downside: depends on the head-to-head results; plain `ppo_search` costs 1.7 points against SimpleAgent.
- **Recommended: A + C, with one cheap piece of B** (raise the turn limit to 30). Real map types only if generation
  takes about a day and needs no naval play.

## Timeline

**Nov 18–19: unblock** (in progress)
- Merge the open branches: game store (#28), `feat/play-vs-agent`, Devin's seat-swap branch, and
  `feat/resumable-training`.
- Read the 18 head-to-head jobs.
- Start main league v1 from the 4M model: SimpleAgent share down from 50% to ~15%, fixed opponents kept
  (`LEAGUE_MAX_SELF`), `AUTO_RESUME=1`.
- Start recruiting human players now: pros need lead time.

**Nov 19–23: recipe experiments, run in parallel**
- **Unit health and status inputs plus attack-outcome features** (`feat/combat-unit-status`): contract bump, warm
  start.
- Turn limit 30.
- Exploiter v1.
- The map-type decision.
- Speed quick wins.
- The play-time search decision.

**Nov 23–24: gate.** The final recipe must win at least 55% head-to-head against league v1 from both seats, with no
SimpleAgent regression.

**Nov 24–Dec 1: all-out training**
- 2–3 seeds plus 1–2 exploiters.
- A daily checkpoint ladder.
- Daily human games, each analysed with `tools/analyze_human_game.py`. Positions it lost become restart points.

**Nov 28–Dec 2: pro games** on the publicly hosted build.

**Dec 2–4: freeze**
- Final measurements: the ladder, canonical results, a single test-pool run if a human asks for one.
- Write-up and demo.

## The original list, reviewed

| # | Item | Verdict |
|---|---|---|
| 1 | Plan | This document. Agree it with Devin and freeze the goal. |
| 2 | Learn from saved games | Medium. Machinery exists (#28). The best use is restarting training from positions in lost games, and the search's choices. Imitating a few dozen human games won't move much. |
| 3 | Per-action attack features | **High, now.** The model also can't see HP or unit status at all (`docs/architecture-explained.md`); both go in one contract bump before the all-out training. |
| 4 | Live win rate | Low. A demo nicety. The current critic sees hidden information, so showing it live would leak it. |
| 5 | Better UI | Medium, only as far as pros need it, plus public hosting. Population display, lighthouses, inspection, animation and replays are done. |
| 6 | Map generation | Medium-high but scoped: decide by Nov 20; no naval maps. |
| 7 | Speed | Yes, time-boxed to 2–3 days. Wall-clock per run is the binding limit. |
| 8 | Alternative architectures | Not the focus: the evidence points at missing inputs and opponents, not network capacity. |
| 9 | Train, train, train | Yes, but start now, not "once we settle". Reduce the 50% SimpleAgent share. |
| 10 | Stress-test agents | High: exploiters, nearly free in spare cores. |
| 11 | Understand the architecture | Done: `docs/architecture-explained.md`. |
| 12 | Port to real Polytopia | Cut for Dec 4: large and risky. Play pros in our engine. |
| 13 | Invite pros | Yes, start this week. Needs public hosting and a fixed protocol. |

**Missing from the original list:**
1. A goal definition with an acceptable turn limit and both seats.
2. Public hosting for human games.
3. A measurement ladder beyond SimpleAgent (head-to-heads built; Java search bots still to wire up).
4. Merge hygiene.
5. Chained runs across the 48 h limit: **done**, `feat/resumable-training`.
6. A queue backstop.
7. The Dec 4 deliverable and a freeze date.
8. The human-game feedback loop.

## Status (Nov 18)

**Done:**
- the play tool: model picker, seats, search option, replays, past games;
- `analyze_human_game.py`;
- search variants and the head-to-head runner;
- stop and resume for training;
- the population display.

**Running:** 18 head-to-head jobs (search variants of the 4M league model against its final, 3M and 2M snapshots).
Queued behind other users on a full cluster.

**In progress:** unit health and status inputs.
