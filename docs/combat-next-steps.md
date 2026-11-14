# Combat: next experiments (built, not yet run)

Three additions for after the `combat_cap_all_v2b` runs finish. All are opt-in: with none of them enabled, training
and evaluation behave exactly as before (fixed-seed trajectory hashes are identical in Phase 1 and combat). Judge
each with the canonical combat evaluation (`tools/evaluate_combat.py`, docs/evaluation.md) and compare map by map
with `tools/compare_combat_evals.py`.

## 1. Outcome-only fine-tuning (CHANGELOG 063)

Continue a trained checkpoint with the shaping faded out, so the policy optimises the game result itself:

```bash
cluster/run_combat_finetune.sh runs/<v2b run>/combat_cap_all_v2b_2m_seed1.cleanrl_model combat_v2b_ft
```

Two dependent jobs: shaping × 0.5 for 500k steps (fresh critic, learning rate 2e-4), then outcome only (win 60,
loss −15, timeout −10, early-win bonus, −0.25 per turn) for 1.5M steps, keeping that critic.
`POLYVISION_COMBAT_REWARD_SHAPING_SCALE` is the underlying setting.

## 2. Self-play league (CHANGELOG 064)

Half the games against SimpleAgent, half against frozen snapshots (the learner's own checkpoints every 100k steps,
plus fixed opponents such as the v2b runs), chosen by prioritised fictitious self-play:

```bash
INIT_FROM=runs/<v2b run>/combat_cap_all_v2b_2m_seed1.cleanrl_model \
LEAGUE_GLOB="runs/Tribes-v0__combat_cap_all_v2b_*/combat_cap_all_v2b_*.cleanrl_model" \
source cluster/combat_selfplay_v2.env && EXP_NAME=combat_selfplay_v2_2m TOTAL_TIMESTEPS=2000000 \
sbatch --cpus-per-task=32 --mem=48G --time=36:00:00 cluster/train_combat_cpu.sbatch
```

W&B shows `end_raw/win_vs_simple`, `end_raw/win_vs_snapshot` and `end_raw/opponent_snapshot`. Snapshot games run
the opponent network in the env workers, so training is slower than against SimpleAgent alone.

## 3. Tactical search: teacher and search at play time (CHANGELOG 065)

`players/heuristics/CombatTactics.java`: a within-turn beam search (depth 3, beam 8) over attacks, captures,
infiltrations and moves near the enemy, on a fogged copy of the state. It values material, cities, sieges, holding
every capital (a win), and which of our units the enemy could kill next turn, using the engine's pathfinder,
so blocking an enemy's path removes the threat. Puzzles in `tests/java/players/CombatTacticsTest.java` (attack
order, taking a capital, blocking) pass on all 50 training maps.

- Teacher (kickstarting PPO; reuses the Phase 1 `--teacher` machinery):
  `source cluster/combat_teacher_v2.env && EXP_NAME=combat_teacher_v2_2m TOTAL_TIMESTEPS=2000000 sbatch ...`
- Search at play time, as an evaluation policy: `tools/evaluate_combat.py --model-path X --policy ppo_argmax
  --policy ppo_search` (ppo_search: the search where it beats stopping, otherwise the policy's best action).
  About 3× slower per game.

## 4. Fog memory (CHANGELOG 066)

`core/game/FogMemory.java` tracks enemy units we have seen after they go into the clouds, follows a move we watched
start (a Rider that attacks and retreats into the fog is remembered where it stopped), and estimates never-seen
units from the enemy cities we know. The policy gets it as extra observation channels; the tactical search places
remembered units it is nearly sure about. Fog here means unexplored tiles only: units on explored tiles are always
visible, so it matters most in the first 15 turns.

- From scratch: `source cluster/combat_fog_memory_v2.env && EXP_NAME=combat_fog_memory_v2_2m TOTAL_TIMESTEPS=2000000
  sbatch --cpus-per-task=32 --mem=48G --time=24:00:00 cluster/train_combat_cpu.sbatch`
- Warm start from a v2b checkpoint: add `INIT_FROM=runs/<v2b run>/<name>.cleanrl_model` (the new inputs start at
  zero weight, so the run begins as that policy).
- `POLYVISION_FOG_MEMORY_SEE_MOVES=0` stops following watched moves into the fog (the unit's possible tiles are then
  the region it could reach). Use it if the real game does not show where a unit moving into the clouds stops.
- Compare with `tools/compare_combat_evals.py` against the v2b evaluation: same maps, so the difference is paired.

