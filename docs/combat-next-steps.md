# Combat: next experiments (built, not yet run)

Opt-in additions for after the `combat_cap_all_v2b` runs finish. All are opt-in: with none of them enabled, training
and evaluation behave exactly as before (fixed-seed trajectory hashes are identical in Phase 1 and combat). Judge
each with the canonical combat evaluation (`tools/evaluate_combat.py`, docs/evaluation.md) and compare map by map
with `tools/compare_combat_evals.py`.

## 1. Outcome-only fine-tuning (CHANGELOG 073)

Continue a trained checkpoint with the shaping switched off, so the policy optimises the game result itself:

```bash
cluster/run_combat_finetune.sh runs/<v2b run>/combat_cap_all_v2b_2m_seed1.cleanrl_model combat_v2b_s1_outcome_2m
```

One job: outcome only (win 60, loss −15, timeout −10, early-win bonus, −0.25 per turn; no end-of-game share
of the lead, so a timeout scores the same however far ahead) for 2M steps, with a fresh
critic and learning rate 2e-4. There is no intermediate stage with partial shaping: the goal is winning games, not
first maximising the economy terms and then winning.
`POLYVISION_COMBAT_REWARD_SHAPING_SCALE` is the underlying setting.

## 2. Self-play league (CHANGELOG 074)

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

## 3. Tactical search: teacher and search at play time (CHANGELOG 075)

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

## 4. Fog memory (CHANGELOG 076)

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
  the region it could reach). Leave it on: Polytopia shows where a unit moving into the clouds stops, so the
  default matches the real game. The switch is for experiments with less information.
- Compare with `tools/compare_combat_evals.py` against the v2b evaluation: same maps, so the difference is paired.


## 5. Full-information critic (CHANGELOG 079)

`ppo.py --privileged-critic` computes advantages with a separate critic that sees the unfogged game state; the
policy still plays from the fogged observation, and its checkpoint is unchanged (the critic is saved to
`X.cleanrl_model.privileged_critic.pt`, needed only to resume training). Under fog the policy's own value head has
to guess what is in the clouds, which makes the advantages noisier; the full-information critic removes that part
of the noise. Since fog here is unexplored tiles only, expect the effect mostly in the opening (first contact,
finding the enemy capital) and with hidden Cloaks.

- A/B against v2b, same settings plus the flag: `source cluster/combat_cap_all_v2.env &&
  EXP_NAME=combat_cap_all_v2_pcritic_2m EXTRA_PPO_ARGS="--privileged-critic" sbatch cluster/train_combat_cpu.sbatch`
  (3 seeds with `SEED=1..3`, as for v2b).
- Watch `losses/explained_variance`: it should be clearly higher than v2b's from early on. If it is not, the
  critic is not using the extra information and the win rate will not move either.
- Slower: in a short local test (Mac, 4 envs, composed network, 6k steps) 140 steps/s against 212 without it, and
  171 with `--overlap-rollout`. Building the unfogged observation costs about 0.6 ms of a 2.75 ms env step; most
  of the rest is the critic's own ResNet in the update. Check the SPS chart on Farmshare, where more envs hide the
  env part.
- Combines with the other items: with fog memory or the teacher, add the flag to their `EXTRA_PPO_ARGS`.
- Compare with `tools/compare_combat_evals.py` against the v2b evaluation.

**Result (2026-11-15, canonical, docs/experiments.md):** from scratch, 3 seeds, 95.1% wins against v2b's 76.1%
(seeds pooled +19.0 [+15.4, +22.8] points, 89 of 120 maps better), mostly by turning timeouts into wins; sampled
play now wins as often as argmax (94.3% against 48.7%). Final `losses/explained_variance` was about the same as
v2b's (0.93–0.94), so that check did not predict the gain. Warm starts from v2b seed 1 with the flag stayed near
their source (−4.5 and +4.0 points, both within noise). On Farmshare the runs took about 16 h against v2b's
10.8 h (about 36 steps/s with 16 envs).
