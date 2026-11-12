# Rewards

Phase 1 optimizes a shaped economy-and-expansion reward, not raw game victory and not final stars per turn (SPT) alone.

At each policy step the current reward is:

```text
scaled change in Bardur SPT
+ city-capture bonus
+ exploration and village-progress shaping
+ optional terminal SPT bonus
```

## SPT component

SPT is computed as the sum of production for Bardur-owned cities. A positive stepwise SPT change is multiplied by 5; a zero or negative change uses multiplier 1. This asymmetry makes productive economic actions more salient during the short horizon.

## Active shaping

The wrapper currently adds signals for:

- capturing new cities (3 to 6, capped by the number captured on the step);
- revealing uncaptured villages (+1 each);
- moving closer to a visible uncaptured village (+0.5);
- moving onto a village (+1);
- movement breadcrumbs toward new village progress (+0.5);
- clearing fog (+0.08 per newly visible tile, capped at five tiles per step);
- moving onto a visible neutral village (+5), or missing that opportunity with the selected move (-2);
- a movement that reveals no fog and makes no useful village progress (-0.35).

Two formerly considered hold/move-off-village shaping terms are present as diagnostics but currently have zero weight. `POLYVISION_RESOURCE_GATHER_UPGRADE_FILTER_ENABLED` controls an action filter, not a reward term, and defaults off.

## Terminal reward

Terminal-SPT reward is part of the standard Phase-1 reward configuration (enabled by default). Completion of Turn 10 adds a terminal return bonus:

```text
base_weight × final_SPT
+ over_10_weight × max(0, final_SPT - 10)
+ over_15_weight × max(0, final_SPT - 15)
```

Default weights are `1.0`, `2.0`, and `3.0` respectively and can be overridden with `POLYVISION_TERMINAL_SPT_BASE_WEIGHT`, `POLYVISION_TERMINAL_SPT_OVER_10_WEIGHT`, and `POLYVISION_TERMINAL_SPT_OVER_15_WEIGHT`.

The terminal bonus can be explicitly disabled for controlled ablations or backward comparisons by setting the environment variable `POLYVISION_TERMINAL_SPT_REWARD_ENABLED=0`.

### Terminal-SPT Reference Experiments

Terminal-SPT reward shaping was first introduced in the historical [Phase 1 v3 Seed3 16M Terminal-SPT Reference Run](results/Phase1_V3_Seed3_16M_TerminalSPT_Reference_Run.md) ($19.34$ test argmax SPT), resolving the historical delayed economic investment hurdle and increasing deterministic Forestry adoption from 0.0% to 99.2%.

The active current reference model, [Phase 1 v5 PARITY002 Seed3 16M Terminal-SPT Reference Run](results/Phase1_V5_PARITY002_Seed3_16M_TerminalSPT_Reference_Run.md), was trained under the `v5_human_information_parity` contract with Terminal-SPT enabled by default ($w_{\text{base}}=1.0$, $w_{>10}=2.0$, $w_{>15}=3.0$), achieving **20.17 mean Turn-10 SPT** on the fixed held-out test pool, 100.0% deterministic Forestry adoption, and 12.79 mean Lumber Huts per map.

> [!CAUTION]
> **Single-Seed Scope:**
> These reference models reflect single-seed experiments (Seed 3). While results demonstrate strong empirical performance under Terminal-SPT shaping, multi-seed variance and confidence bounds across multiple independent training seeds remain an active milestone.

## Training Reward vs. Evaluation Metric

It is critical to distinguish between:
1. **Training Reward Objective:** The dense step-level shaping bonuses and terminal SPT multiplier are used exclusively during training rollouts to compute PPO surrogate loss gradients.
2. **Policy Observations at Evaluation Time:** Reward signals and return accumulations are **never** part of the policy's observation vector ($6,424$-dimensional spatial/economic state under `v5_human_information_parity`) during inference or evaluation.
3. **Primary Evaluation Metric:** Final Turn-10 stars per turn (SPT) on held-out maps is the primary capability metric. Total shaped training return is a diagnostic scalar and must not be treated as a measure of policy capability.

Because Phase 1 is shaped and combat-restricted, its return is not a measure of full-game strength. Report final SPT, city count, expansion timing, research, legality/fallback rates, and raw shaped return separately.

## Research Status on Shaping and Delayed Economic Returns

The interaction between dense step-level shaping rewards (such as immediate fruit/animal gathering deltas and fog clearance bounties) and multi-step delayed economic investments (such as Tier-2 Forestry research) was historically problematic under pure step shaping, where deterministic argmax exhibited an Organization monoculture (analyzed in detail in the historical [Phase 1 v3 Seed3 16M Behavioral Failure Analysis](results/Phase1_V3_Seed3_16M_Behavioral_Failure_Analysis.md)).

Enabling the non-stepwise Terminal-SPT bonus established a strong terminal credit-assignment signal that successfully unlocked deterministic Forestry adoption and supported the current reference benchmark of **20.17 test argmax SPT** in v5. Future work will investigate multi-seed confirmation and whether simplified or normalized reward formulations achieve similar or superior credit assignment.

## Phase 2 combat reward

Combat mode (`POLYVISION_COMBAT_MODE=1`, environment version `combat_v4_capitals_engine_potential_horizon_<T>`)
replaces the Phase 1 reward entirely. Combat runs in the engine's CAPITALS mode: a tribe wins as soon as it
holds every capital (in 1v1, taking the enemy capital while keeping its own), and losing its own capital loses.
Games that reach the turn limit `T` (`POLYVISION_MAX_TURNS`, default 20) are timeouts. Each step pays

```text
reward = outcome + (Φ(s') − Φ(s))
outcome = win (default +10) on taking the enemy capital, loss (default −10) on losing ours, else 0
Φ(s)   = city            × (our cities − their cities)
       + score_per_1000  × (our Tribes score − their score) / 1000
       + kill            × (our unit kills − their unit kills)
```

The second term rewards changes in a weighted "who is ahead" score. Its sum over an episode is
Φ(final) − Φ(start), so a game that reaches the turn limit still pays out the final advantage instead of
zero. That also means the agent optimizes `outcome + Φ(final)`, not the outcome alone: being ahead at the
turn limit is rewarded in its own right. (Textbook potential shaping leaves the optimal policy unchanged
only with γΦ(s′) − Φ(s) and Φ = 0 at episode end; this reward deliberately does neither.)

Φ is computed from the engine's true game state (`_combat_engine_totals`), not from the policy's
observation. Up to `combat_v3` it read the observation, which is fog-limited: enemy cities and kills were
undercounted, so discovering an enemy city lowered Φ and was penalized. The policy never observes Φ.

Weights default to `TribesGymWrapper.COMBAT_REWARD_DEFAULTS` and can be overridden per run without code
changes, e.g. `POLYVISION_COMBAT_REWARD_CITY=2 POLYVISION_COMBAT_REWARD_KILL=0.5 sbatch cluster/train_farmshare_combat.sbatch`:

| Variable | Default | Meaning |
|---|---:|---|
| `POLYVISION_COMBAT_REWARD_WIN` | 10 | terminal reward for a win (must be > 0) |
| `POLYVISION_COMBAT_REWARD_LOSS` | −10 | terminal reward for a loss (must be < 0) |
| `POLYVISION_COMBAT_REWARD_CITY` | 1.0 | potential per city of advantage |
| `POLYVISION_COMBAT_REWARD_SCORE_PER_1000` | 1.0 | potential per 1000 points of Tribes score advantage |
| `POLYVISION_COMBAT_REWARD_KILL` | 0.25 | potential per kill of advantage |

Each step's `info` reports `combat_outcome_reward`, `combat_shaping_reward` and `combat_potential`. The last
step of a game also reports engine-side diagnostics: `combat_end_enemy_cities`, `combat_end_own_cities`,
`combat_end_enemy_capitals_held`, `combat_end_min_unit_dist_to_enemy_capital` (Chebyshev, −1 with no units)
and `combat_first_contact_tick` (first turn an enemy unit or city was visible, −1 if never).
Under `combat_v3`, random legal play against `SimpleAgent` on the 1v1 pool (20-turn horizon), every episode now has a
nonzero return (timeouts range about −7 to +5, losses about −15), where the previous ±1-only reward was
zero in about 89% of episodes. The defaults are a starting point, not a tuned design.

The evaluation metric for combat is win rate against fixed opponents on held-out maps, not shaped return.
