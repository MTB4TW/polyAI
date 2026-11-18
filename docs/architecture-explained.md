# The combat agent, explained at five levels

This describes the model people play against, `combat_pcritic_t1_league_swap_4m` (Devin's 4M-step league run, trained at
commit `fa32201` on `claude/combat-critic-init-96179a`). Numbers come from the code, the checkpoint and its sidecar,
`CHANGELOG.md` and `docs/experiments.md`. Written 2026-11-18.

**The most important finding: the model cannot see unit health.**
- `register_env.py` `_dict_to_array` encodes each unit as type and owner only (+1 ours, −1 the enemy's).
- It has no HP, no veteran status, and nothing saying whether a unit has moved or attacked this turn.
- The 47 per-action features don't include health either.
- So it cannot tell whether an attack kills, or how to order attacks. Adding these inputs is the top model change
  (`feat/combat-unit-status`).

## 1. Soundbite

The agent is a chess player who sees the board only as a photo: where every piece is and whose it is, but not how hurt
any piece is.
- **The coach:** during training, the privileged critic sees through the fog and says after each game which moves
  were really good.
- **The tutor:** early on, the search teacher showed it tactics.
- **Sparring:** now it practises in a league against SimpleAgent and frozen copies of itself, playing both sides of
  the board.
- **The game:** one fixed game, a 20-turn race to take the enemy capital. It has become a near-perfect sprinter at
  that.

## 2. Intuition: what each part does and why

- **The eyes (observation).**
  - **The board picture:** an 11×11 map with 54 layers, one fact per layer: terrain, our units versus enemy units
    by type, city territory, roads, buildings, resources.
  - **The scoreboard:** 161 numbers for stars, techs, cities, and each city's population and income.
  - **Fog:** everything is blanked in fogged tiles.
  - **Why:** a map-shaped input lets the network learn "a rider two tiles from my capital" once and reuse it
    everywhere on the board.
- **The brain (a small ResNet).** Three layers of 3×3 pattern detectors scan the board. Each tile ends up with a
  summary of its 7×7 surroundings, plus an overall summary of the position. Local tactics (adjacency, distance)
  dominate this game.
- **The hands (choosing an action).** Each turn the engine lists every legal move, up to 512. The model scores each
  one from four things:
  - what kind of action it is;
  - 47 hand-made facts about it: does it reveal fog, what does it cost, is it a capture;
  - what the brain saw at its start and target tiles;
  - the overall position.

  It then picks the best-scoring move. Choosing from the legal list, not from all 93,454 possible actions, means it
  never proposes an illegal move and learns faster.
- **The coach (privileged critic).** A second network of the same shape, used only in training, that sees the
  unfogged board and predicts how the game will go. "Was that move better than expected?" is far easier to judge
  when you can see the hidden enemy. The single biggest improvement: 76% to 95%.
- **PopArt** keeps the coach's predictions at a comfortable scale. Without it, 3 of 17 runs ended with a coach that
  predicted the same number for every position.
- **Rewards.** A win is worth a lot (+60, plus a bonus for winning early). Small hints along the way cover cities,
  units, getting closer to the enemy capital and a sensible tech order. With win/loss reward only, almost every game
  gave no signal and nothing was learned.
- **Training (PPO).** Play a batch of games, nudge the network toward moves that turned out better than the coach
  expected, and repeat. Rewards are discounted per turn rather than per action, so a win 15 turns away still counts.
- **The teacher (tactical search).** For the first 30% of training, a short lookahead (three actions deep) shows the
  right tactic and the model is rewarded for copying it. Worth +3.2 points.
- **The league.** Half the games are against SimpleAgent and half against frozen copies of itself and earlier runs,
  weighted toward the copies it struggles against. It plays both seats, because a copy that only ever played one
  seat was easy to beat.

**What it does not understand:**
- health, so it can't tell whether an attack kills or the attacker dies to retaliation;
- the damage formula: it has unit types and terrain but no HP, so it can't calculate;
- the order of attacks;
- which of its units have already moved or attacked;
- reach and threat: no movement-range, zone-of-control or threat map.

It works from learned averages. The tactical search calculates exactly, with the engine's own damage formula and
pathfinder, which is why the teacher helped and why the search found a human game's winning line.

## 3. Eras, and the problem that ended each one

| Era | What it was | The problem that forced the next era |
|---|---|---|
| 0. PolyVision economy, MLP (upstream, May–Aug) | Solo, turn-10 stars per turn. Reached 20.52 (the Phase 1 reference). | Only the economy; a flat network with no sense of where things are on the map |
| 1. Spatial ResNet (fork start) | Map-shaped input | One embedding per action id: the catalog made up 93% of the parameters, and the critic output a constant 90.88 because large raw inputs saturated it |
| 2. Composed actions (Nov 12) | Score actions by type, features and tiles, not by id: 18.15 against 16.68 | Fine for the economy, but no combat yet |
| 3. Combat begins (Nov 12–14) | 1v1, take the capital, 20 turns; potential shaping; per-turn discounting | With win/loss reward only, 89% of games gave no signal. Even with shaping (cap_all) it reached only 47% wins, with 49% timeouts |
| 4. v2 reward + 2.15 rules + composed network (Nov 14) | Heavier weight on the outcome: **76.1%** | Under fog the critic couldn't judge positions |
| 5. Privileged critic (Nov 14–15) | A coach that sees through the fog: **95.1%** (+19 points) | Some coaches collapsed to a constant prediction (3 of 17 runs) |
| 6. Teacher + PopArt (Nov 16) | Search kickstarting: **98.2%**; PopArt prevents the collapse | SimpleAgent stopped being a test, and the agent never played the second seat |
| 7. League + seat swap (Nov 16–17, branch only) | 4M steps: **100%** against SimpleAgent at top-choice play; 79% against its frozen copies in training | First human games: it misses kills and orders attacks badly, and it can't see unit health |
| 8. Next | Health and status inputs, attack-outcome features, decisive-only search at play time, exploiters | |

Dead ends along the way:
- fog memory: 91.5%, no measurable gain;
- outcome-only reward: 62.8%.

## 4. Detailed description, with terms explained

- **Step / episode.** One step is one action. END_TURN passes play to the opponent, which moves inside the same step.
  A game averages about 86 of our decisions, about 9 per turn, and the median win comes on turn 9.
- **Observation (6,695 numbers).**
  - **Planes:** 54 planes × 121 tiles:
    - terrain (1);
    - unit type, signed ±1 for owner (12);
    - territory of each of our 9 possible cities, enemy territory as −1 (9);
    - which city a unit belongs to (9);
    - road (1);
    - buildings, one-hot (19);
    - resource (1);
    - Cloak and Dagger (2).
  - **Scalars (161):** stars, score, cities, kills and turn (scaled); economy figures; 24 tech flags; 9 city slots ×
    10 facts (position, level, population and population needed, production, capacity, capital); 29 diplomacy
    flags.
  - **Fog:** every plane is zeroed on fogged tiles.
  - **Not used for this model:** observation normalisation (`--norm-obs`) and fog memory.
- **Global action id.** A fixed number for every action that can exist (93,454), such as "move from tile a to tile b"
  or "research tech t". It means the same thing in every state.
- **Legal slot.** A position (0–511) in the current state's legal-action list. The policy's probabilities are over
  slots. A slot maps to a global id, which maps to the exact engine action.
- **Action type.** The catalog block an action belongs to: 94 families or variants (MOVE, ATTACK,
  RESEARCH:FORESTRY, …). Each type has a learned 64-number embedding.
- **ResNet backbone.** A 3×3 convolution layer, then 3 residual blocks, at 64 channels. A residual block adds its input
  to its output, which makes it easier to train. The result is pooled into a 128-number board summary, and the
  11×11×64 per-tile grid is kept for scoring actions.
- **State vector h.** The board summary and the scalar summary combined into 64 numbers: the position at a glance.
- **Scoring an action.**
  - Inputs: h (64), type embedding (64), features encoded to 32, and the grid at the source and target tiles
    (64 + 64): 288 numbers in all.
  - A small two-layer network turns them into one score. Invalid slots get −10⁸, and a softmax over slots gives
    the policy π.
- **Privileged critic.** A separate ResNet with a value head that reads the unfogged board and predicts V(s), the
  expected return from this state, used to compute advantages.
- **Advantage / GAE.** How much better an action turned out than V predicted. Generalised Advantage Estimation blends
  short- and long-range estimates (λ = 0.95).
- **PPO.** Proximal Policy Optimization raises the probability of actions with positive advantage, but limits each
  update: the probability ratio is clipped at ±0.2, and an update stops early if the policy moves too far
  (target KL 0.02).
- **PopArt.** The critic predicts standardised returns, tracking their running mean (73.6) and standard deviation (24).
  When those statistics shift, its output layer is rescaled so its predictions don't jump.
- **Potential shaping, Φ.** A score for the position: our total minus theirs over cities, units and closeness to the
  enemy capital. The reward is the change in Φ, which guides learning without changing which strategy is best.
- **Kickstarting.** An extra loss, −log π(teacher's move), that pulls the policy toward the search's choice. Its
  weight fades from 1 to 0 over the first 30% of training.
- **PFSP (prioritised fictitious self-play).** Snapshots are chosen with weight (1 − win rate against them)², so
  harder opponents come up more often.
- **Seat swap.** Half the games are played as the second-moving tribe. The view is relabelled so the model always sees
  itself as tribe 0.
- **Sidecar.** The `.action_interface.json` file next to every checkpoint, recording what the model expects as input,
  so incompatible checkpoints are rejected instead of silently misused.

## 5. Full technical version

- **Environment.**
  - `combat_v5_scaled_scalars_horizon_20+rules_polytopia_2_15_r2`; 11×11 Drylands.
  - Maps: 50 training; 120 validation, 40 for each pairing of Bardur against Imperius, Xin-xi or Oumaji; 120
    held-out test.
  - CAPITALS (Might) mode; turn limit 20; no scripted opening.
  - Catalog fingerprint `8dda48f0…`; 94 action types; 512 legal slots; 47 features
    (`v1_4_parity_spatial_and_cost`).
- **Policy network: 331,682 parameters** (315,105 trained; the policy's own critic head, 16,577, gets no loss when the
  privileged critic is on).

  | Part | Shape | Parameters |
  |---|---|---:|
  | Spatial backbone | stem Conv3×3 54→64; 3 × [Conv3×3 64→64, ReLU, Conv3×3, + skip, ReLU]; global average pool; Linear 64→128, ReLU | 261,056 |
  | Scalar network | 161→64→64, ReLU | 14,528 |
  | Fusion | 192→64, tanh | 12,352 |
  | Type embedding | 94×64 | 6,016 |
  | Feature encoder | 47→32→32, tanh | 2,592 |
  | Scorer | 288→64, tanh, →1 (initial weights std 0.01) | 18,561 |

  All weights start from an orthogonal initialisation.
- **Privileged critic: 292,161 parameters.** Its own backbone, the scalar network, and a 192→64→64→1 head. PopArt v2
  with exponential-average weight 0.1 and the standard deviation floored at 1. The flat-critic guard re-initialises it
  if its predictions' spread stays below 1e-3 for 5 updates (at most 3 times).
- **PPO settings for this run.**

  | Setting | Value |
  |---|---|
  | Environments × steps per environment | 16 × 128 = 2,048 per batch |
  | Minibatches × epochs | 4 × 8 |
  | Learning rate | 2.5e-4, annealed |
  | Clip / target KL | 0.2 / 0.02 |
  | Entropy / value-loss / gradient-norm coefficients | 0.01 / 0.5 / 0.5 |
  | Value clipping | off |
  | Discount | γ = 0.99 per turn; λ = 0.95 |
  | Observation / return normalisation | off |
  | Optimizer | one Adam for policy and critic |

  - Steps: 3,999,744, warm-started from teacher seed 1 (39 tensors copied).
  - Speed: 91 → 53 steps per second; about 20.7 hours on barley-02.
- **Reward (v2).**
  - **Outcome:** win +60, loss −15, timeout −10; early-win bonus 60 × max(0, 1 − turns/20); half of the final Φ paid
    at game end; −0.25 per turn.
  - **Φ weights:** city 1, score per 1000 1, kill 0.25, unit 0.1, unit value 0.05 per star, giant 0.5, tile 0.05,
    road 0.05, rider 0.3, capital proximity 0.3 (late game 0.6), zone of control 0.05.
  - **Event terms:** capital hold 1, kills near the capital 0.5, regional attack 0.3, damage trade 0.02 per HP, and
    tech-order and tech-cost terms.
  - **Typical returns:** wins about 97–114, losses about −20.
- **Teacher.** CombatTactics beam search with depth 3, beam 8, radius 3.

  | Value term | Weight |
  |---|---:|
  | win | 10,000 |
  | material | 1 |
  | cities | 8 |
  | siege | 3 |
  | siege on a capital | 10 |
  | danger | 1 |
  | capital threat | 6 |

  The kickstarting coefficient decayed from 1 to 0 by about 602k steps (iteration 294).
- **League.**
  - **Opponent choice:** SimpleAgent with probability 0.5; otherwise PFSP with weight (1 − p̂)², where
    p̂ = (wins + 1)/(games + 2).
  - **Pool:** 9 fixed opponents (3 seeds each of v2b, privileged critic and teacher) plus the 20 newest snapshots of
    its own, saved every 200k steps. Snapshots sample their moves.
  - **Seats:** seat swap probability 0.5.
  - **Training games:** 46,422 in all: 96.7% won against SimpleAgent and 78.8% against snapshots.
- **Results.**
  - **Canonical-protocol evaluation:** 100.0% at top-choice play, 99.0% sampled, mean win turn 8.8. The folder is in
    Devin's worktree and not yet committed.
  - **Head-to-head games with search variants:** running; see `docs/human-games.md` and `docs/experiments.md`.

## Glossary

| Term | Meaning |
|---|---|
| SPT | Stars per turn: the Phase 1 metric, measured at the end of turn 10 |
| Canonical | The full evaluation protocol; anything smaller is a diagnostic |
| Paired difference | Per-map difference between two evaluations on identical seeds, with a bootstrap CI over maps |
| Φ (potential) | Weighted totals, ours minus theirs; the shaping reward is ΔΦ |
| Snapshot | A frozen checkpoint playing as the opponent |
| Seat | Which tribe the learner controls: 0 moves first, 1 second |
| ZOC | Zone of control: units stop moving next to enemy units |
| CAPITALS mode | The engine's win rule (Polytopia's Might): hold every capital |
| Contract / environment version | A version string that makes incompatible checkpoints be rejected |
| `--init-from` | Warm start from a checkpoint (new inputs get zero weights) |
| `--resume` | Exact continuation of a stopped run (`feat/resumable-training`) |
