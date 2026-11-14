# Rules parity: Tribes engine vs. current Polytopia

Status (November 2026): everything below is implemented behind `POLYVISION_RULES=polytopia_2_15` except the
2.8 naval rework (CHANGELOG entries 052 and 053). Steps 1-3 came first (052); Step 2, Diplomacy, the 2.14.0
starting stars and backwards research followed (053, rules revision 2). The default, `tribes_2020`, is the
original rule set, unchanged.

The Java engine in `pol_env/Tribes/src/` is the Tribes research framework (Perez-Liebana et al., 2020) as
inherited through PolyVision. Our Phase 1 maps are harvested from the real game (`tools/polytopia_harvester`),
but the rules that play out on them are Tribes' rules. This document lists every rule difference we found
between the engine and the current game, when the game changed it, and which part of our work it affects.

## Sources

- Engine: `pol_env/Tribes/src/core/TribesConfig.java`, `Types.java`, `TechnologyTree.java`, `actors/City.java`,
  `actors/Tribe.java`, `game/GameState.java`, and the commands under `actions/`.
- Game: the fandom wiki (read through its MediaWiki API, since page loads are bot-blocked) and the official wiki
  at `wiki.polytopia.com` (Midjiwan's own changelogs, the only source covering 2025-2026 updates).
  - https://polytopia.fandom.com/wiki/Update_History
  - https://wiki.polytopia.com/wiki/update-history
  - https://wiki.polytopia.com/wiki/whatsnew-balance-adjustments-2025 (2.14.0)
- "Current" means the latest version found: 2.18 (2026). Updates 2.16-2.18 contain no balance changes, so the
  current rules are effectively those of **2.15.0 (29 Oct 2025)**.
- Items marked **unverified** came from a single fan source or could not be confirmed.

## Which version does the engine follow?

The engine predates every dated change we could check. Each of these is the pre-change behaviour:

| Engine behaviour | Changed in |
|---|---|
| Ruins can give a Giant (super unit) | 2.0.58 Balance Pass 2 (2021) |
| Mountain defence bonus comes from Meditation | 2.0.58 |
| Literacy-style discount is 20% (but see the bug below) | 2.0.44 (17 Feb 2021) |
| Clear Forest gives +2★ in Forestry | 2.0.49 |
| Shields tech, Knight with 15 HP | 2.2 (May 2022) |
| Whaling tech, whale hunting (+10★), Customs House, Boat/Ship/Battleship | 2.8.0 Path of the Ocean (Nov 2023) |

So the engine follows **Polytopia as of 2020, before 2.0.31 (Dec 2020)**, which matches the Tribes paper. That
is about five years and some fifteen balance updates behind the current game.

## What actually matters for Phase 1

Phase 1 is a narrow slice of the game, so most differences do not touch it. From the 250 validation maps
(counted from the CSVs) and the wrapper's action filter:

- **Terrain:** plains (60%), forest (36%), mountains, 7 villages and 4 ruins per map on average, one Bardur
  capital. No water, no crops, no fish and no whales.
- **Resources:** fruit, wild animals, ore.
- **Allowed actions** (`ALLOWED_ACTION_TYPES` in `register_env.py`): end turn, move, capture, examine (ruins),
  spawn, resource gathering, clear forest, grow forest, level up, research, build. Burn forest, destroy and
  roads are **not** available. Attacks are excluded.
- **No opponents** (solo mode), so meeting tribes, diplomacy, combat and occupied-city rules do not apply.
- **Horizon:** to Turn 10, so tier-3 techs and level-5 cities are reachable but rare.

The reachable economy is therefore: Hunting (start) → Archery, Forestry; Climbing → Mining, Meditation;
Organization → Farming (useless here: no crops), Strategy/Shields; Riding → Roads, Free Spirit; then the tier-3
techs (Mathematics/Sawmill, Smithery/Forge, Philosophy, Spiritualism, Trade, Construction, Chivalry).

## Gap analysis

Phase columns: **P1** = Phase 1 solo Bardur economy (11×11 Drylands, Turn 10); **P2** = 1v1 combat;
**-** = neither (other tribes, water, score only). "High/low" is how often the rule is likely to come into play.

### Economy

| # | Rule | Engine | Current game | Changed in | Affects |
|---|---|---|---|---|---|
| E1 | Lumber Hut cost | 2★ (+1 pop) | 3★ (+1 pop) | 2.8.0 | **P1 high** |
| E2 | Clear Forest reward | +2★ | +1★ | 2.2 (was 2 → 1 in 2.0.49 → 2 in 2.0.58 → 1 in 2.2) | **P1 high** |
| E3 | Literacy discount (needs Philosophy) | `cost * 0.2`, i.e. you pay **20%** of the cost. This is a bug: Tribes meant "20% off" | 33% off, rounded up (Literacy, Philosophy) | 2.0.44 (was 20% off before) | **P1 low**, P2 |
| E4 | Lumber Hut terrain | Building it turns the forest tile into plain (`BuildCommand`) | The tile stays forest | never was plain in the game | **P1 medium**: changes the observation, and allows plain-only buildings and Grow Forest on that tile afterwards |
| E5 | Park level-up reward (level 5+) | 250 points only | +1★ per turn and 250 points | 2.8.0 | **P1 low** (level 5 by Turn 10 is rare) |
| E6 | Customs House → Market (Trade) | Customs House 5★, +2★ per adjacent Port | Market 5★: +1★ per level of each adjacent Sawmill, Windmill and Forge, capped at level 8 (the official wiki's Star page says 2★ per level; the per-level value is from fandom and play) | 2.8.0 (Market added), 2.8.5 (current income rule) | **P1 low/medium**: with no ports the Customs House is useless on our maps, while the Market is a real star source once Trade is reached. Also P2 |
| E7 | Forge placement | Plains only | Plains or forest | 2.14.0 | **P1 low** (Smithery is tier 3) |
| E8 | Negative population | City production equals its (negative) population | Each negative pop removes 1★, floor 0 per city | not dated | - (needs trade-network losses) |
| E9 | Port | 10★, +2 pop, Sailing | 7★, +1 pop, Fishing | 2.8.0 (cost/pop), 2.8.5 (tech) | P2 / - (no water in Phase 1) |
| E10 | Whale hunting | Whaling, 0★, +10★ | Removed; Starfish (Navigation, +8★) | 2.8.0, 2.8.5 | P2 / - |
| E11 | Roads | 2★ | 3★, plus Bridges (5★) | 2.8.0, 2.8.5 | P2 (roads are not a Phase 1 action) |
| E12 | Burn Forest | 5★, Chivalry | 3★, Construction | 2.8.5 (tech), 2.14.0 (cost) | P2 (not a Phase 1 action) |
| E13 | Destroy | Construction | Chivalry | 2.8.5 | P2 (not a Phase 1 action) |
| E14 | Embassies, Peace Treaty, Diplomacy tech | absent | present | 2.2 | P2 |
| E16 | Stars for meeting a tribe | Score table already modernised by PolyVision: 3★ below 1,000 points, then +2★ per 1,000, max 11★ at 4,000+. **Bug:** the tribe being met never gets stars (`meetTribe` is called with the meeting tribe's own ID and returns early) | 1★ + 2★ per 1,000 of the other tribe's score, rounded up to the next thousand, max 11★ (official wiki). Equal to the engine's table except exactly at multiples of 1,000. Reported from play: a tribe meets another only by seeing its unit or city on its own turn; the second player gets 2★ extra (**unverified**, no written source) | 2.0.58 | P2 (no other tribes in Phase 1) |
| E15 | Starting stars | 5★ for every tribe | 5★ default (Bardur 5★); some tribes 6-7★, Luxidoor 2★ plus a level-3 capital | 2.14.0 | P2 only if the opponent is one of those tribes |

Unchanged and matching: harvest fruit/animal 2★ +1 pop; Farm and Mine 5★ +2; Sawmill and Windmill 5★, +1 per
adjacent Lumber Hut/Farm; Forge +2 per adjacent Mine; temples 20★ (Forest Temple 15★) +1 pop; Grow Forest 5★;
city income = level, +1 for the capital, +1 Workshop; population to reach level n is n; level-up choices
(Workshop/Explorer, Wall/5★, +3 pop/borders, Park/Giant); units per city = level + 1; tech cost =
4 + tier × cities; 5★ at the start; occupied cities produce nothing.

### Tech tree

| # | Rule | Engine | Current game | Changed in | Affects |
|---|---|---|---|---|---|
| T1 | Shields | Shields (Defender) | Renamed Strategy (Defender, Peace Treaty), with Diplomacy as its tier-3 child | 2.2 | P1 naming only; P2 |
| T2 | Fishing branch | Fishing → Sailing (Port, boats) → Navigation; Fishing → Whaling → Aquatism | Fishing (Port, Raft) → Sailing (Scout, ocean) → Navigation (Bomber, Starfish); Fishing → Ramming (Rammer) → Aquatism | 2.8.0, 2.8.5, 2.10.0 | P2 / - |
| T3 | Out-of-order research | Only children of researched techs | A tech's parent can be researched "backwards" | 2.14.0 | - for Bardur (needs a tier-2/3 start tech; a ruin only grants researchable techs) |
| T4 | Mountain Temple tech | Meditation | Meditation; moved to Philosophy for Cymanti only (**unverified** whether it applies to all tribes) | 2.15.0 | P1 low if it is general |

The tiers and every land branch other than Shields match.

### Ruins, villages and exploration

| # | Rule | Engine | Current game | Changed in | Affects |
|---|---|---|---|---|---|
| R1 | Ruin unit reward on land | Giant (super unit) | Veteran Swordsman | 2.0.58 | **P1 medium**: 4 ruins per map, 1-in-5 chance each. A Giant also fills a unit slot |
| R2 | Ruin unit reward on water | Battleship carrying a Warrior | Veteran Rammer carrying a Warrior | 2.0.58 → 2.8.0 (Bomber) → 2.8.5 (Rammer) | - (no water) |
| R3 | Ruin Explorer reward | Always possible | Only if a tile in the 5×5 around the ruin is unexplored | not dated | P1 low |
| R5 | Lighthouses | absent; the Explorer task (Eye of God) is Navigation plus clearing all fog | A lighthouse on each map corner, always fogged at the start. Revealing one gives the capital (or the oldest city) +1 population; finding all four completes the Explorer task, with no tech needed | 2.8.0, 2.8.5 (always start fogged) | **P1 medium**: an 11×11 corner is reachable by Turn 10, and 113 of the training maps put the capital within view of a corner |
| R4 | Explorer movement | 15 random steps | 12 steps (**unverified**: fandom only, Feb 2026; reported from play as 11 or 12). Since game version 115 the explorer steers (developer description, shared by Devin): every tile that would reveal fog is scored by how many fog tiles it reveals (1/2/3/4+ → 173/163/153/143, or 140/130/120/110 when one of them is a lighthouse); a 3-step breadth-first search gives tiles further out +100 per step; each step goes to the adjacent tile with the lowest score, ties at random, never straight back, and the explorer vanishes if it cannot move | v115, 2.15.0 ("Improved explorer movement"), 2026 | P1 medium (the Explorer level-up and ruin reward) |

Matching: ruin rewards drawn with equal weight (10★, a random researchable tech, +3 pop to the capital, an
Explorer, a unit); villages and ruins need a unit that started its turn on the tile (`isFresh`).

### Units and combat (Phase 2)

| # | Rule | Engine | Current game | Changed in |
|---|---|---|---|---|
| C1 | Knight | 8★, 15 HP | 8★, 10 HP | 2.2 (cost was 10★ from 2.0.58 to 2.2) |
| C2 | Swordsman fortify | Can fortify (city defence bonus) | Cannot | 2.8.0 |
| C3 | Mountain defence bonus | Needs Meditation | Needs Climbing | 2.0.58 |
| C4 | Naval units | Boat → Ship (+5★) → Battleship (+15★) | Raft → Scout (+5★, Sailing), Rammer (+5★, Ramming) or Bomber (+15★, Navigation; splash, stiff) | 2.8.0, 2.8.5 |
| C5 | Giant on a Port | n/a | Becomes a Juggernaut | 2.8.0 |
| C6 | Cloak, Dagger | absent | present (Diplomacy) | 2.2 |
| C7 | Push direction when a unit is spawned onto an occupied tile (Giant level-up, ruin unit) | Fixed order from the tile: (0,+1), (−1,0), (0,−1), (+1,0), then the diagonals; no memory of the unit's movement (`Board.pushUnit`) | Each unit remembers its last direction: friendly units are pushed the way they last moved, enemy units the opposite way, ranged units the way they last moved or shot, unmoved units towards the map centre (south from the exact centre); if blocked it tries counterclockwise, then clockwise, one step at a time (fandom Movement page, "Forced Unit Spawns") | not dated | P2 (sieges); **P1 low**: under `polytopia_2_15` a ruin's Swordsman pushes the unit that explored it |

Matching: Warrior, Archer, Defender, Rider, Catapult, Mind Bender and Giant stats; the damage formula
(`round(attackForce/total × attack × 4.5)`, retaliation likewise); defence ×1.5 (forest with Archery, water with
Aquatism, city without walls for fortifying units) and ×4 behind walls; veteran after 3 kills (+5 max HP);
healing +2, +4 inside borders; Mind Bender heal +4; any unit can capture (the game allowed this from 2.0.31). A unit that cannot be pushed
anywhere is destroyed (the "Houdini" or "Thanos snap"), as in the game (`GameState.pushUnit`).

### Score only (neither phase)

Temples level every 2 turns (was 3) since 2.8.5 and are worth 100 per level (was 50) since 2.14.0. Phase 1 is
scored by stars per turn and Phase 2 by win rate, so score rules do not matter to either.

## Proposed order of changes

All changes go behind one rules switch with two values:

- `tribes_2020` (default): today's rules, bit for bit. Existing checkpoints and results stay valid and
  reproducible, and `phase1_environment_version` keeps its current value.
- `polytopia_2_15`: the rules below. Selecting it changes `phase1_environment_version` (and the combat
  environment version), so a checkpoint trained under one rule set is rejected under the other.

Proposed mechanism: a `RulesVersion` setting in `TribesConfig` that the rule code consults, selected by an
environment variable (`POLYVISION_RULES`) that the JVM inherits from the wrapper. This avoids touching
`PythonEnv.java`, which most open PRs also edit. The wrapper reads the same variable, folds it into the version
strings, and asks the JVM which rules it is running so that a mismatch fails closed.

## Using the rules switch

```bash
POLYVISION_RULES=polytopia_2_15 .venv/bin/python tools/evaluate_phase1.py ...
```

- `tribes_2020` (default, also when unset): the original rules.
- `polytopia_2_15`: Steps 1-3 and Diplomacy below. The environment version gains a `+rules_polytopia_2_15_r2`
  suffix (`r2` = the second revision of these rules; the first had no suffix), so a checkpoint trained under
  another rule set or an earlier revision is rejected.
- Results under different rule sets are not comparable; record the rule set with every result.
- Not changed even under `polytopia_2_15`: the 2.8 naval rework (C4, C5, E9, E10, T2, R2). Ports, boats and
  whaling keep the 2020 rules and the ruin reward on water is still a Battleship. None of our maps has water.
- Also under `polytopia_2_15`: forest actions (clear, burn, grow) are refused on a tile with a building, as in
  the game. This matters once Forges can stand on forest.

**Step 1: Phase 1 economy (small, well-sourced constant and logic changes)** (implemented)

1. E1 Lumber Hut 3★.
2. E2 Clear Forest +1★.
3. E3 Literacy: 33% off, rounded up. The current behaviour (pay 20%) is kept under `tribes_2020`.
4. R1 Ruin land unit: veteran Swordsman instead of a Giant.
5. E5 Park: +1★ per turn.
6. E7 Forge allowed on forest.
7. R5 Lighthouses: corners start fogged, +1 population to the capital per lighthouse found, Eye of God for
   all four.
8. E16 Meeting: a tribe meets another only when it sees that tribe's unit or territory on its own turn: when
   its view clears during its turn, or at the start of its turn for anything already visible. Being seen is
   not meeting. The tribe later in turn order (in 1v1, the second player) gets 2★ extra (**unverified**,
   reported from play). A tribe never meets itself (under `tribes_2020` it does, while the level loads; the
   start-of-game star reset hides it). Phase 2 only.
9. R4 Explorer: 12 steps (**unverified** count) with the version-115 steering rule above. Mountain fog does not
   count towards a tile's score. When no fog is within reach the explorer walks at random (the description
   does not cover that case).

The scripted opening (two animal harvests and a Workshop on 5★) still completes on every train and validation
map under these rules (opening audit, 5,250 maps), so `phase1_opening_version` can stay. The Phase 1 reference numbers (24.5 SPT, 20.52) would need re-running under
the new rules before comparing.

**Step 2: Phase 1, larger or less certain** (implemented, CHANGELOG 053)

10. E4 Lumber Huts stay on forest. A forest tile with a building cannot be cleared, burnt or grown on.
11. E6 Market replaces the Customs House in the same building slot (`CUSTOMS_HOUSE` in the interfaces, 5★,
    Trade, plain). It must stand next to a Sawmill, Windmill or Forge and earns +1★ per turn for each level of
    each adjacent one of its tribe (a Sawmill's level is its number of adjacent Lumber Huts, and so on), at most
    8★. The income counts in its city's production, so it shows in SPT, and is still paid when the city is
    occupied or infiltrated. Ports no longer pair with anything. The legal-action feature "expected immediate
    SPT delta" stays 0 for Markets.
12. R3 Ruin Explorer only when a tile in the 5×5 around the ruin is still fogged; otherwise another reward is
    drawn.
13. T1 Shields is Strategy (`TECHNOLOGY.getDisplayName()`; the identifier stays `SHIELDS`). It enables peace
    treaties (Diplomacy below).

**Step 3: Phase 2 combat (Kevin's area)** (implemented; please review, Kevin)

14. C1 Knight 10 HP; C2 Swordsman no fortify; C3 mountain defence from Climbing.
    C7 push directions (implemented). Each unit stores the direction of its last move, push or ranged shot.
    For a move of several tiles that is the overall direction (the sign of the displacement on each axis), not
    the last step. "South" is +x (x is the board row). The blocked-tile fallback alternates counterclockwise and
    clockwise, one step further each time. Saved-game JSON does not store the direction.
15. E11-E13 roads 3★, Burn Forest 3★/Construction, Destroy/Chivalry.

**Step 4: the rest (implemented, CHANGELOG 053; Diplomacy is Kevin's area, please review)**

16. E15 starting stars (2.14.0): Xin-xi 7★, Oumaji 6★, Hoodrick 7★, Quetzali 7★, Yădakk 7★, Luxidoor 2★ (its
    level-3 capital was already a Tribes rule), every other tribe 5★. Ai-Mo starts with Philosophy and a Mind
    Bender instead of Meditation and a Warrior. Bardur, our Phase 1 tribe, is unchanged.
17. T3 backwards research (2.14.0): a tech is researchable when its parent is researched or when one of its
    children is (Vengir can go Smithery → Mining). The cost stays the tech's tier cost. This only matters for
    tribes that start with a tier-2 or tier-3 tech, or after a ruin gives one.
18. C6, E14 Diplomacy, described below.

**Not implemented:** the 2.8 naval rework (C4, C5, E9, E10, T2, R2), by decision: none of our maps has water.

## Diplomacy (2.1/2.2) under `polytopia_2_15`

Sources: fandom Diplomacy, Strategy, Embassy, Cloak, Dagger, Tribe Relations and Unit Skills pages (read
November 2026), official wiki Embassy page and the 2.14.0 changelog.

- **Tech:** Diplomacy, tier 3 after Strategy (`SHIELDS`). It unlocks the Cloak and embassies, and gives vision
  of the capitals of the tribes you have met.
- **Relations** (`core.TribeRelations` on the Board): at war if either tribe attacked the other in this or the
  previous round; at peace if both agreed a treaty; neutral otherwise.
- **Peace:** `OFFER_PEACE` needs Strategy; the other tribe answers with `ACCEPT_PEACE` on its turn (Strategy not
  needed), or the offer lapses at the end of that turn. At peace, no attack, capture, conversion or
  infiltration either way; units pass through each other, ignore each other's zones of control and use each
  other's roads. `BREAK_PEACE` ends a treaty: the breaker's units are frozen for the rest of the turn, its units
  inside the other tribe's borders are disbanded without refund, and it cannot be hostile toward that tribe
  until its next turn.
- **Embassy:** `BUILD_EMBASSY` needs Diplomacy. Since 2.14.0 the first embassy a tribe places costs 4★ and each
  later one 2★ more (the changelog says only "begins cheaper, rises with the number of embassies"; the numbers
  were reported by Devin). Destroyed embassies still count toward the next one's cost. It needs the
  other tribe's capital to still be held by that tribe and no war between them, and allows one embassy per
  tribe. It reveals the 3×3 around that capital. Both owner and host earn +2★ per turn, +4★ at peace. An attack
  either way destroys the embassies between the two; capturing the host capital destroys every embassy in it.
  Embassy income is not city production, so it is not part of SPT.
- **Cloak:** 8★, 5 HP, attack 2, defence 0.5, 2 moves, range 1. Hide: invisible to other tribes (except those at
  peace with its owner) after it moves; visible again when trained, after capturing a village or examining a
  ruin, and when an enemy unit tries to move onto its tile (that move is cancelled and costs nothing). Creep:
  forest and mountain cost one move, no roads, no zones of control, passes through enemy units. Scout: sees
  5×5. Stiff (never retaliates), Static (never a veteran). It cannot attack units or capture cities; it can only
  `INFILTRATE` an adjacent enemy city, also after moving if it began the turn hidden (Hide and Dash conflict).
  The game's "eye" warning that a hidden Cloak is adjacent is not modelled.
- **Infiltration:** a unit of the city's owner in the city takes the damage of a full-health Cloak attack; the
  Cloak is consumed; min(city level, 5) Daggers spawn in the city's borders (city centre first, then forest
  with Archery or mountain with Climbing, then other land; never water, where the game spawns Pirates); the
  infiltrator gains stars equal to the city's income, and the city earns nothing on its owner's next turn and
  cannot be infiltrated again until then. Not possible while another tribe's unit occupies the city.
- **Dagger:** 10 HP, attack 2, defence 2, 1 move. Never trained. Dash, Surprise (no retaliation when it attacks),
  Independent (no home city), Static. It cannot act on the turn it spawns.
- **Scripted opponent** (`SimpleAgent`): infiltrates when it can and builds embassies, but never offers,
  accepts or breaks peace, so against it peace is impossible.
- **Interface:** the Cloak is a `TRAIN` unit and Diplomacy a `RESEARCH_TECH` tech. In combat mode the catalog
  gains `INFILTRATE` (Cloak tile × city tile) and `OFFER_PEACE`, `ACCEPT_PEACE`, `BREAK_PEACE` and
  `BUILD_EMBASSY` (one slot per target tribe id, ids 0-3). The 47-d legal-action features are unchanged: the
  new types use the "other type" flag, as `ATTACK` does, and the action embedding tells them apart. The
  observation gains two spatial channels (Cloak, Dagger: own +1, an own hidden Cloak +0.5, visible enemy −1)
  after the resource channel, and 29 scalars after the city block (Diplomacy researched, then for tribe ids
  0-3: peace, war, offer sent, offer received, own embassy there, their embassy here, treaty broken this turn).
  Under 2.15 rules: 54 × 121 + 161 = 6,695 dimensions (6,424 under 2020 rules), in Phase 1 as well.
  `spatial_resnet` reads its sizes from `observation_layout()`.
