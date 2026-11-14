"""Opt-in combat reward terms computed from the engine's true state (`PythonEnv.combatStateJson`).

All functions are pure: they take parsed state dicts and return numbers, so they can be unit-tested without
the JVM. A state dict has "size", "tick", "tribes" {id: {cities, score, kills}}, "units" [{id, tribe, type,
x, y, hp, maxHp, mov, range, cost}], "cities" [{id, tribe, x, y, capital, level, population, populationNeed}],
"tileCity" [x][y] -> city id or -1, and "road" [x][y] -> bool. None of this is part of the observation.

Distances are Chebyshev (king moves). Movement reach ignores terrain, roads and blocking units: it is a
cheap stand-in for the engine's pathfinder, used only for shaping.
"""
from __future__ import annotations


def cheb(ax, ay, bx, by):
    return max(abs(int(ax) - int(bx)), abs(int(ay) - int(by)))


def own_units(state, me):
    return [u for u in state.get("units", []) if int(u["tribe"]) == int(me)]


def enemy_units(state, me):
    return [u for u in state.get("units", []) if int(u["tribe"]) != int(me)]


def enemy_cities(state, me):
    return [c for c in state.get("cities", []) if int(c["tribe"]) != int(me)]


def enemy_capitals(state, me):
    return [c for c in enemy_cities(state, me) if c.get("capital")]


def proximity(d, size):
    """1 on the tile, falling linearly to 0 at the far side of the board."""
    return max(0.0, 1.0 - float(d) / max(1.0, float(size) - 1.0))


def totals(state, me):
    """Per-tribe totals used by the potential: cities, score, kills (engine) and units, unit value, giants,
    territory tiles and roads in territory (derived here)."""
    out = {}
    city_owner = {int(c["id"]): int(c["tribe"]) for c in state.get("cities", [])}
    for tid, t in (state.get("tribes") or {}).items():
        out[str(tid)] = {"numCities": int(t["cities"]), "score": int(t["score"]), "nKills": int(t["kills"]),
                         "units": 0, "unitValue": 0, "giants": 0, "tiles": 0, "roads": 0}
    for u in state.get("units", []):
        t = out.get(str(int(u["tribe"])))
        if t is None:
            continue
        t["units"] += 1
        t["unitValue"] += int(u.get("cost", 0))
        t["giants"] += int(u.get("type") == "SUPERUNIT")
    tile_city, road = state.get("tileCity") or [], state.get("road") or []
    for x, column in enumerate(tile_city):
        for y, cid in enumerate(column):
            owner = city_owner.get(int(cid))
            if owner is None or str(owner) not in out:
                continue
            out[str(owner)]["tiles"] += 1
            if road and road[x][y]:
                out[str(owner)]["roads"] += 1
    return out


# --- potential terms (part of phi, so they telescope) ---------------------------------------------------

def capital_proximity(state, me, targets=None):
    """Sum over our units of proximity to the nearest enemy capital: grows with closeness and unit count.
    `targets` fixes the capital tiles (the env passes the enemy capitals at reset, so taking the capital does
    not make the term drop)."""
    caps = targets if targets is not None else [(c["x"], c["y"]) for c in enemy_capitals(state, me)]
    if not caps:
        return 0.0
    size = state.get("size", 11)
    return sum(proximity(min(cheb(u["x"], u["y"], cx, cy) for cx, cy in caps), size)
               for u in own_units(state, me))


def zone_of_control(state, me, radius=1):
    """Number of tiles in or within `radius` of enemy city territory that at least one of our units can reach
    this turn (Chebyshev distance <= its movement)."""
    size = int(state.get("size", 11))
    enemy_ids = {int(c["id"]) for c in enemy_cities(state, me)}
    tile_city = state.get("tileCity") or []
    if not enemy_ids or not tile_city:
        return 0.0
    near_enemy = set()
    for x in range(size):
        for y in range(size):
            if int(tile_city[x][y]) in enemy_ids:
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if 0 <= x + dx < size and 0 <= y + dy < size:
                            near_enemy.add((x + dx, y + dy))
    reach = set()
    for u in own_units(state, me):
        m = max(1, int(u.get("mov", 1)))
        for dx in range(-m, m + 1):
            for dy in range(-m, m + 1):
                reach.add((int(u["x"]) + dx, int(u["y"]) + dy))
    return float(len(near_enemy & reach))


# --- event terms ------------------------------------------------------------------------------------------

def _region_tiles(state, x, y, radius=2):
    """Tiles of the city/village territory containing (x, y), or a (2*radius+1)^2 window if unclaimed."""
    tile_city = state.get("tileCity") or []
    size = int(state.get("size", 11))
    cid = int(tile_city[x][y]) if tile_city else -1
    if cid >= 0:
        return {(i, j) for i in range(size) for j in range(size) if int(tile_city[i][j]) == cid}
    return {(i, j) for i in range(max(0, x - radius), min(size, x + radius + 1))
            for j in range(max(0, y - radius), min(size, y + radius + 1))}


def regional_superiority(state, me, x, y):
    """+1 if, in the region around (x, y), we have more units or more total unit value than the enemy."""
    region = _region_tiles(state, int(x), int(y))
    ours = [u for u in own_units(state, me) if (int(u["x"]), int(u["y"])) in region]
    theirs = [u for u in enemy_units(state, me) if (int(u["x"]), int(u["y"])) in region]
    more = len(ours) > len(theirs) or sum(u["cost"] for u in ours) > sum(u["cost"] for u in theirs)
    return 1.0 if more else 0.0


def _nearest(units, x, y):
    return min((cheb(x, y, u["x"], u["y"]) for u in units), default=None)


def end_turn_roles(state, me):
    """Positional role score for our units at the end of our turn (sum over units):
    warriors: mean proximity to the nearest enemy unit and nearest enemy city;
    riders: +1 at distance 3-4 from the nearest enemy unit;
    catapults: +1 at distance 3 from the nearest enemy unit, -1 if closer."""
    size = state.get("size", 11)
    foes, foe_cities = enemy_units(state, me), enemy_cities(state, me)
    score = 0.0
    for u in own_units(state, me):
        d_unit = _nearest(foes, u["x"], u["y"])
        if u["type"] == "WARRIOR":
            d_city = min((cheb(u["x"], u["y"], c["x"], c["y"]) for c in foe_cities), default=None)
            parts = [proximity(d, size) for d in (d_unit, d_city) if d is not None]
            score += sum(parts) / len(parts) if parts else 0.0
        elif u["type"] == "RIDER" and d_unit is not None:
            score += 1.0 if d_unit in (3, 4) else 0.0
        elif u["type"] == "CATAPULT" and d_unit is not None:
            score += 1.0 if d_unit == 3 else (-1.0 if d_unit < 3 else 0.0)
    return score


def action_roles(state, me, action):
    """Role score for one of our actions, judged on the state before it:
    archer attack from within the target's range (it can retaliate): -2;
    attacking a rider or archer: +1;
    moving onto a road tile: +0.5."""
    a_type = action.get("type")
    units = {int(u["id"]): u for u in state.get("units", [])}
    if a_type == "ATTACK":
        att, tgt = units.get(int(action.get("unit_id", -1))), units.get(int(action.get("target_unit_id", -1)))
        if att is None or tgt is None:
            return 0.0
        score = 0.0
        if att["type"] == "ARCHER" and cheb(att["x"], att["y"], tgt["x"], tgt["y"]) <= int(tgt.get("range", 1)):
            score -= 2.0
        if tgt["type"] in ("RIDER", "ARCHER"):
            score += 1.0
        return score
    if a_type == "MOVE" and action.get("dst_x") is not None:
        road = state.get("road") or []
        x, y = int(action["dst_x"]), int(action["dst_y"])
        return 0.5 if road and road[x][y] else 0.0
    return 0.0


def hp_lost(prev, cur):
    """HP lost per tribe between two states: drops for surviving units plus all HP of units that vanished
    (killed, or converted to the other side). Healing and new units are not counted."""
    now = {int(u["id"]): u for u in cur.get("units", [])}
    lost = {}
    for u in prev.get("units", []):
        tid = int(u["tribe"])
        v = now.get(int(u["id"]))
        drop = int(u["hp"]) if v is None or int(v["tribe"]) != tid else max(0, int(u["hp"]) - int(v["hp"]))
        lost[tid] = lost.get(tid, 0) + drop
    return lost


def hanging_population(turn_start, end_of_turn, me):
    """Population our cities gained this turn without levelling up (partial upgrades left hanging)."""
    start = {int(c["id"]): c for c in turn_start.get("cities", []) if int(c["tribe"]) == int(me)}
    added = 0
    for c in end_of_turn.get("cities", []):
        s = start.get(int(c["id"]))
        if int(c["tribe"]) != int(me) or s is None or int(c["level"]) != int(s["level"]):
            continue
        added += max(0, int(c["population"]) - int(s["population"]))
    return float(added)


# --- capital terms ---------------------------------------------------------------------------------------

def units_on_enemy_capital(state, me):
    """Our units standing on an enemy-owned capital tile (a capture needs a unit there at the start of our turn)."""
    caps = {(int(c["x"]), int(c["y"])) for c in enemy_capitals(state, me)}
    return float(sum(1 for u in own_units(state, me) if (int(u["x"]), int(u["y"])) in caps))


def kills_in_enemy_capital_region(prev, cur, me):
    """Enemy units that vanished (killed or converted) this step while inside an enemy capital's territory."""
    cap_ids = {int(c["id"]) for c in enemy_capitals(prev, me)}
    tile_city = prev.get("tileCity") or []
    alive = {int(u["id"]) for u in cur.get("units", []) if int(u["tribe"]) != int(me)}
    count = 0
    for u in enemy_units(prev, me):
        if int(u["id"]) not in alive and tile_city and int(tile_city[int(u["x"])][int(u["y"])]) in cap_ids:
            count += 1
    return float(count)


UNIT_TYPES = ("WARRIOR", "RIDER", "DEFENDER", "SWORDMAN", "ARCHER", "CATAPULT", "KNIGHT", "MIND_BENDER",
              "BOAT", "SHIP", "BATTLESHIP", "SUPERUNIT")


def unit_type_counts(state, me):
    """{type: (ours, theirs)} for every unit type."""
    out = {t: [0, 0] for t in UNIT_TYPES}
    for u in state.get("units", []):
        t = out.setdefault(str(u["type"]), [0, 0])
        t[0 if int(u["tribe"]) == int(me) else 1] += 1
    return out


# --- technology terms -------------------------------------------------------------------------------------

TECH_TIER = {t: 1 for t in ("CLIMBING", "FISHING", "HUNTING", "ORGANIZATION", "RIDING")}
TECH_TIER.update({t: 2 for t in ("ARCHERY", "FARMING", "FORESTRY", "FREE_SPIRIT", "MEDITATION", "MINING", "ROADS",
                                 "SAILING", "SHIELDS", "WHALING")})
TECH_TIER.update({t: 3 for t in ("AQUATISM", "CHIVALRY", "CONSTRUCTION", "MATHEMATICS", "NAVIGATION", "SMITHERY",
                                 "SPIRITUALISM", "TRADE", "PHILOSOPHY")})

# Engine TECHNOLOGY enum order (the observation's `technology.researched` list uses it).
TECH_ORDER = ("CLIMBING", "FISHING", "HUNTING", "ORGANIZATION", "RIDING", "ARCHERY", "FARMING", "FORESTRY",
              "FREE_SPIRIT", "MEDITATION", "MINING", "ROADS", "SAILING", "SHIELDS", "WHALING", "AQUATISM",
              "CHIVALRY", "CONSTRUCTION", "MATHEMATICS", "NAVIGATION", "SMITHERY", "SPIRITUALISM", "TRADE",
              "PHILOSOPHY")

# Population a tech unlocks per tile (engine constants): resource id or terrain id on the board.
RESOURCE_FISH, RESOURCE_FRUIT, RESOURCE_ANIMAL, RESOURCE_ORE, RESOURCE_CROPS = 0, 1, 2, 5, 6
TERRAIN_FOREST = 6
POP_TECHS = {
    "ORGANIZATION": ("resource", RESOURCE_FRUIT, 1),   # gather fruit
    "HUNTING": ("resource", RESOURCE_ANIMAL, 1),       # hunt animals
    "FISHING": ("resource", RESOURCE_FISH, 1),         # fish
    "FARMING": ("resource", RESOURCE_CROPS, 2),        # farm on crops
    "MINING": ("resource", RESOURCE_ORE, 2),           # mine on ore
    "FORESTRY": ("terrain", TERRAIN_FOREST, 1),        # lumber hut on forest
}

# Actions that count as using a tech (type, subtype); techs not listed (e.g. CLIMBING) are never "unused".
TECH_USES = {
    "ORGANIZATION": {("RESOURCE_GATHERING", "FRUIT")}, "HUNTING": {("RESOURCE_GATHERING", "ANIMAL")},
    "FISHING": {("RESOURCE_GATHERING", "FISH")}, "WHALING": {("RESOURCE_GATHERING", "WHALES")},
    "FARMING": {("BUILD", "FARM")}, "MINING": {("BUILD", "MINE")},
    "FORESTRY": {("BUILD", "LUMBER_HUT"), ("CLEAR_FOREST", None)},
    "RIDING": {("SPAWN", "RIDER")}, "ARCHERY": {("SPAWN", "ARCHER")}, "SHIELDS": {("SPAWN", "DEFENDER")},
    "SMITHERY": {("SPAWN", "SWORDMAN"), ("BUILD", "FORGE")}, "CHIVALRY": {("SPAWN", "KNIGHT")},
    "MATHEMATICS": {("SPAWN", "CATAPULT"), ("BUILD", "SAWMILL")}, "PHILOSOPHY": {("SPAWN", "MIND_BENDER")},
    "CONSTRUCTION": {("BUILD", "WINDMILL")}, "SAILING": {("BUILD", "PORT")}, "ROADS": {("BUILD_ROAD", None)},
    "TRADE": {("BUILD", "CUSTOMS_HOUSE")}, "FREE_SPIRIT": {("BUILD", "TEMPLE")},
    "MEDITATION": {("BUILD", "MOUNTAIN_TEMPLE")}, "AQUATISM": {("BUILD", "WATER_TEMPLE")},
    "SPIRITUALISM": {("BUILD", "FOREST_TEMPLE"), ("GROW_FOREST", None)},
}


def action_signature(action):
    """(type, subtype) of an action, matching TECH_USES."""
    a_type = action.get("type")
    sub = action.get("resource_type") or action.get("building_type") or action.get("unit_type")
    return a_type, (str(sub) if sub not in (None, "") else None)


def tech_used_by(tech, signatures):
    uses = TECH_USES.get(tech)
    if not uses:
        return True
    return any((t, s) in uses or (t, None) in uses for t, s in signatures)


def pop_unlocked(tech, state, board, me):
    """Population the tech would unlock on unused tiles inside our existing borders (0 for non-pop techs)."""
    spec = POP_TECHS.get(tech)
    if spec is None or not board:
        return 0.0
    kind, value, pop = spec
    ours = {int(c["id"]) for c in state.get("cities", []) if int(c["tribe"]) == int(me)}
    grid = board.get("resource") if kind == "resource" else board.get("terrain")
    building = board.get("building")
    tile_city = state.get("tileCity") or []
    count = 0
    for x, column in enumerate(tile_city):
        for y, cid in enumerate(column):
            if int(cid) in ours and grid[x][y] == value and (building is None or building[x][y] == -1):
                count += 1
    return float(count * pop)


def pop_tech_margin(chosen, alternatives, state, board, me):
    """For a population tech: its unlocked population minus the best other population tech on offer
    (positive if it is the best choice). Non-population techs score 0, so combat techs are not penalised."""
    if chosen not in POP_TECHS:
        return 0.0
    others = [pop_unlocked(t, state, board, me) for t in set(alternatives) if t in POP_TECHS and t != chosen]
    return pop_unlocked(chosen, state, board, me) - (max(others) if others else 0.0)
