"""Action-type table for composed action representations.

The flat global action catalog (`GlobalActionCatalog` in `pol_env/Tribes/py/register_env.py`)
is a sequence of contiguous blocks: one block per (family, variant), each covering every tile
(or tile pair) for that variant. For example `BUILD/SAWMILL` is 121 consecutive IDs, one per
target tile, and `MOVE` is 121*121 IDs, one per (source, target) pair.

An *action type* here is one such block: the action minus its tile(s). Looking up a legal
action's type is a binary search of its global ID over the block start offsets, so it needs no
environment, observation, or feature-contract change. The table is rebuilt from the same
catalog code and vocabulary the environment uses, and checked against the action-space size.
"""

from __future__ import annotations

from typing import List, Tuple

# Families whose IDs carry no tile, so there are no grid features to gather for them. The 2.15 rules
# Diplomacy tribe actions (one ID per target tribe) carry none either.
NON_SPATIAL_FAMILIES = ("END_TURN", "RESEARCH_TECH", "OFFER_PEACE", "ACCEPT_PEACE", "BREAK_PEACE", "BUILD_EMBASSY")

# Must mirror the filters in TribesGymWrapper's catalog construction (register_env.py).
_TRAIN_EXCLUDED_UNITS = ("BOAT", "SHIP", "BATTLESHIP", "SUPERUNIT", "DAGGER")
_CATALOG_RESOURCES = ("ANIMAL", "FISH", "WHALES", "FRUIT", "CROPS", "ORE")


def build_global_action_catalog(width: int, height: int, enable_combat: bool, rules_version: str | None = None):
    """Build the environment's GlobalActionCatalog without starting Java.

    The vocabulary is parsed from `Types.java` exactly as the environment does it, for the given rules
    (default: POLYVISION_RULES), which decide the 2.15-only names and the Diplomacy families.
    """
    from pol_env.Tribes.py.environment_contract import resolve_rules_version, rules_have_diplomacy
    from pol_env.Tribes.py.register_env import GlobalActionCatalog, TribesGymWrapper

    rules_version = resolve_rules_version() if rules_version is None else rules_version
    # _load_action_vocab only reads Types.java and the rules version.
    shell = TribesGymWrapper.__new__(TribesGymWrapper)
    shell.RULES_VERSION = rules_version
    vocab = TribesGymWrapper._load_action_vocab(shell)
    return GlobalActionCatalog(
        width=width,
        height=height,
        tech_types=vocab["TECHNOLOGY"],
        train_unit_types=[u for u in vocab["UNIT"] if u not in _TRAIN_EXCLUDED_UNITS],
        resource_types=[r for r in vocab["RESOURCE"] if r in _CATALOG_RESOURCES],
        building_types=vocab["BUILDING"],
        levelup_choices=vocab["CITY_LEVEL_UP"],
        enable_combat=bool(enable_combat),
        enable_diplomacy=bool(enable_combat) and rules_have_diplomacy(rules_version),
    )


def action_type_table(catalog) -> List[Tuple[str, int, bool]]:
    """Return [(type_name, first_global_id, has_tile)] sorted by first_global_id."""
    n = catalog.n_tiles
    off = catalog.offsets

    def variants(names):
        return list(names) if names else ["ANY"]

    table: List[Tuple[str, int, bool]] = [("END_TURN", off["END_TURN"], False), ("MOVE", off["MOVE"], True)]
    for i, mode in enumerate(("CITY", "VILLAGE", "UNKNOWN")):
        table.append((f"CAPTURE/{mode}", off["CAPTURE"] + i * n * n, True))
    blocks = [
        ("TRAIN", catalog.train_unit_types, n),
        ("RESOURCE_GATHERING", catalog.resource_types, n),
        ("CLEAR_FOREST", None, n),
        ("GROW_FOREST", None, n),
        ("BUILD", catalog.building_types, n),
        ("RESEARCH_TECH", catalog.tech_types, 1),
        ("LEVEL_UP", catalog.levelup_choices, n),
        ("EXAMINE", None, n),
    ]
    if catalog.enable_combat:
        blocks += [("ATTACK", None, n * n), ("RECOVER", None, n)]
    if getattr(catalog, "enable_diplomacy", False):
        from pol_env.Tribes.py.environment_contract import DIPLOMACY_MAX_TRIBES, DIPLOMACY_TRIBE_ACTION_TYPES

        tribe_slots = [f"T{j}" for j in range(DIPLOMACY_MAX_TRIBES)]
        blocks += [("INFILTRATE", None, n * n)] + [(family, tribe_slots, 1) for family in DIPLOMACY_TRIBE_ACTION_TYPES]
    for family, names, size in blocks:
        has_tile = family not in NON_SPATIAL_FAMILIES
        if names is None:
            table.append((family, off[family], has_tile))
            continue
        for i, name in enumerate(variants(names)):
            table.append((f"{family}/{name}", off[family] + i * size, has_tile))
    starts = [s for _, s, _ in table]
    if starts != sorted(starts) or len(set(starts)) != len(starts):
        raise RuntimeError("Action type table is not strictly increasing; catalog layout changed")
    return table


def action_type_table_for_action_space(n_actions: int, width: int, height: int, rules_version: str | None = None) -> List[Tuple[str, int, bool]]:
    """Find the catalog (Phase 1 or combat, active rules) whose size equals n_actions and return its type table."""
    sizes = {}
    for combat in (False, True):
        catalog = build_global_action_catalog(width, height, combat, rules_version)
        if int(catalog.total_size) == int(n_actions):
            return action_type_table(catalog)
        sizes[combat] = int(catalog.total_size)
    raise ValueError(
        f"Composed action representation needs the global action catalog, but action_space_n={n_actions} "
        f"matches neither the Phase 1 ({sizes[False]}) nor the combat ({sizes[True]}) catalog for a "
        f"{width}x{height} board"
    )
