"""Dimension and checkpoint contracts shared by the Tribes environment tooling."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


class MapGeometryError(RuntimeError):
    """Raised when a level cannot satisfy an environment's fixed geometry."""


class CheckpointCompatibilityError(RuntimeError):
    """Raised before loading model tensors when interface metadata is incompatible."""


class ObservationContractError(RuntimeError):
    """Raised when an observation violates the environment interface contract."""


PHASE1_ENVIRONMENT_VERSION = "v5_human_information_parity"

# Game-rules version played by the Java engine (core.RulesVersion), selected with POLYVISION_RULES.
# "tribes_2020" is the original Tribes rule set and leaves every version string unchanged, so existing
# checkpoints stay valid. Any other rule set is folded into the environment version, so a checkpoint trained
# under one rule set is rejected under another. See docs/rules-parity.md.
RULES_ENV_VAR = "POLYVISION_RULES"
DEFAULT_RULES_VERSION = "tribes_2020"
SUPPORTED_RULES_VERSIONS = ("tribes_2020", "polytopia_2_15")


def resolve_rules_version(environ: Mapping[str, str] | None = None) -> str:
    """Return the validated rules version requested by the environment (default ``tribes_2020``)."""
    environ = os.environ if environ is None else environ
    raw = str(environ.get(RULES_ENV_VAR, "") or "").strip().lower()
    if not raw:
        return DEFAULT_RULES_VERSION
    if raw not in SUPPORTED_RULES_VERSIONS:
        raise ValueError(
            f"Unknown {RULES_ENV_VAR}={raw!r}; expected one of {', '.join(SUPPORTED_RULES_VERSIONS)}"
        )
    return raw


# Revision of each non-default rule set. Bump it whenever what the rule set plays, shows or allows changes, so
# checkpoints trained under an earlier revision are rejected. Revision 1 carries no suffix.
#   polytopia_2_15 r1: Step 1 and combat basics (CHANGELOG 052).
#   polytopia_2_15 r2: Step 2, Diplomacy, starting stars, backwards research (CHANGELOG 053).
RULES_REVISIONS = {"polytopia_2_15": 2}


def versioned_for_rules(environment_version: str, rules_version: str) -> str:
    """Fold a non-default rules version (and its revision) into an environment version string."""
    if rules_version not in SUPPORTED_RULES_VERSIONS:
        raise ValueError(f"Unsupported rules version {rules_version!r}")
    if rules_version == DEFAULT_RULES_VERSION:
        return str(environment_version)
    revision = int(RULES_REVISIONS.get(rules_version, 1))
    suffix = f"_r{revision}" if revision > 1 else ""
    return f"{environment_version}+rules_{rules_version}{suffix}"

# Enum constants the Java engine defines only for the 2.15 rules (core.Types ... isActive()). They are appended
# to their enums, so dropping them under other rule sets leaves every older index and catalog unchanged.
RULES_ONLY_VOCAB = {
    "polytopia_2_15": {
        "TECHNOLOGY": ("DIPLOMACY",),
        "UNIT": ("CLOAK", "DAGGER"),
    },
}


def filter_vocab_for_rules(vocab: Mapping[str, Sequence[str]], rules_version: str) -> dict:
    """Drop enum names that do not exist under ``rules_version`` from a Types.java vocabulary."""
    excluded: dict[str, set[str]] = {}
    for rules, per_enum in RULES_ONLY_VOCAB.items():
        if rules == rules_version:
            continue
        for enum_name, names in per_enum.items():
            excluded.setdefault(enum_name, set()).update(names)
    return {k: [n for n in v if n not in excluded.get(k, set())] for k, v in vocab.items()}


ACTION_STAR_COST_SCALE = 50.0

MAX_OWNED_CITIES = 9
CITY_SLOT_FEATURE_DIM = 10
CITY_SLOT_FEATURE_NAMES = (
    "city_present",
    "city_x",
    "city_y",
    "city_level",
    "city_population",
    "city_population_need",
    "city_production",
    "city_supported_unit_count",
    "city_unit_capacity",
    "city_is_capital",
)
CITY_BLOCK_DIM = MAX_OWNED_CITIES * CITY_SLOT_FEATURE_DIM  # 90

SUPPORTED_UNIT_TYPES = (
    "WARRIOR",
    "RIDER",
    "DEFENDER",
    "SWORDMAN",
    "ARCHER",
    "CATAPULT",
    "KNIGHT",
    "MIND_BENDER",
    "BOAT",
    "SHIP",
    "BATTLESHIP",
    "SUPERUNIT",
)
NUM_UNIT_TYPE_CHANNELS = len(SUPPORTED_UNIT_TYPES)  # 12

SUPPORTED_BUILDINGS = (
    "PORT",
    "MINE",
    "FORGE",
    "FARM",
    "WINDMILL",
    "CUSTOMS_HOUSE",
    "LUMBER_HUT",
    "SAWMILL",
    "TEMPLE",
    "WATER_TEMPLE",
    "FOREST_TEMPLE",
    "MOUNTAIN_TEMPLE",
    "ALTAR_OF_PEACE",
    "EMPERORS_TOMB",
    "EYE_OF_GOD",
    "GATE_OF_POWER",
    "GRAND_BAZAR",
    "PARK_OF_FORTUNE",
    "TOWER_OF_WISDOM",
)
NUM_BUILDING_CHANNELS = len(SUPPORTED_BUILDINGS)  # 19

TECHNOLOGY_ORDER = (
    "CLIMBING",
    "FISHING",
    "HUNTING",
    "ORGANIZATION",
    "RIDING",
    "ARCHERY",
    "FARMING",
    "FORESTRY",
    "FREE_SPIRIT",
    "MEDITATION",
    "MINING",
    "ROADS",
    "SAILING",
    "SHIELDS",
    "WHALING",
    "AQUATISM",
    "CHIVALRY",
    "CONSTRUCTION",
    "MATHEMATICS",
    "NAVIGATION",
    "SMITHERY",
    "SPIRITUALISM",
    "TRADE",
    "PHILOSOPHY",
)
NUM_TECHNOLOGIES = len(TECHNOLOGY_ORDER)  # 24

# 2.15 rules only (Diplomacy): extra observation blocks, appended so the 2020 layout is unchanged.
# - DIPLOMACY_UNIT_TYPES: one spatial channel each, placed after the resource channel (still inside the spatial
#   part). Own units are +1 (an own hidden Cloak is +0.5), visible enemy units -1.
# - Diplomacy scalars, placed after the city block: DIPLOMACY researched, then for each tribe id below
#   DIPLOMACY_MAX_TRIBES the controlled tribe's relation flags DIPLOMACY_RELATION_FEATURES (0 for itself).
DIPLOMACY_UNIT_TYPES = ("CLOAK", "DAGGER")
DIPLOMACY_TECHNOLOGIES = ("DIPLOMACY",)
DIPLOMACY_MAX_TRIBES = 4
DIPLOMACY_RELATION_FEATURES = ("peace", "war", "offer_to", "offer_from", "embassy_in", "embassy_from", "broke_peace")
DIPLOMACY_SCALAR_DIM = len(DIPLOMACY_TECHNOLOGIES) + DIPLOMACY_MAX_TRIBES * len(DIPLOMACY_RELATION_FEATURES)  # 29
# Action families the 2.15 rules add to the combat catalog (after RECOVER).
DIPLOMACY_TRIBE_ACTION_TYPES = ("OFFER_PEACE", "ACCEPT_PEACE", "BREAK_PEACE", "BUILD_EMBASSY")
DIPLOMACY_ACTION_TYPES = ("INFILTRATE",) + DIPLOMACY_TRIBE_ACTION_TYPES


def rules_have_diplomacy(rules_version: str | None = None) -> bool:
    rules_version = resolve_rules_version() if rules_version is None else rules_version
    return rules_version == "polytopia_2_15"


NUM_CITY_TERRITORY_CHANNELS = MAX_OWNED_CITIES  # 9
NUM_UNIT_HOME_CITY_CHANNELS = MAX_OWNED_CITIES  # 9
LEGACY_SCALAR_DIM = 6
ECONOMY_SCALAR_DIM = 12


@dataclass(frozen=True)
class OwnedCityState:
    x: int
    y: int
    level: int
    population: int
    population_need: int
    production: int
    supported_unit_count: int
    unit_capacity: int
    is_capital: bool


def extract_owned_cities(
    obs_dict: Mapping,
    tribe_id: int,
) -> list[OwnedCityState]:
    """Extract canonical owned-city states from fog-respecting POV observation dictionary.

    Cities are deterministically sorted by spatial coordinates (x, y) ascending.
    """
    controlled_tribe_id = int(tribe_id)
    cities_map = obs_dict.get("city", {})
    if not isinstance(cities_map, dict):
        return []

    collected: list[OwnedCityState] = []
    for _key, city_data in cities_map.items():
        if not isinstance(city_data, dict):
            continue
        try:
            city_tribe_id = int(city_data.get("tribeID", -1))
        except (TypeError, ValueError):
            continue
        if city_tribe_id != controlled_tribe_id:
            continue

        try:
            x = int(city_data["x"])
            y = int(city_data["y"])
            level = int(city_data["level"])
            population = int(city_data["population"])
            population_need = int(city_data["population_need"])
            production = int(city_data["production"])
            units_list = city_data.get("units", [])
            supported_unit_count = len(units_list) if isinstance(units_list, (list, tuple)) else 0
            unit_capacity = level + 1
            is_capital = bool(city_data.get("isCapital", False))
        except (KeyError, TypeError, ValueError) as exc:
            raise ObservationContractError(f"Malformed city payload in observation: {city_data}") from exc

        collected.append(
            OwnedCityState(
                x=x,
                y=y,
                level=level,
                population=population,
                population_need=population_need,
                production=production,
                supported_unit_count=supported_unit_count,
                unit_capacity=unit_capacity,
                is_capital=is_capital,
            )
        )

    # Sort strictly by (x, y) ascending for deterministic, actor-ID-independent slot mapping
    collected.sort(key=lambda c: (c.x, c.y))
    return collected


def encode_owned_city_slots(
    owned_cities: Sequence[OwnedCityState],
    *,
    width: int,
    height: int,
    max_cities: int = MAX_OWNED_CITIES,
) -> list[float]:
    """Encode owned cities into a fixed-size normalized flat float vector of length max_cities * CITY_SLOT_FEATURE_DIM."""
    if len(owned_cities) > max_cities:
        raise ObservationContractError(
            f"Observed {len(owned_cities)} owned cities, exceeding maximum slot capacity of {max_cities}"
        )

    max_x = float(max(1, width - 1))
    max_y = float(max(1, height - 1))
    encoded: list[float] = []

    for i in range(max_cities):
        if i < len(owned_cities):
            c = owned_cities[i]
            encoded.extend(
                [
                    1.0,  # city_present
                    float(c.x) / max_x,  # city_x
                    float(c.y) / max_y,  # city_y
                    float(c.level),  # city_level
                    float(c.population),  # city_population
                    float(c.population_need),  # city_population_need
                    float(c.production),  # city_production
                    float(c.supported_unit_count),  # city_supported_unit_count
                    float(c.unit_capacity),  # city_unit_capacity
                    1.0 if c.is_capital else 0.0,  # city_is_capital
                ]
            )
        else:
            encoded.extend([0.0] * CITY_SLOT_FEATURE_DIM)

    return encoded


def decode_owned_city_slots(
    city_block: Sequence[float],
    *,
    width: int,
    height: int,
    max_cities: int = MAX_OWNED_CITIES,
) -> list[dict]:
    """Decode a fixed-size city slot tensor back into human-interpretable city dictionaries."""
    expected_dim = max_cities * CITY_SLOT_FEATURE_DIM
    if len(city_block) != expected_dim:
        raise ValueError(f"Expected city block of length {expected_dim}, got {len(city_block)}")

    max_x = float(max(1, width - 1))
    max_y = float(max(1, height - 1))
    cities: list[dict] = []

    for i in range(max_cities):
        offset = i * CITY_SLOT_FEATURE_DIM
        block = city_block
        present = bool(block[offset + 0] >= 0.5)
        if not present:
            continue
        x = int(round(float(block[offset + 1]) * max_x))
        y = int(round(float(block[offset + 2]) * max_y))
        level = int(round(float(block[offset + 3])))
        population = int(round(float(block[offset + 4])))
        population_need = int(round(float(block[offset + 5])))
        production = int(round(float(block[offset + 6])))
        supported_unit_count = int(round(float(block[offset + 7])))
        unit_capacity = int(round(float(block[offset + 8])))
        is_capital = bool(block[offset + 9] >= 0.5)
        cities.append(
            {
                "slot": i,
                "x": x,
                "y": y,
                "level": level,
                "population": population,
                "population_need": population_need,
                "production": production,
                "supported_unit_count": supported_unit_count,
                "unit_capacity": unit_capacity,
                "is_capital": is_capital,
            }
        )
    return cities


@dataclass(frozen=True)
class ObservationLayout:
    width: int
    height: int
    n_tiles: int
    terrain_start: int
    terrain_end: int
    unit_types_start: int
    unit_types_end: int
    city_territory_start: int
    city_territory_end: int
    unit_home_city_start: int
    unit_home_city_end: int
    road_start: int
    road_end: int
    buildings_start: int
    buildings_end: int
    resource_start: int
    resource_end: int
    legacy_scalar_start: int
    legacy_scalar_end: int
    economy_scalar_start: int
    economy_scalar_end: int
    tech_vector_start: int
    tech_vector_end: int
    city_block_start: int
    city_block_end: int
    expected_obs_dim: int
    city_slots: int = MAX_OWNED_CITIES
    city_slot_dim: int = CITY_SLOT_FEATURE_DIM
    # 2.15 rules only; empty ranges (start == end) under the 2020 rules.
    diplomacy_units_start: int = 0
    diplomacy_units_end: int = 0
    diplomacy_scalar_start: int = 0
    diplomacy_scalar_end: int = 0

    @property
    def spatial_channels(self) -> int:
        """Number of per-tile channels, which form one contiguous block at the start of the vector."""
        return self.legacy_scalar_start // self.n_tiles

    @property
    def scalar_dim(self) -> int:
        return self.expected_obs_dim - self.legacy_scalar_start


def observation_layout(width: int, height: int, rules_version: str | None = None) -> ObservationLayout:
    """Observation layout for a board; ``rules_version`` defaults to the POLYVISION_RULES setting."""
    diplomacy = rules_have_diplomacy(rules_version)
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid observation geometry: {width}x{height}")
    n_tiles = width * height

    cur = 0
    terrain_start = cur
    cur += n_tiles
    terrain_end = cur

    unit_types_start = cur
    cur += NUM_UNIT_TYPE_CHANNELS * n_tiles
    unit_types_end = cur

    city_territory_start = cur
    cur += NUM_CITY_TERRITORY_CHANNELS * n_tiles
    city_territory_end = cur

    unit_home_city_start = cur
    cur += NUM_UNIT_HOME_CITY_CHANNELS * n_tiles
    unit_home_city_end = cur

    road_start = cur
    cur += n_tiles
    road_end = cur

    buildings_start = cur
    cur += NUM_BUILDING_CHANNELS * n_tiles
    buildings_end = cur

    resource_start = cur
    cur += n_tiles
    resource_end = cur

    diplomacy_units_start = cur
    if diplomacy:
        cur += len(DIPLOMACY_UNIT_TYPES) * n_tiles
    diplomacy_units_end = cur

    legacy_scalar_start = cur
    cur += LEGACY_SCALAR_DIM
    legacy_scalar_end = cur

    economy_scalar_start = cur
    cur += ECONOMY_SCALAR_DIM
    economy_scalar_end = cur

    tech_vector_start = cur
    cur += NUM_TECHNOLOGIES
    tech_vector_end = cur

    city_block_start = cur
    cur += MAX_OWNED_CITIES * CITY_SLOT_FEATURE_DIM
    city_block_end = cur

    diplomacy_scalar_start = cur
    if diplomacy:
        cur += DIPLOMACY_SCALAR_DIM
    diplomacy_scalar_end = cur

    expected_obs_dim = cur

    return ObservationLayout(
        width=width,
        height=height,
        n_tiles=n_tiles,
        terrain_start=terrain_start,
        terrain_end=terrain_end,
        unit_types_start=unit_types_start,
        unit_types_end=unit_types_end,
        city_territory_start=city_territory_start,
        city_territory_end=city_territory_end,
        unit_home_city_start=unit_home_city_start,
        unit_home_city_end=unit_home_city_end,
        road_start=road_start,
        road_end=road_end,
        buildings_start=buildings_start,
        buildings_end=buildings_end,
        resource_start=resource_start,
        resource_end=resource_end,
        legacy_scalar_start=legacy_scalar_start,
        legacy_scalar_end=legacy_scalar_end,
        economy_scalar_start=economy_scalar_start,
        economy_scalar_end=economy_scalar_end,
        tech_vector_start=tech_vector_start,
        tech_vector_end=tech_vector_end,
        city_block_start=city_block_start,
        city_block_end=city_block_end,
        expected_obs_dim=expected_obs_dim,
        city_slots=MAX_OWNED_CITIES,
        city_slot_dim=CITY_SLOT_FEATURE_DIM,
        diplomacy_units_start=diplomacy_units_start,
        diplomacy_units_end=diplomacy_units_end,
        diplomacy_scalar_start=diplomacy_scalar_start,
        diplomacy_scalar_end=diplomacy_scalar_end,
    )


def validate_fixed_square_geometry(
    loaded_width: int,
    loaded_height: int,
    expected_width: int | None = None,
    expected_height: int | None = None,
    *,
    level_path: str | None = None,
) -> tuple[int, int]:
    loaded_width = int(loaded_width)
    loaded_height = int(loaded_height)
    where = f"\n  level: {level_path}" if level_path else ""
    if loaded_width <= 0 or loaded_height <= 0:
        raise MapGeometryError(
            f"Invalid map geometry: {loaded_width}x{loaded_height}.{where}"
        )
    if loaded_width != loaded_height:
        raise MapGeometryError(
            "Rectangular maps are unsupported.\n"
            f"  loaded map geometry: {loaded_width}x{loaded_height}{where}\n\n"
            "Use a square, dimension-homogeneous level pool."
        )
    if expected_width is not None and expected_height is not None:
        expected_width = int(expected_width)
        expected_height = int(expected_height)
        if loaded_width != expected_width or loaded_height != expected_height:
            raise MapGeometryError(
                "Map geometry mismatch:\n"
                f"  environment geometry: {expected_width}x{expected_height}\n"
                f"  loaded map geometry: {loaded_width}x{loaded_height}{where}\n\n"
                "Mixed-dimension level pools are unsupported."
            )
    return loaded_width, loaded_height


CHECKPOINT_REQUIRED_FIELDS = (
    "map_width",
    "map_height",
    "observation_dim",
    "action_space_n",
    "action_catalog_fingerprint",
    "actor_mode",
    "legal_action_feature_version",
    "legal_action_feature_dim",
    "catalog_version",
    "canonicalizer_version",
    "phase1_opening_version",
    "phase1_environment_version",
    "max_legal_actions",
)

_INTEGER_FIELDS = {
    "map_width",
    "map_height",
    "observation_dim",
    "action_space_n",
    "legal_action_feature_dim",
    "max_legal_actions",
}


def environment_compatibility_metadata(
    wrapper,
    *,
    actor_mode: str,
    max_legal_actions: int | None = None,
) -> dict:
    catalog = getattr(wrapper, "_catalog", None)
    if catalog is None:
        raise RuntimeError("Environment action catalog is not initialized.")
    obs_shape = tuple(int(v) for v in wrapper.observation_space.shape)
    if len(obs_shape) != 1:
        raise RuntimeError(f"Expected a flat observation space, got shape={obs_shape}")
    metadata = {
        "map_width": int(catalog.width),
        "map_height": int(catalog.height),
        "observation_dim": int(obs_shape[0]),
        "action_space_n": int(wrapper.action_space.n),
        "action_catalog_fingerprint": str(wrapper._catalog_fingerprint),
        "actor_mode": str(actor_mode).strip().lower(),
        "legal_action_feature_version": str(wrapper.LEGAL_ACTION_FEATURE_VERSION),
        "legal_action_feature_dim": int(wrapper.ACTION_FEATURE_DIM),
        "catalog_version": str(wrapper.CATALOG_VERSION),
        "canonicalizer_version": str(wrapper.CANONICALIZER_VERSION),
        "phase1_opening_version": str(wrapper.PHASE1_OPENING_VERSION),
        "phase1_environment_version": str(getattr(wrapper, "PHASE1_ENVIRONMENT_VERSION", PHASE1_ENVIRONMENT_VERSION)),
    }
    actual_max_legal_actions = getattr(wrapper, "_max_legal_actions", max_legal_actions)
    if actual_max_legal_actions is not None:
        metadata["max_legal_actions"] = int(actual_max_legal_actions)
    return metadata


def read_checkpoint_metadata(model_path: str) -> dict:
    meta_path = model_path + ".action_interface.json"
    if not os.path.isfile(meta_path):
        raise CheckpointCompatibilityError(
            "Checkpoint metadata is insufficient to establish compatibility.\n"
            f"Missing metadata file: {meta_path}"
        )
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise CheckpointCompatibilityError(
            "Checkpoint metadata is insufficient to establish compatibility.\n"
            f"Could not read {meta_path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise CheckpointCompatibilityError(
            "Checkpoint metadata is insufficient to establish compatibility.\n"
            f"Expected a JSON object in: {meta_path}"
        )
    return data


def _normalized_value(field: str, value):
    if field in _INTEGER_FIELDS:
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    return str(value).strip().lower() if field == "actor_mode" else str(value)


def validate_checkpoint_compatibility(
    checkpoint: Mapping,
    environment: Mapping,
) -> None:
    missing_checkpoint = [k for k in CHECKPOINT_REQUIRED_FIELDS if checkpoint.get(k) is None]
    missing_environment = [k for k in CHECKPOINT_REQUIRED_FIELDS if environment.get(k) is None]
    if missing_checkpoint or missing_environment:
        details = []
        if missing_checkpoint:
            details.append("checkpoint missing: " + ", ".join(missing_checkpoint))
        if missing_environment:
            details.append("environment missing: " + ", ".join(missing_environment))
        raise CheckpointCompatibilityError(
            "Checkpoint metadata is insufficient to establish compatibility.\n"
            + "\n".join(f"  {line}" for line in details)
        )

    fields = list(CHECKPOINT_REQUIRED_FIELDS)
    differences = []
    for field in fields:
        checkpoint_value = _normalized_value(field, checkpoint.get(field))
        environment_value = _normalized_value(field, environment.get(field))
        if checkpoint_value != environment_value:
            differences.append((field, checkpoint_value, environment_value))
    if differences:
        rows = "\n".join(
            f"  {field}: checkpoint={checkpoint_value!r}, environment={environment_value!r}"
            for field, checkpoint_value, environment_value in differences
        )
        raise CheckpointCompatibilityError(
            "Checkpoint is incompatible with the current environment.\n\n"
            f"Differences:\n{rows}\n\n"
            "Use a checkpoint trained for the same environment geometry/action interface."
        )


def compute_level_pool_identity(
    level_paths: Iterable[str],
    *,
    relative_to: str | None = None,
) -> tuple[str, list[dict]]:
    root = os.path.abspath(relative_to) if relative_to else None
    rows = []
    for path in sorted({os.path.abspath(p) for p in level_paths}):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        label = os.path.relpath(path, root) if root else os.path.basename(path)
        rows.append(
            {
                "path": label.replace("\\", "/"),
                "size_bytes": int(os.path.getsize(path)),
                "sha256": h.hexdigest(),
            }
        )
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), rows
