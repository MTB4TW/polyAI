"""Pure helpers for the combat (Phase 2) evaluator: pool manifest, schedule, aggregation, paired statistics.

No Java is needed here; `tools/evaluate_combat.py` runs the games and `tools/compare_combat_evals.py` compares two
evaluations. The headline metric is the win rate against the scripted opponent; loss and timeout rates, and the
turn on which wins happen, are reported alongside it.

Statistics are per map: each map's rate is the mean over its replicates, and confidence intervals bootstrap over
maps (fixed seed), because maps are the unit that varies most. Two evaluations with the same pool, seed and
replicate count play exactly the same (map, episode seed) games, so they can be compared map by map.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.phase1_eval_core import bootstrap_ci, sha256_file, stable_seed

SCHEMA_VERSION = "polyvision-combat-evaluation-v1"
POOL_ROOT_REL = "pol_env/Tribes/levels/phase2_pool_1v1"
MANIFEST_REL = f"{POOL_ROOT_REL}/eval_manifest.json"
EVAL_POOLS = ("eval_validation", "eval_test")
TEST_POOLS = ("eval_test",)
STOCHASTIC_POLICIES = ("ppo_sampled", "random_legal")
ALL_POLICIES = ("ppo_argmax", "ppo_sampled", "random_legal")
# Optional, not part of the canonical set: tactical search where it applies, otherwise ppo_argmax.
OPTIONAL_POLICIES = ("ppo_search",)
OUTCOMES = ("win", "loss", "timeout")

# A canonical combat evaluation: the full eval_validation pool, 5 replicates, seed 42, all three policies.
CANONICAL = {"pool": "eval_validation", "maps": 120, "repeats_per_map": 5, "seed": 42, "policies": set(ALL_POLICIES)}

# Engine tribe keys (core.Types.TRIBE). A map names each tribe on its capital tile as `c:<key>`; the engine
# numbers tribes in the order their capitals appear row by row, and our policy plays tribe 0.
TRIBE_KEYS = {0: "XIN_XI", 1: "IMPERIUS", 2: "BARDUR", 3: "OUMAJI", 4: "KICKOO", 5: "HOODRICK", 6: "LUXIDOOR",
              7: "VENGIR", 8: "ZEBASI", 9: "AI_MO", 10: "QUETZALI", 11: "YADAKK"}
_SEED_RE = re.compile(r"map_2p_(\d+)\.csv$")


@dataclass(frozen=True)
class CombatMap:
    filename: str
    path: str
    csv_sha256: str
    map_seed: int
    our_tribe: str
    opponent_tribe: str


@dataclass(frozen=True)
class CombatGame:
    policy: str
    map: CombatMap
    replicate_index: int
    episode_seed: int
    policy_seed: int | None


def map_tribes(path: Path) -> tuple[str, ...]:
    """Tribes of a map in engine order (tribe 0 first), read from its capital tiles like core.game.LevelLoader."""
    tribes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        for tile in line.split(","):
            parts = tile.split(":")
            if parts[0].startswith("c") and len(parts) > 1 and parts[1].strip():
                tribes.append(TRIBE_KEYS[int(parts[1])])
    return tuple(tribes)


def build_manifest(repo_root: Path, pools: Sequence[str] = EVAL_POOLS) -> dict:
    """Manifest of the evaluation pools: every map's file name, SHA-256, seed and opponent tribe."""
    out = {"schema_version": "polyvision-combat-eval-pools-v1", "pools": {}}
    for pool in pools:
        files = sorted((repo_root / POOL_ROOT_REL / pool).glob("map_2p_*.csv"))
        if not files:
            raise FileNotFoundError(f"No maps in {POOL_ROOT_REL}/{pool}")
        start = int(_SEED_RE.search(files[0].name).group(1))
        rows = []
        for f in files:
            tribes = map_tribes(f)
            if len(tribes) != 2:
                raise RuntimeError(f"{f}: expected two capitals, found {tribes}")
            rows.append({"filename": f.name, "csv_sha256": sha256_file(f), "map_seed": int(_SEED_RE.search(f.name).group(1)),
                         "our_tribe": tribes[0], "opponent_tribe": tribes[1]})
        out["pools"][pool] = {"start_seed": start, "maps": rows}
    return out


def load_verified_pool(repo_root: Path, pool: str, max_maps: int | None = None) -> tuple[dict, list[CombatMap], Path]:
    """Maps of a pool, each checked against the committed manifest (count and SHA-256)."""
    manifest_path = repo_root / MANIFEST_REL
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["pools"].get(pool)
    if entry is None:
        raise ValueError(f"Pool {pool!r} is not in {MANIFEST_REL}")
    on_disk = sorted(p.name for p in (repo_root / POOL_ROOT_REL / pool).glob("map_2p_*.csv"))
    if on_disk != [m["filename"] for m in entry["maps"]]:
        raise RuntimeError(f"{pool}: the maps on disk do not match {MANIFEST_REL}")
    maps = []
    for m in entry["maps"]:
        path = repo_root / POOL_ROOT_REL / pool / m["filename"]
        if sha256_file(path) != m["csv_sha256"]:
            raise RuntimeError(f"{path}: SHA-256 differs from the manifest")
        maps.append(CombatMap(m["filename"], str(path.resolve()), m["csv_sha256"], int(m["map_seed"]),
                              m["our_tribe"], m["opponent_tribe"]))
    if max_maps is not None:
        maps = maps[: int(max_maps)]
    return manifest, maps, manifest_path


def build_schedule(maps: Sequence[CombatMap], policies: Sequence[str], repeats: int, seed: int) -> list[CombatGame]:
    """Every policy plays every map `repeats` times. The opponent is random, so deterministic policies also need
    replicates. The episode seed (map dynamics and the opponent's choices) depends only on map and replicate, so
    every policy and every evaluation with the same seed meets the same games."""
    games = []
    for policy in policies:
        for m in maps:
            for rep in range(int(repeats)):
                episode_seed = stable_seed(seed, "combat_episode", m.csv_sha256, rep)
                policy_seed = stable_seed(seed, "combat_policy", policy, m.csv_sha256, rep) if policy in STOCHASTIC_POLICIES else None
                games.append(CombatGame(policy, m, rep, episode_seed, policy_seed))
    return games


def is_canonical(pool: str, n_maps: int, repeats: int, seed: int, policies: Sequence[str], max_maps) -> bool:
    return (pool == CANONICAL["pool"] and n_maps == CANONICAL["maps"] and max_maps is None
            and repeats == CANONICAL["repeats_per_map"] and seed == CANONICAL["seed"]
            and set(policies) == CANONICAL["policies"])


def per_map_rates(episodes: Sequence[Mapping[str, Any]]) -> dict:
    """{policy: {map_filename: {"win", "loss", "timeout", "games", "mean_win_turn"}}}."""
    out: dict = {}
    for e in episodes:
        cell = out.setdefault(e["policy"], {}).setdefault(e["map_filename"], {"games": 0, "win": 0, "loss": 0, "timeout": 0,
                                                                               "win_turns": [], "matchup": f"{e.get('our_tribe')} vs {e.get('opponent_tribe')}"})
        cell["games"] += 1
        cell[e["outcome"]] += 1
        if e["outcome"] == "win":
            cell["win_turns"].append(int(e["end_turn"]))
    for policy in out.values():
        for cell in policy.values():
            g = cell["games"]
            for k in OUTCOMES:
                cell[k] = cell[k] / g
            turns = cell.pop("win_turns")
            cell["mean_win_turn"] = float(np.mean(turns)) if turns else None
    return out


def aggregate(episodes: Sequence[Mapping[str, Any]], seed: int) -> tuple[dict, list[dict]]:
    """Per-policy summary (rates with bootstrap CIs over maps, by opponent tribe) and per-map rows."""
    rates = per_map_rates(episodes)
    summary, rows = {}, []
    for policy, maps in rates.items():
        names = sorted(maps)
        entry = {"maps": len(names), "games": int(sum(maps[n]["games"] for n in names))}
        for k in OUTCOMES:
            vals = [maps[n][k] for n in names]
            entry[f"{k}_rate"] = float(np.mean(vals))
            entry[f"{k}_rate_ci95"] = bootstrap_ci(vals, stable_seed(seed, "ci", policy, k))
        win_turns = [e["end_turn"] for e in episodes if e["policy"] == policy and e["outcome"] == "win"]
        entry["mean_win_turn"] = float(np.mean(win_turns)) if win_turns else None
        entry["median_win_turn"] = float(np.median(win_turns)) if win_turns else None
        by_matchup = {}
        for matchup in sorted({maps[n]["matchup"] for n in names}):
            sel = [maps[n] for n in names if maps[n]["matchup"] == matchup]
            by_matchup[matchup] = {k + "_rate": float(np.mean([c[k] for c in sel])) for k in OUTCOMES}
            by_matchup[matchup]["maps"] = len(sel)
        entry["by_matchup"] = by_matchup
        summary[policy] = entry
        for n in names:
            rows.append({"policy": policy, "map_filename": n, **{k: v for k, v in maps[n].items()}})
    return summary, rows


def paired_difference(left: Mapping[str, Mapping], right: Mapping[str, Mapping], metric: str, seed: int) -> dict | None:
    """Map-by-map difference left - right of a per-map rate (maps present in both), with a bootstrap CI."""
    shared = sorted(set(left) & set(right))
    if not shared:
        return None
    diffs = np.asarray([float(left[m][metric]) - float(right[m][metric]) for m in shared])
    return {
        "metric": metric, "maps": len(shared), "mean_difference": float(diffs.mean()),
        "ci95": bootstrap_ci(diffs, stable_seed(seed, "paired", metric)),
        "maps_better": int((diffs > 0).sum()), "maps_equal": int((diffs == 0).sum()), "maps_worse": int((diffs < 0).sum()),
    }
