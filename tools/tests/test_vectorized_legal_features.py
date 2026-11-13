"""The vectorised legal-action feature builder must match the previous per-tile builder byte for byte.

Offline tests run on random synthetic boards and actions (no Java). The live test is opt-in:
POLYVISION_RUN_JAVA_TESTS=1 python -m unittest tools.tests.test_vectorized_legal_features
"""
import os
import random
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pol_env.Tribes.py.register_env import GlobalActionCatalog, TribesGymWrapper

ROOT = Path(__file__).resolve().parents[2]

TECHS = ["CLIMBING", "FORESTRY", "ORGANIZATION", "FISHING", "HUNTING"]
RESOURCES = ["ANIMAL", "FRUIT", "FISH", "CROPS", "ORE", "WHALES"]
BUILDINGS = ["LUMBER_HUT", "SAWMILL", "MINE", "FARM"]
LEVELUPS = ["WORKSHOP", "EXPLORER", "CITY_WALL", "RESOURCES"]


def bare_wrapper(catalog):
    w = TribesGymWrapper.__new__(TribesGymWrapper)
    w._catalog = catalog
    w._max_legal_actions = 64
    w._profile_sps_enabled = False
    w._feature_equiv_check_enabled = False
    w._profile_feature_build_active = False
    w._profile_feature_build_repr_parse_s = 0.0
    w._last_step_feature_profile = {}
    w._feature_action_meta_cache = {}
    w._feature_static_row_cache = {}
    w._current_action_mask = None
    w._unit_previous_tiles = {}
    w._validate_visible_uncaptured_villages = lambda obs, positions: None
    return w


def make_catalog(size):
    return GlobalActionCatalog(size, size, TECHS, ["WARRIOR"], RESOURCES, BUILDINGS, LEVELUPS)


def random_obs(rng, size):
    terrain = [[rng.choice([0, 0, 1, 2, 3, 4, 7, 7, 7]) for _ in range(size)] for _ in range(size)]
    city_ids = [[-1] * size for _ in range(size)]
    cities = {}
    for cid in range(rng.randrange(0, 4)):
        cities[str(cid)] = {
            "x": rng.randrange(-1, size + 1), "y": rng.randrange(-1, size + 1),
            "tribeID": rng.choice([0, 0, 1]), "bound": rng.choice([-1, 0, 1, 2, 3]),
            "population": rng.randrange(0, 4), "population_need": rng.randrange(0, 4),
        }
    units = {str(uid): {"x": rng.randrange(size), "y": rng.randrange(size), "type": rng.choice([0, 0, 1, 10])}
             for uid in range(rng.randrange(0, 5))}
    tribes = {"0": {"capitalID": rng.choice(list(cities) + [None, "99"]) if cities else None}}
    return {"board": {"terrain": terrain, "cityID": city_ids}, "city": cities, "unit": units, "tribes": tribes}


def random_action(rng, size):
    def maybe(d, key, value, p=0.85):
        if rng.random() < p:
            d[key] = value

    a_type = rng.choice(["MOVE", "MOVE", "MOVE", "RESEARCH_TECH", "RESOURCE_GATHERING", "BUILD", "LEVEL_UP",
                         "SPAWN", "CAPTURE", "EXAMINE", "END_TURN", "CLEAR_FOREST", "GROW_FOREST", "BUILD_ROAD",
                         "TRAIN", "mystery"])
    a = {"type": a_type, "star_cost": rng.choice([0, 2, 5, 10]), "schema_version": 1, "idx": rng.randrange(9)}
    uid, city = rng.randrange(-1, 6), rng.randrange(-1, 5)
    tx, ty = rng.randrange(-1, size + 1), rng.randrange(-1, size + 1)
    if a_type == "MOVE":
        a["repr"] = f"MOVE by unit {uid} to {tx} : {ty}"
        maybe(a, "unit_id", uid)
        maybe(a, "dst_x", tx)
        maybe(a, "dst_y", ty)
        maybe(a, "src_x", rng.randrange(-1, size + 1), 0.6)
        maybe(a, "src_y", rng.randrange(-1, size + 1), 0.6)
    elif a_type == "RESEARCH_TECH":
        tech = rng.choice(TECHS + ["UNKNOWN_TECH"])
        a["repr"] = f"RESEARCH_TECH by tribe 0 : {tech}"
        maybe(a, "tech_type", tech)
    elif a_type == "RESOURCE_GATHERING":
        res = rng.choice(RESOURCES)
        a["repr"] = f"RESOURCE_GATHERED by city {city} : {res}"
        maybe(a, "resource_type", res)
        maybe(a, "city_id", city)
        maybe(a, "city_x", rng.randrange(size), 0.5)
        maybe(a, "city_y", rng.randrange(size), 0.5)
        maybe(a, "target_x", tx)
        maybe(a, "target_y", ty)
    elif a_type == "BUILD":
        b = rng.choice(BUILDINGS)
        a["repr"] = f"BUILD by city {city} at {tx} : {ty} : {b}"
        maybe(a, "building_type", b)
        maybe(a, "city_id", city)
        maybe(a, "target_x", tx)
        maybe(a, "target_y", ty)
    elif a_type == "LEVEL_UP":
        c = rng.choice(LEVELUPS)
        a["repr"] = f"LEVEL_UP by city {city} with bonus {c}"
        maybe(a, "levelup_choice", c)
        maybe(a, "city_id", city)
        maybe(a, "city_x", rng.randrange(size), 0.5)
        maybe(a, "city_y", rng.randrange(size), 0.5)
    else:
        a["repr"] = f"{a_type} by unit {uid} at {tx} : {ty}"
        maybe(a, "unit_id", uid, 0.5)
        maybe(a, "city_id", city, 0.5)
        maybe(a, "src_x", rng.randrange(size), 0.5)
        maybe(a, "src_y", rng.randrange(size), 0.5)
        maybe(a, "target_x", tx, 0.5)
        maybe(a, "target_y", ty, 0.5)
    return a


def random_legal_payload(rng, max_slots, n_gids=40):
    n = rng.randrange(0, min(max_slots, 30) + 1)
    gids = rng.sample(range(n_gids), n) if n else []
    ids = np.zeros(max_slots, dtype=np.int32)
    valid = np.zeros(max_slots, dtype=bool)
    ids[:n] = gids
    valid[:n] = True
    if rng.random() < 0.2 and n:  # sparse, non-prefix mask
        valid[rng.randrange(n)] = False
    return ids, valid, gids


class VectorizedLegalFeatureTests(unittest.TestCase):
    def test_step_context_matches_per_tile_loops(self):
        rng = random.Random(0)
        for _ in range(300):
            size = rng.choice([1, 2, 5, 8, 11, 14])
            obs = random_obs(rng, size)
            new = bare_wrapper(None)._build_feature_step_context(obs)
            ref = bare_wrapper(None)._build_feature_step_context_loops(obs)
            self.assertEqual(set(new), set(ref))
            for key in ref:
                a, b = new[key], ref[key]
                if isinstance(b, np.ndarray):
                    self.assertEqual((a.dtype, a.shape), (b.dtype, b.shape), key)
                    self.assertEqual(a.tobytes(), b.tobytes(), key)
                else:
                    self.assertEqual(a, b, key)

    def test_malformed_boards(self):
        for obs in ({}, {"board": {}}, {"board": {"terrain": []}},
                    {"board": {"terrain": [[0, 7], [7]]}, "city": {}, "unit": {}},
                    {"board": {"terrain": [[7, "x"], [7, 3]]}, "city": {"0": "bad"}, "unit": {"u": {}}}):
            try:
                ref = bare_wrapper(None)._build_feature_step_context_loops(obs)
            except Exception as exc:  # the vectorised builder must fail the same way
                with self.assertRaises(type(exc)):
                    bare_wrapper(None)._build_feature_step_context(obs)
                continue
            new = bare_wrapper(None)._build_feature_step_context(obs)
            for key in ref:
                if isinstance(ref[key], np.ndarray):
                    self.assertEqual(new[key].tobytes(), ref[key].tobytes(), key)
                else:
                    self.assertEqual(new[key], ref[key], key)

    def test_feature_rows_match_previous_builder_bytes(self):
        rng = random.Random(1)
        for catalog_size in (None, 11):
            catalog = make_catalog(catalog_size) if catalog_size else None
            # One long-lived wrapper per path so the per-gid caches see the same gid with changing actions.
            fast, slow = bare_wrapper(catalog), bare_wrapper(catalog)
            for step in range(400):
                size = rng.choice([8, 11, 14])
                obs = random_obs(rng, size)
                prev = {int(u): (rng.randrange(size), rng.randrange(size)) for u in range(4) if rng.random() < 0.5}
                fast._unit_previous_tiles = slow._unit_previous_tiles = prev
                ids, valid, gids = random_legal_payload(rng, fast._max_legal_actions)
                actions = [random_action(rng, size) for _ in gids]
                id_to_raw = {g: i for i, g in enumerate(gids)}
                if gids and rng.random() < 0.2:  # unmapped / out-of-range raw indices give zero rows
                    id_to_raw[gids[0]] = len(actions) + 3
                args = (ids, valid, id_to_raw, actions, obs)
                got = fast._build_legal_action_features_padded(*args)
                want = slow._build_legal_action_features_padded_loops(*args)
                self.assertEqual(got.dtype, want.dtype)
                self.assertEqual(got.shape, want.shape)
                self.assertEqual(got.tobytes(), want.tobytes(), f"step={step}")

    def test_static_cache_tracks_action_changes(self):
        w = bare_wrapper(make_catalog(11))
        a = {"type": "RESEARCH_TECH", "tech_type": "FORESTRY", "repr": "RESEARCH_TECH by tribe 0 : FORESTRY", "star_cost": 5}
        _, row1 = w._static_legal_action_feature_entry(3, a)
        b = dict(a, tech_type="ORGANIZATION")
        _, row2 = w._static_legal_action_feature_entry(3, b)
        self.assertEqual((row1[24], row1[23]), (1.0, 0.0))
        self.assertEqual((row2[24], row2[23]), (0.0, 1.0))
        c = dict(b)
        del c["star_cost"]
        self.assertEqual(w._static_legal_action_feature_entry(3, c)[1][42], 0.0)
        # A float star cost equal to an int one is a different input type and is not served from cache.
        d = dict(b, star_cost=5.0)
        w._static_legal_action_feature_entry(3, d)
        self.assertIs(w._feature_static_row_cache[3][0][0][-1], 5.0)
        # Callers get copies of cached rows only through _legal_action_feature_row.
        obs = random_obs(random.Random(2), 11)
        row = w._legal_action_feature_row(d, obs, w._build_feature_step_context(obs), 3)
        row[0] = 99.0
        self.assertEqual(w._static_legal_action_feature_entry(3, d)[1][0], 0.0)

    def test_clip_matches_numpy(self):
        for v in (-3.5, -1.0, -0.0, 0.0, 0.25, 1.0, 7.0, float("inf"), float("-inf")):
            for lo, hi in ((-1.0, 1.0), (0.0, 1.0)):
                got = TribesGymWrapper._clip_float(v, lo, hi)
                want = float(np.clip(v, lo, hi))
                self.assertEqual(np.float64(got).tobytes(), np.float64(want).tobytes(), (v, lo, hi))
        self.assertTrue(np.isnan(TribesGymWrapper._clip_float(float("nan"), -1.0, 1.0)))


@unittest.skipUnless(os.environ.get("POLYVISION_RUN_JAVA_TESTS") == "1", "requires local Java gateway")
class LiveVectorizedLegalFeatureTests(unittest.TestCase):
    def test_random_play_passes_in_env_equivalence_check(self):
        # POLYVISION_FEATURE_EQUIV_CHECK=1 rebuilds every step's features with the per-tile builder and raises
        # on any byte difference.
        settings = dict(POLYVISION_LEVEL_POOL_GLOB="levels/phase1_pool_bardur_real/train/*.csv",
                        POLYVISION_SOLO_NO_OPPONENT_MODE="1", POLYVISION_INFO_MODE="fast",
                        POLYVISION_FEATURE_EQUIV_CHECK="1")
        jdks = sorted((ROOT / ".runtime").glob("jdk-*/Contents/Home"))
        if jdks:
            settings["JAVA_HOME"] = str(jdks[-1])
            settings["PATH"] = str(jdks[-1] / "bin") + os.pathsep + os.environ["PATH"]
        with patch.dict(os.environ, settings):
            env = TribesGymWrapper()
            try:
                rng = random.Random(5)
                _, info = env.reset(seed=5)
                for _ in range(400):
                    legal = np.asarray(info["legal_global_ids_padded"])[np.asarray(info["legal_action_valid_mask"])]
                    _, _, term, trunc, info = env.step(int(legal[rng.randrange(len(legal))]))
                    if term or trunc:
                        _, info = env.reset(seed=rng.randrange(1000))
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
