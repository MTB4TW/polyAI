"""The cached board-dimension and village scans must match the original per-tile lookups exactly."""
import copy
import random
import unittest

from pol_env.Tribes.py.register_env import TribesGymWrapper


def bare_wrapper():
    # The scans need no Java gateway; skip __init__.
    return TribesGymWrapper.__new__(TribesGymWrapper)


def reference_villages(w, obs):
    """Original algorithm: one _board_get_int_by_java_coord call per tile."""
    dims = TribesGymWrapper._compute_board_dimensions(obs) if isinstance(obs, dict) else None
    if dims is None:
        return None
    occupied = set()
    for c in (obs.get("city") or {}).values():
        if isinstance(c, dict):
            try:
                cx, cy = int(c.get("x", -1)), int(c.get("y", -1))
            except Exception:
                continue
            if cx >= 0 and cy >= 0:
                occupied.add((cx, cy))
    out = set()
    for x in range(dims[0]):
        for y in range(dims[1]):
            t = w._board_get_int_by_java_coord(obs, "terrain", x, y, default=-1)
            c = w._board_get_int_by_java_coord(obs, "cityID", x, y, default=-1)
            if t == 4 and c == -1 and (x, y) not in occupied:
                out.add((x, y))
    return out


def random_obs(rng, size=11):
    terrain = [[rng.choice([0, 1, 2, 4, 4, 7, None, "4", "x"]) for _ in range(size)] for _ in range(size)]
    city = [[rng.choice([-1, -1, -1, 3, None, "-1", "bad"]) for _ in range(size)] for _ in range(size)]
    if rng.random() < 0.2:  # ragged column
        terrain[rng.randrange(size)] = terrain[0][: rng.randrange(size)]
    if rng.random() < 0.2:
        city[rng.randrange(size)] = "not-a-list"
    cities = {str(i): {"x": rng.randrange(size), "y": rng.randrange(size)} for i in range(rng.randrange(4))}
    return {"board": {"terrain": terrain, "cityID": city}, "city": cities}


class BoardScanCacheTests(unittest.TestCase):
    def test_fast_scan_matches_reference_on_random_boards(self):
        rng = random.Random(0)
        for _ in range(300):
            obs = random_obs(rng)
            self.assertEqual(bare_wrapper()._scan_visible_uncaptured_villages(obs), reference_villages(bare_wrapper(), obs))

    def test_malformed_observations(self):
        w = bare_wrapper()
        for obs in ({}, {"board": {}}, {"board": {"terrain": []}}, {"board": {"terrain": "x"}}, None):
            self.assertIsNone(w._scan_visible_uncaptured_villages(obs) if isinstance(obs, dict) else None)
            self.assertIsNone(w._board_dimensions_from_obs(obs))

    def test_dimension_cache_is_per_observation_object(self):
        w = bare_wrapper()
        small = {"board": {"terrain": [[0] * 3] * 3}}
        big = {"board": {"terrain": [[0] * 5] * 5}}
        self.assertEqual(w._board_dimensions_from_obs(small), (3, 3))
        self.assertEqual(w._board_dimensions_from_obs(big), (5, 5))
        self.assertEqual(w._board_dimensions_from_obs(small), (3, 3))

    def test_village_cache_is_per_observation_object(self):
        w = bare_wrapper()
        w._validate_visible_uncaptured_villages = lambda obs, positions: None
        obs = {"board": {"terrain": [[4, 0], [0, 0]], "cityID": [[-1, -1], [-1, -1]]}, "city": {}}
        self.assertEqual(w._get_visible_uncaptured_village_positions(obs), {(0, 0)})
        changed = copy.deepcopy(obs)
        changed["board"]["terrain"][1][1] = 4
        self.assertEqual(w._get_visible_uncaptured_village_positions(changed), {(0, 0), (1, 1)})
        # Callers may mutate the returned set without corrupting the cache.
        w._get_visible_uncaptured_village_positions(obs).add((9, 9))
        self.assertEqual(w._get_visible_uncaptured_village_positions(obs), {(0, 0)})


if __name__ == "__main__":
    unittest.main()
