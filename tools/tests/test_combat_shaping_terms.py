"""Unit tests for the opt-in combat reward terms (pure functions, no JVM)."""
import os
import unittest

from pol_env.Tribes.py import combat_shaping as cs


def unit(uid, tribe, kind, x, y, hp=10, mov=1, rng=1, cost=2):
    return {"id": uid, "tribe": tribe, "type": kind, "x": x, "y": y, "hp": hp, "maxHp": 10,
            "mov": mov, "range": rng, "cost": cost}


def state(units, cities=None, size=11, tile_city=None, road=None):
    tile_city = tile_city or [[-1] * size for _ in range(size)]
    road = road or [[False] * size for _ in range(size)]
    return {"size": size, "tick": 0, "units": units, "cities": cities or [], "tileCity": tile_city, "road": road,
            "tribes": {"0": {"cities": 1, "score": 0, "kills": 0}, "1": {"cities": 1, "score": 0, "kills": 0}}}


ENEMY_CAP = {"id": 9, "tribe": 1, "x": 10, "y": 10, "capital": True, "level": 1, "population": 0, "populationNeed": 2}


class TermTests(unittest.TestCase):
    def test_capital_proximity_grows_with_closeness_and_count(self):
        far = cs.capital_proximity(state([unit(1, 0, "WARRIOR", 0, 0)], [ENEMY_CAP]), 0)
        near = cs.capital_proximity(state([unit(1, 0, "WARRIOR", 9, 9)], [ENEMY_CAP]), 0)
        two = cs.capital_proximity(state([unit(1, 0, "WARRIOR", 9, 9), unit(2, 0, "WARRIOR", 9, 10)], [ENEMY_CAP]), 0)
        self.assertAlmostEqual(far, 0.0)
        self.assertAlmostEqual(near, 0.9)
        self.assertAlmostEqual(two, 1.8)

    def test_capital_proximity_fixed_targets(self):
        s = state([unit(1, 0, "WARRIOR", 10, 10)], [])  # capital already ours, so no enemy capital in state
        self.assertAlmostEqual(cs.capital_proximity(s, 0), 0.0)
        self.assertAlmostEqual(cs.capital_proximity(s, 0, [(10, 10)]), 1.0)

    def test_zone_of_control_counts_reachable_tiles_near_enemy_territory(self):
        tc = [[-1] * 11 for _ in range(11)]
        tc[5][5] = 9
        s = state([unit(1, 0, "RIDER", 3, 5, mov=2)], [dict(ENEMY_CAP, x=5, y=5)], tile_city=tc)
        # enemy tile (5,5) plus its 8 neighbours; rider at (3,5) reaches x in 1..5 -> tiles x=4,5 of the window
        self.assertEqual(cs.zone_of_control(s, 0), 6.0)

    def test_regional_superiority(self):
        s = state([unit(1, 0, "WARRIOR", 5, 5), unit(2, 0, "WARRIOR", 5, 6), unit(3, 1, "WARRIOR", 6, 6)])
        self.assertEqual(cs.regional_superiority(s, 0, 6, 6), 1.0)
        s2 = state([unit(1, 0, "WARRIOR", 5, 5), unit(3, 1, "KNIGHT", 6, 6, cost=8), unit(4, 1, "WARRIOR", 6, 7)])
        self.assertEqual(cs.regional_superiority(s2, 0, 6, 6), 0.0)

    def test_end_turn_roles(self):
        foe = unit(9, 1, "WARRIOR", 5, 5)
        self.assertEqual(cs.end_turn_roles(state([unit(1, 0, "RIDER", 5, 8), foe]), 0), 1.0)    # distance 3
        self.assertEqual(cs.end_turn_roles(state([unit(1, 0, "RIDER", 5, 6), foe]), 0), 0.0)    # distance 1
        self.assertEqual(cs.end_turn_roles(state([unit(1, 0, "CATAPULT", 5, 8), foe]), 0), 1.0)
        self.assertEqual(cs.end_turn_roles(state([unit(1, 0, "CATAPULT", 5, 7), foe]), 0), -1.0)
        warrior = cs.end_turn_roles(state([unit(1, 0, "WARRIOR", 5, 6), foe]), 0)
        self.assertAlmostEqual(warrior, 0.9)  # no enemy city in state: proximity to the unit only

    def test_action_roles(self):
        archer, foe_w, foe_r = unit(1, 0, "ARCHER", 5, 5, rng=2), unit(2, 1, "WARRIOR", 5, 6), unit(3, 1, "RIDER", 5, 7)
        s = state([archer, foe_w, foe_r])
        self.assertEqual(cs.action_roles(s, 0, {"type": "ATTACK", "unit_id": 1, "target_unit_id": 2}), -2.0)
        self.assertEqual(cs.action_roles(s, 0, {"type": "ATTACK", "unit_id": 1, "target_unit_id": 3}), 1.0)
        road = [[False] * 11 for _ in range(11)]
        road[4][4] = True
        s_road = state([archer], road=road)
        self.assertEqual(cs.action_roles(s_road, 0, {"type": "MOVE", "unit_id": 1, "dst_x": 4, "dst_y": 4}), 0.5)
        self.assertEqual(cs.action_roles(s_road, 0, {"type": "MOVE", "unit_id": 1, "dst_x": 4, "dst_y": 5}), 0.0)

    def test_hp_lost(self):
        prev = state([unit(1, 0, "WARRIOR", 1, 1, hp=10), unit(2, 1, "WARRIOR", 2, 2, hp=10), unit(3, 1, "ARCHER", 3, 3, hp=8)])
        cur = state([unit(1, 0, "WARRIOR", 1, 1, hp=6), unit(3, 1, "ARCHER", 3, 3, hp=10)])  # 2 killed, 3 healed
        self.assertEqual(cs.hp_lost(prev, cur), {0: 4, 1: 10})

    def test_hanging_population(self):
        start = state([], [{"id": 1, "tribe": 0, "level": 1, "population": 0}, {"id": 2, "tribe": 0, "level": 1, "population": 1}])
        end = state([], [{"id": 1, "tribe": 0, "level": 1, "population": 1}, {"id": 2, "tribe": 0, "level": 2, "population": 0}])
        self.assertEqual(cs.hanging_population(start, end, 0), 1.0)  # city 2 levelled up, so it does not count

    def test_totals(self):
        tc = [[-1] * 11 for _ in range(11)]
        road = [[False] * 11 for _ in range(11)]
        tc[0][0] = tc[0][1] = 9
        road[0][1] = True
        s = state([unit(1, 0, "SUPERUNIT", 1, 1, cost=10), unit(2, 1, "WARRIOR", 2, 2)], [ENEMY_CAP], tile_city=tc, road=road)
        t = cs.totals(s, 0)
        self.assertEqual((t["0"]["units"], t["0"]["unitValue"], t["0"]["giants"]), (1, 10, 1))
        self.assertEqual((t["1"]["tiles"], t["1"]["roads"]), (2, 1))

    def test_pop_tech_margin(self):
        tc = [[-1] * 11 for _ in range(11)]
        for x, y in ((0, 0), (0, 1), (0, 2)):
            tc[x][y] = 1
        resource = [[-1] * 11 for _ in range(11)]
        resource[0][0] = resource[0][1] = cs.RESOURCE_FRUIT
        resource[0][2] = cs.RESOURCE_ANIMAL
        resource[5][5] = cs.RESOURCE_ANIMAL  # outside our borders: ignored
        board = {"resource": resource, "terrain": [[0] * 11 for _ in range(11)], "building": [[-1] * 11 for _ in range(11)]}
        s = state([], [{"id": 1, "tribe": 0, "x": 0, "y": 1, "capital": True, "level": 1, "population": 0, "populationNeed": 2}], tile_city=tc)
        offered = ["ORGANIZATION", "HUNTING", "RIDING"]
        self.assertEqual(cs.pop_tech_margin("ORGANIZATION", offered, s, board, 0), 1.0)   # 2 fruit vs 1 animal
        self.assertEqual(cs.pop_tech_margin("HUNTING", offered, s, board, 0), -1.0)
        self.assertEqual(cs.pop_tech_margin("RIDING", offered, s, board, 0), 0.0)         # non-pop tech: neutral

    def test_tech_used_by(self):
        self.assertTrue(cs.tech_used_by("ORGANIZATION", {("RESOURCE_GATHERING", "FRUIT")}))
        self.assertFalse(cs.tech_used_by("ORGANIZATION", {("SPAWN", "WARRIOR")}))
        self.assertTrue(cs.tech_used_by("ROADS", {("BUILD_ROAD", None)}))
        self.assertTrue(cs.tech_used_by("CLIMBING", set()))  # no defined use: never counted as unused
        self.assertEqual(cs.action_signature({"type": "BUILD", "building_type": "FARM"}), ("BUILD", "FARM"))

    def test_capital_terms(self):
        tc = [[-1] * 11 for _ in range(11)]
        for x in range(9, 11):
            for y in range(9, 11):
                tc[x][y] = 9
        prev = state([unit(1, 0, "WARRIOR", 10, 10), unit(5, 1, "DEFENDER", 9, 9), unit(6, 1, "WARRIOR", 2, 2)],
                     [ENEMY_CAP], tile_city=tc)
        cur = state([unit(1, 0, "WARRIOR", 10, 10)], [ENEMY_CAP], tile_city=tc)  # both enemy units gone
        self.assertEqual(cs.units_on_enemy_capital(prev, 0), 1.0)
        self.assertEqual(cs.kills_in_enemy_capital_region(prev, cur, 0), 1.0)  # only the one inside the capital region

    def test_unit_type_counts(self):
        s = state([unit(1, 0, "WARRIOR", 0, 0), unit(2, 0, "WARRIOR", 0, 1), unit(3, 1, "ARCHER", 5, 5)])
        counts = cs.unit_type_counts(s, 0)
        self.assertEqual(counts["WARRIOR"], [2, 0])
        self.assertEqual(counts["ARCHER"], [0, 1])
        self.assertEqual(counts["SUPERUNIT"], [0, 0])


if __name__ == "__main__":
    unittest.main()
