"""Tests for the combat self-play league (pol_env/Tribes/py/combat_selfplay.py). Pure parts; no Java."""

import os
import tempfile
import unittest

import numpy as np

from pol_env.Tribes.py import combat_selfplay as S


class SwapTests(unittest.TestCase):
    def test_swap_relabels_every_tribe_reference(self):
        obs = {"activeTribeID": 1, "tick": 3,
               "tribes": {"0": {"star": 5, "diplomacy": {"peace": [False, True], "war": [False, False]}},
                          "1": {"star": 9, "diplomacy": {"peace": [True, False], "war": [True, False]}}},
               "unit": {"7": {"tribeId": 0, "x": 1}, "8": {"tribeId": 1, "x": 2}},
               "city": {"3": {"tribeID": 1}, "4": {"tribeID": 0}}}
        actions = [{"type": "OFFER_PEACE", "tribe_id": 1, "target_tribe_id": 0}, {"type": "MOVE", "unit_id": 8}]
        s_obs, s_act = S.swap_tribes(obs, actions, 0, 1)
        self.assertEqual(s_obs["activeTribeID"], 0)
        self.assertEqual(s_obs["tribes"]["0"]["star"], 9)
        self.assertEqual(s_obs["tribes"]["1"]["star"], 5)
        self.assertEqual(s_obs["tribes"]["0"]["diplomacy"]["peace"], [False, True])  # its own slot is now 0
        self.assertEqual(s_obs["tribes"]["0"]["diplomacy"]["war"], [False, True])
        self.assertEqual(s_obs["unit"]["8"]["tribeId"], 0)
        self.assertEqual(s_obs["city"]["4"]["tribeID"], 1)
        self.assertEqual((s_act[0]["tribe_id"], s_act[0]["target_tribe_id"]), (0, 1))
        self.assertEqual(obs["unit"]["8"]["tribeId"], 1, "input left untouched")
        twice, _ = S.swap_tribes(s_obs, s_act, 0, 1)
        self.assertEqual(twice, obs)


class LeagueTests(unittest.TestCase):
    def make_snapshots(self, folder, n):
        paths = []
        for i in range(n):
            p = os.path.join(folder, f"model_checkpoint_{(i + 1) * 1000}.cleanrl_model")
            for f in (p, p + ".action_interface.json"):
                open(f, "w").close()
            os.utime(p, (i, i))
            paths.append(p)
        return paths

    def test_pool_keeps_newest_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_snapshots(tmp, 5)
            league = S.CombatLeague({"POLYVISION_COMBAT_LEAGUE_SELF_DIR": tmp,
                                     "POLYVISION_COMBAT_LEAGUE_MAX_SNAPSHOTS": "3"})
            self.assertEqual(league.candidates(), paths[-3:])

    def test_pfsp_prefers_snapshots_the_learner_beats_less(self):
        league = S.CombatLeague({}, rng=np.random.default_rng(0))
        for _ in range(20):
            league.record("easy", True)
            league.record("hard", False)
        w = league.weights(["easy", "hard", "new"])
        self.assertGreater(w[1], w[2])
        self.assertGreater(w[2], w[0])
        self.assertAlmostEqual(float(w.sum()), 1.0)

    def test_no_snapshots_means_simple_agent(self):
        league = S.CombatLeague({"POLYVISION_COMBAT_LEAGUE_SIMPLE_PROB": "0"})
        self.assertIsNone(league.choose(wrapper=None))

    def test_unloadable_snapshot_is_rejected_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_snapshots(tmp, 1)
            league = S.CombatLeague({"POLYVISION_COMBAT_LEAGUE_SELF_DIR": tmp, "POLYVISION_COMBAT_LEAGUE_SIMPLE_PROB": "0"})
            self.assertIsNone(league.choose(wrapper=None))  # the empty file cannot load
            self.assertEqual(len(league.rejected), 1)
            league.refresh()
            self.assertEqual(league.pool, [])


class DeferredVillageCaptureTests(unittest.TestCase):
    """A snapshot opponent with one city gets the wrapper's deferred village captures at END_TURN (CHANGELOG 089)."""

    def make(self, queued, cities=1):
        from types import SimpleNamespace
        actions = [{"type": "MOVE", "repr": "MOVE unit 7"}, {"type": "CAPTURE", "repr": "CAPTURE unit 7", "village": True},
                   {"type": "CAPTURE", "repr": "CAPTURE unit 9", "village": True}, {"type": "END_TURN", "repr": "END_TURN"}]
        stepped = []
        state = {"cities": cities}

        class Java:
            def stepByIndex(self, i):
                stepped.append(actions[i]["repr"])
                if actions[i]["type"] == "CAPTURE":
                    state["cities"] += 1

        env = SimpleNamespace(_env=Java(), get_observation=lambda relabel=True: {"tribes": {}},
                              list_actions=lambda relabel=True: [dict(a) for a in actions])
        inner = SimpleNamespace(
            _queued_village_capture_unit_ids=set(queued),
            _get_city_count=lambda obs: state["cities"],
            _is_capture_of_village=lambda a, obs: bool(a.get("village")),
            _parse_unit_id_from_action_repr=lambda r: int(r.split()[-1]) if r.split()[-1].isdigit() else None)
        return env, SimpleNamespace(view=inner), stepped

    def test_queued_unit_captures_before_the_turn_ends(self):
        env, view, stepped = self.make({7})
        self.assertEqual(S.play_deferred_village_captures(env, view, 1), [1])
        self.assertEqual(stepped, ["CAPTURE unit 7"])
        self.assertEqual(view.view._queued_village_capture_unit_ids, set())

    def test_stops_once_the_tribe_has_two_cities(self):
        env, view, stepped = self.make({7, 9})
        S.play_deferred_village_captures(env, view, 1)
        self.assertEqual(len(stepped), 1)  # the second capture would be the wrapper's 3rd city: not deferred

    def test_nothing_queued_or_two_cities_means_no_capture(self):
        env, view, stepped = self.make({7}, cities=2)
        self.assertEqual(S.play_deferred_village_captures(env, view, 1), [])
        self.assertEqual(stepped, [])


class OwnRulesTests(unittest.TestCase):
    def league(self, prob):
        return S.CombatLeague(environ={"POLYVISION_COMBAT_LEAGUE_OWN_RULES_PROB": str(prob)},
                              rng=np.random.default_rng(0))

    def test_off_by_default_and_never_for_a_snapshot_with_the_runs_rules(self):
        from types import SimpleNamespace
        old = SimpleNamespace(trained_rules_off=frozenset())
        self.assertIsNone(self.league(0).choose_rules(old, frozenset({"fishing_mask"})))
        same = SimpleNamespace(trained_rules_off=frozenset({"fishing_mask"}))
        self.assertIsNone(self.league(1).choose_rules(same, frozenset({"fishing_mask"})))

    def test_share_of_games_with_the_snapshots_own_rules(self):
        from types import SimpleNamespace
        old = SimpleNamespace(trained_rules_off=frozenset())
        league = self.league(0.5)
        picks = [league.choose_rules(old, frozenset({"village_forced_moves"})) for _ in range(2000)]
        own = [p for p in picks if p is not None]
        self.assertTrue(all(p == frozenset() for p in own))  # all its rules on
        self.assertAlmostEqual(len(own) / len(picks), 0.5, delta=0.04)


if __name__ == "__main__":
    unittest.main()
