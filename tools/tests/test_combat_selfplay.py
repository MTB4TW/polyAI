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


if __name__ == "__main__":
    unittest.main()
