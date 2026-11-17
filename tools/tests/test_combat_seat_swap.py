"""Combat league seat swap (CHANGELOG 072): the learner plays tribe 1 in some games.

Pure tests always run; the live-engine tests need POLYVISION_RUN_JAVA_TESTS=1 (as test_combat_integration.py).
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pol_env.Tribes.py import combat_selfplay as S

ROOT = Path(__file__).resolve().parents[2]


class SeatChoiceTests(unittest.TestCase):
    def test_default_is_seat_zero_without_touching_the_rng(self):
        rng = np.random.default_rng(3)
        league = S.CombatLeague({}, rng=rng)
        before = rng.bit_generator.state
        self.assertEqual({league.choose_seat() for _ in range(50)}, {0})
        self.assertEqual(rng.bit_generator.state, before)

    def test_swap_prob_share(self):
        league = S.CombatLeague({"POLYVISION_COMBAT_LEAGUE_SWAP_PROB": "0.5"}, rng=np.random.default_rng(0))
        share = np.mean([league.choose_seat() for _ in range(4000)])
        self.assertAlmostEqual(share, 0.5, delta=0.03)
        self.assertEqual(S.CombatLeague({"POLYVISION_COMBAT_LEAGUE_SWAP_PROB": "1"}).choose_seat(), 1)
        with self.assertRaises(ValueError):
            S.CombatLeague({"POLYVISION_COMBAT_LEAGUE_SWAP_PROB": "1.5"})

    def test_swap_combat_state(self):
        state = {"tribes": {"0": {"cities": 1}, "1": {"cities": 3}},
                 "units": [{"tribe": 1, "x": 2}, {"tribe": 0, "x": 4}], "cities": [{"tribe": 1, "capital": True}]}
        S.swap_combat_state(state, 0, 1)
        self.assertEqual(state["tribes"]["0"]["cities"], 3)
        self.assertEqual([u["tribe"] for u in state["units"]], [0, 1])
        self.assertEqual(state["cities"][0]["tribe"], 0)

    def test_inplace_swap_matches_copy(self):
        obs = {"activeTribeID": 1, "tribes": {"0": {"star": 1}, "1": {"star": 2}}, "unit": {"5": {"tribeId": 1}}}
        copied, _ = S.swap_tribes(obs, [], 0, 1)
        inplace, _ = S.swap_tribes(obs, [], 0, 1, inplace=True)
        self.assertIs(inplace, obs)
        self.assertEqual(copied, inplace)


class PoolAndStatsTests(unittest.TestCase):
    def _touch(self, path, t):
        for f in (path, path + ".action_interface.json"):
            open(f, "w").close()
        os.utime(path, (t, t))

    def test_max_self_keeps_fixed_opponents(self):
        with tempfile.TemporaryDirectory() as tmp:
            own_dir, fixed_dir = os.path.join(tmp, "own"), os.path.join(tmp, "Tribes-v0__teacher_s1__1__99")
            os.makedirs(own_dir)
            os.makedirs(fixed_dir)
            fixed = os.path.join(fixed_dir, "teacher_s1.cleanrl_model")
            self._touch(fixed, 1)  # the oldest file
            own = [os.path.join(own_dir, f"model_checkpoint_{i}.cleanrl_model") for i in range(1, 6)]
            for i, p in enumerate(own):
                self._touch(p, 10 + i)
            env = {"POLYVISION_COMBAT_LEAGUE_SELF_DIR": own_dir, "POLYVISION_COMBAT_LEAGUE_GLOB": fixed,
                   "POLYVISION_COMBAT_LEAGUE_MAX_SNAPSHOTS": "3"}
            self.assertNotIn(fixed, S.CombatLeague(env).candidates(), "old behaviour: newest 3 of all")
            league = S.CombatLeague({**env, "POLYVISION_COMBAT_LEAGUE_MAX_SELF": "2"})
            self.assertEqual(league.candidates(), [fixed] + own[-2:])
            self.assertEqual(league.label(fixed), "teacher_s1")
            self.assertEqual(league.label(own[0]), "self/model_checkpoint_1")
            self.assertEqual(league.label(None), "simple")

    def test_stats_tracker_aggregates_rows(self):
        t = S.LeagueStatsTracker(window=100)
        rows = ([{"combat_league_opponent": "hard", "combat_league_learner_won": 0.0, "combat_end_raw_learner_seat": 1.0}] * 4
                + [{"combat_league_opponent": "easy", "combat_league_learner_won": 1.0, "combat_end_raw_learner_seat": 0.0}] * 4
                + [{"combat_league_opponent": "simple", "combat_league_learner_won": 1.0, "combat_end_raw_learner_seat": 0.0}] * 2
                + [None, {"combat_outcome": "win"}])
        self.assertEqual(sum(t.add(r) for r in rows), 10)
        out = t.scalars()
        self.assertAlmostEqual(out["league/pick_share/hard"], 0.4)
        self.assertAlmostEqual(out["league/snapshot_pick_share/hard"], 0.5)
        self.assertEqual(out["league/win_rate/hard"], 0.0)
        self.assertEqual(out["league/win_rate/easy"], 1.0)
        self.assertGreater(out["league/pfsp_weight/hard"], out["league/pfsp_weight/easy"])
        self.assertNotIn("league/pfsp_weight/simple", out)
        self.assertAlmostEqual(out["league/win_rate_seat1"], 0.0)
        self.assertAlmostEqual(out["league/win_rate_seat0"], 1.0)
        self.assertEqual(out["league/games/hard"], 4.0)
        self.assertEqual(t.table()[0]["games"], 4)


@unittest.skipUnless(os.environ.get("POLYVISION_RUN_JAVA_TESTS") == "1", "requires local Java gateway")
class SeatSwapEngineTests(unittest.TestCase):
    """A real combat game with the learner as tribe 1, against SimpleAgent through the league path."""

    def make_env(self, swap_prob, max_turns=3):
        from pol_env.Tribes.py.register_env import TribesGymWrapper

        jdks = sorted((ROOT / ".runtime").glob("jdk-*/Contents/Home"))
        settings = dict(POLYVISION_PHASE="2", POLYVISION_COMBAT_MODE="1", POLYVISION_SOLO_NO_OPPONENT_MODE="0",
                        POLYVISION_MAX_TURNS=str(max_turns), POLYVISION_INFO_MODE="fast",
                        POLYVISION_LEVEL_POOL_GLOB="levels/phase2_pool_1v1/train/*.csv",
                        POLYVISION_COMBAT_LEAGUE="1", POLYVISION_COMBAT_LEAGUE_SIMPLE_PROB="1",
                        POLYVISION_COMBAT_LEAGUE_SWAP_PROB=str(swap_prob))
        if jdks:
            settings["JAVA_HOME"] = str(jdks[-1])
            settings["PATH"] = str(jdks[-1] / "bin") + os.pathsep + os.environ["PATH"]
        p = patch.dict(os.environ, settings)
        p.start()
        self.addCleanup(p.stop)
        env = TribesGymWrapper()
        self.addCleanup(env.close)
        return env

    def java(self, env):
        return env.tribes_env._env

    def test_reset_puts_learner_in_seat_one_with_relabelled_view(self):
        env = self.make_env(1.0)
        obs, info = env.reset(seed=5)
        java = self.java(env)
        self.assertEqual(env.tribes_env.learner_seat, 1)
        self.assertEqual(int(java.getActiveTribeID()), 1, "the opponent (tribe 0) has already moved")
        raw = env.tribes_env.get_observation(relabel=False)
        view = env.tribes_env.get_observation()
        self.assertEqual(view["tribes"]["0"]["type"], raw["tribes"]["1"]["type"])
        self.assertEqual(view["activeTribeID"], 0)
        # Every unit the learner may act with is its own (Java tribe 1), shown as tribe 0.
        mine = {k for k, u in raw["unit"].items() if int(u["tribeId"]) == 1}
        self.assertEqual(mine, {k for k, u in view["unit"].items() if int(u["tribeId"]) == 0})
        mask = np.asarray(info["legal_action_valid_mask"], dtype=bool)
        self.assertTrue(mask.any())
        self.assertIn(env._catalog.end_turn_id, set(np.asarray(info["legal_global_ids_padded"])[mask].tolist()))
        for a in env.tribes_env.list_actions():
            if "unit_id" in a:
                self.assertIn(str(a["unit_id"]), mine)

    def test_each_side_plays_max_turns(self):
        env = self.make_env(1.0, max_turns=3)
        java = self.java(env)
        played = []
        orig = env.tribes_env.play_opponent_turn

        def counting():
            if not bool(java.isDone()) and int(java.getActiveTribeID()) != 1:
                played.append(int(java.getTick()))
            orig()

        env.tribes_env.play_opponent_turn = counting
        env.reset(seed=7)
        self.assertEqual(played, [0], "the opponent's first turn is played in the reset")
        self.assertEqual(int(java.getTick()), 0)
        for turn in range(1, 4):
            _, _, term, trunc, info = env.step(env._catalog.end_turn_id)
            if turn < 3:
                self.assertFalse(term or trunc)
                self.assertEqual(int(java.getActiveTribeID()), 1, "the opponent played its turn")
                self.assertEqual(int(java.getTick()), turn)
        self.assertTrue(trunc)
        self.assertEqual(info["combat_outcome"], "timeout")
        self.assertEqual(int(java.getTick()), 3)
        # 3 opponent turns (ticks 0, 1, 2) and none after the learner's 3rd; the wrapper then ends the opponent's
        # turn without actions, so the final view is the learner's, at the start of its 4th turn as in seat 0.
        self.assertEqual(played, [0, 1, 2])
        self.assertEqual(int(java.getActiveTribeID()), 1)
        self.assertEqual(info["combat_end_raw_learner_seat"], 1.0)
        self.assertEqual(info["combat_end_raw_win_seat1"], 0.0)
        self.assertEqual(info["combat_league_opponent"], "simple")

    def _capture(self, env, java_tribe, x, y):
        gs = self.java(env).getGameState()
        self.assertTrue(gs.getBoard().capture(gs, gs.getTribe(java_tribe), int(x), int(y)))

    def _capital_pos(self, env, java_tribe):
        from py4j.java_gateway import get_field

        gs = self.java(env).getGameState()
        pos = gs.getActor(gs.getTribe(java_tribe).getCapitalID()).getPosition()
        return int(get_field(pos, "x")), int(get_field(pos, "y"))

    def test_win_in_seat_one_is_positive(self):
        from pol_env.Tribes.py.register_env import TribesGymWrapper

        env = self.make_env(1.0)
        env.reset(seed=9)
        self._capture(env, 1, *self._capital_pos(env, 0))  # the learner (Java tribe 1) takes tribe 0's capital
        _, reward, term, _, info = env.step(env._catalog.end_turn_id)
        self.assertTrue(term)
        self.assertEqual(info["combat_outcome"], "win")
        self.assertEqual(info["combat_outcome_reward"], TribesGymWrapper.COMBAT_REWARD_DEFAULTS["win"])
        self.assertGreater(info["combat_outcome_reward"], 0)
        self.assertEqual(info["combat_end_enemy_capitals_held"], 1)
        self.assertEqual(info["combat_end_raw_win_seat1"], 1.0)

    def test_loss_in_seat_one_is_negative(self):
        env = self.make_env(1.0)
        env.reset(seed=11)
        self._capture(env, 0, *self._capital_pos(env, 1))  # the opponent takes the learner's capital
        _, _, term, _, info = env.step(env._catalog.end_turn_id)
        self.assertTrue(term)
        self.assertEqual(info["combat_outcome"], "loss")
        self.assertLess(info["combat_outcome_reward"], 0)
        self.assertEqual(info["combat_end_own_cities"], 0)

    def test_seat_zero_unchanged(self):
        env = self.make_env(0.0, max_turns=2)
        env.reset(seed=1)
        java = self.java(env)
        self.assertEqual(env.tribes_env.learner_seat, 0)
        self.assertEqual(int(java.getControlledTribe()), 0)
        self.assertEqual(int(java.getActiveTribeID()), 0)
        _, _, term, trunc, info = env.step(env._catalog.end_turn_id)
        self.assertEqual(info["tick"], 1)
        _, _, term, trunc, info = env.step(env._catalog.end_turn_id)
        self.assertTrue(trunc)
        self.assertEqual(int(java.getActiveTribeID()), 0, "seat 0: the opponent's last turn is played as before")
        self.assertEqual(info["combat_end_raw_learner_seat"], 0.0)


if __name__ == "__main__":
    unittest.main()
