"""Engine-backed checks of the play-time search helpers (PythonEnv.tacticalAnalyze / attackPreviews)."""

import math
import os
import shutil
import unittest

from tools import play_vs_agent as P

HAVE_JAVA = shutil.which("java") is not None or bool(os.environ.get("JAVA_HOME"))


@unittest.skipUnless(HAVE_JAVA, "needs the Java engine")
class SearchAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env, _ = P.make_env(None, "eval_validation", "polytopia_2_15", 20)
        cls.maps = P.load_maps("eval_validation")

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_previews_and_analysis_are_read_only_and_well_formed(self):
        from tools.search_variants import BEST, FORCED_VALUE, STOP_VALUE
        env = self.env
        env._level_pool, env._level_pool_size = [str(self.maps[0].path)], 1
        env._select_combat_opponent = lambda: None
        env.reset(seed=11)
        te = env.tribes_env
        java, gw = te._env, te._gateway
        # play a few turns of END_TURN so units meet less often; previews must line up with the action list either way
        before = te.get_observation(full_visibility=True)
        actions = te.list_actions()
        prev = list(java.attackPreviews())
        self.assertEqual(len(prev), 5 * len(actions))
        for i, a in enumerate(actions):
            if a.get("type") != "ATTACK":
                self.assertTrue(all(math.isnan(v) for v in prev[i * 5:i * 5 + 5]))
        legal = list(range(len(actions)))
        arr = gw.new_array(gw.jvm.int, len(legal))
        for i, r in enumerate(legal):
            arr[i] = r
        end = next(i for i, a in enumerate(actions) if a["type"] == "END_TURN")
        move = next((i for i, a in enumerate(actions) if a["type"] == "MOVE"), -1)
        for forced, reply in ((-1, 0), (move, 0), (move, 4), (end, 2)):
            res = list(java.tacticalAnalyze(None, arr, forced, 3, 4, 3, 7, reply))
            self.assertEqual(len(res), 18)
            self.assertTrue(int(res[BEST]) == -1 or 0 <= int(res[BEST]) < len(actions))
            self.assertTrue(math.isfinite(res[STOP_VALUE]))
            if forced == move and move >= 0:
                self.assertTrue(math.isfinite(res[FORCED_VALUE]), "a non-tactical forced move is still searched")
            if forced == end:
                self.assertTrue(math.isnan(res[FORCED_VALUE]), "END_TURN is 'stopping now', not a line")
        self.assertEqual(te.get_observation(full_visibility=True), before, "analysis must not change the live game")


if __name__ == "__main__":
    unittest.main()
