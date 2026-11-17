"""The combat evaluator never plays a league opponent or runs training-only inputs (tools/evaluate_combat.py)."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.evaluate_combat import assert_evaluation_environment, clear_training_only_environment

ROOT = Path(__file__).resolve().parents[2]

TRAINING_SETTINGS = {
    "POLYVISION_COMBAT_LEAGUE": "1",
    "POLYVISION_COMBAT_LEAGUE_GLOB": "runs/x/*.cleanrl_model",
    "POLYVISION_COMBAT_LEAGUE_SIMPLE_PROB": "0.5",
    "POLYVISION_COMBAT_REWARD_WIN": "30",
    "POLYVISION_TEACHER": "combat_tactics",
    "POLYVISION_PRIVILEGED_OBS": "1",
    "POLYVISION_FOG_MEMORY_SEE_MOVES": "0",
}


class ClearEnvironmentTests(unittest.TestCase):
    def test_training_only_settings_are_removed_and_the_league_forced_off(self):
        environ = dict(TRAINING_SETTINGS, POLYVISION_RULES="polytopia_2_15", PATH="/bin")
        removed = clear_training_only_environment(environ)
        self.assertEqual(set(removed), set(TRAINING_SETTINGS))
        self.assertEqual(environ, {"POLYVISION_COMBAT_LEAGUE": "0", "POLYVISION_RULES": "polytopia_2_15", "PATH": "/bin"})

    def test_clean_environment_is_unchanged_apart_from_the_league_switch(self):
        environ = {"POLYVISION_RULES": "polytopia_2_15"}
        self.assertEqual(clear_training_only_environment(environ), [])
        self.assertEqual(environ, {"POLYVISION_RULES": "polytopia_2_15", "POLYVISION_COMBAT_LEAGUE": "0"})

    def test_assert_rejects_league_teacher_and_privileged_obs(self):
        assert_evaluation_environment(SimpleNamespace(_combat_league=None, _teacher_enabled=False, _teacher_mode=None))
        for bad in ({"_combat_league": object()}, {"_teacher_enabled": True}, {"_teacher_mode": "combat_tactics"},
                    {"_privileged_obs": True}):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                assert_evaluation_environment(SimpleNamespace(**bad))


@unittest.skipUnless(os.environ.get("POLYVISION_RUN_JAVA_TESTS") == "1", "requires local Java gateway")
class WrapperAfterClearingTests(unittest.TestCase):
    def test_wrapper_built_after_clearing_has_no_league(self):
        from pol_env.Tribes.py.register_env import TribesGymWrapper

        with tempfile.TemporaryDirectory() as snapshots:
            settings = dict(TRAINING_SETTINGS, POLYVISION_COMBAT_LEAGUE_GLOB=os.path.join(snapshots, "*.cleanrl_model"),
                            POLYVISION_PHASE="2", POLYVISION_COMBAT_MODE="1", POLYVISION_SOLO_NO_OPPONENT_MODE="0",
                            POLYVISION_MAX_TURNS="2", POLYVISION_INFO_MODE="fast",
                            POLYVISION_LEVEL_POOL_GLOB="levels/phase2_pool_1v1/train/*.csv")
            with patch.dict(os.environ, settings):
                clear_training_only_environment(os.environ)
                env = TribesGymWrapper()
                try:
                    assert_evaluation_environment(env)
                    self.assertIsNone(env._combat_league)
                    env.reset(seed=1)
                    self.assertIsNone(env._combat_opponent)
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()
