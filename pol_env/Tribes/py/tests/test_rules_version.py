"""Tests for the POLYVISION_RULES switch (docs/rules-parity.md).

Covers the version strings, the wrapper/JVM consistency check, checkpoint rejection across rule sets, and
runs the Java rule tests in tests/java/core/RulesParityTest.java under both rule sets.
"""

import glob
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pol_env.Tribes.py.environment_contract import (
    DEFAULT_RULES_VERSION,
    PHASE1_ENVIRONMENT_VERSION,
    SUPPORTED_RULES_VERSIONS,
    CheckpointCompatibilityError,
    resolve_rules_version,
    validate_checkpoint_compatibility,
    versioned_for_rules,
)
from pol_env.Tribes.py.register_env import TribesGymWrapper
from pol_env.Tribes.py.tests.test_environment_contract import compatibility_metadata


REPO_ROOT = Path(__file__).resolve().parents[4]
TRIBES_ROOT = REPO_ROOT / "pol_env" / "Tribes"
OUT_DIR = TRIBES_ROOT / "out"
JSON_JAR = TRIBES_ROOT / "lib" / "json.jar"
JAVA_TEST = TRIBES_ROOT / "tests" / "java" / "core" / "RulesParityTest.java"
# A training map whose capital (2, 2) sees the (0, 0) corner at the start, for the lighthouse fog rule.
CORNER_LEVEL = str(TRIBES_ROOT / "levels" / "phase1_pool_bardur_real" / "train" / "map_000132.csv")
# A two-tribe Phase 2 map whose tribes have not met at the start, for the meeting-stars rule.
DUEL_LEVEL = str(TRIBES_ROOT / "levels" / "phase2_pool_1v1" / "train" / "map_2p_010000.csv")
PHASE1_ENV = {
    "POLYVISION_LEVEL_POOL_GLOB": "levels/phase1_pool_bardur_real/train/*.csv",
    "POLYVISION_SOLO_NO_OPPONENT_MODE": "1",
}


class RulesVersionContractTests(unittest.TestCase):
    def test_default_is_tribes_2020(self):
        self.assertEqual(DEFAULT_RULES_VERSION, "tribes_2020")
        self.assertEqual(resolve_rules_version({}), "tribes_2020")
        self.assertEqual(resolve_rules_version({"POLYVISION_RULES": "  "}), "tribes_2020")

    def test_known_values_are_normalised(self):
        self.assertEqual(resolve_rules_version({"POLYVISION_RULES": " Polytopia_2_15 "}), "polytopia_2_15")
        for rules in SUPPORTED_RULES_VERSIONS:
            self.assertEqual(resolve_rules_version({"POLYVISION_RULES": rules}), rules)

    def test_unknown_value_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Unknown POLYVISION_RULES"):
            resolve_rules_version({"POLYVISION_RULES": "polytopia_3"})

    def test_default_rules_leave_environment_versions_unchanged(self):
        self.assertEqual(versioned_for_rules(PHASE1_ENVIRONMENT_VERSION, "tribes_2020"), PHASE1_ENVIRONMENT_VERSION)
        combat = "combat_v3_potential_shaping_horizon_20"
        self.assertEqual(versioned_for_rules(combat, "tribes_2020"), combat)

    def test_modern_rules_change_environment_versions(self):
        self.assertEqual(
            versioned_for_rules(PHASE1_ENVIRONMENT_VERSION, "polytopia_2_15"),
            "v5_human_information_parity+rules_polytopia_2_15",
        )
        self.assertEqual(
            versioned_for_rules("combat_v3_potential_shaping_horizon_20", "polytopia_2_15"),
            "combat_v3_potential_shaping_horizon_20+rules_polytopia_2_15",
        )

    def test_checkpoints_do_not_cross_rule_sets(self):
        old = compatibility_metadata()
        new = compatibility_metadata(
            phase1_environment_version=versioned_for_rules(PHASE1_ENVIRONMENT_VERSION, "polytopia_2_15")
        )
        validate_checkpoint_compatibility(old, old)
        validate_checkpoint_compatibility(new, new)
        with self.assertRaisesRegex(CheckpointCompatibilityError, "phase1_environment_version"):
            validate_checkpoint_compatibility(old, new)
        with self.assertRaisesRegex(CheckpointCompatibilityError, "phase1_environment_version"):
            validate_checkpoint_compatibility(new, old)


@unittest.skipUnless((OUT_DIR / "core" / "RulesVersion.class").exists(), "compile pol_env/Tribes/out first")
class RulesVersionWrapperTests(unittest.TestCase):
    def _wrapper(self, rules):
        env = dict(PHASE1_ENV)
        if rules is not None:
            env["POLYVISION_RULES"] = rules
        with mock.patch.dict(os.environ, env, clear=False):
            if rules is None:
                os.environ.pop("POLYVISION_RULES", None)
            return TribesGymWrapper()

    def test_default_wrapper_reports_original_version(self):
        wrapper = self._wrapper(None)
        try:
            self.assertEqual(wrapper.RULES_VERSION, "tribes_2020")
            self.assertEqual(str(wrapper.tribes_env._jvm.core.RulesVersion.activeId()), "tribes_2020")
            _, info = wrapper.reset(seed=42)
            self.assertEqual(info["phase1_environment_version"], "v5_human_information_parity")
        finally:
            wrapper.close()

    def test_modern_wrapper_plays_modern_rules(self):
        wrapper = self._wrapper("polytopia_2_15")
        try:
            self.assertEqual(wrapper.RULES_VERSION, "polytopia_2_15")
            self.assertEqual(str(wrapper.tribes_env._jvm.core.RulesVersion.activeId()), "polytopia_2_15")
            obs, info = wrapper.reset(seed=42)
            self.assertEqual(
                info["phase1_environment_version"], "v5_human_information_parity+rules_polytopia_2_15"
            )
            self.assertEqual(wrapper.tribes_env._jvm.core.Types.BUILDING.LUMBER_HUT.getCost(), 3)
            # A short episode runs end to end under the new rules.
            import numpy as np

            rng = np.random.default_rng(0)
            for _ in range(60):
                legal = info["legal_global_ids_padded"][info["legal_action_valid_mask"]]
                obs, _, terminated, truncated, info = wrapper.step(int(rng.choice(legal)))
                if terminated or truncated:
                    obs, info = wrapper.reset()
        finally:
            wrapper.close()

    def test_wrapper_rejects_a_jvm_playing_other_rules(self):
        wrapper = self._wrapper(None)
        try:
            wrapper.RULES_VERSION = "polytopia_2_15"
            with self.assertRaisesRegex(RuntimeError, "plays rules 'tribes_2020'"):
                wrapper._verify_java_rules_version()
        finally:
            wrapper.close()


@unittest.skipUnless(shutil.which("javac") and shutil.which("java"), "JDK not on PATH")
@unittest.skipUnless((OUT_DIR / "core" / "RulesVersion.class").exists(), "compile pol_env/Tribes/out first")
class JavaRulesParityTests(unittest.TestCase):
    def test_java_rules_parity(self):
        levels = sorted(glob.glob(str(TRIBES_ROOT / "levels" / "phase1_pool_bardur_real" / "train" / "*.csv")))
        self.assertTrue(levels, "no Phase 1 training levels found")
        classpath = os.pathsep.join([str(OUT_DIR), str(JSON_JAR)])
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                ["javac", "-nowarn", "-cp", classpath, "-d", tmp, str(JAVA_TEST)],
                check=True,
                capture_output=True,
                text=True,
            )
            result = subprocess.run(
                ["java", "-cp", os.pathsep.join([classpath, tmp]), "core.RulesParityTest", levels[0], CORNER_LEVEL, DUEL_LEVEL],
                cwd=TRIBES_ROOT,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("RulesParityTest: PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
