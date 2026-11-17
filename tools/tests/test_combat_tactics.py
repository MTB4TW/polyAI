"""Runs the Java tactical-search puzzles (pol_env/Tribes/tests/java/players/CombatTacticsTest.java) and checks the
combat teacher mode wiring. Skipped without a JDK or a compiled engine."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRIBES = REPO / "pol_env" / "Tribes"
OUT = TRIBES / "out"


@unittest.skipUnless(shutil.which("javac") and shutil.which("java"), "JDK not on PATH")
@unittest.skipUnless((OUT / "players" / "heuristics" / "CombatTactics.class").exists(), "compile pol_env/Tribes/out first")
class CombatTacticsPuzzles(unittest.TestCase):
    def test_puzzles(self):
        cp = os.pathsep.join([str(OUT), str(TRIBES / "lib" / "json.jar")])
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["javac", "-nowarn", "-cp", cp, "-d", tmp,
                            str(TRIBES / "tests" / "java" / "players" / "CombatTacticsTest.java")],
                           check=True, capture_output=True, text=True)
            for level in ("map_2p_010000.csv", "map_2p_010042.csv"):
                r = subprocess.run(["java", "-cp", os.pathsep.join([cp, tmp]), "players.CombatTacticsTest",
                                    f"levels/phase2_pool_1v1/train/{level}"], cwd=TRIBES, capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                self.assertIn("CombatTacticsTest: PASS", r.stdout)


class TeacherModeTests(unittest.TestCase):
    def test_modes_fit_their_phase(self):
        from py_rl.cleanrl.cleanrl.teacher_kickstart import validate_teacher_args

        self.assertEqual(validate_teacher_args("tactical_search", 1.0, 0.5, "legal_features", True), "tactical_search")
        with self.assertRaises(RuntimeError):
            validate_teacher_args("tactical_search", 1.0, 0.5, "legal_features", False)
        with self.assertRaises(RuntimeError):
            validate_teacher_args("lookahead_visible", 1.0, 0.5, "legal_features", True)


if __name__ == "__main__":
    unittest.main()
