"""Combat fog memory (core.game.FogMemory, POLYVISION_FOG_MEMORY=1): the observation contract, checkpoint
versioning, the Java puzzles (pol_env/Tribes/tests/java/players/FogMemoryTest.java) and a live combat game.
The Java and live parts are skipped without a JDK or a compiled engine."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pol_env.Tribes.py import environment_contract as EC

REPO = Path(__file__).resolve().parents[2]
TRIBES = REPO / "pol_env" / "Tribes"
OUT = TRIBES / "out"
HAVE_ENGINE = bool(shutil.which("java")) and (OUT / "core" / "game" / "FogMemory.class").exists()
COMBAT = {"POLYVISION_COMBAT_MODE": "1", "POLYVISION_PHASE": "2"}


class ContractTests(unittest.TestCase):
    def test_layout_adds_channels_and_scalars_only_when_on(self):
        for rules in EC.SUPPORTED_RULES_VERSIONS:
            off = EC.observation_layout(11, 11, rules, False)
            on = EC.observation_layout(11, 11, rules, True)
            self.assertEqual(off.fog_memory_end - off.fog_memory_start, 0)
            self.assertEqual(off.fog_scalar_end - off.fog_scalar_start, 0)
            self.assertEqual(on.expected_obs_dim, off.expected_obs_dim + 4 * 121 + 3)
            self.assertEqual(on.spatial_channels, off.spatial_channels + 4)
            self.assertEqual(on.scalar_dim, off.scalar_dim + 3)
            self.assertEqual(on.fog_memory_start, off.legacy_scalar_start)  # right after the existing channels
            self.assertEqual(on.fog_scalar_end, on.expected_obs_dim)          # at the very end

    def test_default_layout_is_unchanged(self):
        self.assertEqual(EC.observation_layout(11, 11, "tribes_2020", False).expected_obs_dim, 6424)
        self.assertEqual(EC.observation_layout(11, 11, "polytopia_2_15", False).expected_obs_dim, 6695)

    def test_combat_only(self):
        self.assertFalse(EC.resolve_fog_memory({}))
        self.assertFalse(EC.resolve_fog_memory({"POLYVISION_FOG_MEMORY": "0"}))
        self.assertTrue(EC.resolve_fog_memory({"POLYVISION_FOG_MEMORY": "1", **COMBAT}))
        with self.assertRaises(ValueError):
            EC.resolve_fog_memory({"POLYVISION_FOG_MEMORY": "1"})  # Phase 1
        with self.assertRaises(ValueError):
            EC.resolve_fog_memory({"POLYVISION_FOG_MEMORY": "yes", **COMBAT})
        self.assertTrue(EC.fog_memory_sees_moves({}))
        self.assertFalse(EC.fog_memory_sees_moves({"POLYVISION_FOG_MEMORY_SEE_MOVES": "0"}))

    def test_version_suffix_round_trips_with_rules(self):
        from py_rl.cleanrl.cleanrl.init_from import rules_of_version

        base = "combat_v1_h20"
        v = EC.versioned_for_rules(EC.versioned_for_fog_memory(base, True), "polytopia_2_15")
        self.assertTrue(EC.fog_memory_of_version(v))
        self.assertEqual(rules_of_version(v), "polytopia_2_15")
        self.assertFalse(EC.fog_memory_of_version(EC.versioned_for_rules(base, "polytopia_2_15")))
        self.assertEqual(EC.versioned_for_fog_memory(base, False), base)

    def test_init_from_maps_a_checkpoint_without_fog_memory_into_one_with_it(self):
        from py_rl.cleanrl.cleanrl.init_from import observation_index_map, observation_layout_for

        src_meta = {"phase1_environment_version": EC.versioned_for_rules("combat_v1_h20", "polytopia_2_15")}
        tgt_meta = {"phase1_environment_version": EC.versioned_for_rules(
            EC.versioned_for_fog_memory("combat_v1_h20", True), "polytopia_2_15")}
        src, tgt = observation_layout_for(src_meta), observation_layout_for(tgt_meta)
        index = observation_index_map(src, tgt)
        self.assertIsNotNone(index)
        self.assertEqual(len(index), src.expected_obs_dim)
        fog = set(range(tgt.fog_memory_start, tgt.fog_memory_end)) | set(range(tgt.fog_scalar_start, tgt.fog_scalar_end))
        self.assertFalse(fog & set(index))  # the new inputs receive nothing (zero weight)
        self.assertEqual(index[src.legacy_scalar_start], tgt.legacy_scalar_start)


@unittest.skipUnless(HAVE_ENGINE and shutil.which("javac"), "JDK or compiled engine missing")
class JavaPuzzles(unittest.TestCase):
    def test_puzzles(self):
        cp = os.pathsep.join([str(OUT), str(TRIBES / "lib" / "json.jar")])
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["javac", "-nowarn", "-cp", cp, "-d", tmp,
                            str(TRIBES / "tests" / "java" / "players" / "FogMemoryTest.java")],
                           check=True, capture_output=True, text=True)
            for rules in ("polytopia_2_15", "tribes_2020"):
                for level in ("map_2p_010000.csv", "map_2p_010042.csv"):
                    r = subprocess.run(["java", "-cp", os.pathsep.join([cp, tmp]), "players.FogMemoryTest",
                                        f"levels/phase2_pool_1v1/train/{level}"], cwd=TRIBES, capture_output=True,
                                       text=True, env={**os.environ, "POLYVISION_RULES": rules})
                    self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                    self.assertIn("FogMemoryTest: PASS", r.stdout)


LIVE_GAME = r"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
from pol_env.Tribes.py.register_env import TribesGymWrapper
from pol_env.Tribes.py.environment_contract import observation_layout
env = TribesGymWrapper()
layout = observation_layout(11, 11, env.RULES_VERSION, env.FOG_MEMORY)
rng = np.random.default_rng(3)
obs, info = env.reset(seed=11)
report = {"version": env.PHASE1_ENVIRONMENT_VERSION, "dim": int(obs.shape[0]), "expected": layout.expected_obs_dim,
          "fog_on_seen_tile": 0, "max_presence": 0.0, "max_unseen": 0.0, "steps": 0}
n = layout.n_tiles
for _ in range(400):
    if layout.fog_memory_end > layout.fog_memory_start:
        terrain = obs[layout.terrain_start:layout.terrain_end]
        fog = obs[layout.fog_memory_start:layout.fog_memory_end].reshape(4, n)
        seen = terrain != 7
        report["fog_on_seen_tile"] += int((fog[[0, 1, 3]][:, seen] > 0).sum())  # never belief on a tile we see
        report["max_presence"] = max(report["max_presence"], float(fog[0].max()))
        report["max_unseen"] = max(report["max_unseen"], float(fog[3].max()))
    ids = np.asarray(info["legal_global_ids_padded"]).reshape(-1)
    valid = np.flatnonzero(np.asarray(info["legal_action_valid_mask"], bool).reshape(-1))
    obs, r, term, trunc, info = env.step(int(ids[int(rng.choice(valid))]))
    report["steps"] += 1
    if term or trunc:
        obs, info = env.reset(seed=12)
print("REPORT" + json.dumps(report))
"""


@unittest.skipUnless(HAVE_ENGINE, "compiled engine missing")
class LiveGame(unittest.TestCase):
    def run_game(self, fog: str) -> dict:
        env = {**os.environ, **COMBAT, "POLYVISION_SOLO_NO_OPPONENT_MODE": "0", "POLYVISION_INFO_MODE": "fast",
               "POLYVISION_LEVEL_POOL_GLOB": "levels/phase2_pool_1v1/eval_validation/*.csv",
               "POLYVISION_RULES": "polytopia_2_15", "POLYVISION_MAX_LEGAL_ACTIONS": "512",
               "POLYVISION_FOG_MEMORY": fog}
        r = subprocess.run([sys.executable, "-c", LIVE_GAME], cwd=REPO, env=env, capture_output=True, text=True,
                           timeout=600)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        line = [l for l in r.stdout.splitlines() if l.startswith("REPORT")][-1]
        return json.loads(line[len("REPORT"):])

    def test_fog_memory_game(self):
        on = self.run_game("1")
        self.assertIn("+fog_memory_v1", on["version"])
        self.assertEqual(on["dim"], on["expected"])
        self.assertEqual(on["fog_on_seen_tile"], 0)
        self.assertGreater(on["max_unseen"], 0.0)  # the enemy capital starts unfound
        off = self.run_game("0")
        self.assertNotIn("fog_memory", off["version"])
        self.assertEqual(off["dim"], 6695)


if __name__ == "__main__":
    unittest.main()
