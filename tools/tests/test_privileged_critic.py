"""Privileged (full-information) critic for combat PPO (--privileged-critic, POLYVISION_PRIVILEGED_OBS=1):
the combat-only switch, the critic network, the info batching, a live combat game (the privileged observation
reveals what fog hides and agrees with the policy observation on every seen tile) and a tiny training run whose
policy checkpoint has the same keys as one trained without it. The live parts are skipped without a compiled
engine."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pol_env.Tribes.py import environment_contract as EC

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "pol_env" / "Tribes" / "out"
HAVE_ENGINE = bool(shutil.which("java")) and (OUT / "core" / "game" / "PythonEnv.class").exists()
COMBAT = {"POLYVISION_COMBAT_MODE": "1", "POLYVISION_PHASE": "2"}


class SwitchTests(unittest.TestCase):
    def test_combat_only(self):
        self.assertFalse(EC.resolve_privileged_obs({}))
        self.assertFalse(EC.resolve_privileged_obs({"POLYVISION_PRIVILEGED_OBS": "0", **COMBAT}))
        self.assertTrue(EC.resolve_privileged_obs({"POLYVISION_PRIVILEGED_OBS": "1", **COMBAT}))
        with self.assertRaises(ValueError):
            EC.resolve_privileged_obs({"POLYVISION_PRIVILEGED_OBS": "1"})  # Phase 1
        with self.assertRaises(ValueError):
            EC.resolve_privileged_obs({"POLYVISION_PRIVILEGED_OBS": "yes", **COMBAT})


class CriticTests(unittest.TestCase):
    def setUp(self):
        from py_rl.cleanrl.cleanrl.spatial_agent import observation_sizes

        c, s = observation_sizes()
        self.obs_dim = c * 121 + s

    def test_shapes_and_normaliser(self):
        from py_rl.cleanrl.cleanrl.privileged_critic import PrivilegedCritic

        torch.manual_seed(0)
        x = torch.randn(7, self.obs_dim) * 5 + 3
        plain = PrivilegedCritic(self.obs_dim)
        self.assertIsNone(plain.obs_normalizer)
        self.assertEqual(tuple(plain(x).shape), (7, 1))
        self.assertEqual(tuple(plain.values(x, chunk=3).shape), (7,))
        torch.testing.assert_close(plain.values(x, chunk=3), plain(x).view(-1))
        normed = PrivilegedCritic(self.obs_dim, norm_obs=True)
        before = normed.obs_normalizer.mean.clone()
        normed.update_normalizer(x)
        self.assertFalse(torch.equal(before, normed.obs_normalizer.mean))
        self.assertIn("obs_normalizer.mean", normed.state_dict())
        with self.assertRaises(ValueError):
            PrivilegedCritic(self.obs_dim + 1)

    def test_extract_from_vector_info(self):
        from py_rl.cleanrl.cleanrl.privileged_critic import extract_privileged_obs

        rows = np.arange(2 * self.obs_dim, dtype=np.float32).reshape(2, self.obs_dim)
        out = extract_privileged_obs({"privileged_obs": rows, "_privileged_obs": np.array([True, True])}, 2,
                                     self.obs_dim, "cpu")
        self.assertTrue(torch.equal(out, torch.from_numpy(rows)))
        with self.assertRaises(RuntimeError):
            extract_privileged_obs({}, 2, self.obs_dim, "cpu")
        with self.assertRaises(RuntimeError):
            extract_privileged_obs({"privileged_obs": rows, "_privileged_obs": np.array([True, False])}, 2,
                                   self.obs_dim, "cpu")

    def test_sidecar_name(self):
        from py_rl.cleanrl.cleanrl.privileged_critic import privileged_critic_path

        self.assertEqual(privileged_critic_path("runs/x/m.cleanrl_model"), "runs/x/m.cleanrl_model.privileged_critic.pt")

    def test_init_from_loads_saved_critic(self):
        from py_rl.cleanrl.cleanrl.privileged_critic import (
            PrivilegedCritic, load_privileged_critic_for_init, privileged_critic_path)

        torch.manual_seed(1)
        src = PrivilegedCritic(self.obs_dim, norm_obs=True)
        src.update_normalizer(torch.randn(5, self.obs_dim))
        with tempfile.TemporaryDirectory() as d:
            model = os.path.join(d, "m.cleanrl_model")
            # No critic file: stays fresh.
            torch.manual_seed(2)
            dst = PrivilegedCritic(self.obs_dim, norm_obs=True)
            fresh = {k: v.clone() for k, v in dst.state_dict().items()}
            self.assertFalse(load_privileged_critic_for_init(dst, model))
            for k, v in dst.state_dict().items():
                self.assertTrue(torch.equal(v, fresh[k]), k)
            # Saved critic: loaded exactly, normaliser statistics included.
            torch.save(src.state_dict(), privileged_critic_path(model))
            self.assertTrue(load_privileged_critic_for_init(dst, model))
            for k, v in src.state_dict().items():
                self.assertTrue(torch.equal(dst.state_dict()[k], v), k)
            # Missing keys (no normaliser in the run) raise.
            with self.assertRaises(RuntimeError):
                load_privileged_critic_for_init(PrivilegedCritic(self.obs_dim, norm_obs=False), model)
            # Mismatched shapes raise.
            with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
                load_privileged_critic_for_init(PrivilegedCritic(self.obs_dim, norm_obs=True, hidden_channels=32), model)


LIVE_GAME = r"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
from pol_env.Tribes.py.register_env import TribesGymWrapper
from pol_env.Tribes.py.environment_contract import observation_layout
env = TribesGymWrapper()
layout = observation_layout(11, 11, env.RULES_VERSION, env.FOG_MEMORY)
n = layout.n_tiles
rng = np.random.default_rng(5)
obs, info = env.reset(seed=21)
report = {"version": env.PHASE1_ENVIRONMENT_VERSION, "has_key": "privileged_obs" in info, "steps": 0,
          "fog_in_privileged": 0, "seen_terrain_mismatch": 0, "hidden_enemy_units_revealed": 0, "max_abs_diff": 0.0}
for _ in range(300):
    if "privileged_obs" not in info:
        report["has_key"] = False
        break
    priv = np.asarray(info["privileged_obs"], np.float32)
    assert priv.shape == obs.shape, (priv.shape, obs.shape)
    terrain, pterrain = obs[:n], priv[:n]
    seen = terrain != 7
    report["fog_in_privileged"] += int((pterrain == 7).sum())
    report["seen_terrain_mismatch"] += int((terrain[seen] != pterrain[seen]).sum())
    units = obs[n:13 * n].reshape(12, n)
    punits = priv[n:13 * n].reshape(12, n)
    report["hidden_enemy_units_revealed"] += int((punits[:, ~seen] < 0).sum())
    report["max_abs_diff"] = max(report["max_abs_diff"], float(np.abs(priv - obs).max()))
    ids = np.asarray(info["legal_global_ids_padded"]).reshape(-1)
    valid = np.flatnonzero(np.asarray(info["legal_action_valid_mask"], bool).reshape(-1))
    obs, r, term, trunc, info = env.step(int(ids[int(rng.choice(valid))]))
    report["steps"] += 1
    if term or trunc:
        obs, info = env.reset(seed=22)
print("REPORT" + json.dumps(report))
"""


def _combat_env(**extra):
    return {**os.environ, **COMBAT, "POLYVISION_SOLO_NO_OPPONENT_MODE": "0", "POLYVISION_INFO_MODE": "fast",
            "POLYVISION_LEVEL_POOL_GLOB": "levels/phase2_pool_1v1/eval_validation/*.csv",
            "POLYVISION_RULES": "polytopia_2_15", "POLYVISION_MAX_LEGAL_ACTIONS": "512", **extra}


@unittest.skipUnless(HAVE_ENGINE, "compiled engine missing")
class LiveGame(unittest.TestCase):
    def run_game(self, on: str) -> dict:
        r = subprocess.run([sys.executable, "-c", LIVE_GAME], cwd=REPO, env=_combat_env(POLYVISION_PRIVILEGED_OBS=on),
                           capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        line = [l for l in r.stdout.splitlines() if l.startswith("REPORT")][-1]
        return json.loads(line[len("REPORT"):])

    def test_privileged_obs_reveals_fog_only(self):
        on = self.run_game("1")
        self.assertTrue(on["has_key"])
        self.assertEqual(on["steps"], 300)
        self.assertEqual(on["fog_in_privileged"], 0)       # nothing is fogged in the true state
        self.assertEqual(on["seen_terrain_mismatch"], 0)   # and it agrees with the policy on every seen tile
        self.assertGreater(on["max_abs_diff"], 0.0)        # but does show more than the policy sees
        off = self.run_game("0")
        self.assertFalse(off["has_key"])
        self.assertEqual(on["version"], off["version"])    # training-only: no version change


@unittest.skipUnless(HAVE_ENGINE, "compiled engine missing")
class TinyTraining(unittest.TestCase):
    def train(self, out_dir: str, privileged: bool, init_from=None, name=None):
        model = Path(out_dir) / (name or ("p.cleanrl_model" if privileged else "b.cleanrl_model"))
        cmd = [sys.executable, "py_rl/cleanrl/cleanrl/ppo.py", "--env-id", "Tribes-v0", "--network-arch",
               "spatial_resnet_composed", "--num-envs", "2", "--num-steps", "16", "--num-minibatches", "2",
               "--update-epochs", "1", "--total-timesteps", "64", "--max-legal-actions", "512", "--norm-obs",
               "--no-clip-vloss", "--norm-returns", "--save-model", "--model-path", str(model),
               "--exp-name", "pcritic_smoke", "--seed", "1"]
        if privileged:
            cmd.append("--privileged-critic")
        if init_from is not None:
            cmd += ["--init-from", str(init_from)]
        r = subprocess.run(cmd, cwd=REPO, env=_combat_env(WANDB_MODE="disabled"), capture_output=True, text=True,
                           timeout=1500)
        self.assertEqual(r.returncode, 0, (r.stdout[-2000:], r.stderr[-4000:]))
        if init_from is not None:
            return model, r.stdout
        return model

    def test_checkpoint_keys_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.train(d, True)
            b = self.train(d, False)
            sp, sb = torch.load(p, map_location="cpu"), torch.load(b, map_location="cpu")
            self.assertEqual(sorted(sp), sorted(sb))
            side = Path(str(p) + ".privileged_critic.pt")
            self.assertTrue(side.exists())
            self.assertFalse(Path(str(b) + ".privileged_critic.pt").exists())
            self.assertIn("value_head.4.weight", torch.load(side, map_location="cpu"))
            meta = json.loads(Path(str(p) + ".action_interface.json").read_text())
            self.assertEqual(meta.get("privileged_critic"), "privileged_critic_v1")
            self.assertNotIn("privileged_critic", json.loads(Path(str(b) + ".action_interface.json").read_text()))
            # Warm start with the flag: the source's critic file is loaded; without one the critic starts fresh.
            w, out = self.train(d, True, init_from=p, name="w.cleanrl_model")
            self.assertIn("init: loaded saved critic", out)
            meta = json.loads(Path(str(w) + ".action_interface.json").read_text())
            self.assertEqual(meta.get("privileged_critic_init"), "loaded")
            f, out = self.train(d, True, init_from=b, name="f.cleanrl_model")
            self.assertIn("starting a fresh critic", out)
            meta = json.loads(Path(str(f) + ".action_interface.json").read_text())
            self.assertEqual(meta.get("privileged_critic_init"), "fresh")


if __name__ == "__main__":
    unittest.main()
