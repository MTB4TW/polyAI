"""Tests for `ppo.py --init-from` (py_rl/cleanrl/cleanrl/init_from.py). Offline: no Java needed."""

import json
import os
import re
import tempfile
import unittest
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from py_rl.cleanrl.cleanrl import init_from as I
from py_rl.cleanrl.cleanrl.spatial_agent import SpatialResNetAgent

PHASE1_N, COMBAT_N = 63913, 78675
PHASE1_VERSION = "v5_human_information_parity"
COMBAT_VERSION = "combat_v5_scaled_scalars_horizon_20"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def meta(n, version, arch="spatial_resnet"):
    return {"network_arch": arch, "actor_mode": "legal_features", "observation_dim": "6424",
            "legal_action_feature_dim": 47, "map_width": "11", "map_height": "11",
            "legal_action_feature_version": "v1_4_parity_spatial_and_cost",
            "action_space_n": str(n), "phase1_environment_version": version}


def save(tmp, name, module, sidecar):
    path = os.path.join(tmp, name + ".cleanrl_model")
    torch.save(module.state_dict(), path)
    with open(path + ".action_interface.json", "w") as fh:
        json.dump(sidecar, fh)
    return path


def fake_envs(n):
    return SimpleNamespace(single_observation_space=gym.spaces.Box(-1e9, 1e9, (6424,), np.float32),
                           single_action_space=gym.spaces.Discrete(n))


class CatalogTest(unittest.TestCase):
    def test_combat_catalog_extends_phase1(self):
        self.assertTrue(I.catalog_extends(PHASE1_N, COMBAT_N, 11, 11))
        self.assertTrue(I.catalog_extends(PHASE1_N, PHASE1_N, 11, 11))

    def test_reverse_or_unknown_is_not_a_prefix(self):
        self.assertFalse(I.catalog_extends(COMBAT_N, PHASE1_N, 11, 11))
        self.assertFalse(I.catalog_extends(PHASE1_N, COMBAT_N + 1, 11, 11))

    def test_scale_constants_match_environment(self):
        with open(os.path.join(REPO, "pol_env", "Tribes", "py", "register_env.py"), encoding="utf-8") as fh:
            line = next(l for l in fh if "self.COMBAT_LEGACY_SCALAR_SCALES = (" in l)
        # Fixed scales are float literals; the tick scale is the combat horizon (MAX_TURNS).
        self.assertEqual([float(x) for x in re.findall(r"\b\d+\.\d+\b", line)],
                         [s for s in I.COMBAT_LEGACY_SCALAR_BASE_SCALES if s is not None])
        self.assertIn("MAX_TURNS", line.split(",")[4] + line.split(",")[5])
        self.assertEqual(I.legacy_scalar_scales(COMBAT_VERSION), (50.0, 1000.0, 6.0, 10.0, 20.0, 1.0))
        self.assertIsNone(I.legacy_scalar_scales(PHASE1_VERSION))
        self.assertIsNone(I.legacy_scalar_scales("combat_v3_horizon_20"))  # pre-scaling combat
        # The rules switch appends a suffix; the scaling must still be recognised.
        self.assertEqual(I.legacy_scalar_scales(COMBAT_VERSION + "+rules_polytopia_2_15"),
                         (50.0, 1000.0, 6.0, 10.0, 20.0, 1.0))
        self.assertIsNone(I.legacy_scalar_scales(PHASE1_VERSION + "+rules_polytopia_2_15"))


class SpatialTransferTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tmp = tempfile.mkdtemp()
        self.src = SpatialResNetAgent(envs=None, action_space_n=PHASE1_N, local_action_features=True)
        self.path = save(self.tmp, "p1", self.src, meta(PHASE1_N, PHASE1_VERSION, "spatial_resnet_local"))

    def target(self):
        torch.manual_seed(1)
        return SpatialResNetAgent(envs=None, action_space_n=COMBAT_N, local_action_features=True)

    def test_phase1_to_combat(self):
        tgt = self.target()
        fresh_new_rows = tgt.action_embedding.weight[PHASE1_N:].clone()
        fresh_critic = {k: v.clone() for k, v in tgt.state_dict().items() if k.startswith("critic.")}
        report = I.apply_init_from(tgt, self.path, target_meta=meta(COMBAT_N, COMBAT_VERSION, "spatial_resnet_local"),
                                   network_arch="spatial_resnet_local")
        s, t = self.src.state_dict(), tgt.state_dict()
        # Action table: Phase 1 rows copied, combat rows untouched.
        self.assertTrue(torch.equal(t["action_embedding.weight"][:PHASE1_N], s["action_embedding.weight"]))
        self.assertTrue(torch.equal(t["action_embedding.weight"][PHASE1_N:], fresh_new_rows))
        self.assertEqual(report["extended"], ["action_embedding.weight"])
        # Critic reset automatically (environment version differs).
        self.assertTrue(report["reset"] and all(k.startswith("critic.") for k in report["reset"]))
        for k, v in fresh_critic.items():
            self.assertTrue(torch.equal(t[k], v))
        # Everything else copied, except the scalar branch, which is folded.
        for k in s:
            if k.startswith("critic.") or k == "action_embedding.weight":
                continue
            if k == "scalar_backbone.0.weight":
                continue
            self.assertTrue(torch.equal(t[k], s[k]), k)
        self.assertEqual(report["folded"], ["scalar_backbone.0.weight"])

    def test_fold_preserves_the_scalar_branch_function(self):
        tgt = self.target()
        I.apply_init_from(tgt, self.path, target_meta=meta(COMBAT_N, COMBAT_VERSION, "spatial_resnet_local"),
                          network_arch="spatial_resnet_local")
        raw = torch.rand(5, 132) * 100.0
        scaled = raw.clone()
        for j, s in enumerate((50.0, 1000.0, 6.0, 10.0, 20.0, 1.0)):
            scaled[:, j] /= s
        with torch.no_grad():
            self.assertTrue(torch.allclose(self.src.scalar_backbone(raw), tgt.scalar_backbone(scaled), atol=1e-4))

    def test_warm_policy_matches_source_on_phase1_actions(self):
        tgt = self.target()
        I.apply_init_from(tgt, self.path, target_meta=meta(COMBAT_N, COMBAT_VERSION, "spatial_resnet_local"),
                          network_arch="spatial_resnet_local")
        torch.manual_seed(3)
        x = torch.rand(4, 6424)
        x[:, 6292:6298] = torch.tensor([12.0, 2100.0, 3.0, 1.0, 7.0, 0.0])
        xs = x.clone()
        for j, s in enumerate((50.0, 1000.0, 6.0, 10.0, 20.0, 1.0)):
            xs[:, 6292 + j] /= s
        ids = torch.randint(0, PHASE1_N, (4, 256))
        valid = torch.zeros(4, 256, dtype=torch.bool)
        valid[:, :30] = True
        feats = torch.rand(4, 256, 47)
        with torch.no_grad():
            a = self.src.get_action_distribution(x, legal_global_ids=ids, legal_action_valid_mask=valid,
                                                 legal_action_features=feats)
            b = tgt.get_action_distribution(xs, legal_global_ids=ids, legal_action_valid_mask=valid,
                                            legal_action_features=feats)
        self.assertTrue(torch.allclose(a.probs, b.probs, atol=1e-5))

    def test_no_fold_and_explicit_reset(self):
        tgt = self.target()
        report = I.apply_init_from(tgt, self.path, target_meta=meta(COMBAT_N, COMBAT_VERSION, "spatial_resnet_local"),
                                   network_arch="spatial_resnet_local", reset="none", fold_scalars=False)
        self.assertEqual(report["reset"], [])
        self.assertIsNone(report["scalar_conversion"])
        self.assertTrue(torch.equal(tgt.state_dict()["scalar_backbone.0.weight"],
                                    self.src.state_dict()["scalar_backbone.0.weight"]))
        self.assertTrue(torch.equal(tgt.state_dict()["critic.0.weight"], self.src.state_dict()["critic.0.weight"]))

    def test_same_environment_is_an_exact_copy(self):
        torch.manual_seed(2)
        tgt = SpatialResNetAgent(envs=None, action_space_n=PHASE1_N, local_action_features=True)
        report = I.apply_init_from(tgt, self.path, target_meta=meta(PHASE1_N, PHASE1_VERSION, "spatial_resnet_local"),
                                   network_arch="spatial_resnet_local")
        self.assertEqual(report["reset"], [])
        self.assertEqual(report["kept_fresh"], [])
        for k, v in self.src.state_dict().items():
            self.assertTrue(torch.equal(tgt.state_dict()[k], v), k)

    def test_mismatches_are_rejected(self):
        tgt = self.target()
        with self.assertRaises(I.InitFromError):
            I.apply_init_from(tgt, self.path, target_meta=meta(COMBAT_N, COMBAT_VERSION, "spatial_resnet"),
                              network_arch="spatial_resnet")
        bad = meta(COMBAT_N + 7, COMBAT_VERSION, "spatial_resnet_local")
        with self.assertRaises(I.InitFromError):
            I.apply_init_from(tgt, self.path, target_meta=bad, network_arch="spatial_resnet_local")
        os.remove(self.path + ".action_interface.json")
        with self.assertRaises(I.InitFromError):
            I.apply_init_from(tgt, self.path, target_meta=meta(COMBAT_N, COMBAT_VERSION, "spatial_resnet_local"),
                              network_arch="spatial_resnet_local")


class MlpTransferTest(unittest.TestCase):
    def test_mlp_first_layers_are_folded(self):
        from py_rl.cleanrl.cleanrl.ppo import Agent
        tmp = tempfile.mkdtemp()
        torch.manual_seed(0)
        src = Agent(fake_envs(PHASE1_N))
        path = save(tmp, "mlp", src, meta(PHASE1_N, PHASE1_VERSION, "mlp"))
        torch.manual_seed(1)
        tgt = Agent(fake_envs(COMBAT_N))
        report = I.apply_init_from(tgt, path, target_meta=meta(COMBAT_N, COMBAT_VERSION, "mlp"), network_arch="mlp")
        self.assertIn("state_encoder.0.weight", report["folded"])
        x = torch.rand(3, 6424) * 50.0
        xs = x.clone()
        for j, s in enumerate((50.0, 1000.0, 6.0, 10.0, 20.0, 1.0)):
            xs[:, 6292 + j] /= s
        with torch.no_grad():
            self.assertTrue(torch.allclose(src.state_encoder(x), tgt.state_encoder(xs), atol=1e-4))
        self.assertTrue(torch.equal(tgt.action_embedding.weight[:PHASE1_N], src.action_embedding.weight))


class _Normalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(6424))
        self.register_buffer("var", torch.ones(6424))
        self.register_buffer("count", torch.tensor(1.0))


class _Wrapped(nn.Module):
    """Same state_dict layout as ObsNormalizedAgent (--norm-obs): policy.* plus obs_normalizer.*."""

    def __init__(self, n):
        super().__init__()
        self.policy = SpatialResNetAgent(envs=None, action_space_n=n, local_action_features=True)
        self.obs_normalizer = _Normalizer()


class NormalizedTransferTest(unittest.TestCase):
    def test_scaling_goes_into_the_normalizer_not_the_weights(self):
        tmp = tempfile.mkdtemp()
        torch.manual_seed(0)
        src = _Wrapped(PHASE1_N)
        src.obs_normalizer.mean.uniform_(0, 100)
        src.obs_normalizer.var.uniform_(1, 400)
        path = save(tmp, "norm", src, meta(PHASE1_N, PHASE1_VERSION, "spatial_resnet_local"))
        tgt = _Wrapped(COMBAT_N)
        report = I.apply_init_from(tgt, path, target_meta=meta(COMBAT_N, COMBAT_VERSION, "spatial_resnet_local"),
                                   network_arch="spatial_resnet_local")
        self.assertEqual(sorted(report["folded"]), ["obs_normalizer.mean", "obs_normalizer.var"])
        self.assertTrue(torch.equal(tgt.policy.scalar_backbone[0].weight, src.policy.scalar_backbone[0].weight))
        x = torch.rand(4, 6424) * 100.0
        xs = x.clone()
        for j, s in enumerate((50.0, 1000.0, 6.0, 10.0, 20.0, 1.0)):
            xs[:, 6292 + j] /= s
        norm = lambda m, v: (v - m.obs_normalizer.mean) / torch.sqrt(m.obs_normalizer.var)
        self.assertTrue(torch.allclose(norm(src, x)[:, 6292:6298], norm(tgt, xs)[:, 6292:6298], atol=1e-4))

    def test_normalisation_mismatch_is_rejected(self):
        tmp = tempfile.mkdtemp()
        src = SpatialResNetAgent(envs=None, action_space_n=PHASE1_N, local_action_features=True)
        path = save(tmp, "plain", src, meta(PHASE1_N, PHASE1_VERSION, "spatial_resnet_local"))
        with self.assertRaises(I.InitFromError):
            I.apply_init_from(_Wrapped(COMBAT_N), path,
                              target_meta=meta(COMBAT_N, COMBAT_VERSION, "spatial_resnet_local"),
                              network_arch="spatial_resnet_local")


if __name__ == "__main__":
    unittest.main()
