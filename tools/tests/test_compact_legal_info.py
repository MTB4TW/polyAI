"""Compact legal-action infos (valid rows only) must rebuild exactly the padded trainer tensors.

The live test is opt-in: POLYVISION_RUN_JAVA_TESTS=1 python -m unittest tools.tests.test_compact_legal_info
"""
import os
import random
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

from py_rl.cleanrl.cleanrl.ppo import (
    COMPACT_LEGAL_INFO_KEY,
    CompactLegalInfoWrapper,
    _extract_vector_action_mask,
    _extract_vector_legal_feature_tensors,
    _extract_vector_legal_tensors,
    _normalize_final_infos,
    compact_legal_info,
    expand_compact_legal_rows,
    make_training_vector_env,
)

ROOT = Path(__file__).resolve().parents[2]
SLOTS, DIM, ACTIONS = 256, 47, 900
CPU = torch.device("cpu")


def padded_info(rng, n):
    ids = np.zeros(SLOTS, dtype=np.int32)
    valid = np.zeros(SLOTS, dtype=np.bool_)
    feats = np.zeros((SLOTS, DIM), dtype=np.float32)
    ids[:n] = rng.choice(ACTIONS, size=n, replace=False)
    valid[:n] = True
    feats[:n] = rng.standard_normal((n, DIM)).astype(np.float32)
    feats[:n, ::3] = 0.0  # zero (and negative-zero) cells inside valid rows must survive as-is
    if n:
        feats[0, 1] = -0.0
    return {"legal_global_ids_padded": ids, "legal_action_valid_mask": valid,
            "legal_action_features_padded": feats, "legal_action_count": n, "spt": 2.0}


def extract_all(infos, num_envs, action_dim=ACTIONS):
    ids, valid = _extract_vector_legal_tensors(infos, num_envs, SLOTS, action_dim, CPU)
    feats = _extract_vector_legal_feature_tensors(infos, num_envs, SLOTS, DIM, CPU)
    mask = _extract_vector_action_mask(infos, num_envs, action_dim, CPU)
    return ids, valid, feats, mask


class RandomLegalEnv(gym.Env):
    """Emits padded legal infos with a random number of valid slots; ends episodes now and then."""

    observation_space = gym.spaces.Box(-1, 1, (4,), dtype=np.float32)
    action_space = gym.spaces.Discrete(ACTIONS)

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def _info(self):
        return padded_info(self.rng, int(self.rng.choice([0, 1, 18, 40, 100, SLOTS])))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(4, dtype=np.float32), self._info()

    def step(self, action):
        done = bool(self.rng.random() < 0.2)
        return np.ones(4, dtype=np.float32), 0.0, done, False, self._info()


def env_a():
    return RandomLegalEnv(1)


def env_b():
    return RandomLegalEnv(2)


def compact_env_a():
    return CompactLegalInfoWrapper(RandomLegalEnv(1))


def compact_env_b():
    return CompactLegalInfoWrapper(RandomLegalEnv(2))


class CompactLegalInfoTests(unittest.TestCase):
    def test_round_trip_is_bit_identical(self):
        rng = np.random.default_rng(0)
        for n in (0, 1, 18, 100, SLOTS):
            info = padded_info(rng, n)
            compact = compact_legal_info(info)
            for key in ("legal_global_ids_padded", "legal_action_valid_mask", "legal_action_features_padded"):
                self.assertNotIn(key, compact)
            self.assertEqual(compact["legal_action_count"], n)
            ids, valid, feats = expand_compact_legal_rows(compact[COMPACT_LEGAL_INFO_KEY])
            for got, key in ((ids, "legal_global_ids_padded"), (valid, "legal_action_valid_mask"),
                             (feats, "legal_action_features_padded")):
                want = info[key]
                self.assertEqual((got.dtype, got.shape), (want.dtype, want.shape), key)
                self.assertEqual(got.tobytes(), want.tobytes(), key)
            # The compact rows are copies, not views that pin the padded buffers.
            self.assertIsNone(compact[COMPACT_LEGAL_INFO_KEY][1].base)

    def test_unexpected_layouts_stay_padded(self):
        rng = np.random.default_rng(1)
        bad = []
        info = padded_info(rng, 5)
        info["legal_action_features_padded"][7, 3] = 1.0  # nonzero padding row
        bad.append(info)
        info = padded_info(rng, 5)
        info["legal_action_features_padded"][9, 0] = -0.0  # sign bit in padding
        bad.append(info)
        info = padded_info(rng, 5)
        info["legal_action_valid_mask"][2] = False  # not a prefix
        bad.append(info)
        info = padded_info(rng, 5)
        info["legal_global_ids_padded"][200] = 4  # padding id
        bad.append(info)
        info = padded_info(rng, 5)
        info["legal_action_features_padded"] = info["legal_action_features_padded"].astype(np.float64)
        bad.append(info)
        for info in bad:
            self.assertIs(compact_legal_info(info), info)
        partial = {"legal_global_ids_padded": np.zeros(SLOTS, np.int32)}
        self.assertIs(compact_legal_info(partial), partial)
        self.assertIsNone(compact_legal_info(None))

    def test_vector_env_tensors_match_padded_path(self):
        torch.set_num_threads(1)
        padded = make_training_vector_env([env_a, env_b])
        compact = make_training_vector_env([compact_env_a, compact_env_b])
        try:
            _, infos_p = padded.reset(seed=3)
            _, infos_c = compact.reset(seed=3)
            self.assertIn(COMPACT_LEGAL_INFO_KEY, infos_c)
            self.assertNotIn("legal_action_features_padded", infos_c)
            finals = 0
            for step in range(60):
                for got, want in zip(extract_all(infos_c, 2), extract_all(infos_p, 2)):
                    self.assertEqual(got.dtype, want.dtype)
                    self.assertTrue(torch.equal(got, want), f"step={step}")
                    self.assertEqual(got.numpy().tobytes(), want.numpy().tobytes())
                actions = np.array([1, 2])
                _, _, term_p, _, infos_p = padded.step(actions)
                _, _, term_c, _, infos_c = compact.step(actions)
                np.testing.assert_array_equal(term_p, term_c)
                infos_p = _normalize_final_infos(infos_p, 2)
                infos_c = _normalize_final_infos(infos_c, 2)
                self.assertEqual("final_info" in infos_p, "final_info" in infos_c)
                finals += int(bool(term_p.any()))
                for i in range(2):
                    fp = infos_p.get("final_info", [None, None])[i]
                    fc = infos_c.get("final_info", [None, None])[i]
                    if fp is None:
                        self.assertIsNone(fc)
                        continue
                    ids, valid, feats = expand_compact_legal_rows(fc[COMPACT_LEGAL_INFO_KEY])
                    self.assertEqual(feats.tobytes(), fp["legal_action_features_padded"].tobytes())
                    self.assertEqual(ids.tobytes(), fp["legal_global_ids_padded"].tobytes())
            self.assertGreater(finals, 0)  # SAME_STEP autoreset rows were exercised
        finally:
            padded.close()
            compact.close()

    def test_mixed_compact_and_padded_rows(self):
        rng = np.random.default_rng(4)
        infos_list = [padded_info(rng, 3), padded_info(rng, 6), {"spt": 1.0}]
        infos_list[1]["legal_action_features_padded"][50, 0] = 2.0  # forces env 1 to fall back to padded
        # env 2 sends no legal tensors at all; Gymnasium leaves zero rows with a False presence mask.

        def batch(rows):
            out = {}
            for i, row in enumerate(rows):
                out = _add_info(out, row, i, len(rows))
            return out

        for count in (2, 3):
            padded = batch(infos_list[:count])
            mixed = batch([compact_legal_info(info) for info in infos_list[:count]])
            self.assertIn(COMPACT_LEGAL_INFO_KEY, mixed)
            self.assertIn("legal_action_features_padded", mixed)
            for got, want in zip(extract_all(mixed, count), extract_all(padded, count)):
                self.assertTrue(torch.equal(got, want))

    def test_env_var_disables_wrapper(self):
        from py_rl.cleanrl.cleanrl import ppo
        with patch.dict(os.environ, {"POLYVISION_COMPACT_LEGAL_INFO": "0"}):
            self.assertFalse(ppo._compact_legal_info_enabled())
        with patch.dict(os.environ, {"POLYVISION_COMPACT_LEGAL_INFO": "1"}):
            self.assertTrue(ppo._compact_legal_info_enabled())


def _add_info(vector_infos, env_info, env_num, num_envs):
    """Gymnasium's VectorEnv._add_info batching rule, without building a vector env."""
    holder = type("H", (), {"num_envs": num_envs, "_add_info": gym.vector.VectorEnv._add_info})()
    return holder._add_info(vector_infos, env_info, env_num)


@unittest.skipUnless(os.environ.get("POLYVISION_RUN_JAVA_TESTS") == "1", "requires local Java gateway")
class LiveCompactLegalInfoTests(unittest.TestCase):
    def test_real_env_infos_round_trip(self):
        from pol_env.Tribes.py.register_env import TribesGymWrapper

        settings = dict(POLYVISION_LEVEL_POOL_GLOB="levels/phase1_pool_bardur_real/train/*.csv",
                        POLYVISION_SOLO_NO_OPPONENT_MODE="1", POLYVISION_INFO_MODE="fast")
        jdks = sorted((ROOT / ".runtime").glob("jdk-*/Contents/Home"))
        if jdks:
            settings["JAVA_HOME"] = str(jdks[-1])
            settings["PATH"] = str(jdks[-1] / "bin") + os.pathsep + os.environ["PATH"]
        with patch.dict(os.environ, settings):
            env = TribesGymWrapper()
            try:
                rng = random.Random(9)
                _, info = env.reset(seed=9)
                compacted = 0
                for _ in range(300):
                    compact = compact_legal_info(info)
                    compacted += COMPACT_LEGAL_INFO_KEY in compact
                    n_actions = int(env.action_space.n)
                    want = extract_all(_add_info({}, info, 0, 1), 1, n_actions)
                    got = extract_all(_add_info({}, compact, 0, 1), 1, n_actions)
                    for g, w in zip(got, want):
                        self.assertEqual(g.numpy().tobytes(), w.numpy().tobytes())
                    legal = np.asarray(info["legal_global_ids_padded"])[np.asarray(info["legal_action_valid_mask"])]
                    _, _, term, trunc, info = env.step(int(legal[rng.randrange(len(legal))]))
                    if term or trunc:
                        _, info = env.reset(seed=rng.randrange(1000))
                self.assertEqual(compacted, 300)  # the env always produces the compactable layout
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
