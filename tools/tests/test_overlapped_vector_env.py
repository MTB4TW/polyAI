"""--overlap-rollout: two half vector envs must reproduce one vector env exactly (outputs, infos, buffers)."""
import os
import random
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

from py_rl.cleanrl.cleanrl.overlapped_vector_env import (
    OverlappedRolloutStepper, OverlappedVectorEnv, merge_vector_infos, split_sizes,
)
from py_rl.cleanrl.cleanrl.ppo import (
    Args, _extract_vector_legal_feature_tensors, _extract_vector_legal_tensors, _normalize_final_infos,
    make_env, make_training_vector_env,
)

ROOT = Path(__file__).resolve().parents[2]
SAME_STEP = gym.vector.AutoresetMode.SAME_STEP


class VariedInfoGame(gym.Env):
    """Episode lengths, info keys, nesting and dtypes vary with the env index and the step, so the merge
    sees keys present in only one half, nested dicts, ndarray values, objects and SAME_STEP final_info."""

    def __init__(self, idx):
        self.idx = idx
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(-1e6, 1e6, (4,), dtype=np.float32)
        self.t = 0
        self.episode = 0

    def _obs(self):
        return np.array([self.idx, self.t, self.episode, self.np_random.random()], dtype=np.float32)

    def _info(self):
        info = {"t": self.t, "ratio": self.t / 3.0, "flag": bool(self.t % 2), "name": f"e{self.idx}",
                "legal_global_ids_padded": np.array([0, 1, 2, 3], dtype=np.int64),
                "legal_action_valid_mask": np.array([True, True, self.t % 2 == 0, False])}
        if self.idx % 2:
            info["odd_only"] = np.int64(self.idx)
        if self.t % 3 == 0:
            info["nested"] = {"a": self.t, "arr": np.full(2, self.idx, dtype=np.float32)}
        if self.idx >= 2:
            info["late_key"] = "only envs >= 2"
        return info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.episode += 1
        return self._obs(), self._info()

    def step(self, action):
        self.t += 1
        ended = self.t >= 2 + (self.idx + self.episode) % 4
        info = self._info()
        info["action"] = int(action)
        if ended:
            info["episode"] = {"r": np.float32(self.t), "l": np.int64(self.t)}
        return self._obs(), float(action) - self.idx, ended and self.idx % 3 != 2, ended and self.idx % 3 == 2, info


def _fns(n):
    return [(lambda i=i: VariedInfoGame(i)) for i in range(n)]


class _SyncWithAsyncApi(gym.vector.SyncVectorEnv):
    """SyncVectorEnv with AsyncVectorEnv's split reset/step calls (the work happens in the *_wait call)."""

    def reset_async(self, **kwargs):
        self._pending = ("reset", kwargs)

    def reset_wait(self):
        return self.reset(**self._pending[1])

    def step_async(self, actions):
        self._pending = ("step", actions)

    def step_wait(self):
        return self.step(self._pending[1])


def _sync(fns):
    return _SyncWithAsyncApi(fns, autoreset_mode=SAME_STEP)


def assert_same(test, a, b, path="root"):
    if isinstance(a, dict):
        test.assertIsInstance(b, dict, path)
        test.assertEqual(list(a.keys()), list(b.keys()), path)
        for k in a:
            assert_same(test, a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (tuple, list)):
        test.assertEqual(len(a), len(b), path)
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(test, x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray):
        test.assertIsInstance(b, np.ndarray, path)
        test.assertEqual((a.dtype, a.shape), (b.dtype, b.shape), path)
        if a.dtype == object:
            for i, (x, y) in enumerate(zip(a.reshape(-1), b.reshape(-1))):
                assert_same(test, x, y, f"{path}[{i}]")
        else:
            test.assertEqual(a.tobytes(), b.tobytes(), path)
    else:
        test.assertEqual(type(a), type(b), path)
        test.assertEqual(a, b, path)


class MergeTests(unittest.TestCase):
    def test_split_sizes(self):
        self.assertEqual(split_sizes(2), (1, 1))
        self.assertEqual(split_sizes(5), (3, 2))
        self.assertEqual(split_sizes(8), (4, 4))
        with self.assertRaises(ValueError):
            split_sizes(1)

    def test_merge_matches_one_vector_env_for_even_and_odd_n(self):
        for n in (2, 3, 5, 6):
            single, split = _sync(_fns(n)), OverlappedVectorEnv(_fns(n), _sync)
            try:
                assert_same(self, single.reset(seed=11), split.reset(seed=11), f"reset n={n}")
                rng = np.random.default_rng(n)
                for t in range(40):
                    a = rng.integers(0, 5, size=n)
                    assert_same(self, single.step(a), split.step(a), f"step {t} n={n}")
            finally:
                single.close()
                split.close()

    def test_mixed_dtypes_take_the_first_envs_dtype(self):
        a = {"x": np.array([1, 2]), "_x": np.array([True, True])}
        b = {"x": np.array([0.5]), "_x": np.array([True]), "y": np.array(["s"], dtype=object), "_y": np.array([True])}
        merged = merge_vector_infos(a, 2, b, 1)
        self.assertEqual(list(merged), ["x", "_x", "y", "_y"])
        self.assertEqual(merged["x"].dtype, np.array([1]).dtype)
        self.assertEqual(merged["y"].tolist(), [None, None, "s"])
        np.testing.assert_array_equal(merged["_y"], [False, False, True])


class StepperTests(unittest.TestCase):
    """Fixed action table: the default loop (all envs, then step) and the double-buffered stepper must
    write identical rollout buffers and see identical trajectories."""

    def _inputs(self, obs, infos, n):
        infos = _normalize_final_infos(infos, n)
        ids, valid = _extract_vector_legal_tensors(infos, n, 4, 5, torch.device("cpu"))
        return {"obs": torch.Tensor(obs), "legal_global_ids": ids, "legal_action_valid_mask": valid}

    def _rollouts(self, n, iterations=3, steps=7, overlap=False):
        table = np.random.default_rng(3).integers(0, 5, size=(iterations, steps, n))
        envs = OverlappedVectorEnv(_fns(n), _sync) if overlap else _sync(_fns(n))
        buf = {k: torch.zeros(iterations, steps, n) for k in ("act", "rew", "done", "obs0")}
        legal = torch.zeros(iterations, steps, n, 4, dtype=torch.bool)
        seen_inputs = []
        try:
            obs, infos = envs.reset(seed=5)
            inputs = self._inputs(obs, infos, n)
            for it in range(iterations):
                def act(step, sl, inp, it=it):
                    seen_inputs.append((it, step, sl.start, inp["obs"].clone()))
                    a = torch.as_tensor(table[it, step, sl])
                    buf["act"][it, step, sl] = a.float()
                    return a
                stepper = OverlappedRolloutStepper(envs, self._inputs, act) if overlap else None
                for step in range(steps):
                    buf["obs0"][it, step] = inputs["obs"][:, 0]
                    legal[it, step] = inputs["legal_action_valid_mask"]
                    if overlap:
                        (obs, rew, term, trunc, infos), action = stepper.step(step, inputs, step == steps - 1)
                    else:
                        action = act(step, slice(0, n), inputs)
                        obs, rew, term, trunc, infos = envs.step(action.numpy())
                    np.testing.assert_array_equal(action.numpy(), table[it, step])
                    buf["rew"][it, step] = torch.tensor(rew, dtype=torch.float32)
                    buf["done"][it, step] = torch.tensor(np.logical_or(term, trunc), dtype=torch.float32)
                    inputs = self._inputs(obs, infos, n)
        finally:
            envs.close()
        return buf, legal, seen_inputs

    def test_buffers_identical(self):
        for n in (2, 3, 5):
            want, want_legal, _ = self._rollouts(n)
            got, got_legal, seen = self._rollouts(n, overlap=True)
            for k in want:
                self.assertTrue(torch.equal(want[k], got[k]), f"{k} n={n}")
            self.assertTrue(torch.equal(want_legal, got_legal), f"legal n={n}")
            self.assertGreater(float(want["done"].sum()), 0)
            # Policy order: A(0), B(0), A(1), B(1), ...; half A of a rollout's step 0 is never prefetched.
            self.assertEqual([(it, s, start) for it, s, start, _ in seen][:4], [(0, 0, 0), (0, 0, split_sizes(n)[0]),
                                                                                (0, 1, 0), (0, 1, split_sizes(n)[0])])

    def test_prefetch_inputs_are_checked(self):
        envs = OverlappedVectorEnv(_fns(2), _sync)
        try:
            obs, infos = envs.reset(seed=1)
            inputs = self._inputs(obs, infos, 2)
            stepper = OverlappedRolloutStepper(envs, self._inputs, lambda s, sl, inp: torch.zeros(sl.stop - sl.start,
                                                                                                    dtype=torch.long))
            (obs, _, _, _, infos), _ = stepper.step(0, inputs, last_step=False)
            bad = self._inputs(obs, infos, 2)
            bad["obs"] = bad["obs"] + 1
            with self.assertRaises(RuntimeError):
                stepper.step(1, bad, last_step=True)
        finally:
            envs.close()

    def test_async_training_vector_env_matches(self):
        single = make_training_vector_env(_fns(3))
        split = make_training_vector_env(_fns(3), overlap_rollout=True)
        try:
            self.assertIsInstance(split, OverlappedVectorEnv)
            assert_same(self, single.reset(seed=2), split.reset(seed=2))
            for t in range(12):
                a = np.array([t % 5, (t + 1) % 5, (t + 2) % 5])
                assert_same(self, single.step(a), split.step(a), f"step {t}")
        finally:
            single.close()
            split.close()


def _strip_timing(x, parent=None):
    """Drop wall-clock values: profile_* timers and RecordEpisodeStatistics' episode time `t`."""
    if isinstance(x, dict):
        return {k: _strip_timing(v, k) for k, v in x.items()
                if not str(k).lstrip("_").startswith("profile_") and not (parent == "episode" and k in ("t", "_t"))}
    return x


@unittest.skipUnless(os.environ.get("POLYVISION_RUN_JAVA_TESTS") == "1", "requires local Java gateway")
class LiveOverlapTests(unittest.TestCase):
    def test_real_envs_same_trajectory_and_tensors(self):
        settings = dict(POLYVISION_LEVEL_POOL_GLOB="levels/phase1_pool_bardur_real/train/*.csv",
                        POLYVISION_SOLO_NO_OPPONENT_MODE="1", POLYVISION_INFO_MODE="fast")
        jdks = sorted((ROOT / ".runtime").glob("jdk-*/Contents/Home"))
        if jdks:
            settings["JAVA_HOME"] = str(jdks[-1])
            settings["PATH"] = str(jdks[-1] / "bin") + os.pathsep + os.environ["PATH"]
        n, steps = 3, int(os.environ.get("POLYVISION_OVERLAP_TEST_STEPS", "200"))
        with patch.dict(os.environ, settings):
            fns = [make_env("Tribes-v0", i, False, "overlap_test", 0.0, 0.1) for i in range(n)]
            runs = []
            for overlap in (False, True):
                envs = make_training_vector_env(fns, overlap_rollout=overlap)
                rng = random.Random(4)
                trace, episodes = [], 0
                try:
                    obs, infos = envs.reset(seed=21)
                    trace.append((obs, _strip_timing(infos)))
                    for _ in range(steps):
                        norm = _normalize_final_infos(infos, n)
                        ids, valid = _extract_vector_legal_tensors(norm, n, 256, envs.single_action_space.n,
                                                                   torch.device("cpu"))
                        feats = _extract_vector_legal_feature_tensors(norm, n, 256, Args().legal_action_feature_dim,
                                                                      torch.device("cpu"))
                        trace.append((ids.numpy(), valid.numpy(), feats.numpy()))
                        action = np.array([int(ids[i][valid[i]][rng.randrange(int(valid[i].sum()))]) for i in range(n)])
                        obs, rew, term, trunc, infos = envs.step(action)
                        episodes += int(np.logical_or(term, trunc).sum())
                        trace.append((action, obs, rew, term, trunc, _strip_timing(infos)))
                finally:
                    envs.close()
                runs.append((trace, episodes))
            self.assertGreater(runs[0][1], 0, "trajectory should include episode ends")
            self.assertEqual(runs[0][1], runs[1][1])
            assert_same(self, runs[0][0], runs[1][0])


if __name__ == "__main__":
    unittest.main()
