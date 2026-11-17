"""Opt-in PPO experiment options (CHANGELOG 053): --norm-returns, --norm-obs, EXTRA_PPO_ARGS passthrough."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
import tyro

from py_rl.cleanrl.cleanrl.agent_factory import make_agent
from py_rl.cleanrl.cleanrl.gae import END_TURN_GLOBAL_ID, per_turn_discounts
from py_rl.cleanrl.cleanrl.obs_normalization import (
    OBS_NORMALIZATION_VERSION,
    ObsNormalizedAgent,
    RunningObsNormalizer,
    observation_normalization_from_metadata,
)
from py_rl.cleanrl.cleanrl.ppo import Args
from py_rl.cleanrl.cleanrl.return_normalization import ReturnNormalizer, RunningMeanStd

REPO_ROOT = Path(__file__).resolve().parents[2]
OBS_DIM = 6424  # 52x11x11 spatial planes + 132 scalars
N_ACTIONS = 500
MAX_LEGAL = 8
FEAT_DIM = 47


def _adapter():
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(low=-1e5, high=1e5, shape=(OBS_DIM,)),
        single_action_space=gym.spaces.Discrete(N_ACTIONS),
    )


def _agent(arch, observation_normalization=None, seed=0):
    torch.manual_seed(seed)
    return make_agent(_adapter(), network_arch=arch, actor_mode="legal_features", max_legal_actions=MAX_LEGAL,
                      legal_action_feature_dim=FEAT_DIM, observation_normalization=observation_normalization)


def _batch(n=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    obs = torch.rand((n, OBS_DIM), generator=g) * 3.0
    obs[:, -5:] *= 400.0  # unscaled raw city numbers
    ids = torch.randint(0, N_ACTIONS, (n, MAX_LEGAL), generator=g)
    valid = torch.ones((n, MAX_LEGAL), dtype=torch.bool)
    valid[:, -2:] = False
    feats = torch.rand((n, MAX_LEGAL, FEAT_DIM), generator=g)
    slots = torch.randint(0, MAX_LEGAL - 2, (n,), generator=g)
    return obs, dict(legal_global_ids=ids, legal_action_valid_mask=valid, legal_action_features=feats), slots


class FlagParsingTests(unittest.TestCase):
    def test_defaults_are_off(self):
        args = Args()
        self.assertFalse(args.norm_returns)
        self.assertFalse(args.norm_obs)
        # Existing defaults unchanged.
        self.assertEqual((args.gamma, args.learning_rate, args.update_epochs), (0.99, 2.5e-4, 4))
        self.assertTrue(args.clip_vloss)
        self.assertFalse(args.gamma_per_turn)
        self.assertIsNone(args.target_kl)

    def test_cli_flags(self):
        args = tyro.cli(Args, args=["--norm-returns", "--norm-obs", "--gamma-per-turn"])
        self.assertTrue(args.norm_returns and args.norm_obs and args.gamma_per_turn)
        self.assertEqual(tyro.cli(Args, args=[]), Args())


class RunningMeanStdTests(unittest.TestCase):
    def test_batched_updates_match_full_data(self):
        rng = np.random.default_rng(0)
        data = rng.normal(3.0, 2.0, size=(1000, 4))
        rms = RunningMeanStd(shape=(4,), epsilon=1e-12)
        for chunk in np.array_split(data, 7):
            rms.update(chunk)
        np.testing.assert_allclose(rms.mean, data.mean(axis=0), rtol=1e-9)
        np.testing.assert_allclose(rms.var, data.var(axis=0), rtol=1e-6)


class ReturnNormalizerTests(unittest.TestCase):
    def reference(self, rewards, dones, discounts, eps=1e-8, clip=10.0):
        """Straightforward per-env loop of the baselines VecNormalize reward path."""
        T, N = rewards.shape
        ret = np.zeros(N)
        carry = np.full(N, 0.99)
        seen = []
        out = np.zeros_like(rewards)
        for t in range(T):
            ret = ret * carry + rewards[t]
            seen.append(ret.copy())
            allr = np.concatenate(seen)
            # epsilon-count prior (count 1e-4, mean 0, var 1) merged with all returns so far
            c0, n = 1e-4, allr.size
            mean = allr.mean() * n / (c0 + n)
            var = (1.0 * c0 + allr.var() * n + allr.mean() ** 2 * c0 * n / (c0 + n)) / (c0 + n)
            out[t] = np.clip(rewards[t] / np.sqrt(var + eps), -clip, clip)
            carry = discounts[t].astype(float)
            ret[dones[t]] = 0.0
        return out

    def test_matches_reference_constant_gamma(self):
        rng = np.random.default_rng(1)
        rewards = rng.normal(size=(50, 3)) * 5
        dones = rng.random((50, 3)) < 0.1
        norm = ReturnNormalizer(3, 0.99)
        got = np.stack([norm(rewards[t], dones[t]) for t in range(50)])
        np.testing.assert_allclose(got, self.reference(rewards, dones, np.full((50, 3), 0.99)), rtol=1e-6, atol=1e-9)

    def test_matches_reference_per_turn(self):
        rng = np.random.default_rng(2)
        rewards = rng.normal(size=(60, 2))
        dones = rng.random((60, 2)) < 0.05
        actions = np.where(rng.random((60, 2)) < 0.3, END_TURN_GLOBAL_ID, 7)
        discounts = per_turn_discounts(actions, 0.9)
        norm = ReturnNormalizer(2, 0.9)
        got = np.stack([norm(rewards[t], dones[t], discounts[t]) for t in range(60)])
        ref = self.reference(rewards, dones, discounts)
        # reference starts with carry 0.99 but ret is 0 at t=0, so the first carry is irrelevant
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-9)

    def test_per_turn_return_accumulates_without_discount_inside_turn(self):
        norm = ReturnNormalizer(1, 0.5)
        for a, r in [(7, 1.0), (7, 1.0), (END_TURN_GLOBAL_ID, 1.0)]:
            norm([r], [False], per_turn_discounts(np.array([a]), 0.5))
        self.assertAlmostEqual(norm.returns[0], 3.0)  # no discount within a turn
        norm([1.0], [False], per_turn_discounts(np.array([7]), 0.5))
        self.assertAlmostEqual(norm.returns[0], 2.5)  # 3.0 * 0.5 + 1 after END_TURN

    def test_scaling_reduces_large_rewards_and_resets_on_done(self):
        norm = ReturnNormalizer(2, 0.99)
        for _ in range(200):
            scaled = norm([100.0, 100.0], [False, True])
        self.assertLess(abs(scaled[0]), 10.0)
        self.assertEqual(norm.returns[1], 0.0)
        self.assertGreater(norm.scale, 10.0)


class ObsNormalizerTests(unittest.TestCase):
    def test_update_matches_numpy_and_freezes_between_updates(self):
        x = torch.randn(300, 5) * torch.tensor([1.0, 10.0, 0.1, 100.0, 1.0]) + 4.0
        norm = RunningObsNormalizer(5, epsilon=1e-8, initial_count=1e-12)
        norm.update(x[:100])
        norm.update(x[100:])
        np.testing.assert_allclose(norm.mean.numpy(), x.numpy().mean(0), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(norm.var.numpy(), x.numpy().var(0), rtol=1e-4)
        before = norm(x).clone()
        _ = norm(x)  # forward never updates
        torch.testing.assert_close(norm(x), before)
        self.assertLessEqual(float(norm(x * 1e6).abs().max()), 10.0)

    def test_sidecar_field_parsing(self):
        self.assertIsNone(observation_normalization_from_metadata({}))
        self.assertIsNone(observation_normalization_from_metadata({"actor_mode": "legal_features"}))
        self.assertIsNone(observation_normalization_from_metadata(None))
        self.assertEqual(observation_normalization_from_metadata({"observation_normalization": OBS_NORMALIZATION_VERSION}),
                         OBS_NORMALIZATION_VERSION)
        with self.assertRaises(ValueError):
            observation_normalization_from_metadata({"observation_normalization": "something_else_v9"})

    def test_default_agent_is_unwrapped_and_identical(self):
        for arch in ("mlp", "spatial_resnet", "spatial_resnet_local"):
            with self.subTest(arch=arch):
                a, b = _agent(arch), _agent(arch, observation_normalization=None)
                self.assertNotIsInstance(a, ObsNormalizedAgent)
                self.assertEqual(list(a.state_dict()), list(b.state_dict()))
                for k, v in a.state_dict().items():
                    torch.testing.assert_close(v, b.state_dict()[k])

    def test_identity_stats_match_unwrapped_policy(self):
        # With mean 0 / var 1 (before any update) the wrapper is the identity on in-range inputs.
        plain = _agent("spatial_resnet_local", seed=3)
        wrapped = _agent("spatial_resnet_local", OBS_NORMALIZATION_VERSION, seed=3)
        wrapped.obs_normalizer.epsilon.fill_(0.0)
        obs, legal, slots = _batch()
        with torch.no_grad():
            p = plain.get_action_and_value(obs.clamp(-10, 10), selected_slot=slots, **legal)
            w = wrapped.get_action_and_value(obs, selected_slot=slots, **legal)
        for x, y in zip(p[2:], w[2:]):
            torch.testing.assert_close(x, y)

    def test_round_trip_save_load(self):
        for arch in ("mlp", "spatial_resnet", "spatial_resnet_local"):
            with self.subTest(arch=arch):
                agent = _agent(arch, OBS_NORMALIZATION_VERSION, seed=4)
                obs, legal, slots = _batch(seed=5)
                agent.obs_normalizer.update(obs * 2.0 + 1.0)
                self.assertEqual(sorted(k for k in agent.state_dict() if k.startswith("obs_normalizer.")),
                                 ["obs_normalizer.clip", "obs_normalizer.count", "obs_normalizer.epsilon",
                                  "obs_normalizer.mean", "obs_normalizer.var"])
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "m.cleanrl_model")
                    torch.save(agent.state_dict(), path)
                    with open(path + ".action_interface.json", "w") as f:
                        json.dump({"network_arch": arch, "observation_normalization": OBS_NORMALIZATION_VERSION}, f)
                    meta = json.loads(Path(path + ".action_interface.json").read_text())
                    loaded = _agent(arch, observation_normalization_from_metadata(meta), seed=99)
                    loaded.load_state_dict(torch.load(path, map_location="cpu"))
                loaded.eval()
                agent.eval()
                torch.testing.assert_close(loaded.obs_normalizer.mean, agent.obs_normalizer.mean)
                with torch.no_grad():
                    a = agent.get_action_and_value(obs, selected_slot=slots, **legal)
                    b = loaded.get_action_and_value(obs, selected_slot=slots, **legal)
                    torch.testing.assert_close(agent.get_value(obs), loaded.get_value(obs))
                    torch.testing.assert_close(
                        agent.get_action_distribution(obs, **legal).logits,
                        loaded.get_action_distribution(obs, **legal).logits)
                for x, y in zip(a[2:], b[2:]):
                    torch.testing.assert_close(x, y)

    def test_wrapper_normalises_exactly_once(self):
        agent = _agent("spatial_resnet_local", OBS_NORMALIZATION_VERSION, seed=6)
        obs, legal, slots = _batch(seed=7)
        agent.obs_normalizer.update(obs * 5.0 + 2.0)
        with torch.no_grad():
            wrapped = agent.get_action_and_value(obs, selected_slot=slots, **legal)
            manual = agent.policy.get_action_and_value(agent.normalize(obs), selected_slot=slots, **legal)
        for x, y in zip(wrapped, manual):
            torch.testing.assert_close(x, y)

    def test_mismatched_checkpoints_are_rejected(self):
        plain = _agent("spatial_resnet")
        wrapped = _agent("spatial_resnet", OBS_NORMALIZATION_VERSION)
        with self.assertRaises(RuntimeError):
            plain.load_state_dict(wrapped.state_dict())
        with self.assertRaises(RuntimeError):
            wrapped.load_state_dict(plain.state_dict())

    def test_attribute_delegation(self):
        wrapped = _agent("spatial_resnet", OBS_NORMALIZATION_VERSION)
        self.assertEqual(wrapped.actor_mode, "legal_features")
        self.assertEqual(wrapped.max_legal_actions, MAX_LEGAL)
        with self.assertRaises(AttributeError):
            wrapped.critic  # submodule access would bypass normalisation
        # Buffers are not parameters, so the optimiser never touches them.
        self.assertFalse(any(p is wrapped.obs_normalizer.mean for p in wrapped.parameters()))


@unittest.skipIf(shutil.which("bash") is None, "bash unavailable")
class ExtraPpoArgsTests(unittest.TestCase):
    """run_experiment.sh with a stub sbatch; train_farmshare.sbatch with a stub python."""

    def _sandbox(self, tmp):
        root = Path(tmp) / "repo"
        (root / "cluster").mkdir(parents=True)
        for name in ("run_experiment.sh", "train_farmshare.sbatch"):
            shutil.copy(REPO_ROOT / "cluster" / name, root / "cluster" / name)
        bin_dir = Path(tmp) / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "sbatch"
        stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "EXTRA=${EXTRA_PPO_ARGS-UNSET}" >> "$SBATCH_LOG"\necho 123\n')
        stub.chmod(0o755)
        return root, bin_dir

    def _submit(self, extra):
        with tempfile.TemporaryDirectory() as tmp:
            root, bin_dir = self._sandbox(tmp)
            log = Path(tmp) / "sbatch.log"
            env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", SBATCH_LOG=str(log))
            env.pop("EXTRA_PPO_ARGS", None)
            if extra is not None:
                env["EXTRA_PPO_ARGS"] = extra
            proc = subprocess.run(["bash", str(root / "cluster/run_experiment.sh"), "exp1", "spatial_resnet", "1000"],
                                  env=env, capture_output=True, text=True, cwd=tmp)
            record = None
            if (root / "outputs/experiments/exp1.json").exists():
                record = json.loads((root / "outputs/experiments/exp1.json").read_text())
            return proc, record, log.read_text() if log.exists() else ""

    def test_run_experiment_records_and_forwards(self):
        proc, record, log = self._submit("--norm-obs --learning-rate 3e-4")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(record["extra_ppo_args"], "--norm-obs --learning-rate 3e-4")
        self.assertIn("EXTRA=--norm-obs --learning-rate 3e-4", log.splitlines()[0])

    def test_run_experiment_empty(self):
        proc, record, log = self._submit(None)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(record["extra_ppo_args"], "")
        self.assertEqual(log.splitlines()[0], "EXTRA=")

    def test_run_experiment_rejects_unsafe_or_reserved(self):
        for bad in ('--exp-name "x y"', "--lr $(id)", "--model-path foo", "--seed=3"):
            with self.subTest(bad=bad):
                proc, record, _ = self._submit(bad)
                self.assertEqual(proc.returncode, 2)
                self.assertIsNone(record)

    def _train_args(self, extra):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self._sandbox(tmp)
            (root / ".venv/bin").mkdir(parents=True)
            # `python -c ...` prints the device; the trainer call prints one argument per line.
            (root / ".venv/bin/activate").write_text(
                'python() { if [ "$1" = -c ]; then echo CPU; else printf "%s\\n" "$@" > "$ARGS_LOG"; fi; }\n')
            log = Path(tmp) / "args.log"
            env = dict(os.environ, ARGS_LOG=str(log), SLURM_JOB_ID="1")
            env.pop("EXTRA_PPO_ARGS", None)
            if extra is not None:
                env["EXTRA_PPO_ARGS"] = extra
            proc = subprocess.run(["bash", "cluster/train_farmshare.sbatch"], env=env, capture_output=True,
                                  text=True, cwd=root)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            return log.read_text().splitlines()

    def test_sbatch_appends_extra_args_last(self):
        base = self._train_args(None)
        self.assertEqual(base, self._train_args(""))
        extended = self._train_args("--norm-obs  --update-epochs 8")
        self.assertEqual(extended, base + ["--norm-obs", "--update-epochs", "8"])


if __name__ == "__main__":
    unittest.main()
