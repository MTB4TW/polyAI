"""Tests for py_rl/cleanrl/cleanrl/resume.py: resume bundles, argument checks and stop requests (no Java)."""

import json
import os
import random
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from py_rl.cleanrl.cleanrl import resume as R
from py_rl.cleanrl.cleanrl.return_normalization import ReturnNormalizer


def args(**kw):
    base = dict(env_id="Tribes-v0", network_arch="spatial_resnet", actor_mode="legal_features", num_envs=2,
                num_steps=64, max_legal_actions=512, legal_action_feature_dim=47, norm_obs=True, norm_returns=True,
                privileged_critic=False, privileged_critic_value_norm=True, gamma=0.99, gamma_per_turn=True,
                learning_rate=5e-4, total_timesteps=640, anneal_lr=True, teacher=None, teacher_coef=1.0,
                teacher_decay_frac=0.3, num_minibatches=4, update_epochs=8, ent_coef=0.02, vf_coef=0.5,
                clip_coef=0.2, target_kl=0.03)
    base.update(kw)
    return SimpleNamespace(**base)


class ResumeBundleTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.model = torch.nn.Linear(4, 2)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss = self.model(torch.randn(8, 4)).pow(2).mean()
        loss.backward()
        self.opt.step()
        self.rn = ReturnNormalizer(2, 0.99)
        self.rn(np.array([1.0, -2.0]), np.array([False, True]))
        self.ckpt = os.path.join(self.dir, "model_checkpoint_256.cleanrl_model")
        torch.save(self.model.state_dict(), self.ckpt)

    def test_round_trip_restores_optimizer_schedule_returns_and_rng(self):
        random.seed(1); np.random.seed(2); torch.manual_seed(3)
        R.save_resume_bundle(self.ckpt, global_step=256, iteration=2, optimizer=self.opt, args=args(),
                             return_normalizer=self.rn, extras={"privileged_critic_reinits": 1}, reason="SIGUSR1")
        expect = (random.random(), np.random.rand(), torch.rand(1).item())
        state = R.load_resume_bundle(R.find_resume_checkpoint(self.dir))
        self.assertEqual((state["global_step"], state["iteration"], state["reason"]), (256, 2, "SIGUSR1"))
        self.assertEqual(state["extras"]["privileged_critic_reinits"], 1)
        opt2 = torch.optim.Adam(torch.nn.Linear(4, 2).parameters(), lr=1e-3)
        opt2.load_state_dict(state["optimizer"])
        a, b = self.opt.state_dict()["state"][0], opt2.state_dict()["state"][0]
        self.assertTrue(torch.equal(a["exp_avg"], b["exp_avg"]) and torch.equal(a["exp_avg_sq"], b["exp_avg_sq"]))
        rn2 = ReturnNormalizer(2, 0.99)
        R.restore_return_normalizer(rn2, state["return_normalizer"])
        self.assertEqual(rn2.scale, self.rn.scale)
        np.testing.assert_array_equal(rn2.returns, self.rn.returns)
        random.seed(9); np.random.seed(9); torch.manual_seed(9)
        R.restore_rng(state["rng"])
        self.assertEqual((random.random(), np.random.rand(), torch.rand(1).item()), expect)
        latest = json.load(open(os.path.join(self.dir, R.LATEST_FILE)))
        self.assertEqual(latest["global_step"], 256)

    def test_find_prefers_latest_pointer_then_highest_step(self):
        R.save_resume_bundle(self.ckpt, global_step=256, iteration=2, optimizer=self.opt, args=args())
        other = os.path.join(self.dir, "model_checkpoint_512.cleanrl_model")
        torch.save(self.model.state_dict(), other)
        R.save_resume_bundle(other, global_step=512, iteration=4, optimizer=self.opt, args=args())
        self.assertEqual(R.find_resume_checkpoint(self.dir), os.path.abspath(other))
        os.remove(os.path.join(self.dir, R.LATEST_FILE))
        self.assertTrue(R.find_resume_checkpoint(self.dir).endswith("model_checkpoint_512.cleanrl_model"))
        bare = os.path.join(self.dir, "bare.cleanrl_model")
        torch.save(self.model.state_dict(), bare)
        with self.assertRaises(FileNotFoundError):
            R.find_resume_checkpoint(bare)  # no bundle: use --init-from for a warm start

    def test_structural_changes_are_refused_schedule_changes_warned(self):
        saved = R._plain_args(args())
        self.assertEqual(R.check_resume_args(saved, args()), [])
        warnings = R.check_resume_args(saved, args(learning_rate=2e-4, total_timesteps=10_000))
        self.assertEqual(len(warnings), 2)
        with self.assertRaises(SystemExit):
            R.check_resume_args(saved, args(network_arch="mlp"))
        with self.assertRaises(SystemExit):
            R.check_resume_args(saved, args(num_envs=8))

    def test_stop_file_and_graceful_signal(self):
        stop = R.StopRequests(self.dir)
        try:
            self.assertIsNone(stop.requested())
            open(stop.stop_file(), "w").close()
            self.assertEqual(stop.requested(), "STOP file")
            os.remove(stop.stop_file())
            if hasattr(R.signal, "SIGUSR1"):
                os.kill(os.getpid(), R.signal.SIGUSR1)
                self.assertEqual(stop.requested(), "SIGUSR1")
        finally:
            stop.restore()


if __name__ == "__main__":
    unittest.main()
