"""GAE helper: identical to the original inline PPO loop by default, correct with per-turn discounts."""
import unittest

import numpy as np

from py_rl.cleanrl.cleanrl.gae import END_TURN_GLOBAL_ID, compute_gae, per_turn_discounts


def original_loop(rewards, values, dones, next_value, next_done, gamma, lam):
    """The inline loop ppo.py used before compute_gae existed."""
    advantages = np.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
    return advantages, advantages + values


class GaeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.T, self.N = 64, 5
        self.rewards = rng.normal(size=(self.T, self.N))
        self.values = rng.normal(size=(self.T, self.N))
        self.dones = (rng.random((self.T, self.N)) < 0.05).astype(float)
        self.next_value = rng.normal(size=(1, self.N))
        self.next_done = (rng.random(self.N) < 0.5).astype(float)

    def test_default_matches_original_loop(self):
        args = (self.rewards, self.values, self.dones, self.next_value, self.next_done, 0.99, 0.95)
        adv, ret = compute_gae(*args)
        ref_adv, ref_ret = original_loop(*args)
        np.testing.assert_allclose(adv, ref_adv)
        np.testing.assert_allclose(ret, ref_ret)

    def test_constant_step_discounts_equal_scalar_gamma(self):
        args = (self.rewards, self.values, self.dones, self.next_value, self.next_done, 0.99, 0.95)
        adv, _ = compute_gae(*args)
        adv_vec, _ = compute_gae(*args, step_discounts=np.full_like(self.rewards, 0.99))
        np.testing.assert_allclose(adv, adv_vec)

    def test_per_turn_discount_counts_turns_not_actions(self):
        # One env, 6 steps, three actions per turn (END_TURN at steps 2 and 5); reward 1 only at the end.
        actions = np.array([[7], [7], [END_TURN_GLOBAL_ID], [7], [7], [END_TURN_GLOBAL_ID]])
        rewards = np.zeros((6, 1))
        rewards[5] = 1.0
        zeros = np.zeros((6, 1))
        gamma = 0.9
        discounts = per_turn_discounts(actions, gamma)
        np.testing.assert_allclose(discounts[:, 0], [1, 1, gamma, 1, 1, gamma])
        # Episode ends after the last step; zero values and lambda=1 make advantages pure discounted returns.
        adv, _ = compute_gae(rewards, zeros, zeros, np.zeros((1, 1)), np.ones(1), gamma, 1.0, discounts)
        np.testing.assert_allclose(adv[:, 0], [gamma, gamma, gamma, 1, 1, 1])

    def test_per_turn_discounts_are_floats(self):
        out = per_turn_discounts(np.array([0, 3, 0]), 0.5)
        np.testing.assert_allclose(out, [0.5, 1.0, 0.5])


if __name__ == "__main__":
    unittest.main()
