"""Reward scaling by a running std of discounted returns (opt-in `--norm-returns`).

This is the reward path of openai-baselines `VecNormalize` / gymnasium `NormalizeReward`, as used by
CleanRL's continuous-action PPO: keep a per-env running discounted return, track its variance, and
divide each reward by that std (no mean subtraction), then clip. Value targets are therefore in the
scaled units. It is training-only: checkpoints and evaluation see unscaled observations and the
policy is unaffected at inference time.

With `--gamma-per-turn` the discount between step t-1 and step t is the per-step discount of step
t-1 (gamma after END_TURN, 1.0 otherwise), matching `gae.per_turn_discounts`.

Torch-free so it can be unit-tested with numpy.
"""

import numpy as np


class RunningMeanStd:
    """Running mean/variance with the parallel (Chan et al.) merge, float64."""

    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.update_from_moments(x.mean(axis=0), x.var(axis=0), x.shape[0])

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        batch_mean = np.asarray(batch_mean, dtype=np.float64)
        batch_var = np.asarray(batch_var, dtype=np.float64)
        batch_count = float(batch_count)
        if batch_count <= 0:
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta ** 2 * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total


class ReturnNormalizer:
    """Per-env discounted-return accumulator plus a scalar RunningMeanStd of those returns."""

    def __init__(self, num_envs, gamma, epsilon=1e-8, clip=10.0):
        self.num_envs = int(num_envs)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.clip = float(clip)
        self.returns = np.zeros(self.num_envs, dtype=np.float64)
        # Discount to apply to the accumulated return before adding the next reward.
        self._carry_discount = np.full(self.num_envs, self.gamma, dtype=np.float64)
        self.rms = RunningMeanStd(shape=())

    @property
    def scale(self):
        return float(np.sqrt(self.rms.var + self.epsilon))

    def __call__(self, rewards, dones, step_discounts=None):
        """Scale one step of rewards (shape [num_envs]).

        dones[i]: the episode in env i ended with this step (the accumulator restarts afterwards).
        step_discounts[i]: discount of this step's transition (None = gamma for every env).
        """
        rewards = np.asarray(rewards, dtype=np.float64).reshape(self.num_envs)
        dones = np.asarray(dones, dtype=bool).reshape(self.num_envs)
        self.returns = self.returns * self._carry_discount + rewards
        self.rms.update(self.returns)
        scaled = np.clip(rewards / self.scale, -self.clip, self.clip)
        if step_discounts is None:
            self._carry_discount = np.full(self.num_envs, self.gamma, dtype=np.float64)
        else:
            self._carry_discount = np.asarray(step_discounts, dtype=np.float64).reshape(self.num_envs).copy()
        self.returns[dones] = 0.0
        return scaled
