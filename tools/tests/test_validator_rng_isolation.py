"""The action-interface validator must not change the training RNG streams.

It samples from the global RNGs only when it actually runs (not on a cache hit), so without
isolation PPO minibatch shuffles depended on whether the validation cache existed.
"""

import random
import unittest

import numpy as np
import torch

from py_rl.cleanrl.cleanrl.ppo import _restore_global_rngs, _snapshot_global_rngs


def _draws():
    return (random.random(), float(np.random.rand()), float(torch.rand(1)))


class ValidatorRngIsolationTest(unittest.TestCase):
    def test_restore_returns_streams_to_snapshot(self):
        random.seed(3)
        np.random.seed(3)
        torch.manual_seed(3)
        expected = _draws()

        random.seed(3)
        np.random.seed(3)
        torch.manual_seed(3)
        state = _snapshot_global_rngs()
        # Simulate a validator run consuming randomness.
        np.random.choice(np.arange(100), size=50)
        random.random()
        torch.rand(10)
        _restore_global_rngs(state)
        self.assertEqual(_draws(), expected)


if __name__ == "__main__":
    unittest.main()
