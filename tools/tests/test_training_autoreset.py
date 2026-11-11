"""Regressions for terminal empty masks and Gymnasium SAME_STEP final_info."""
import unittest
import gymnasium as gym
import numpy as np
import torch
from py_rl.cleanrl.cleanrl.ppo import (
    make_training_vector_env, _normalize_final_infos, _extract_vector_field,
    _extract_vector_legal_tensors,
)
from py_rl.cleanrl.cleanrl.agent_factory import make_agent


class OneStepGame(gym.Env):
    def __init__(self, timeout=False):
        self.timeout = timeout
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(-1, 1, (6424,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(6424, dtype=np.float32), self.info(False)

    def info(self, ended):
        return {'legal_global_ids_padded': np.array([0, 1]),
                'legal_action_valid_mask': np.array([not ended, False]),
                'combat_outcome': ('timeout' if self.timeout else 'loss') if ended else 'ongoing',
                'spt': 17 if ended else 2}

    def step(self, action):
        assert action == 0
        info = self.info(True)
        info['episode'] = {'r': np.float32(-1), 'l': np.int64(1)}
        return np.ones(6424, dtype=np.float32), -1., not self.timeout, self.timeout, info


def terminal_game():
    return OneStepGame()


def timeout_game():
    return OneStepGame(timeout=True)


class AutoresetTests(unittest.TestCase):
    def test_repeated_terminal_masks_for_both_architectures(self):
        torch.set_num_threads(1)
        envs = make_training_vector_env([terminal_game, timeout_game])
        try:
            for arch in ('spatial_resnet', 'spatial_resnet_local'):
                agent = make_agent(envs, network_arch=arch, max_legal_actions=2)
                obs, infos = envs.reset(seed=1)
                for _ in range(4):
                    ids, valid = _extract_vector_legal_tensors(infos, 2, 2, 2, torch.device('cpu'))
                    self.assertTrue(valid.any(dim=1).all())
                    with torch.no_grad():
                        result = agent.get_action_and_value(torch.tensor(obs), legal_global_ids=ids,
                            legal_action_valid_mask=valid, legal_action_features=torch.zeros(2, 2, 47))
                    obs, reward, term, trunc, infos = envs.step(result[0].numpy())
                    infos = _normalize_final_infos(infos, 2)
                    np.testing.assert_array_equal(term, [True, False])
                    np.testing.assert_array_equal(trunc, [False, True])
                    np.testing.assert_array_equal(obs, np.zeros((2, 6424)))
                    self.assertEqual(_extract_vector_field(infos, 'combat_outcome', 2), ['loss', 'timeout'])
                    self.assertEqual(_extract_vector_field(infos, 'spt', 2), [17, 17])
                    self.assertEqual(infos['final_info'][0]['episode']['l'], 1)
                    self.assertFalse(infos['final_info'][0]['legal_action_valid_mask'].any())
        finally:
            envs.close()

    def test_nested_masks_and_missing_rows(self):
        infos = {'final_info': {'spt': np.array([17, 0]), '_spt': np.array([True, False]),
                 'episode': {'r': np.array([-1, 0]), '_r': np.array([True, False])},
                 '_episode': np.array([True, False])}, '_final_info': np.array([True, False]),
                 'spt': np.array([2, 8])}
        normalized = _normalize_final_infos(infos, 2)
        self.assertIsNone(normalized['final_info'][1])
        self.assertEqual(_extract_vector_field(normalized, 'spt', 2), [17, 8])
        self.assertEqual(normalized['final_info'][0]['episode']['r'], -1)
        np.testing.assert_array_equal(infos['spt'], [2, 8])


if __name__ == '__main__':
    unittest.main()
