import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from torch.distributions import Categorical
from pol_env.Tribes.py.register_env import TribesGymWrapper
from py_rl.cleanrl.cleanrl.agent_factory import make_agent
from py_rl.cleanrl.cleanrl.spatial_agent import SpatialResNetAgent
from tools.lookahead_policy import TacticalLookaheadPolicy
from tools.evaluate_superhuman import human_reference


class TrainingImprovementsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_combat_results_and_unknown_terminal(self):
        for key, reward, outcome in [(0, 1, 'win'), (1, -1, 'loss')]:
            self.assertEqual(TribesGymWrapper._combat_result({'tribes': {'0': {'winner': key}}}, True),
                             (reward, outcome))
        self.assertEqual(TribesGymWrapper._combat_result({}, False), (0, 'ongoing'))
        with self.assertRaises(RuntimeError):
            TribesGymWrapper._combat_result({'tribes': {'0': {'winner': 2}}}, True)

    def test_spatial_single_encode_and_matching_log_prob(self):
        for arch in ('spatial_resnet', 'spatial_resnet_local'):
            adapter = SimpleNamespace(single_action_space=SimpleNamespace(n=16))
            agent = make_agent(adapter, network_arch=arch, max_legal_actions=3)
            x = torch.randn(2, 6424)
            ids = torch.tensor([[1, 2, 0], [3, 4, 0]])
            valid = torch.tensor([[True, True, False]] * 2)
            features = torch.rand(2, 3, 47)
            kwargs = dict(legal_global_ids=ids, legal_action_valid_mask=valid, legal_action_features=features)
            with patch.object(agent.spatial_backbone, 'forward', wraps=agent.spatial_backbone.forward) as forward:
                result = agent.get_action_and_value(x, selected_slot=torch.tensor([0, 1]), **kwargs)
                self.assertEqual(forward.call_count, 1)
            dist = agent.get_action_distribution(x, **kwargs)
            torch.testing.assert_close(result[2], dist.log_prob(torch.tensor([0, 1])))
            self.assertTrue(torch.equal(result[0], torch.tensor([1, 4])))
            self.assertTrue(torch.equal(dist.probs[:, 2], torch.zeros(2)))
            (result[2].sum() + result[4].sum()).backward()
            self.assertIsNotNone(agent.spatial_backbone.stem[0].weight.grad)

    def test_local_coordinates_are_x_major(self):
        grid = torch.arange(121.).reshape(1, 1, 11, 11)
        features = torch.zeros(1, 1, 47)
        features[0, 0, 43:] = torch.tensor([.2, .7, .9, .1])
        torch.testing.assert_close(SpatialResNetAgent.gather_local_features(grid, features),
                                   torch.tensor([[[29., 100.]]]))

    def test_unknown_architecture_and_unsupported_actor_fail(self):
        with self.assertRaises(ValueError):
            make_agent(None, network_arch='typo')
        with self.assertRaises(ValueError):
            SpatialResNetAgent(actor_mode='dense_debug')

    def test_reveal_heuristic_does_not_reward_all_moves(self):
        agent = SimpleNamespace(get_action_distribution=lambda *a, **k: Categorical(logits=torch.zeros(1, 2)))
        policy = TacticalLookaheadPolicy(agent, torch.device('cpu'))
        features = np.zeros((2, 47), dtype=np.float32)
        features[:, 0] = 1  # Both actions are moves.
        features[1, 1] = 1  # Only the second reveals fog.
        info = {'legal_global_ids_padded': np.array([7, 9]),
                'legal_action_valid_mask': np.array([True, True]),
                'legal_action_features_padded': features}
        self.assertEqual(policy.choose_action(np.zeros(6424), info), (9, 1))

    def test_human_reference_requires_matching_seed_and_map(self):
        attempt = {'participant_kind': 'human', 'status': 'completed', 'is_first_completed_attempt': True,
                   'map': {'csv_sha256': 'abc'}, 'configuration': {'episode_seed': 42},
                   'attempt_id': 'record', 'result': {'final_t10_spt': 27}}
        record = SimpleNamespace(csv_sha256='abc', filename='map.csv')
        self.assertEqual(human_reference([attempt], record, 42)['spt'], 27)
        self.assertIsNone(human_reference([attempt], record, 1))
        record.csv_sha256 = 'different'
        self.assertIsNone(human_reference([attempt], record, 42))


if __name__ == '__main__':
    unittest.main()
