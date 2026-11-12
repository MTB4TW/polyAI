"""Opt-in live JVM checks: POLYVISION_RUN_JAVA_TESTS=1 python -m unittest ..."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from pol_env.Tribes.py.register_env import TribesGymWrapper

ROOT = Path(__file__).resolve().parents[2]

@unittest.skipUnless(os.environ.get('POLYVISION_RUN_JAVA_TESTS') == '1', 'requires local Java gateway')
class CombatIntegrationTests(unittest.TestCase):
    def setUp(self):
        jdks = sorted((ROOT / '.runtime').glob('jdk-*/Contents/Home'))
        settings = dict(POLYVISION_PHASE='2', POLYVISION_COMBAT_MODE='1',
                        POLYVISION_SOLO_NO_OPPONENT_MODE='0', POLYVISION_MAX_TURNS='2',
                        POLYVISION_INFO_MODE='fast',
                        POLYVISION_LEVEL_POOL_GLOB='levels/phase2_pool_1v1/train/*.csv')
        if jdks:
            settings['JAVA_HOME'] = str(jdks[-1])
            settings['PATH'] = str(jdks[-1] / 'bin') + os.pathsep + os.environ['PATH']
        self.settings = patch.dict(os.environ, settings)
        self.settings.start()
        self.addCleanup(self.settings.stop)
        self.env = TribesGymWrapper()
        self.addCleanup(self.env.close)
        self.env.reset(seed=1)
        self.env.tribes_env._env.setOpponentAgent('donothing', 1)

    def test_clock_income_and_timeout(self):
        _, reward, term, trunc, info = self.env.step(self.env._catalog.end_turn_id)
        self.assertFalse(term or trunc)
        self.assertEqual(info['combat_outcome_reward'], 0)
        self.assertAlmostEqual(reward, info['combat_shaping_reward'])
        self.assertEqual(info['tick'], 1)
        self.assertGreater(info['stars'], 5)
        _, reward, term, trunc, info = self.env.step(self.env._catalog.end_turn_id)
        self.assertFalse(term)
        self.assertTrue(trunc)
        self.assertEqual(info['combat_outcome'], 'timeout')
        self.assertEqual(info['turn_count'], 2)
        self.assertEqual(info['combat_outcome_reward'], 0)
        self.assertAlmostEqual(reward, info['combat_shaping_reward'])

    def test_win_ends_episode_before_horizon(self):
        state = self.env.tribes_env._env.getGameState()
        result = self.env.tribes_env._jvm.core.Types.RESULT
        state.getTribe(1).setWinner(result.LOSS)
        _, reward, term, trunc, info = self.env.step(self.env._catalog.end_turn_id)
        self.assertTrue(term)
        self.assertFalse(trunc)
        self.assertEqual(info['combat_outcome_reward'], TribesGymWrapper.COMBAT_REWARD_DEFAULTS['win'])
        self.assertAlmostEqual(reward, info['combat_outcome_reward'] + info['combat_shaping_reward'])
        self.assertEqual(info['combat_outcome'], 'win')
        self.assertFalse(any(info['legal_action_valid_mask']))

    def test_enemy_ownership_and_fog_encoding(self):
        import copy
        obs = copy.deepcopy(self.env.tribes_env._last_obs)
        x, y = 0, 0
        obs['board']['terrain'][x][y] = 0
        obs['board']['cityID'][x][y] = 999
        obs['city']['999'] = {'tribeID': 1, 'x': x, 'y': y}
        obs['unit']['999'] = {'tribeId': 1, 'type': 0, 'x': x, 'y': y, 'cityID': 999}
        planes = self.env._dict_to_array(obs)[:6292].reshape(52, 11, 11)
        self.assertEqual(planes[1, x, y], -1)
        self.assertEqual(planes[13, x, y], -1)
        obs['board']['terrain'][x][y] = 7
        planes = self.env._dict_to_array(obs)[:6292].reshape(52, 11, 11)
        self.assertEqual(planes[1, x, y], 0)
        self.assertEqual(planes[13, x, y], 0)

    def test_loss_ends_episode(self):
        state = self.env.tribes_env._env.getGameState()
        result = self.env.tribes_env._jvm.core.Types.RESULT
        state.getTribe(0).setWinner(result.LOSS)
        _, reward, term, trunc, info = self.env.step(self.env._catalog.end_turn_id)
        self.assertTrue(term)
        self.assertFalse(trunc)
        self.assertEqual(info['combat_outcome_reward'], TribesGymWrapper.COMBAT_REWARD_DEFAULTS['loss'])
        self.assertAlmostEqual(reward, info['combat_outcome_reward'] + info['combat_shaping_reward'])
        self.assertEqual(info['combat_outcome'], 'loss')
