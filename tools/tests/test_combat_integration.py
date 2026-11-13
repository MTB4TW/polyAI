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

    def _capture(self, tribe_id, x, y):
        state = self.env.tribes_env._env.getGameState()
        self.assertTrue(state.getBoard().capture(state, state.getTribe(tribe_id), int(x), int(y)))

    def _capital_pos(self, tribe_id):
        state = self.env.tribes_env._env.getGameState()
        from py4j.java_gateway import get_field
        pos = state.getActor(state.getTribe(tribe_id).getCapitalID()).getPosition()
        return int(get_field(pos, 'x')), int(get_field(pos, 'y'))

    def _give_enemy_second_city(self):
        state = self.env.tribes_env._env.getGameState()
        board, village = state.getBoard(), self.env.tribes_env._jvm.core.Types.TERRAIN.VILLAGE
        size = board.getSize()
        for x in range(size):
            for y in range(size):
                if board.getTerrainAt(x, y) == village:
                    self._capture(1, x, y)
                    self.assertEqual(state.getTribe(1).getNumCities(), 2)
                    return
        self.skipTest('map has no village to give the enemy a second city')

    def test_capturing_enemy_capital_wins_even_if_they_keep_another_city(self):
        self.assertEqual(self.env.tribes_env._last_obs['gameMode'], 0)  # Types.GAME_MODE.CAPITALS key
        self._give_enemy_second_city()
        self._capture(0, *self._capital_pos(1))
        _, reward, term, trunc, info = self.env.step(self.env._catalog.end_turn_id)
        self.assertTrue(term)
        self.assertFalse(trunc)
        self.assertEqual(info['combat_outcome'], 'win')
        self.assertEqual(info['combat_outcome_reward'], TribesGymWrapper.COMBAT_REWARD_DEFAULTS['win'])
        self.assertAlmostEqual(reward, info['combat_outcome_reward'] + info['combat_shaping_reward'])
        self.assertEqual(info['combat_end_enemy_capitals_held'], 1)
        self.assertEqual(info['combat_end_enemy_cities'], 1)
        self.assertFalse(any(info['legal_action_valid_mask']))

    def test_legacy_scalars_are_scaled_in_combat(self):
        # Raw score (~500-3000) saturated the critic; combat divides the six legacy scalars.
        vec = self.env._dict_to_array(self.env.tribes_env._last_obs)
        legacy = vec[6292:6298]
        tribe0 = self.env.tribes_env._last_obs['tribes']['0']
        self.assertAlmostEqual(float(legacy[1]), float(tribe0['score']) / 1000.0, places=5)
        self.assertLess(float(abs(vec).max()), 10.0)

    def test_potential_counts_enemy_cities_hidden_by_fog(self):
        obs_enemy = len(self.env.tribes_env._last_obs['tribes']['1']['citiesID'])
        totals = self.env._combat_engine_totals()['tribes']
        self.assertEqual(totals['1']['numCities'], 1)
        self.assertLessEqual(obs_enemy, 1)
        before = TribesGymWrapper._combat_potential({'tribes': totals}, self.env._combat_reward_weights)
        self._give_enemy_second_city()
        after = TribesGymWrapper._combat_potential(self.env._combat_engine_totals(), self.env._combat_reward_weights)
        self.assertLess(after, before)

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

    def test_losing_own_capital_ends_episode_as_loss(self):
        self._capture(1, *self._capital_pos(0))
        _, reward, term, trunc, info = self.env.step(self.env._catalog.end_turn_id)
        self.assertTrue(term)
        self.assertFalse(trunc)
        self.assertEqual(info['combat_outcome_reward'], TribesGymWrapper.COMBAT_REWARD_DEFAULTS['loss'])
        self.assertAlmostEqual(reward, info['combat_outcome_reward'] + info['combat_shaping_reward'])
        self.assertEqual(info['combat_outcome'], 'loss')
