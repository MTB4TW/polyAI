"""Unit tests for the Phase 2 combat reward: outcome weights and potential-based shaping."""
import unittest

from pol_env.Tribes.py.register_env import TribesGymWrapper

W = TribesGymWrapper.COMBAT_REWARD_DEFAULTS


def tribe(cities=1, score=0, kills=0):
    return {"citiesID": list(range(cities)), "score": score, "nKills": kills}


class CombatRewardWeightsTests(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(TribesGymWrapper._load_combat_reward_weights({}), W)

    def test_environment_overrides(self):
        weights = TribesGymWrapper._load_combat_reward_weights(
            {"POLYVISION_COMBAT_REWARD_CITY": "2.5", "POLYVISION_COMBAT_REWARD_WIN": "20"})
        self.assertEqual(weights["city"], 2.5)
        self.assertEqual(weights["win"], 20.0)
        self.assertEqual(weights["kill"], W["kill"])

    def test_rejects_bad_values(self):
        for env in ({"POLYVISION_COMBAT_REWARD_KILL": "lots"},
                    {"POLYVISION_COMBAT_REWARD_SCORE_PER_1000": "nan"},
                    {"POLYVISION_COMBAT_REWARD_WIN": "-1"},
                    {"POLYVISION_COMBAT_REWARD_LOSS": "5"},
                    {"POLYVISION_COMBAT_REWARD_TIMEOUT": "-10"},
                    {"POLYVISION_COMBAT_REWARD_TIMEOUT": "10"},
                    {"POLYVISION_COMBAT_REWARD_TERMINAL_POTENTIAL": "1.5"}):
            with self.assertRaises(ValueError, msg=str(env)):
                TribesGymWrapper._load_combat_reward_weights(env)


class CombatPotentialTests(unittest.TestCase):
    def potential(self, me, *others):
        tribes = {"0": me, **{str(i + 1): t for i, t in enumerate(others)}}
        return TribesGymWrapper._combat_potential({"tribes": tribes}, W)

    def test_even_position_is_zero(self):
        self.assertEqual(self.potential(tribe(2, 1500, 1), tribe(2, 1500, 1)), 0.0)

    def test_each_term(self):
        self.assertAlmostEqual(self.potential(tribe(cities=3), tribe(cities=1)), 2 * W["city"])
        self.assertAlmostEqual(self.potential(tribe(score=2500), tribe(score=500)), 2 * W["score_per_1000"])
        self.assertAlmostEqual(self.potential(tribe(kills=4), tribe(kills=0)), 4 * W["kill"])

    def test_symmetric(self):
        a, b = tribe(3, 2000, 2), tribe(1, 800, 0)
        self.assertAlmostEqual(self.potential(a, b), -self.potential(b, a))

    def test_sums_over_several_opponents(self):
        self.assertAlmostEqual(self.potential(tribe(cities=4), tribe(cities=1), tribe(cities=1)), 2 * W["city"])

    def test_missing_data_is_zero(self):
        self.assertEqual(TribesGymWrapper._combat_potential({}, W), 0.0)
        self.assertEqual(TribesGymWrapper._combat_potential({"tribes": None}, W), 0.0)
        self.assertEqual(TribesGymWrapper._combat_potential({"tribes": {"0": None, "1": {}}}, W), 0.0)

    def test_shaping_telescopes_to_final_advantage(self):
        states = [(tribe(1), tribe(1)), (tribe(2), tribe(1)), (tribe(2, 900), tribe(1)), (tribe(1, 900), tribe(2))]
        potentials = [self.potential(*s) for s in states]
        shaping = [b - a for a, b in zip(potentials, potentials[1:])]
        self.assertAlmostEqual(sum(shaping), potentials[-1] - potentials[0])


if __name__ == "__main__":
    unittest.main()
