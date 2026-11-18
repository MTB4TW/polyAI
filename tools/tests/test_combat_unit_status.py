"""Combat unit-status inputs (POLYVISION_UNIT_STATUS=1): HP / veteran / has-acted planes and the attack outcome.

Pure contract tests always run. The live-engine tests need POLYVISION_RUN_JAVA_TESTS=1 (as the other combat tests).
"""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pol_env.Tribes.py import environment_contract as C

ROOT = Path(__file__).resolve().parents[2]


class ContractTests(unittest.TestCase):
    def test_layout_adds_planes_before_the_scalars(self):
        off = C.observation_layout(11, 11, "polytopia_2_15", False, False)
        on = C.observation_layout(11, 11, "polytopia_2_15", False, True)
        self.assertEqual(on.expected_obs_dim - off.expected_obs_dim, len(C.UNIT_STATUS_CHANNELS) * 121)
        self.assertEqual(on.unit_status_end - on.unit_status_start, 5 * 121)
        self.assertEqual(on.unit_status_end, on.legacy_scalar_start)
        self.assertEqual(on.spatial_channels, off.spatial_channels + 5)
        self.assertEqual(off.unit_status_start, off.unit_status_end)

    def test_version_suffix_and_combat_only(self):
        v = C.versioned_for_unit_status("combat_v5_x", True)
        self.assertTrue(C.unit_status_of_version(v))
        self.assertFalse(C.unit_status_of_version("combat_v5_x"))
        self.assertTrue(C.resolve_unit_status({"POLYVISION_UNIT_STATUS": "1", "POLYVISION_COMBAT_MODE": "1"}))
        self.assertFalse(C.resolve_unit_status({}))
        with self.assertRaises(ValueError):
            C.resolve_unit_status({"POLYVISION_UNIT_STATUS": "1"})  # Phase 1 stays unchanged


@unittest.skipUnless(os.environ.get("POLYVISION_RUN_JAVA_TESTS") == "1", "requires local Java gateway")
class UnitStatusEngineTests(unittest.TestCase):
    def make_env(self, unit_status: bool):
        from pol_env.Tribes.py.register_env import TribesGymWrapper

        settings = dict(POLYVISION_PHASE="2", POLYVISION_COMBAT_MODE="1", POLYVISION_SOLO_NO_OPPONENT_MODE="0",
                        POLYVISION_MAX_TURNS="20", POLYVISION_INFO_MODE="fast", POLYVISION_RULES="polytopia_2_15",
                        POLYVISION_LEVEL_POOL_GLOB="levels/phase2_pool_1v1/train/*.csv",
                        POLYVISION_UNIT_STATUS="1" if unit_status else "0", POLYVISION_MAX_LEGAL_ACTIONS="512")
        p = patch.dict(os.environ, settings)
        p.start()
        self.addCleanup(p.stop)
        env = TribesGymWrapper()
        self.addCleanup(env.close)
        return env

    def planes(self, env, obs):
        layout = C.observation_layout(11, 11, "polytopia_2_15", False, True)
        block = np.asarray(obs)[layout.unit_status_start:layout.unit_status_end].reshape(5, 11, 11)
        return dict(zip(C.UNIT_STATUS_CHANNELS, block))

    def test_planes_follow_the_engine(self):
        env = self.make_env(True)
        obs, info = env.reset(seed=3)
        self.assertEqual(np.asarray(obs).shape[0], C.observation_layout(11, 11, "polytopia_2_15", False, True).expected_obs_dim)
        self.assertTrue(C.unit_status_of_version(env.PHASE1_ENVIRONMENT_VERSION))
        self.assertEqual(env.ACTION_FEATURE_DIM, 47 + 4)
        view = env.tribes_env.get_observation()
        p = self.planes(env, obs)
        own = [u for u in view["unit"].values() if int(u["tribeId"]) == 0]
        self.assertTrue(own)
        for u in own:
            x, y = int(u["x"]), int(u["y"])
            self.assertAlmostEqual(p["unit_hp_fraction"][x, y], u["currentHP"] / u["maxHP"], places=5)
            self.assertAlmostEqual(p["unit_hp_absolute"][x, y], u["currentHP"] / C.UNIT_STATUS_HP_SCALE, places=5)
            self.assertEqual(p["unit_can_move"][x, y], 1.0)  # fresh at the start of the turn
        # Move one unit: it can no longer move this turn.
        acts, to_raw = env._current_legal_actions, env._current_legal_id_to_raw_index
        move = next(g for g, r in to_raw.items() if acts[r].get("type") == "MOVE")
        unit_id = acts[to_raw[move]]["unit_id"]
        obs, *_ = env.step(int(move))
        u = env.tribes_env.get_observation()["unit"][str(unit_id)]
        self.assertEqual(self.planes(env, obs)["unit_can_move"][int(u["x"]), int(u["y"])], 0.0)

    def test_attack_features_match_what_the_attack_does(self):
        env = self.make_env(True)
        rng = np.random.default_rng(0)
        base = 47
        checked = 0
        for seed in range(1, 40):
            obs, info = env.reset(seed=seed)
            done = False
            while not done and checked < 6:
                acts, to_raw = env._current_legal_actions, env._current_legal_id_to_raw_index
                ids = np.asarray(info["legal_global_ids_padded"]).reshape(-1)
                valid = np.asarray(info["legal_action_valid_mask"], dtype=bool).reshape(-1)
                feats = np.asarray(info["legal_action_features_padded"]).reshape(ids.size, -1)
                attacks = [k for k in np.flatnonzero(valid) if acts[to_raw[int(ids[k])]].get("type") == "ATTACK"]
                if attacks:
                    k = attacks[0]
                    a = acts[to_raw[int(ids[k])]]
                    before = env.tribes_env.get_observation(full_visibility=True)["unit"]
                    target, attacker = str(a["target_unit_id"]), str(a["unit_id"])
                    self.assertAlmostEqual(feats[k, base], a["attack_damage"] / 10.0, places=5)
                    obs, _r, term, trunc, info = env.step(int(ids[k]))
                    after = env.tribes_env.get_observation(full_visibility=True)["unit"]
                    killed = target not in after
                    self.assertEqual(bool(feats[k, base + 2]), killed)
                    if not killed:
                        self.assertEqual(before[target]["currentHP"] - after[target]["currentHP"], a["attack_damage"])
                        if attacker in after:
                            self.assertEqual(before[attacker]["currentHP"] - after[attacker]["currentHP"],
                                             a["attack_retaliation"])
                        self.assertEqual(bool(feats[k, base + 3]), attacker not in after)
                    checked += 1
                    done = term or trunc
                    continue
                legal = np.flatnonzero(valid)
                # Prefer moving towards contact: random non-end-turn actions, ending the turn now and then.
                k = int(rng.choice(legal))
                obs, _r, term, trunc, info = env.step(int(ids[k]))
                done = term or trunc
            if checked >= 6:
                break
        self.assertGreaterEqual(checked, 3, "random play should reach a few attacks")

    def test_warm_start_from_an_old_checkpoint_keeps_its_choices_and_old_snapshots_still_play(self):
        import torch

        from pol_env.Tribes.py.combat_selfplay import SnapshotOpponent
        from py_rl.cleanrl.cleanrl.agent_factory import make_agent
        from py_rl.cleanrl.cleanrl.init_from import apply_init_from

        old_env = self.make_env(False)
        old_obs, old_info = old_env.reset(seed=4)
        old_meta = C.environment_compatibility_metadata(old_env, actor_mode="legal_features", max_legal_actions=512)
        old_meta.update(network_arch="spatial_resnet_composed", actor_mode="legal_features", max_legal_actions=512)
        adapter = SimpleNamespace(single_observation_space=old_env.observation_space, single_action_space=old_env.action_space)
        with patch.dict(os.environ, {"POLYVISION_UNIT_STATUS": "0"}):
            old_agent = make_agent(adapter, network_arch="spatial_resnet_composed", actor_mode="legal_features",
                                   max_legal_actions=512, legal_action_feature_dim=47)
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "old.cleanrl_model")
        torch.save(old_agent.state_dict(), path)
        import json
        Path(path + ".action_interface.json").write_text(json.dumps(old_meta, default=str))

        new_env = self.make_env(True)
        new_obs, new_info = new_env.reset(seed=4)
        new_meta = C.environment_compatibility_metadata(new_env, actor_mode="legal_features", max_legal_actions=512)
        new_meta.update(network_arch="spatial_resnet_composed")
        adapter = SimpleNamespace(single_observation_space=new_env.observation_space, single_action_space=new_env.action_space)
        new_agent = make_agent(adapter, network_arch="spatial_resnet_composed", actor_mode="legal_features",
                               max_legal_actions=512, legal_action_feature_dim=51)
        report = apply_init_from(new_agent, path, target_meta=new_meta, network_arch="spatial_resnet_composed", reset="none")
        self.assertTrue(report["extended"] or report["remapped"])

        def logits(agent, obs, info, dim):
            ids = torch.as_tensor(np.asarray(info["legal_global_ids_padded"])).reshape(1, -1)
            valid = torch.as_tensor(np.asarray(info["legal_action_valid_mask"], dtype=bool)).reshape(1, -1)
            feats = torch.as_tensor(np.asarray(info["legal_action_features_padded"], dtype=np.float32)).reshape(1, ids.shape[1], dim)
            with torch.no_grad():
                return agent.get_action_distribution(torch.as_tensor(obs).reshape(1, -1).float(), legal_global_ids=ids,
                                                     legal_action_valid_mask=valid, legal_action_features=feats).logits
        # Same game, same seed: the warm-started network chooses exactly like the old one before any training.
        torch.testing.assert_close(logits(old_agent, old_obs, old_info, 47), logits(new_agent, new_obs, new_info, 51),
                                   rtol=1e-5, atol=1e-5)
        # An old snapshot still plays in the new environment (league fixed opponents).
        snap = SnapshotOpponent(path, new_env)
        ids = np.asarray(new_info["legal_global_ids_padded"]).reshape(-1)
        valid = np.asarray(new_info["legal_action_valid_mask"], dtype=bool).reshape(-1)
        feats = np.asarray(new_info["legal_action_features_padded"], dtype=np.float32).reshape(ids.size, -1)
        self.assertIn(snap.act(np.asarray(new_obs), ids, valid, feats), set(ids[valid].tolist()))


if __name__ == "__main__":
    unittest.main()
