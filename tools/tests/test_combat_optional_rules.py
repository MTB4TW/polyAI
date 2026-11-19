"""Optional wrapper rules (POLYVISION_COMBAT_RULES_OFF, CHANGELOG 090).

Contract tests always run; the engine tests need POLYVISION_RUN_JAVA_TESTS=1.
"""

import os
import unittest
from unittest.mock import patch

import numpy as np

from pol_env.Tribes.py import environment_contract as C

V = "combat_v5_scaled_scalars_horizon_20+unit_status_v1+rules_polytopia_2_15_r2"


class ContractTests(unittest.TestCase):
    def test_names_all_and_none(self):
        combat = {"POLYVISION_COMBAT_MODE": "1"}
        self.assertEqual(C.resolve_combat_rules_off(combat), frozenset())
        self.assertEqual(C.resolve_combat_rules_off({**combat, "POLYVISION_COMBAT_RULES_OFF": "none"}), frozenset())
        self.assertEqual(C.resolve_combat_rules_off({**combat, "POLYVISION_COMBAT_RULES_OFF": "all"}),
                         frozenset(C.COMBAT_OPTIONAL_RULES))
        self.assertEqual(C.resolve_combat_rules_off({**combat, "POLYVISION_COMBAT_RULES_OFF": "fishing_mask, unit2_backtrack"}),
                         {"fishing_mask", "unit2_backtrack"})
        with self.assertRaises(ValueError):
            C.resolve_combat_rules_off({**combat, "POLYVISION_COMBAT_RULES_OFF": "no_such_rule"})
        with self.assertRaises(ValueError):  # Phase 1 stays unchanged
            C.resolve_combat_rules_off({"POLYVISION_COMBAT_RULES_OFF": "all"})

    def test_version_round_trip_keeps_the_game_rules_suffix_last(self):
        from py_rl.cleanrl.cleanrl.init_from import rules_of_version

        off = {"village_forced_moves", "fishing_mask"}
        w = C.with_rules_off(V, off)
        self.assertTrue(w.endswith("+rules_polytopia_2_15_r2"))
        self.assertEqual(C.rules_off_of_version(w), off)
        self.assertEqual(rules_of_version(w), "polytopia_2_15")
        self.assertEqual(C.with_rules_off(w, ()), V)
        self.assertEqual(C.rules_off_of_version(V), frozenset())
        # tribes_2020 combat has no game-rules suffix: the rules-off part must not be read as one.
        self.assertEqual(rules_of_version(C.versioned_for_rules_off("combat_v5_x", {"fishing_mask"})), "tribes_2020")


class ExtraActionContractTests(unittest.TestCase):
    def test_names_and_versions(self):
        combat = {"POLYVISION_COMBAT_MODE": "1"}
        self.assertEqual(C.resolve_combat_extra_actions({**combat, "POLYVISION_COMBAT_EXTRA_ACTIONS": "make_veteran,BUILD_ROAD"}),
                         ("BUILD_ROAD", "MAKE_VETERAN"))  # catalog order, whatever order they are given in
        self.assertEqual(C.resolve_combat_extra_actions({**combat, "POLYVISION_COMBAT_EXTRA_ACTIONS": "all"}),
                         tuple(C.COMBAT_EXTRA_ACTION_FAMILIES))
        with self.assertRaises(ValueError):
            C.resolve_combat_extra_actions({"POLYVISION_COMBAT_EXTRA_ACTIONS": "BUILD_ROAD"})  # Phase 1 unchanged
        v = C.with_rules_off(C.with_extra_actions(V, ("BUILD_ROAD",)), {"fishing_mask"})
        self.assertEqual(v, "combat_v5_scaled_scalars_horizon_20+unit_status_v1+extra_actions_build_road"
                            "+wrapper_rules_off_fishing_mask+rules_polytopia_2_15_r2")
        self.assertEqual(C.extra_actions_of_version(v), ("BUILD_ROAD",))
        self.assertEqual(C.with_extra_actions(v, ()), C.with_rules_off(V, {"fishing_mask"}))

    def test_extra_blocks_come_after_every_existing_id(self):
        from py_rl.cleanrl.cleanrl.action_types import action_type_table, build_global_action_catalog

        old = build_global_action_catalog(11, 11, True, "polytopia_2_15")
        new = build_global_action_catalog(11, 11, True, "polytopia_2_15", ("BUILD_ROAD", "CONVERT", "SEND_STARS"))
        self.assertEqual({k: v for k, v in new.offsets.items() if k in old.offsets}, old.offsets)
        self.assertEqual(new.offsets["BUILD_ROAD"], old.total_size)
        self.assertEqual(new.total_size, old.total_size + 121 + 121 * 121 + C.DIPLOMACY_MAX_TRIBES * C.SEND_STARS_MAX)
        self.assertEqual(new.id_extra("SEND_STARS", target_tribe=1, stars=3),
                         new.offsets["SEND_STARS"] + C.SEND_STARS_MAX + 2)
        old_names = [n for n, _, _ in action_type_table(old)]
        new_names = [n for n, _, _ in action_type_table(new)]
        self.assertEqual(new_names[: len(old_names)], old_names)
        self.assertIn("BUILD_ROAD", new_names)
        self.assertIn("SEND_STARS/T1", new_names)


@unittest.skipUnless(os.environ.get("POLYVISION_RUN_JAVA_TESTS") == "1", "requires local Java gateway")
class EngineTests(unittest.TestCase):
    def make_env(self, rules_off="", extra=""):
        from pol_env.Tribes.py.register_env import TribesGymWrapper

        settings = dict(POLYVISION_PHASE="2", POLYVISION_COMBAT_MODE="1", POLYVISION_SOLO_NO_OPPONENT_MODE="0",
                        POLYVISION_MAX_TURNS="20", POLYVISION_INFO_MODE="fast", POLYVISION_RULES="polytopia_2_15",
                        POLYVISION_LEVEL_POOL_GLOB="levels/phase2_pool_1v1/train/*.csv", POLYVISION_MAX_LEGAL_ACTIONS="512",
                        POLYVISION_COMBAT_RULES_OFF=rules_off, POLYVISION_UNIT_STATUS="0",
                        POLYVISION_COMBAT_EXTRA_ACTIONS=extra)
        p = patch.dict(os.environ, settings)
        p.start()
        self.addCleanup(p.stop)
        env = TribesGymWrapper()
        self.addCleanup(env.close)
        return env

    def test_rules_off_leaves_every_engine_action_of_an_allowed_type(self):
        env = self.make_env("all")
        self.assertEqual(env.RULES_OFF, frozenset(C.COMBAT_OPTIONAL_RULES))
        self.assertIn("+wrapper_rules_off_", env.PHASE1_ENVIRONMENT_VERSION)
        rng = np.random.default_rng(0)
        for seed in range(3):
            obs, info = env.reset(seed=seed)
            done = False
            while not done:
                acts, o = env._current_legal_actions, env.tribes_env._last_obs
                allowed = env._filter_allowed_raw_indices(acts, o)
                def kept(a):
                    kind = str(a.get("type", "")).upper()
                    return kind in env.ALLOWED_ACTION_TYPES and (kind != "MOVE" or env._is_move_destination_within_board(a, o))
                self.assertEqual(sorted(allowed), [i for i, a in enumerate(acts) if kept(a)])
                self.assertEqual(env._queued_village_capture_unit_ids, set())
                ids = np.asarray(info["legal_global_ids_padded"]).reshape(-1)
                valid = np.asarray(info["legal_action_valid_mask"], dtype=bool).reshape(-1)
                obs, _r, term, trunc, info = env.step(int(rng.choice(ids[valid])))
                done = term or trunc

    def test_village_rules_restrict_only_with_them_on(self):
        on, off = self.make_env(""), None
        rng = np.random.default_rng(1)
        restricted = 0
        for seed in range(4):
            obs, info = on.reset(seed=seed)
            done = False
            while not done:
                acts, o = on._current_legal_actions, on.tribes_env._last_obs
                with patch.object(on, "RULES_OFF", frozenset(C.COMBAT_OPTIONAL_RULES)):
                    free = set(on._filter_allowed_raw_indices(acts, o))
                real = set(on._filter_allowed_raw_indices(acts, o))
                self.assertLessEqual(real, free)
                restricted += int(real != free)
                ids = np.asarray(info["legal_global_ids_padded"]).reshape(-1)
                valid = np.asarray(info["legal_action_valid_mask"], dtype=bool).reshape(-1)
                obs, _r, term, trunc, info = on.step(int(rng.choice(ids[valid])))
                done = term or trunc
        self.assertGreater(restricted, 0, "random play should meet the village rules at least once")

    def test_an_old_snapshot_plays_with_the_rules_off(self):
        from types import SimpleNamespace

        import torch

        from pol_env.Tribes.py.combat_selfplay import SnapshotOpponent
        from py_rl.cleanrl.cleanrl.agent_factory import make_agent

        old_env = self.make_env("")
        old_env.reset(seed=2)
        meta = C.environment_compatibility_metadata(old_env, actor_mode="legal_features", max_legal_actions=512)
        meta.update(network_arch="spatial_resnet_composed")
        agent = make_agent(SimpleNamespace(single_observation_space=old_env.observation_space,
                                           single_action_space=old_env.action_space),
                           network_arch="spatial_resnet_composed", actor_mode="legal_features", max_legal_actions=512,
                           legal_action_feature_dim=47)
        import json
        import tempfile
        path = os.path.join(tempfile.mkdtemp(), "old.cleanrl_model")
        torch.save(agent.state_dict(), path)
        with open(path + ".action_interface.json", "w") as fh:
            json.dump(meta, fh, default=str)
        new_env = self.make_env("all")
        obs, info = new_env.reset(seed=2)
        snap = SnapshotOpponent(path, new_env)
        ids = np.asarray(info["legal_global_ids_padded"]).reshape(-1)
        valid = np.asarray(info["legal_action_valid_mask"], dtype=bool).reshape(-1)
        feats = np.asarray(info["legal_action_features_padded"], dtype=np.float32).reshape(ids.size, -1)
        self.assertIn(snap.act(np.asarray(obs), ids, valid, feats), set(ids[valid].tolist()))

    def test_extra_actions_are_offered_and_do_what_they_say(self):
        env = self.make_env("", "BUILD_ROAD,DISBAND")
        self.assertEqual(env._catalog.extra_actions, ("BUILD_ROAD", "DISBAND"))
        rng = np.random.default_rng(3)
        seen = set()
        for seed in range(8):
            obs, info = env.reset(seed=seed)
            done = False
            while not done and seen != {"BUILD_ROAD", "DISBAND"}:
                acts, to_raw = env._current_legal_actions, env._current_legal_id_to_raw_index
                extra = [(g, acts[r]) for g, r in to_raw.items() if acts[r].get("type") in ("BUILD_ROAD", "DISBAND")]
                for g, a in extra:
                    lo = env._catalog.offsets[a["type"]]
                    self.assertTrue(lo <= g < lo + 121)
                    self.assertEqual(g - lo, env._catalog.tile_id(a["target_x"], a["target_y"]))
                fresh = [(g, a) for g, a in extra if a["type"] not in seen]
                if fresh:
                    g, a = fresh[0]
                    seen.add(a["type"])
                    before = env.tribes_env.get_observation()
                    obs, _r, term, trunc, info = env.step(int(g))
                    after = env.tribes_env.get_observation()
                    if a["type"] == "DISBAND":
                        self.assertNotIn(str(a["unit_id"]), after["unit"])
                    else:
                        self.assertEqual(after["tribes"]["0"]["star"], before["tribes"]["0"]["star"] - a["star_cost"])
                    done = term or trunc
                    continue
                ids = np.asarray(info["legal_global_ids_padded"]).reshape(-1)
                valid = np.asarray(info["legal_action_valid_mask"], dtype=bool).reshape(-1)
                obs, _r, term, trunc, info = env.step(int(rng.choice(ids[valid])))
                done = term or trunc
            if seen == {"BUILD_ROAD", "DISBAND"}:
                break
        self.assertEqual(seen, {"BUILD_ROAD", "DISBAND"}, "random play should reach a road and a disband")

    def test_an_old_snapshot_plays_where_extra_actions_are_offered(self):
        from types import SimpleNamespace

        import json
        import tempfile

        import torch

        from pol_env.Tribes.py.combat_selfplay import SnapshotOpponent
        from py_rl.cleanrl.cleanrl.agent_factory import make_agent

        old_env = self.make_env("")
        old_env.reset(seed=2)
        meta = C.environment_compatibility_metadata(old_env, actor_mode="legal_features", max_legal_actions=512)
        meta.update(network_arch="spatial_resnet_composed")
        agent = make_agent(SimpleNamespace(single_observation_space=old_env.observation_space,
                                           single_action_space=old_env.action_space),
                           network_arch="spatial_resnet_composed", actor_mode="legal_features", max_legal_actions=512,
                           legal_action_feature_dim=47)
        path = os.path.join(tempfile.mkdtemp(), "old.cleanrl_model")
        torch.save(agent.state_dict(), path)
        with open(path + ".action_interface.json", "w") as fh:
            json.dump(meta, fh, default=str)
        new_env = self.make_env("", "all")
        obs, info = new_env.reset(seed=2)
        snap = SnapshotOpponent(path, new_env)
        self.assertIsNotNone(snap.id_map)
        rng = np.random.default_rng(0)
        for _ in range(60):
            ids = np.asarray(info["legal_global_ids_padded"]).reshape(-1)
            valid = np.asarray(info["legal_action_valid_mask"], dtype=bool).reshape(-1)
            feats = np.asarray(info["legal_action_features_padded"], dtype=np.float32).reshape(ids.size, -1)
            g = snap.act(np.asarray(obs), ids, valid, feats)
            self.assertIn(g, set(ids[valid].tolist()))
            self.assertLess(g, new_env._catalog.offsets["BUILD_ROAD"])  # never a family it has not seen
            obs, _r, term, trunc, info = new_env.step(int(rng.choice(ids[valid])))
            if term or trunc:
                obs, info = new_env.reset(seed=3)


if __name__ == "__main__":
    unittest.main()
