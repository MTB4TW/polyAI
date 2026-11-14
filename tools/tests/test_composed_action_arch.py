import io
import unittest
from types import SimpleNamespace

import torch

from py_rl.cleanrl.cleanrl.action_types import action_type_table, build_global_action_catalog
from py_rl.cleanrl.cleanrl.agent_factory import make_agent

PHASE1_N = 63913
COMBAT_N = 78675
# Fingerprints recorded by the real environment in checkpoint sidecars.
PHASE1_FINGERPRINT = "c849a4abf7b0bee073ccc56b63ae65917ea30e77068ad648c472130693dfe6e4"
COMBAT_FINGERPRINT = "9162f7d914d3f433422474169d150c7915681a11db5c5734fba0476eaa80d9b3"


def adapter(n):
    return SimpleNamespace(single_action_space=SimpleNamespace(n=n))


def n_params(agent):
    return sum(p.numel() for p in agent.parameters())


class ComposedActionArchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.catalog = build_global_action_catalog(11, 11, enable_combat=False)

    def make(self, n=PHASE1_N, arch="spatial_resnet_composed", seed=0, **kwargs):
        torch.manual_seed(seed)
        return make_agent(adapter(n), network_arch=arch, max_legal_actions=kwargs.pop("max_legal_actions", 6), **kwargs)

    def batch(self, seed=1):
        g = torch.Generator().manual_seed(seed)
        cat = self.catalog
        t = lambda x, y: cat.tile_id(x, y)
        ids = torch.tensor([
            [cat.id_end_turn(), cat.id_move(t(2, 7), t(3, 7)), cat.id_research("ORGANIZATION"),
             cat.id_build("SAWMILL", t(5, 5)), 0, 0],
            [cat.id_train("WARRIOR", t(4, 4)), cat.id_move(t(4, 4), t(4, 5)), cat.id_capture(t(1, 1), t(1, 2), "VILLAGE"),
             cat.id_examine(t(9, 9)), cat.id_end_turn(), 0],
        ])
        valid = torch.tensor([[True] * 4 + [False] * 2, [True] * 5 + [False]])
        features = torch.rand(2, 6, 47, generator=g)
        features[..., 43:] = torch.randint(0, 11, (2, 6, 4), generator=g) / 10.0
        x = torch.randn(2, 6424, generator=g)
        return x, dict(legal_global_ids=ids, legal_action_valid_mask=valid, legal_action_features=features)

    def test_type_table_matches_environment_catalog(self):
        self.assertEqual(self.catalog.total_size, PHASE1_N)
        self.assertEqual(self.catalog.fingerprint(), PHASE1_FINGERPRINT)
        combat = build_global_action_catalog(11, 11, enable_combat=True)
        self.assertEqual(combat.total_size, COMBAT_N)
        self.assertEqual(combat.fingerprint(), COMBAT_FINGERPRINT)
        self.assertEqual(len(action_type_table(self.catalog)), 73)
        self.assertEqual(len(action_type_table(combat)), 75)

    def test_every_global_id_maps_to_its_catalog_block(self):
        agent = self.make()
        cat = self.catalog
        names = agent.action_type_names
        types = agent.action_types(torch.arange(PHASE1_N))
        self.assertEqual(int(types.min()), 0)
        self.assertEqual(int(types.max()), len(names) - 1)
        t = lambda x, y: cat.tile_id(x, y)
        expected = {
            cat.id_end_turn(): "END_TURN",
            cat.id_move(t(0, 0), t(0, 1)): "MOVE",
            cat.id_move(t(10, 10), t(10, 9)): "MOVE",
            cat.id_capture(t(3, 3), t(3, 4), "CITY"): "CAPTURE/CITY",
            cat.id_capture(t(3, 3), t(3, 4), "VILLAGE"): "CAPTURE/VILLAGE",
            cat.id_train("WARRIOR", t(10, 10)): "TRAIN/WARRIOR",
            cat.id_resource("FRUIT", t(6, 2)): "RESOURCE_GATHERING/FRUIT",
            cat.id_clear_forest(t(10, 10)): "CLEAR_FOREST",
            cat.id_build("LUMBER_HUT", t(0, 0)): "BUILD/LUMBER_HUT",
            cat.id_build("SAWMILL", t(10, 10)): "BUILD/SAWMILL",
            cat.id_research("ORGANIZATION"): "RESEARCH_TECH/ORGANIZATION",
            cat.id_levelup("WORKSHOP", t(5, 5)): "LEVEL_UP/WORKSHOP",
            cat.id_examine(t(10, 10)): "EXAMINE",
            PHASE1_N - 1: "EXAMINE",
        }
        for gid, name in expected.items():
            self.assertEqual(names[int(types[gid])], name, gid)
        # The same move on different tiles shares one type.
        self.assertEqual(len({int(types[cat.id_move(t(x, 0), t(x, 1))]) for x in range(11)}), 1)

    def test_forward_backward_shapes_and_masking(self):
        agent = self.make()
        x, kw = self.batch()
        gid, slot, logp, ent, value = agent.get_action_and_value(x, **kw)
        self.assertEqual(gid.shape, (2,))
        self.assertEqual(slot.shape, (2,))
        self.assertEqual(logp.shape, (2,))
        self.assertEqual(ent.shape, (2,))
        self.assertEqual(value.shape, (2, 1))
        self.assertTrue(torch.equal(gid, kw["legal_global_ids"].gather(1, slot[:, None]).squeeze(1)))
        dist = agent.get_action_distribution(x, **kw)
        self.assertEqual(dist.probs.shape, (2, 6))
        self.assertTrue(torch.equal(dist.probs[~kw["legal_action_valid_mask"]], torch.zeros(3)))
        torch.testing.assert_close(dist.probs.sum(-1), torch.ones(2))
        for _ in range(20):
            self.assertTrue(bool(kw["legal_action_valid_mask"].gather(1, dist.sample()[:, None]).all()))
        fixed = torch.tensor([1, 2])
        result = agent.get_action_and_value(x, selected_slot=fixed, **kw)
        torch.testing.assert_close(result[2], dist.log_prob(fixed))
        (result[2].sum() + result[4].sum() - result[3].sum()).backward()
        for name in ("action_type_embedding.weight", "spatial_backbone.stem.0.weight",
                     "action_feature_encoder.0.weight", "action_scorer.0.weight"):
            grad = dict(agent.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertGreater(float(grad.abs().sum()), 0.0, name)

    def test_tileless_actions_ignore_grid_at_origin(self):
        agent = self.make()
        x, kw = self.batch()
        h, _, grid = agent.encode_state(x, return_grid=True)
        grid2 = grid.clone()
        grid2[:, :, 0, 0] += 5.0  # Tile (0,0), where zero coordinates point.
        # Row 0 slots 0 and 2 are END_TURN / RESEARCH_TECH; give them zero coordinates as the env does.
        kw0 = dict(kw)
        kw0["legal_action_features"] = kw["legal_action_features"].clone()
        kw0["legal_action_features"][0, [0, 2], 43:] = 0.0
        base = agent._distribution(h, grid, **kw0).logits
        changed = agent._distribution(h, grid2, **kw0).logits
        torch.testing.assert_close(base[0, [0, 2]], changed[0, [0, 2]])

    def test_parameter_count_and_no_per_id_table(self):
        composed = self.make(max_legal_actions=256)
        resnet = self.make(arch="spatial_resnet", max_legal_actions=256)
        local = self.make(arch="spatial_resnet_local", max_legal_actions=256)
        self.assertIsNone(composed.action_embedding)
        self.assertNotIn("action_embedding.weight", composed.state_dict())
        self.assertEqual(n_params(resnet), 4_404_898)
        self.assertEqual(n_params(local), 4_413_090)
        self.assertEqual(n_params(composed), 327_330)
        # Independent of the catalog size: combat adds only two action types.
        combat = self.make(n=COMBAT_N, max_legal_actions=256)
        self.assertEqual(n_params(combat) - n_params(composed), 2 * 64)

    def test_checkpoint_round_trip_and_cross_arch_rejection(self):
        agent = self.make(seed=3)
        x, kw = self.batch()
        buf = io.BytesIO()
        torch.save(agent.state_dict(), buf)
        buf.seek(0)
        state = torch.load(buf, map_location="cpu")
        self.assertIn("action_type_starts", state)
        loaded = self.make(seed=99)
        loaded.load_state_dict(state)
        for key, value in agent.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)
        torch.testing.assert_close(agent.get_action_distribution(x, **kw).logits,
                                   loaded.get_action_distribution(x, **kw).logits)
        for arch in ("spatial_resnet", "spatial_resnet_local"):
            with self.assertRaises(RuntimeError):
                self.make(arch=arch).load_state_dict(state)
            with self.assertRaises(RuntimeError):
                self.make().load_state_dict(self.make(arch=arch).state_dict())

    def test_old_architectures_unchanged(self):
        # Parameter names, shapes and seeded initial values match the pre-composed layout.
        for arch, scorer_in in (("spatial_resnet", 160), ("spatial_resnet_local", 288)):
            agent = self.make(arch=arch, seed=7)
            state = agent.state_dict()
            self.assertEqual(state["action_embedding.weight"].shape, (PHASE1_N, 64))
            self.assertEqual(state["action_scorer.0.weight"].shape, (64, scorer_in))
            self.assertFalse(any(k.startswith("action_type") for k in state))
            again = self.make(arch=arch, seed=7).state_dict()
            for key in state:
                self.assertTrue(torch.equal(state[key], again[key]), key)

    def test_composed_requires_legal_features_and_known_catalog(self):
        with self.assertRaises(ValueError):
            self.make(actor_mode="legal_only")
        with self.assertRaises(ValueError):
            self.make(n=16)


if __name__ == "__main__":
    unittest.main()
