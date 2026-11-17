"""Saved-game engine (py_rl/cleanrl/cleanrl/game_store): recording, selection, storage, learners. No JVM."""
import tempfile
import unittest
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from py_rl.cleanrl.cleanrl.agent_factory import make_agent
from py_rl.cleanrl.cleanrl.game_store import Codec, EpisodeRecorder, GameRecord, GameStore, discounted_returns
from py_rl.cleanrl.cleanrl.game_store import make_learner, make_selector
from py_rl.cleanrl.cleanrl.game_store.engine import GameReplayEngine, GameStoreConfig
from py_rl.cleanrl.cleanrl.game_store.learners import batch_to_torch, vtrace_targets
from py_rl.cleanrl.cleanrl.game_store.selectors import AnyOf

OBS_DIM = 6424
N_ACTIONS = 500
MAX_LEGAL = 8
FEAT_DIM = 47


def rollout(num_steps=6, num_envs=2, seed=0, dones_at=(), next_done=None):
    """Synthetic rollout buffers with a non-prefix valid mask (slot 0 sometimes padded)."""
    rng = np.random.default_rng(seed)
    obs = rng.random((num_steps, num_envs, OBS_DIM), dtype=np.float32)
    ids = rng.integers(1, N_ACTIONS, (num_steps, num_envs, MAX_LEGAL))
    valid = rng.random((num_steps, num_envs, MAX_LEGAL)) < 0.6
    valid[..., 3] = True
    slots = np.zeros((num_steps, num_envs), dtype=np.int64)
    for t in range(num_steps):
        for e in range(num_envs):
            slots[t, e] = rng.choice(np.nonzero(valid[t, e])[0])
    actions = np.take_along_axis(ids, slots[..., None], axis=-1)[..., 0]
    feats = rng.random((num_steps, num_envs, MAX_LEGAL, FEAT_DIM), dtype=np.float32)
    dones = np.zeros((num_steps, num_envs), dtype=np.float32)
    for t, e in dones_at:
        dones[t, e] = 1.0
    nd = np.zeros(num_envs, dtype=np.float32) if next_done is None else np.asarray(next_done, dtype=np.float32)
    return dict(obs=obs, legal_ids=ids, legal_valid=valid, selected_slots=slots, actions=actions,
                logprobs=rng.normal(size=(num_steps, num_envs)).astype(np.float32),
                values=rng.normal(size=(num_steps, num_envs)).astype(np.float32),
                rewards=rng.normal(size=(num_steps, num_envs)).astype(np.float32),
                dones=dones, next_done=nd, legal_features=feats)


def final_info(num_envs, env, **fields):
    infos = [None] * num_envs
    infos[env] = dict(fields)
    mask = np.zeros(num_envs, dtype=bool)
    mask[env] = True
    return {"final_info": infos, "_final_info": mask}


def mlp_agent(seed=0):
    torch.manual_seed(seed)
    adapter = SimpleNamespace(single_observation_space=gym.spaces.Box(low=-1e5, high=1e5, shape=(OBS_DIM,)),
                              single_action_space=gym.spaces.Discrete(N_ACTIONS))
    return make_agent(adapter, network_arch="mlp", actor_mode="legal_features", max_legal_actions=MAX_LEGAL,
                      legal_action_feature_dim=FEAT_DIM)


class CodecTests(unittest.TestCase):
    def test_round_trip_is_lossless(self):
        rng = np.random.default_rng(0)
        base = (rng.random((1, 300)) > 0.8).astype(np.float16)
        rows = np.repeat(base, 50, axis=0)
        rows[np.arange(50), rng.integers(0, 300, 50)] = 3.5
        for name in Codec.NAMES:
            for arr in (rows, rows.astype(np.float32), rng.integers(0, 9, (7, 5)).astype(np.int32), rows[:1]):
                packed = Codec(name).encode(arr)
                out = Codec.decode(packed)
                self.assertEqual(out.dtype, arr.dtype)
                np.testing.assert_array_equal(out, arr)

    def test_xor_beats_plain_zlib_on_slowly_changing_rows(self):
        # Rows wider than zlib's 32 KB window, so plain zlib cannot match one row against the previous one.
        rng = np.random.default_rng(1)
        rows = np.repeat(rng.random((1, 20000)).astype(np.float16), 40, axis=0)
        rows[np.arange(40), rng.integers(0, 20000, 40)] = 1.0
        xor = Codec.nbytes(Codec("xor_zlib").encode(rows))
        plain = Codec.nbytes(Codec("zlib").encode(rows))
        self.assertLess(xor, plain / 5)


class RecorderTests(unittest.TestCase):
    def test_games_are_cut_at_dones_and_across_iterations(self):
        kept = []
        rec = EpisodeRecorder(2, kept.append, obs_dtype=np.float32, feature_dtype=np.float32)
        # Iteration 1: env 0's first game ends after step 2 (dones[3]); env 1 runs on.
        r1 = rollout(seed=1, dones_at=[(3, 0)])
        rec.note_infos(2, final_info(2, 0, combat_outcome="win", map_id="m0", episode_seed=11))
        self.assertEqual(rec.ingest(**r1, global_step=12), 1)
        # Iteration 2: env 1's game (begun in iteration 1) ends after step 1; env 0 ends on the last step.
        r2 = rollout(seed=2, dones_at=[(2, 1)], next_done=[1, 0])
        rec.note_infos(1, final_info(2, 1, combat_outcome="loss", map_id="m1"))
        rec.note_infos(5, final_info(2, 0, combat_outcome="timeout", map_id="m2"))
        self.assertEqual(rec.ingest(**r2, global_step=24), 2)

        by_map = {r.meta["map_id"]: r for r in kept}
        self.assertEqual(by_map["m0"].length, 3)
        self.assertEqual(by_map["m0"].outcome, "win")
        self.assertEqual(by_map["m0"].meta["episode_seed"], 11)
        self.assertEqual(by_map["m1"].length, 6 + 2)
        self.assertEqual(by_map["m2"].length, 3 + 6)
        np.testing.assert_array_equal(by_map["m1"].actions, np.concatenate([r1["actions"][:, 1], r2["actions"][:2, 1]]))

        # Ragged legal rows and the remapped slot reproduce the chosen action and its features.
        g = by_map["m1"].tensors()
        for t in range(by_map["m1"].length):
            src, step = (r1, t) if t < 6 else (r2, t - 6)
            valid = src["legal_valid"][step, 1]
            rows = g["legal_ids"][g["legal_offsets"][t]:g["legal_offsets"][t + 1]]
            np.testing.assert_array_equal(rows, src["legal_ids"][step, 1][valid])
            self.assertEqual(rows[g["slot"][t]], src["actions"][step, 1])
            feats = g["legal_features"][g["legal_offsets"][t]:g["legal_offsets"][t + 1]]
            np.testing.assert_array_equal(feats, src["legal_features"][step, 1][valid])
            np.testing.assert_array_equal(g["obs"][t], src["obs"][step, 1])

    def test_torch_inputs_match_numpy_inputs(self):
        out_np, out_t = [], []
        r = rollout(seed=3, dones_at=[(4, 0), (2, 1)])
        EpisodeRecorder(2, out_np.append, obs_dtype=np.float32).ingest(**r)
        EpisodeRecorder(2, out_t.append, obs_dtype=np.float32).ingest(**{k: torch.as_tensor(v) for k, v in r.items()})
        self.assertEqual(len(out_np), 2)
        for a, b in zip(out_np, out_t):
            ta, tb = a.tensors(), b.tensors()
            for key in ("obs", "legal_ids", "legal_offsets", "slot", "legal_features", "reward"):
                np.testing.assert_array_equal(ta[key], tb[key])

    def test_selector_discards_before_compression_and_actions_payload(self):
        kept = []
        rec = EpisodeRecorder(2, kept.append, selector=make_selector("win"), payload="actions")
        rec.note_infos(1, final_info(2, 0, combat_outcome="loss"))
        rec.note_infos(1, final_info(2, 1, combat_outcome="win"))
        rec.ingest(**rollout(seed=4, dones_at=[(2, 0), (2, 1)]))
        self.assertEqual(rec.games_seen, 2)
        self.assertEqual([r.outcome for r in kept], ["win"])
        self.assertFalse(kept[0].has_payload)
        self.assertLess(kept[0].nbytes(), 400)


def make_record(outcome="win", length=10, priority=1.0, seed=0, reward_scale=1.0):
    kept = []
    rec = EpisodeRecorder(1, kept.append, obs_dtype=np.float16)
    r = rollout(num_steps=length, num_envs=1, seed=seed, next_done=[1])
    rec.note_infos(length - 1, final_info(1, 0, combat_outcome=outcome))
    rec.ingest(**r, reward_scale=reward_scale)
    kept[0].priority = priority
    return kept[0]


class SelectorTests(unittest.TestCase):
    def test_specs(self):
        game = {"reward": np.array([0.0, 0.0, 1.0], dtype=np.float32), "value": np.zeros(3, dtype=np.float32)}
        self.assertIsNone(make_selector("win").score({"outcome": "loss"}, game))
        self.assertEqual(make_selector("outcome:keep=win+timeout").score({"outcome": "timeout"}, game), 1.0)
        sel = make_selector("surprise:min_error=0.5|win")
        self.assertIsInstance(sel, AnyOf)
        self.assertIsNone(sel.score({"outcome": "loss"}, dict(game, reward=np.zeros(3, dtype=np.float32))))
        self.assertGreater(sel.score({"outcome": "loss"}, dict(game, reward=np.array([2.0, 2.0, 2.0], np.float32))), 0.5)
        self.assertIsNone(make_selector("win&short_win:max_length=5").score({"outcome": "win", "length": 9}, game))
        with self.assertRaises(ValueError):
            make_selector("nope")

    def test_top_return_keeps_the_top_fraction(self):
        sel = make_selector("top_return:fraction=0.2,window=1000,warmup=0")
        for v in np.random.default_rng(0).random(500):
            sel.score({"train_return": float(v)}, {})
        self.assertIsNone(sel.score({"train_return": 0.5}, {}))
        self.assertGreater(sel.score({"train_return": 0.95}, {}), 0.8)
        self.assertIsNone(sel.score({"other": 1.0}, {}))


class StoreTests(unittest.TestCase):
    def test_budget_demotes_lowest_priority_then_max_games_drops(self):
        one = make_record().nbytes()
        store = GameStore(budget_mb=2.5 * one / 2 ** 20, max_games=4)
        for i, p in enumerate([5.0, 1.0, 3.0]):
            store.add(make_record(priority=p, seed=i))
        self.assertLessEqual(store.payload_bytes, store.budget_bytes)
        demoted = [r.priority for r in store.records() if not r.has_payload]
        self.assertEqual(demoted, [1.0])
        for i in range(3):
            store.add(make_record(priority=10.0 + i, seed=10 + i))
        self.assertEqual(len(store), 4)
        self.assertNotIn(1.0, [r.priority for r in store.records()])

    def test_half_life_prefers_recent_games(self):
        store = GameStore(half_life=0.5, max_games=2, budget_mb=100)
        store.add(make_record(priority=3.0, seed=0))
        store.add(make_record(priority=1.0, seed=1))
        store.add(make_record(priority=1.0, seed=2))
        # Ages 3, 2, 1 games: effective priorities 3/64, 1/16, 1/4, so the oldest (highest raw) game goes.
        self.assertEqual(sorted(r.priority for r in store.records()), [1.0, 1.0])

    def test_sample_steps_shapes_and_rescaled_returns(self):
        store = GameStore(budget_mb=100, alpha=0.0, seed=1)
        rec = make_record(length=12, reward_scale=2.0)
        store.add(rec)
        batch = store.sample_steps(32, games_per_batch=1, gamma=0.9, reward_scale_now=4.0)
        self.assertEqual(batch["obs"].shape, (32, OBS_DIM))
        self.assertEqual(batch["legal_ids"].shape[0], 32)
        expected = discounted_returns(rec.tensors()["reward"], 0.9) * 0.5
        np.testing.assert_allclose(batch["return"], expected[batch["step"]], rtol=1e-6)
        for i in range(32):
            self.assertTrue(batch["legal_valid"][i, batch["slot"][i]])
            self.assertEqual(batch["legal_ids"][i, batch["slot"][i]], rec.actions[batch["step"][i]])

    def test_save_and_load(self):
        store = GameStore(budget_mb=100)
        store.add(make_record(outcome="win", seed=1))
        store.add(make_record(outcome="loss", seed=2).strip_payload())
        with tempfile.TemporaryDirectory() as d:
            store.save(d, shard_games=1)
            again = GameStore(budget_mb=100)
            self.assertEqual(again.load(d), 2)
        self.assertEqual(sorted(r.outcome for r in again.records()), ["loss", "win"])
        a = next(r for r in store.records() if r.has_payload).tensors()
        b = next(r for r in again.records() if r.has_payload).tensors()
        np.testing.assert_array_equal(a["obs"], b["obs"])


class LearnerTests(unittest.TestCase):
    def setUp(self):
        self.store = GameStore(budget_mb=100, seed=0)
        for i in range(3):
            self.store.add(make_record(seed=i, length=8))
        self.ctx = {"device": "cpu", "gamma": 0.99, "per_turn": False, "rng": np.random.default_rng(0)}

    def test_every_learner_gives_a_finite_differentiable_loss(self):
        for spec in ("sil", "awr:beta=0.5", "bc", "value", "replay_ppo", "vtrace:max_steps=5"):
            agent = mlp_agent()
            learner = make_learner(spec)
            if learner.kind == "games":
                batch = self.store.sample_games(2)
            else:
                batch = batch_to_torch(self.store.sample_steps(16, 3), "cpu")
            loss, metrics = learner.loss(agent, batch, self.ctx)
            self.assertTrue(torch.isfinite(loss), spec)
            loss.backward()
            grads = [p.grad for p in agent.parameters() if p.grad is not None]
            self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads), spec)
            self.assertTrue(all(np.isfinite(v) for v in metrics.values()), spec)

    def test_sil_ignores_steps_worse_than_expected(self):
        agent = mlp_agent()
        batch = batch_to_torch(self.store.sample_steps(16, 3), "cpu")
        batch["return"] = torch.full_like(batch["return"], -1e6)
        loss, metrics = make_learner("sil").loss(agent, batch, self.ctx)
        self.assertEqual(metrics["active_frac"], 0.0)
        self.assertEqual(float(loss), 0.0)

    def test_vtrace_on_policy_equals_monte_carlo(self):
        rewards = torch.tensor([0.0, 1.0, 0.0, 2.0])
        values = torch.tensor([0.3, -0.2, 0.5, 0.1])
        disc = torch.full((4,), 0.9)
        vs, _ = vtrace_targets(rewards, values, disc, torch.ones(4), lam=1.0)
        mc = torch.as_tensor(discounted_returns(rewards.numpy(), 0.9))
        torch.testing.assert_close(vs, mc)
        # Truncated importance weights pull the target back towards V.
        vs_off, _ = vtrace_targets(rewards, values, disc, torch.full((4,), 0.1), lam=1.0)
        self.assertLess(float((vs_off - values).abs().sum()), float((vs - values).abs().sum()))


class EngineTests(unittest.TestCase):
    def test_ingest_then_update_changes_the_agent(self):
        cfg = GameStoreConfig(selector="all", learner="sil", min_games=1, updates=2, batch_size=8, games_per_batch=2)
        engine = GameReplayEngine(cfg, num_envs=2, gamma=0.99, per_turn=False, device="cpu")
        agent = mlp_agent()
        r = rollout(seed=5, dones_at=[(3, 0), (4, 1)])
        r["rewards"][:] = 5.0  # returns far above the untrained critic, so SIL is active
        engine.ingest(**r)
        self.assertEqual(len(engine.store), 2)
        before = [p.detach().clone() for p in agent.parameters()]
        opt = torch.optim.Adam(agent.parameters(), lr=1e-3)
        metrics = engine.update(agent, opt, global_step=100)
        self.assertIn("policy_loss", metrics)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, agent.parameters())))
        self.assertIn("store_games", engine.log_metrics())

    def test_not_ready_before_min_games(self):
        cfg = GameStoreConfig(min_games=5)
        engine = GameReplayEngine(cfg, num_envs=2, gamma=0.99, per_turn=False, device="cpu")
        self.assertEqual(engine.update(mlp_agent(), torch.optim.Adam(mlp_agent().parameters()), 0), {})


class TrainerFlagTests(unittest.TestCase):
    def test_off_by_default_and_parses(self):
        import tyro
        from py_rl.cleanrl.cleanrl.ppo import Args

        self.assertFalse(Args().game_store)
        args = tyro.cli(Args, args=["--game-store", "--game-store-selector", "surprise|win",
                                    "--game-store-learner", "vtrace:lam=0.9", "--game-store-half-life", "500"])
        self.assertTrue(args.game_store)
        self.assertEqual(args.game_store_half_life, 500.0)
        cfg = GameStoreConfig(selector=args.game_store_selector, learner=args.game_store_learner)
        engine = GameReplayEngine(cfg, num_envs=1, gamma=0.99, per_turn=False, device="cpu")
        self.assertEqual(engine.learner.lam, 0.9)


if __name__ == "__main__":
    unittest.main()
