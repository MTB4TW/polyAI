"""Kickstarting PPO from the Phase 1 lookahead teacher (CHANGELOG 056).

Pure-Python tests run by default. The live Java tests (teacher info in a real env, no Java search when the teacher is
off) are opt-in with POLYVISION_RUN_JAVA_TESTS=1, like tools/tests/test_compact_legal_info.py.
"""
import json
import os
import re
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tyro

from pol_env.Tribes.py import search_teacher
from pol_env.Tribes.py.search_teacher import (
    FEATURE_NAMES, TUNED_VISIBLE_WEIGHTS, decision_seed, parse_teacher_mode, select_lookahead_slot,
)
from py_rl.cleanrl.cleanrl.obs_normalization import OBS_NORMALIZATION_VERSION
from py_rl.cleanrl.cleanrl.ppo import Args, compact_legal_info, expand_compact_legal_rows
from py_rl.cleanrl.cleanrl.teacher_kickstart import (
    extract_teacher_slots, kickstart_cross_entropy, teacher_coef_at, validate_teacher_args,
)
from tools.tests import test_ppo_experiment_options as _opts
from tools.tests.test_ppo_experiment_options import _batch
from py_rl.cleanrl.cleanrl.agent_factory import make_agent

PHASE1_N_ACTIONS = 63913  # the composed architecture needs the real Phase 1 catalog size


def _agent(arch, observation_normalization=None, seed=0):
    torch.manual_seed(seed)
    adapter = _opts.SimpleNamespace(
        single_observation_space=_opts.gym.spaces.Box(low=-1e5, high=1e5, shape=(_opts.OBS_DIM,)),
        single_action_space=_opts.gym.spaces.Discrete(PHASE1_N_ACTIONS),
    )
    return make_agent(adapter, network_arch=arch, actor_mode="legal_features", max_legal_actions=_opts.MAX_LEGAL,
                      legal_action_feature_dim=_opts.FEAT_DIM, observation_normalization=observation_normalization)

REPO_ROOT = Path(__file__).resolve().parents[2]
TUNED_EVAL = REPO_ROOT / "outputs/evaluations/20261112_lookahead_visible_tuned_validation250_visible-search-diagnostic"


def _reference_choice(scores, ids, valid, is_end, raw_map, eps=1e-9):
    """LookaheadSearchPolicy.choose_action's selection, verbatim from PR #11 (before it moved to search_teacher)."""
    slots = np.flatnonzero(valid)
    end_slots = [int(s) for s in slots if is_end[s]]
    best_slot, best_score = None, -np.inf
    for s in slots:
        raw = raw_map.get(int(ids[s]))
        if raw is None or raw >= len(scores):
            continue
        v = float(scores[raw])
        if v > best_score + eps:
            best_slot, best_score = int(s), v
    if end_slots:
        end_raw = raw_map.get(int(ids[end_slots[0]]))
        end_score = float(scores[end_raw]) if end_raw is not None and end_raw < len(scores) else -np.inf
        if best_slot is None or best_score <= end_score + eps:
            best_slot = end_slots[0]
    if best_slot is None:
        best_slot = int(slots[0])
    return best_slot


class TeacherSelectionTests(unittest.TestCase):
    def test_matches_pr11_selection_on_random_cases(self):
        rng = np.random.default_rng(0)
        for case in range(3000):
            slots, n_valid = 12, int(rng.integers(1, 12))
            ids = np.zeros(slots, dtype=np.int64)
            ids[:n_valid] = rng.choice(500, n_valid, replace=False)
            valid = np.zeros(slots, dtype=bool)
            valid[:n_valid] = True
            is_end = np.zeros(slots, dtype=bool)
            if rng.random() < 0.8:
                is_end[int(rng.integers(0, n_valid))] = True
            n_raw = n_valid + int(rng.integers(0, 4))
            raws = rng.permutation(n_raw)[:n_valid]
            raw_map = {int(g): int(r) for g, r in zip(ids[:n_valid], raws) if rng.random() > 0.05}
            scores = np.round(rng.normal(size=n_raw), 1)  # rounding forces ties
            scores[rng.random(n_raw) < 0.1] = -np.inf
            scores = scores[: n_raw - int(rng.integers(0, 2))]  # occasionally a raw index past the end
            with self.subTest(case=case):
                self.assertEqual(select_lookahead_slot(list(scores), ids, valid, is_end, raw_map),
                                 _reference_choice(list(scores), ids, valid, is_end, raw_map))

    def test_end_turn_wins_ties(self):
        ids, valid = np.array([7, 3, 0]), np.array([True, True, False])
        is_end = np.array([False, True, False])
        self.assertEqual(select_lookahead_slot([5.0, 5.0], ids, valid, is_end, {7: 0, 3: 1}), 1)
        self.assertEqual(select_lookahead_slot([5.1, 5.0], ids, valid, is_end, {7: 0, 3: 1}), 0)


class TeacherConfigTests(unittest.TestCase):
    def test_tuned_weights_match_committed_evaluation(self):
        config = json.loads((TUNED_EVAL / "config.json").read_text())
        self.assertEqual(config["heuristic_weights"], TUNED_VISIBLE_WEIGHTS)
        self.assertEqual(config["feature_names"], list(FEATURE_NAMES))
        self.assertFalse(config["privileged_information"])

    def test_feature_order_matches_java(self):
        src = (REPO_ROOT / "pol_env/Tribes/src/players/heuristics/Phase1SptHeuristic.java").read_text()
        block = src[src.index("FEATURE_NAMES = {"):]
        block = block[: block.index("};")]
        self.assertEqual(tuple(re.findall(r'"([a-z_]+)"', block)), FEATURE_NAMES)

    def test_teacher_mode_is_fog_respecting(self):
        weights, full_information = search_teacher.TEACHER_MODES["lookahead_visible"]
        self.assertIs(weights, TUNED_VISIBLE_WEIGHTS)
        self.assertFalse(full_information)

    def test_parse_teacher_mode(self):
        for off in (None, "", "0", "none", "OFF"):
            self.assertIsNone(parse_teacher_mode(off))
        self.assertEqual(parse_teacher_mode(" Lookahead_Visible "), "lookahead_visible")
        with self.assertRaises(ValueError):
            parse_teacher_mode("lookahead_full")

    def test_decision_seed(self):
        self.assertEqual(decision_seed(5, 3), decision_seed(5, 3))
        seeds = {decision_seed(e, d) for e in range(20) for d in range(50)}
        self.assertEqual(len(seeds), 1000)
        self.assertTrue(all(0 <= s < 2 ** 62 for s in seeds))


class TrainerTeacherTests(unittest.TestCase):
    def test_flags(self):
        args = Args()
        self.assertIsNone(args.teacher)
        self.assertEqual((args.teacher_coef, args.teacher_decay_frac), (1.0, 0.5))
        self.assertTrue(args.teacher_disable_after_decay)
        args = tyro.cli(Args, args=["--teacher", "lookahead_visible", "--teacher-coef", "2.0",
                                    "--teacher-decay-frac", "0.25"])
        self.assertEqual((args.teacher, args.teacher_coef, args.teacher_decay_frac), ("lookahead_visible", 2.0, 0.25))

    def test_validate_args(self):
        self.assertIsNone(validate_teacher_args(None, 1.0, 0.5, "legal_features", False))
        self.assertEqual(validate_teacher_args("lookahead_visible", 1.0, 0.5, "legal_features", False),
                         "lookahead_visible")
        for bad in (dict(combat=True), dict(actor_mode="dense_debug"), dict(coef=-1.0), dict(decay=-0.1)):
            kw = dict(coef=1.0, decay=0.5, actor_mode="legal_features", combat=False, **{})
            kw.update(bad)
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                validate_teacher_args("lookahead_visible", kw["coef"], kw["decay"], kw["actor_mode"], kw["combat"])

    def test_coefficient_schedule(self):
        n = 100
        self.assertEqual(teacher_coef_at(1, n, 1.0, 0.5), 1.0)
        self.assertAlmostEqual(teacher_coef_at(26, n, 1.0, 0.5), 0.5)
        self.assertEqual(teacher_coef_at(51, n, 1.0, 0.5), 0.0)
        self.assertEqual(teacher_coef_at(100, n, 1.0, 0.5), 0.0)
        self.assertEqual(teacher_coef_at(100, n, 2.0, 0.0), 2.0)
        self.assertEqual(teacher_coef_at(1, n, 0.0, 0.5), 0.0)
        coefs = [teacher_coef_at(i, n, 1.0, 0.5) for i in range(1, n + 1)]
        self.assertTrue(all(a >= b for a, b in zip(coefs, coefs[1:])))

    def test_extract_teacher_slots(self):
        valid = torch.zeros((3, 4), dtype=torch.bool)
        valid[:, :2] = True
        infos = {"teacher_slot": np.array([1, 0, 0]), "_teacher_slot": np.array([True, True, False])}
        self.assertEqual(extract_teacher_slots(infos, 3, valid).tolist(), [1, 0, -1])
        self.assertEqual(extract_teacher_slots({}, 3, valid).tolist(), [-1, -1, -1])
        self.assertEqual(extract_teacher_slots({"teacher_slot": np.array([-1, 1, 0])}, 3, valid).tolist(), [-1, 1, 0])
        with self.assertRaises(RuntimeError):
            extract_teacher_slots({"teacher_slot": np.array([2, 0, 0])}, 3, valid)  # padded slot
        with self.assertRaises(RuntimeError):
            extract_teacher_slots({"teacher_slot": np.array([4, 0, 0])}, 3, valid)  # out of range

    def test_compact_ipc_keeps_teacher_slot(self):
        slots, n = 6, 3
        ids = np.zeros(slots, dtype=np.int32)
        ids[:n] = [40, 7, 12]
        valid = np.zeros(slots, dtype=np.bool_)
        valid[:n] = True
        feats = np.zeros((slots, 5), dtype=np.float32)
        feats[:n] = np.arange(15, dtype=np.float32).reshape(3, 5) + 1
        info = {"legal_global_ids_padded": ids, "legal_action_valid_mask": valid,
                "legal_action_features_padded": feats, "teacher_slot": 2}
        compact = compact_legal_info(info)
        self.assertNotIn("legal_global_ids_padded", compact)
        self.assertEqual(compact["teacher_slot"], 2)
        ids2, valid2, _ = expand_compact_legal_rows(compact["legal_compact"])
        self.assertTrue(valid2[2])
        self.assertEqual(int(ids2[compact["teacher_slot"]]), 12)

    def test_cross_entropy_matches_masked_softmax(self):
        for arch in ("mlp", "spatial_resnet_local", "spatial_resnet_composed"):
            for norm in (None, OBS_NORMALIZATION_VERSION):
                with self.subTest(arch=arch, norm=norm):
                    agent = _agent(arch, norm, seed=2)
                    obs, legal, slots = _batch(seed=3)
                    plain = agent.get_action_and_value(obs, selected_slot=slots, **legal)
                    out = agent.get_action_and_value(obs, selected_slot=slots, return_distribution=True, **legal)
                    self.assertEqual((len(plain), len(out)), (5, 6))
                    for x, y in zip(plain, out[:5]):
                        torch.testing.assert_close(x, y)
                    dist = out[5]
                    torch.testing.assert_close(dist.log_prob(slots), out[2])
                    teacher = slots.clone()
                    teacher[0] = -1
                    ce, n = kickstart_cross_entropy(dist, teacher)
                    self.assertEqual(n, len(slots) - 1)
                    logits = agent.get_action_distribution(obs, **legal).logits
                    expected = F.cross_entropy(logits[1:], teacher[1:])
                    torch.testing.assert_close(ce, expected)
                    agent.zero_grad()
                    ce.backward()
                    grads = [p.grad for p in agent.parameters() if p.grad is not None]
                    self.assertTrue(grads and any(float(g.abs().sum()) > 0 for g in grads))

    def test_cross_entropy_without_labels_is_zero(self):
        agent = _agent("mlp", seed=1)
        obs, legal, slots = _batch(seed=1)
        dist = agent.get_action_and_value(obs, selected_slot=slots, return_distribution=True, **legal)[5]
        ce, n = kickstart_cross_entropy(dist, torch.full_like(slots, -1))
        self.assertEqual((float(ce.detach()), n), (0.0, 0))


@unittest.skipIf(_opts.shutil.which("bash") is None, "bash unavailable")
class TeacherExtraArgsTests(unittest.TestCase):
    """EXTRA_PPO_ARGS carries the teacher flags through run_experiment.sh and train_farmshare.sbatch."""

    # Reuse the sandbox helpers without re-running that class's own tests here.
    _sandbox = _opts.ExtraPpoArgsTests._sandbox
    _submit = _opts.ExtraPpoArgsTests._submit
    _train_args = _opts.ExtraPpoArgsTests._train_args

    FLAGS = "--teacher lookahead_visible --teacher-coef 1.0 --teacher-decay-frac 0.5 --norm-obs"

    def test_teacher_flags_survive_filter(self):
        proc, record, log = self._submit(self.FLAGS)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(record["extra_ppo_args"], self.FLAGS)
        self.assertEqual(self._train_args(self.FLAGS)[-len(self.FLAGS.split()):], self.FLAGS.split())


@unittest.skipUnless(os.environ.get("POLYVISION_RUN_JAVA_TESTS") == "1", "requires local Java gateway")
class LiveTeacherEnvTests(unittest.TestCase):
    class _CountingJava:
        def __init__(self, inner):
            self.inner, self.search_calls = inner, 0

        def __getattr__(self, name):
            attr = getattr(self.inner, name)
            if name.startswith(("lookahead", "mcts", "heuristic")):
                self.search_calls += 1
            return attr

    def _play(self, teacher, steps=150):
        os.environ.setdefault("POLYVISION_LEVEL_POOL_GLOB", "levels/phase1_pool_bardur_real/train/*.csv")
        os.environ.setdefault("POLYVISION_SOLO_NO_OPPONENT_MODE", "1")
        if teacher:
            os.environ["POLYVISION_TEACHER"] = teacher
        else:
            os.environ.pop("POLYVISION_TEACHER", None)
        from pol_env.Tribes.py.register_env import TribesGymWrapper
        env = TribesGymWrapper()
        counter = self._CountingJava(env.tribes_env._env)
        env.tribes_env._env = counter
        infos = []
        try:
            obs, info = env.reset(seed=11)
            infos.append(info)
            rng = np.random.default_rng(0)
            for _ in range(steps):
                ids = np.asarray(info["legal_global_ids_padded"])[np.asarray(info["legal_action_valid_mask"], bool)]
                obs, _, term, trunc, info = env.step(int(rng.choice(ids)))
                infos.append((info, term or trunc))
                if term or trunc:
                    obs, info = env.reset(seed=12)
                    infos.append(info)
        finally:
            os.environ.pop("POLYVISION_TEACHER", None)
            env.tribes_env._env = counter.inner
            env.close()
        return infos, counter.search_calls

    def test_off_adds_nothing_and_never_searches(self):
        infos, calls = self._play(None)
        self.assertEqual(calls, 0)
        for item in infos:
            info = item[0] if isinstance(item, tuple) else item
            self.assertNotIn("teacher_slot", info)

    def test_on_reports_valid_slot_every_decision(self):
        infos, calls = self._play("lookahead_visible")
        decisions = 0
        for item in infos:
            info, done = item if isinstance(item, tuple) else (item, False)
            slot = info["teacher_slot"]
            if done:
                self.assertEqual(slot, -1)
                continue
            decisions += 1
            self.assertTrue(np.asarray(info["legal_action_valid_mask"], bool)[slot])
        self.assertEqual(calls, decisions)


if __name__ == "__main__":
    unittest.main()
