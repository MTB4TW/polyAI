"""Tests for the combat evaluator's pure parts (tools/combat_eval_core.py, tools/compare_combat_evals.py). No Java."""

import json
import tempfile
import unittest
from pathlib import Path

from tools import combat_eval_core as C
from tools.compare_combat_evals import compare

REPO = Path(__file__).resolve().parents[2]


def episode(policy, map_name, outcome, turn=20, rep=0, our="BARDUR", opp="IMPERIUS"):
    return {"policy": policy, "map_filename": map_name, "outcome": outcome, "end_turn": turn, "replicate_index": rep,
            "our_tribe": our, "opponent_tribe": opp}


class PoolTests(unittest.TestCase):
    def test_eval_pools_match_the_manifest(self):
        for pool in C.EVAL_POOLS:
            _, maps, _ = C.load_verified_pool(REPO, pool)
            self.assertEqual(len(maps), 120)
            self.assertEqual({m.our_tribe for m in maps}, {"BARDUR"})
            counts = {t: sum(m.opponent_tribe == t for m in maps) for t in ("IMPERIUS", "XIN_XI", "OUMAJI")}
            self.assertEqual(counts, {"IMPERIUS": 40, "XIN_XI": 40, "OUMAJI": 40})

    def test_manifest_is_reproducible(self):
        committed = json.loads((REPO / C.MANIFEST_REL).read_text(encoding="utf-8"))
        self.assertEqual(C.build_manifest(REPO), committed)

    def test_eval_maps_are_disjoint_from_training(self):
        train = {C.sha256_file(p) for p in (REPO / C.POOL_ROOT_REL / "train").glob("*.csv")}
        for pool in C.EVAL_POOLS:
            _, maps, _ = C.load_verified_pool(REPO, pool)
            self.assertFalse(train & {m.csv_sha256 for m in maps})

    def test_map_tribes_follow_engine_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "map_2p_000001.csv"
            p.write_text(".:,c:3,.:\n.:,.:,.:\nc:2,.:,.:\n", encoding="utf-8")
            self.assertEqual(C.map_tribes(p), ("OUMAJI", "BARDUR"))


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        _, self.maps, _ = C.load_verified_pool(REPO, "eval_validation", max_maps=3)

    def test_episode_seeds_are_shared_across_policies_and_distinct_across_replicates(self):
        games = C.build_schedule(self.maps, C.ALL_POLICIES, 2, 42)
        self.assertEqual(len(games), 3 * 3 * 2)
        by_policy = {}
        for g in games:
            by_policy.setdefault(g.policy, []).append((g.map.filename, g.replicate_index, g.episode_seed))
        self.assertEqual(by_policy["ppo_argmax"], by_policy["random_legal"])
        seeds = [s for _, _, s in by_policy["ppo_argmax"]]
        self.assertEqual(len(set(seeds)), len(seeds))
        self.assertTrue(all(g.policy_seed is None for g in games if g.policy == "ppo_argmax"))
        self.assertTrue(all(g.policy_seed is not None for g in games if g.policy == "random_legal"))
        self.assertEqual([g.episode_seed for g in C.build_schedule(self.maps, ["ppo_argmax"], 2, 42)], seeds)
        self.assertNotEqual([g.episode_seed for g in C.build_schedule(self.maps, ["ppo_argmax"], 2, 7)], seeds)

    def test_canonical_classification(self):
        self.assertTrue(C.is_canonical("eval_validation", 120, 5, 42, C.ALL_POLICIES, None))
        self.assertFalse(C.is_canonical("eval_validation", 120, 5, 42, ["ppo_argmax"], None))
        self.assertFalse(C.is_canonical("eval_validation", 3, 5, 42, C.ALL_POLICIES, 3))
        self.assertFalse(C.is_canonical("eval_test", 120, 5, 42, C.ALL_POLICIES, None))


class StatsTests(unittest.TestCase):
    def test_rates_are_per_map_means(self):
        eps = [episode("ppo_argmax", "a", "win", 12), episode("ppo_argmax", "a", "timeout", rep=1),
               episode("ppo_argmax", "b", "loss"), episode("ppo_argmax", "b", "loss", rep=1)]
        summary, rows = C.aggregate(eps, 42)
        s = summary["ppo_argmax"]
        self.assertAlmostEqual(s["win_rate"], 0.25)
        self.assertAlmostEqual(s["loss_rate"], 0.5)
        self.assertAlmostEqual(s["timeout_rate"], 0.25)
        self.assertEqual(s["mean_win_turn"], 12)
        self.assertEqual(s["by_matchup"]["BARDUR vs IMPERIUS"]["maps"], 2)
        self.assertEqual(len(rows), 2)

    def test_paired_difference(self):
        left = {"a": {"win": 1.0}, "b": {"win": 0.5}, "c": {"win": 0.0}}
        right = {"a": {"win": 0.0}, "b": {"win": 0.5}, "c": {"win": 0.5}, "d": {"win": 1.0}}
        d = C.paired_difference(left, right, "win", 42)
        self.assertEqual(d["maps"], 3)
        self.assertAlmostEqual(d["mean_difference"], 1.0 / 6)
        self.assertEqual((d["maps_better"], d["maps_equal"], d["maps_worse"]), (1, 1, 1))


class CompareTests(unittest.TestCase):
    def write_eval(self, root, name, outcomes, **config):
        folder = Path(root) / name
        folder.mkdir()
        cfg = {"pool": "eval_validation", "manifest_sha256": "x", "evaluation_seed": 42, "repeats_per_map": 1,
               "max_turns": 20, "rules": "polytopia_2_15", **config}
        (folder / "config.json").write_text(json.dumps(cfg))
        with (folder / "episodes.jsonl").open("w") as fh:
            for m, o in outcomes.items():
                fh.write(json.dumps(episode("ppo_argmax", m, o)) + "\n")
        return folder

    def test_compare_pairs_maps(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self.write_eval(tmp, "base", {"a": "timeout", "b": "win"})
            cand = self.write_eval(tmp, "cand", {"a": "win", "b": "win"})
            result = compare(base, cand, "ppo_argmax")
            self.assertAlmostEqual(result["differences"]["win"]["mean_difference"], 0.5)
            self.assertAlmostEqual(result["differences"]["timeout"]["mean_difference"], -0.5)
            self.assertTrue(result["same_rules"])

    def test_compare_refuses_unpaired_evaluations(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self.write_eval(tmp, "base", {"a": "win"})
            other = self.write_eval(tmp, "other", {"a": "win"}, evaluation_seed=7)
            with self.assertRaises(SystemExit):
                compare(base, other, "ppo_argmax")


if __name__ == "__main__":
    unittest.main()
