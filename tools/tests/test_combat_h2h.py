"""Head-to-head combat evaluator (tools/evaluate_combat_h2h.py): schedule, outcome flipping, summary, and a live
tiny run (a freshly trained tiny checkpoint against itself on one map). The live part is skipped without a
compiled engine."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools import combat_eval_core as C
from tools import evaluate_combat_h2h as H

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "pol_env" / "Tribes" / "out"
HAVE_ENGINE = bool(shutil.which("java")) and (OUT / "core" / "game" / "PythonEnv.class").exists()


def ep(seat, map_name, outcome, turn=10):
    return {"policy": seat, "map_filename": map_name, "outcome": outcome, "end_turn": turn,
            "our_tribe": "BARDUR", "opponent_tribe": "IMPERIUS"}


class PureTests(unittest.TestCase):
    def test_schedule_both_seats_same_episode_seeds(self):
        _, maps, _ = C.load_verified_pool(REPO, "eval_validation", 3)
        games = H.build_h2h_schedule(maps, 2, 42)
        self.assertEqual(len(games), 3 * 2 * 2)
        by_seat = {s: [(g.map.filename, g.replicate_index, g.episode_seed) for g in games if g.seat == s] for s in H.SEATS}
        self.assertEqual(by_seat["a_tribe0"], by_seat["a_tribe1"])
        # Same episode seeds as the canonical evaluator, so results line up with it map by map.
        canonical = {(g.map.filename, g.replicate_index): g.episode_seed
                     for g in C.build_schedule(maps, ["ppo_argmax"], 2, 42)}
        for name, rep, seed in by_seat["a_tribe0"]:
            self.assertEqual(canonical[(name, rep)], seed)
        self.assertEqual(len({(g.seed_tribe0, g.seed_tribe1) for g in games}), len(games))

    def test_outcome_for_a(self):
        self.assertEqual(H.outcome_for_a("a_tribe0", "win"), "win")
        self.assertEqual(H.outcome_for_a("a_tribe1", "win"), "loss")
        self.assertEqual(H.outcome_for_a("a_tribe1", "loss"), "win")
        self.assertEqual(H.outcome_for_a("a_tribe1", "timeout"), "timeout")
        with self.assertRaises(ValueError):
            H.outcome_for_a("a_tribe0", "draw")

    def test_summary_and_margin(self):
        episodes = [ep("a_tribe0", "m1", "win"), ep("a_tribe1", "m1", "loss"),
                    ep("a_tribe0", "m2", "win"), ep("a_tribe1", "m2", "win")]
        summary, rows = H.summarize_h2h(episodes, 42)
        self.assertAlmostEqual(summary["overall"]["win_rate"], 0.75)
        self.assertAlmostEqual(summary["overall"]["loss_rate"], 0.25)
        self.assertAlmostEqual(summary["by_seat"]["a_tribe0"]["win_rate"], 1.0)
        self.assertAlmostEqual(summary["by_seat"]["a_tribe1"]["win_rate"], 0.5)
        self.assertAlmostEqual(summary["margin_a_minus_b_win"]["mean_difference"], 0.5)  # m1: 0, m2: +1
        self.assertAlmostEqual(summary["seat_gap_a_win_tribe0_minus_tribe1"]["mean_difference"], 0.5)
        self.assertEqual({r["policy"] for r in rows}, {"overall", "a_tribe0", "a_tribe1"})

    def test_canonical_label(self):
        self.assertTrue(H.is_h2h_canonical("eval_validation", 120, 5, 42, "sampled", None))
        self.assertFalse(H.is_h2h_canonical("eval_validation", 120, 5, 42, "argmax", None))
        self.assertFalse(H.is_h2h_canonical("eval_validation", 1, 5, 42, "sampled", 1))

    def test_refuses_test_pool(self):
        with self.assertRaises(SystemExit):
            H.main(["--model-a", "x", "--model-b", "y", "--pool", "eval_test"])


COMBAT_ENV = {"POLYVISION_COMBAT_MODE": "1", "POLYVISION_PHASE": "2", "POLYVISION_SOLO_NO_OPPONENT_MODE": "0",
              "POLYVISION_INFO_MODE": "fast", "POLYVISION_LEVEL_POOL_GLOB": "levels/phase2_pool_1v1/train/*.csv",
              "POLYVISION_RULES": "polytopia_2_15", "WANDB_MODE": "disabled"}


@unittest.skipUnless(HAVE_ENGINE, "compiled engine missing")
class LiveTiny(unittest.TestCase):
    def test_tiny_model_against_itself(self):
        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / "t.cleanrl_model"
            cmd = [sys.executable, "py_rl/cleanrl/cleanrl/ppo.py", "--env-id", "Tribes-v0", "--network-arch",
                   "spatial_resnet_composed", "--num-envs", "1", "--num-steps", "16", "--num-minibatches", "1",
                   "--update-epochs", "1", "--total-timesteps", "16", "--max-legal-actions", "512",
                   "--save-model", "--model-path", str(model), "--exp-name", "h2h_smoke", "--seed", "1"]
            r = subprocess.run(cmd, cwd=REPO, env={**os.environ, **COMBAT_ENV}, capture_output=True, text=True, timeout=1500)
            self.assertEqual(r.returncode, 0, (r.stdout[-2000:], r.stderr[-4000:]))
            out_root = Path(d) / "evals"
            r = subprocess.run([sys.executable, "tools/evaluate_combat_h2h.py", "--model-a", str(model), "--model-b",
                                str(model), "--max-maps", "1", "--repeats-per-map", "1", "--output-root", str(out_root),
                                "--evaluation-id", "h2h_smoke"],
                               cwd=REPO, env={**os.environ, "POLYVISION_COMBAT_LEAGUE": "1"}, capture_output=True,
                               text=True, timeout=1500)
            self.assertEqual(r.returncode, 0, (r.stdout[-2000:], r.stderr[-4000:]))
            folder = out_root / "h2h_smoke"
            rows = [json.loads(l) for l in (folder / "episodes.jsonl").read_text().splitlines()]
            self.assertEqual([row["policy"] for row in rows], ["a_tribe0", "a_tribe1"])
            for row in rows:
                self.assertEqual(row["outcome"], H.outcome_for_a(row["policy"], row["tribe0_outcome"]))
            summary = json.loads((folder / "summary.json").read_text())
            self.assertFalse(summary["canonical"])
            config = json.loads((folder / "config.json").read_text())
            self.assertNotIn("POLYVISION_COMBAT_LEAGUE", config["polyvision_environment"])


if __name__ == "__main__":
    unittest.main()
