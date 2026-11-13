"""Evaluate search-based Phase 1 policies (heuristic lookahead) on a verified map pool.

These policies use the Java forward model on a copy of the game state, so they are NOT ordinary policy-visible
baselines. Two information modes:

  --information visible  The state copy has the player's fog applied (fogged tiles read as FOG), so the search
                         only knows what Bardur can see. Still a planning policy with a forward model, so report it
                         separately from PPO/greedy rows.
  --information full     Privileged oracle: the copy includes hidden tiles. Report as a privileged-oracle diagnostic.

Every move is still chosen from the wrapper's policy-visible legal set and executed through env.step(global_id),
with the same scripted opening, legal filters and Turn-10 contract checks as tools/evaluate_phase1.py. Results are
always labelled as diagnostics; they are never canonical PPO results.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.phase1_eval_core import (  # noqa: E402
    FEATURE, EvaluationPolicy, MapRecord, aggregate_results, build_schedule, legal_tensors, load_verified_pool,
    sha256_file,
)


def load_pool(pool: str, max_maps: int | None, offset: int = 0):
    """Validation/human_benchmark via the verified loader; train maps (for tuning) straight from the manifest."""
    if pool != "train":
        manifest, maps, manifest_path = load_verified_pool(REPO_ROOT, pool, None)
        return manifest, maps[offset:offset + max_maps] if max_maps else maps[offset:], manifest_path
    manifest_path = REPO_ROOT / "pol_env/Tribes/levels/phase1_pool_bardur_real/split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = sorted((r for r in manifest["maps"] if r["pool"] == "train"), key=lambda r: (r["canonical_map_sha256"], r["filename"]))
    rows = rows[offset:offset + max_maps] if max_maps else rows[offset:]
    maps = []
    for r in rows:
        path = (manifest_path.parent / r["relative_path"]).resolve()
        if sha256_file(path) != r["csv_sha256"]:
            raise RuntimeError(f"Map hash mismatch: {path}")
        maps.append(MapRecord(r["filename"], str(path), r["canonical_map_sha256"], r["csv_sha256"], "train"))
    return manifest, maps, manifest_path

# Weights, feature order and the lookahead's slot choice live in pol_env/Tribes/py/search_teacher.py, shared with the
# training-time teacher (POLYVISION_TEACHER=lookahead_visible), so both play identically. DEFAULT_WEIGHTS are the
# hand-set weights; TUNED_VISIBLE_WEIGHTS were tuned on training-pool maps (see --pool train), never on validation.
from pol_env.Tribes.py.search_teacher import (  # noqa: E402
    DEFAULT_WEIGHTS, FEATURE_NAMES, TUNED_VISIBLE_WEIGHTS, decision_seed, java_weight_array, lookahead_scores_seeded,
    select_lookahead_slot,
)


class LookaheadSearchPolicy(EvaluationPolicy):
    """One-step heuristic lookahead over the policy-visible legal set."""

    def __init__(self, env, weights: dict, full_information: bool, eps: float = 1e-9, seeded: bool = False):
        self.env = env
        # seeded=False: PR #11 behaviour (unseeded state copies). seeded=True: the reproducible teacher used for
        # training (POLYVISION_TEACHER), seeded per decision from the episode seed and the decision index.
        self.seeded = bool(seeded)
        self._seed_episode, self._seed_decision = None, 0
        self.full_information = bool(full_information)
        self.name = f"lookahead_{'full' if full_information else 'visible'}"
        tribes = env.tribes_env
        self.java = tribes._env
        self.java_weights = java_weight_array(tribes._gateway, weights)
        self.eps = eps

    def turns_remaining(self) -> int:
        return max(0, int(self.env.MAX_TURNS) - int(self.env._turn_count))

    def choose_action(self, obs, info):
        ids, valid, feats = legal_tensors(info)
        raw_map = self.env._current_legal_id_to_raw_index
        if self.seeded:
            if self._seed_episode != self.env._episode_index:
                self._seed_episode, self._seed_decision = self.env._episode_index, 0
            seed = decision_seed(self.env._last_reset_seed, self._seed_decision)
            self._seed_decision += 1
            scores = lookahead_scores_seeded(self.java, self.java_weights, self.turns_remaining(),
                                             self.full_information, seed)
        else:
            scores = list(self.java.lookaheadScores(self.java_weights, self.turns_remaining(), self.full_information))

        best_slot = select_lookahead_slot(scores, ids, valid, feats[:, FEATURE["is_end_turn"]] > 0.5, raw_map, self.eps)
        return int(ids[best_slot]), best_slot


class TurnMCTSPolicy(LookaheadSearchPolicy):
    """Within-turn MCTS (Java Phase1TurnMCTS) over the policy-visible legal set, valued by the same heuristic."""

    def __init__(self, env, weights: dict, full_information: bool, iterations: int, ucb_c: float,
                 max_depth: int, rollout_steps: int, seed: int):
        super().__init__(env, weights, full_information)
        self.name = f"mcts_{'full' if full_information else 'visible'}"
        self.iterations, self.ucb_c = int(iterations), float(ucb_c)
        self.max_depth, self.rollout_steps = int(max_depth), int(rollout_steps)
        self.seed, self.decision = int(seed), 0

    def choose_action(self, obs, info):
        ids, valid, feats = legal_tensors(info)
        slots = [int(s) for s in np.flatnonzero(valid)]
        raw_map = self.env._current_legal_id_to_raw_index
        raw_to_slot = {}
        for s in slots:
            raw = raw_map.get(int(ids[s]))
            if raw is not None:
                raw_to_slot.setdefault(int(raw), s)
        if not raw_to_slot:
            return int(ids[slots[0]]), slots[0]
        g = self.env.tribes_env._gateway
        allowed = g.new_array(g.jvm.int, len(raw_to_slot))
        for i, raw in enumerate(sorted(raw_to_slot)):
            allowed[i] = raw
        self.decision += 1
        raw = int(self.java.mctsChooseIndex(
            self.java_weights, self.turns_remaining(), self.full_information, allowed, self.iterations,
            self.ucb_c, self.max_depth, self.rollout_steps, self.seed * 1_000_003 + self.decision))
        slot = raw_to_slot[raw]
        return int(ids[slot]), slot


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", choices=["validation", "train", "human_benchmark"], default="validation",
                   help="Tune on train; report on validation. The test pool is deliberately not offered here.")
    p.add_argument("--map-offset", type=int, default=0, help="Skip this many maps of the sorted pool")
    p.add_argument("--information", choices=["visible", "full"], default="visible")
    p.add_argument("--search", choices=["lookahead", "mcts"], default="lookahead")
    p.add_argument("--iterations", type=int, default=200, help="MCTS iterations per decision")
    p.add_argument("--ucb-c", type=float, default=0.7)
    p.add_argument("--max-depth", type=int, default=8, help="MCTS: max actions per branch within a turn")
    p.add_argument("--rollout-steps", type=int, default=2, help="MCTS: greedy steps after each expanded leaf")
    p.add_argument("--search-seeding", choices=["none", "per_decision"], default="none",
                   help="lookahead only. none: unseeded state copies (PR #11 default, not reproducible when actions "
                        "have random outcomes); per_decision: the reproducible teacher used by POLYVISION_TEACHER")
    p.add_argument("--max-maps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weights", type=str,
                   help="JSON object overriding entries of DEFAULT_WEIGHTS, or 'tuned_visible' for TUNED_VISIBLE_WEIGHTS")
    p.add_argument("--max-steps-per-episode", type=int, default=512)
    p.add_argument("--progress-every", type=int, default=5)
    p.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs/evaluations")
    p.add_argument("--evaluation-id")
    return p.parse_args()


def main():
    args = parse_args()
    weights = dict(DEFAULT_WEIGHTS)
    if args.weights == "tuned_visible":
        weights = dict(TUNED_VISIBLE_WEIGHTS)
    elif args.weights:
        override = json.loads(args.weights)
        unknown = set(override) - set(weights)
        if unknown:
            raise SystemExit(f"Unknown weight names: {sorted(unknown)}")
        weights.update({k: float(v) for k, v in override.items()})

    # Same environment configuration as tools/evaluate_phase1.py.
    os.environ["POLYVISION_LEVEL_POOL_GLOB"] = f"levels/phase1_pool_bardur_real/{args.pool}/*.csv"
    os.environ["POLYVISION_LEVEL_SELECTION_MODE"] = "round_robin"
    os.environ["POLYVISION_INFO_MODE"] = "fast"
    os.environ["POLYVISION_SOLO_NO_OPPONENT_MODE"] = "1"
    os.environ["POLYVISION_PHASE"] = "1"
    os.environ["POLYVISION_COMBAT_MODE"] = "0"

    from pol_env.Tribes.py.register_env import TribesGymWrapper
    from tools.evaluate_phase1 import run_episode, runtime_provenance, utc_now, write_outputs, print_summary

    manifest, maps, manifest_path = load_pool(args.pool, args.max_maps, args.map_offset)
    full = args.information == "full"
    policy_name = f"{args.search}_{args.information}"
    schedule = build_schedule(maps, [policy_name], 1, args.seed)
    label = f"{'oracle' if full else 'visible-search'}-diagnostic"
    evaluation_id = args.evaluation_id or (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{args.pool}_{policy_name}_{len(maps)}maps_{label}")
    out_dir = args.output_root / evaluation_id
    if out_dir.exists():
        raise SystemExit(f"Output directory already exists: {out_dir}")

    env = TribesGymWrapper()
    try:
        if args.search == "mcts":
            policy = TurnMCTSPolicy(env, weights, full, args.iterations, args.ucb_c, args.max_depth,
                                    args.rollout_steps, args.seed)
        else:
            policy = LookaheadSearchPolicy(env, weights, full, seeded=args.search_seeding == "per_decision")
        started, episodes = utc_now(), []
        for i, record in enumerate(schedule, 1):
            episodes.append(run_episode(env, policy, record, args.max_steps_per_episode))
            if args.progress_every and (i % args.progress_every == 0 or i == len(schedule)):
                spts = [e["final_spt_t10"] for e in episodes]
                print(f"{policy_name}: {i}/{len(schedule)} maps, running mean SPT {np.mean(spts):.2f}", flush=True)
        summary_policies, per_map = aggregate_results(episodes, args.seed)
        config = {
            "evaluation_id": evaluation_id, "status": "complete", "canonical": False, "classification": label,
            "privileged_information": full, "information_mode": args.information,
            "policy": policy_name, "heuristic_weights": weights,
            "search": args.search, "search_seeding": args.search_seeding if args.search == "lookahead" else None,
            "mcts": ({"iterations": args.iterations, "ucb_c": args.ucb_c,
                                              "max_depth": args.max_depth, "rollout_steps": args.rollout_steps}
                                             if args.search == "mcts" else None), "feature_names": list(FEATURE_NAMES),
            "started_utc": started, "ended_utc": utc_now(), "pool": args.pool, "maps": len(maps),
            "seed": args.seed, "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
            "pool_identity": manifest.get("pool_identities", {}).get(args.pool), "map_offset": args.map_offset, "runtime": runtime_provenance(),
            "polyvision_environment": {k: v for k, v in os.environ.items() if k.startswith("POLYVISION_")},
        }
        summary = {"policies": summary_policies, "paired": {}}
        write_outputs(out_dir, config, episodes, summary, per_map, [])
        print_summary(args.pool, len(maps), summary, out_dir)
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
