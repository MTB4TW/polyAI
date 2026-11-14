"""Tune Phase1SptHeuristic weights for the lookahead policy by coordinate ascent on TRAINING-pool maps.

Never tunes on validation or test maps. For each weight in turn, all candidate values are scored in parallel
(one process and one JVM per worker), and the best is kept. Every trial is appended to --log; rerunning with the
same --log reuses trials already scored on the same maps, so an interrupted run resumes where it stopped.
Prints the best weights as JSON for `tools/evaluate_search_policy.py --weights '...'`.

Example:
  python tools/tune_search_heuristic.py --information visible --maps 60 --rounds 3 --workers 5 --log tune.jsonl
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ENV_VARS = {
    "POLYVISION_LEVEL_POOL_GLOB": "levels/phase1_pool_bardur_real/train/*.csv",
    "POLYVISION_LEVEL_SELECTION_MODE": "round_robin",
    "POLYVISION_INFO_MODE": "fast",
    "POLYVISION_SOLO_NO_OPPONENT_MODE": "1",
    "POLYVISION_PHASE": "1",
    "POLYVISION_COMBAT_MODE": "0",
}

_worker = {}


def _init_worker(maps_count: int, map_offset: int, information: str, seed: int):
    os.environ.update(ENV_VARS)
    from pol_env.Tribes.py.register_env import TribesGymWrapper
    from tools.evaluate_search_policy import load_pool
    from tools.phase1_eval_core import build_schedule

    _, maps, _ = load_pool("train", maps_count, map_offset)
    _worker["schedule"] = build_schedule(maps, [f"lookahead_{information}"], 1, seed)
    _worker["full"] = information == "full"
    _worker["env"] = TribesGymWrapper()


def _score(weights: dict) -> tuple[dict, float, float]:
    from tools.evaluate_phase1 import run_episode
    from tools.evaluate_search_policy import LookaheadSearchPolicy

    env = _worker["env"]
    policy = LookaheadSearchPolicy(env, weights, _worker["full"])
    t0 = time.perf_counter()
    spts = [run_episode(env, policy, r, 512)["final_spt_t10"] for r in _worker["schedule"]]
    return weights, float(np.mean(spts)), time.perf_counter() - t0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--information", choices=["visible", "full"], default="visible")
    p.add_argument("--maps", type=int, default=60, help="Number of training maps per trial")
    p.add_argument("--map-offset", type=int, default=0)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start", type=str, help="JSON weights to start from (default: best in --log, else DEFAULT_WEIGHTS)")
    p.add_argument("--log", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.update(ENV_VARS)
    from tools.evaluate_search_policy import DEFAULT_WEIGHTS, FEATURE_NAMES

    def key(w: dict) -> tuple:
        return tuple(round(float(w[k]), 6) for k in FEATURE_NAMES)

    cache: dict[tuple, float] = {}
    if args.log.exists():
        for line in args.log.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if (row.get("maps") == args.maps and row.get("information") == args.information
                    and row.get("map_offset", 0) == args.map_offset and set(row["weights"]) == set(FEATURE_NAMES)):
                cache[key(row["weights"])] = float(row["mean_spt"])

    weights = dict(DEFAULT_WEIGHTS)
    if args.start:
        weights.update({k: float(v) for k, v in json.loads(args.start).items()})
    elif cache:
        best_key = max(cache, key=cache.get)
        weights = dict(zip(FEATURE_NAMES, best_key))
    print(f"resuming with {len(cache)} cached trials", flush=True)

    log = args.log.open("a", encoding="utf-8")
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(args.workers, initializer=_init_worker,
                    initargs=(args.maps, args.map_offset, args.information, args.seed))

    def score_many(candidates: list[dict]) -> list[float]:
        todo = [w for w in candidates if key(w) not in cache]
        for w, mean_spt, secs in pool.imap_unordered(_score, todo):
            cache[key(w)] = mean_spt
            log.write(json.dumps({"weights": w, "mean_spt": mean_spt, "maps": args.maps, "map_offset": args.map_offset,
                                  "information": args.information, "seconds": secs}) + "\n")
            log.flush()
        return [cache[key(w)] for w in candidates]

    try:
        best = score_many([weights])[0]
        print(f"start: {best:.3f}", flush=True)
        for rnd in range(args.rounds):
            improved = False
            for k in FEATURE_NAMES:
                if k == "spt":
                    continue  # fixes the scale
                base = weights[k]
                values = {base * 0.5, base * 1.5, base * 2.5, 0.0, -base} if base != 0 else {0.5, 2.0, -0.5}
                trials = [dict(weights, **{k: v}) for v in sorted(values)]
                for trial, s in zip(trials, score_many(trials)):
                    if s > best + 1e-9:
                        best, weights, improved = s, trial, True
                print(f"round {rnd + 1} {k:20s} -> {weights[k]:8.3f}   best {best:.3f}", flush=True)
            if not improved:
                break
        print("BEST", json.dumps(weights), f"{best:.3f}", flush=True)
    finally:
        log.close()
        pool.terminate()


if __name__ == "__main__":
    main()
