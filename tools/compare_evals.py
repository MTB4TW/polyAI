#!/usr/bin/env python
"""Paired, map-by-map comparison of two Phase 1 evaluation folders.

Both folders must come from tools/evaluate_phase1.py with the same pool and
seed, so every map was played with the same episode seeds. The script checks
that before reporting anything.

Usage:
    python tools/compare_evals.py BASELINE CANDIDATE [--policy ppo_argmax] [--output out.json]

BASELINE and CANDIDATE are folder paths or folder names under outputs/evaluations/.
Uses only the standard library, so it runs without the project virtualenv.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / "outputs" / "evaluations"


def resolve(path_or_id: str) -> Path:
    for candidate in (Path(path_or_id), EVAL_ROOT / path_or_id):
        if (candidate / "per_map.csv").is_file():
            return candidate
    raise SystemExit(f"No evaluation folder with per_map.csv found for: {path_or_id}")


def per_map_scores(folder: Path, policy: str) -> dict[str, float]:
    with (folder / "per_map.csv").open(encoding="utf-8") as f:
        scores = {row["map_identity"]: float(row["mean_spt"]) for row in csv.DictReader(f) if row["policy"] == policy}
    if not scores:
        raise SystemExit(f"{folder.name}: no rows for policy {policy!r}")
    return scores


def episode_seeds(folder: Path, policy: str) -> set[tuple[str, int, int]]:
    seeds = set()
    with (folder / "episodes.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["policy"] == policy:
                seeds.add((row["map_identity"], int(row["replicate_index"]), int(row["episode_seed"])))
    return seeds


def read_config(folder: Path) -> dict:
    path = folder / "config.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--policy", default="ppo_argmax", help="policy row to compare (default: ppo_argmax)")
    p.add_argument("--bootstrap", type=int, default=5000, help="bootstrap resamples over maps")
    p.add_argument("--seed", type=int, default=42, help="bootstrap RNG seed")
    p.add_argument("--output", type=Path, help="optional JSON file for the result")
    args = p.parse_args()

    base_dir, cand_dir = resolve(args.baseline), resolve(args.candidate)
    base_cfg, cand_cfg = read_config(base_dir), read_config(cand_dir)
    warnings = []
    for name, cfg in (("baseline", base_cfg), ("candidate", cand_cfg)):
        if not cfg.get("canonical", False):
            warnings.append(f"{name} evaluation is not canonical ({cfg.get('classification', 'unknown')})")
    if base_cfg.get("pool") != cand_cfg.get("pool"):
        raise SystemExit(f"Pools differ: {base_cfg.get('pool')} vs {cand_cfg.get('pool')}")

    base, cand = per_map_scores(base_dir, args.policy), per_map_scores(cand_dir, args.policy)
    if set(base) != set(cand):
        raise SystemExit(f"Map sets differ ({len(base)} vs {len(cand)} maps); not a paired comparison")
    if episode_seeds(base_dir, args.policy) != episode_seeds(cand_dir, args.policy):
        raise SystemExit("Episode seeds differ between the two evaluations; not a paired comparison")

    maps = sorted(base)
    diffs = [cand[m] - base[m] for m in maps]
    rng = random.Random(args.seed)
    boots = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(args.bootstrap))
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]

    result = {
        "policy": args.policy,
        "pool": base_cfg.get("pool"),
        "maps": len(maps),
        "baseline": {"evaluation": base_dir.name, "mean_spt": statistics.mean(base.values()),
                     "network_arch": (base_cfg.get("checkpoint_metadata") or {}).get("network_arch", "mlp")},
        "candidate": {"evaluation": cand_dir.name, "mean_spt": statistics.mean(cand.values()),
                      "network_arch": (cand_cfg.get("checkpoint_metadata") or {}).get("network_arch", "mlp")},
        "mean_difference": statistics.mean(diffs),
        "difference_ci95": [lo, hi],
        "bootstrap": {"samples": args.bootstrap, "seed": args.seed, "unit": "map"},
        "wins": sum(d > 0 for d in diffs),
        "ties": sum(d == 0 for d in diffs),
        "losses": sum(d < 0 for d in diffs),
        "warnings": warnings,
    }

    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    print(f"Policy {args.policy} on {len(maps)} {result['pool']} maps (paired)")
    print(f"  baseline  {base_dir.name}: {result['baseline']['mean_spt']:.2f} SPT")
    print(f"  candidate {cand_dir.name}: {result['candidate']['mean_spt']:.2f} SPT")
    print(f"  difference (candidate - baseline): {result['mean_difference']:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]")
    print(f"  W/T/L: {result['wins']}/{result['ties']}/{result['losses']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"  written: {args.output}")


if __name__ == "__main__":
    main()
