#!/usr/bin/env python
"""Compare two combat evaluations map by map (`tools/evaluate_combat.py` output folders).

    python tools/compare_combat_evals.py BASELINE_DIR CANDIDATE_DIR [--policy ppo_argmax]

Both evaluations must use the same pool manifest, seed and replicate count, so they played the same games
(map and episode seed). Reports candidate minus baseline for win, loss and timeout rates with bootstrap CIs over
maps. Different rule sets are allowed but flagged: the maps and opponent seeds match, the games do not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.combat_eval_core import OUTCOMES, paired_difference, per_map_rates  # noqa: E402


def load(folder: Path):
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    with (folder / "episodes.jsonl").open(encoding="utf-8") as fh:
        episodes = [json.loads(line) for line in fh if line.strip()]
    return config, per_map_rates(episodes)


def compare(baseline: Path, candidate: Path, policy: str) -> dict:
    base_cfg, base_rates = load(baseline)
    cand_cfg, cand_rates = load(candidate)
    for key in ("pool", "manifest_sha256", "evaluation_seed", "repeats_per_map", "max_turns"):
        if base_cfg.get(key) != cand_cfg.get(key):
            raise SystemExit(f"Not pairable: {key} differs ({base_cfg.get(key)!r} vs {cand_cfg.get(key)!r})")
    if policy not in base_rates or policy not in cand_rates:
        raise SystemExit(f"Policy {policy!r} missing from one of the evaluations")
    seed = int(base_cfg["evaluation_seed"])
    return {
        "baseline": str(baseline), "candidate": str(candidate), "policy": policy,
        "same_rules": base_cfg.get("rules") == cand_cfg.get("rules"),
        "rules": {"baseline": base_cfg.get("rules"), "candidate": cand_cfg.get("rules")},
        "differences": {k: paired_difference(cand_rates[policy], base_rates[policy], k, seed) for k in OUTCOMES},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("baseline", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--policy", default="ppo_argmax")
    p.add_argument("--json", action="store_true", help="Print the result as JSON")
    args = p.parse_args()
    result = compare(args.baseline, args.candidate, args.policy)
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"{args.policy}: candidate {args.candidate.name} minus baseline {args.baseline.name}")
    if not result["same_rules"]:
        print(f"WARNING: different rules {result['rules']}: same maps and seeds, but not the same games")
    for k, d in result["differences"].items():
        print(f"  {k:8s} {d['mean_difference']:+7.1%}  95% CI [{d['ci95'][0]:+.1%}, {d['ci95'][1]:+.1%}]  "
              f"maps {d['maps_better']} higher / {d['maps_equal']} equal / {d['maps_worse']} lower (of {d['maps']})")


if __name__ == "__main__":
    main()
