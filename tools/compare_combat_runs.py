#!/usr/bin/env python3
"""Compare combat training runs by their W&B curves (diagnostic; there is no combat evaluator yet).

Groups runs by name prefix (everything before `-seedN`), and for each group reports, per seed and averaged:
- the final rolling-100-game win / loss / timeout rates,
- the mean win rate over the second half of training (less noisy than the last point),
- the final rolling end-of-game potential,
- games completed.

These are stochastic training-time results against the training opponent, not a held-out evaluation.

Usage:
    python tools/compare_combat_runs.py PREFIX [PREFIX ...] [--entity polyai-stanford] [--project polyAI]
e.g.
    python tools/compare_combat_runs.py combat-scratch-1m combat-warm-1m combat-scratch-gentle-1m combat-warm-gentle-1m
"""

import argparse
import re
import statistics as st

KEYS = {
    "win": "combat/win_rate_last100",
    "loss": "combat/loss_rate_last100",
    "timeout": "combat/timeout_rate_last100",
    "potential": "combat/episode_end_potential_last100",
    "games": "combat/games_completed",
}


def summarize(run):
    rows = [r for r in run.scan_history(keys=["global_step", KEYS["win"]]) if r.get(KEYS["win"]) is not None]
    if not rows:
        return None
    max_step = max(r["global_step"] for r in rows)
    second_half = [r[KEYS["win"]] for r in rows if r["global_step"] >= max_step / 2]
    s = run.summary
    return {
        "step": int(max_step),
        "win": s.get(KEYS["win"]),
        "loss": s.get(KEYS["loss"]),
        "timeout": s.get(KEYS["timeout"]),
        "win_2nd_half": st.mean(second_half) if second_half else None,
        "potential": s.get(KEYS["potential"]),
        "games": s.get(KEYS["games"]),
        "state": run.state,
    }


def fmt(v, pct=False):
    if v is None:
        return "   -  "
    return f"{100 * v:5.1f}%" if pct else f"{v:6.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("prefixes", nargs="+")
    ap.add_argument("--entity", default="polyai-stanford")
    ap.add_argument("--project", default="polyAI")
    args = ap.parse_args()

    import wandb

    api = wandb.Api(timeout=120)
    runs = list(api.runs(f"{args.entity}/{args.project}"))
    print("Combat training curves (rolling 100 games vs the training opponent; diagnostic)")
    print(f"{'group':30s} {'seed':>4s} {'steps':>9s} {'win':>7s} {'loss':>7s} {'timeout':>8s} "
          f"{'win 2nd half':>12s} {'potential':>9s} {'games':>7s}")
    for prefix in args.prefixes:
        pattern = re.compile(rf"(^|__){re.escape(prefix)}-seed(\d+)(__|$)")
        found = []
        for r in runs:
            m = pattern.search(r.name)
            if m and r.job_type not in ("evaluation", "upload-checkpoint"):
                summary = summarize(r)
                if summary:
                    found.append((int(m.group(2)), summary))
        if not found:
            print(f"{prefix:30s} no runs found")
            continue
        for seed, s in sorted(found):
            note = "" if s["state"] == "finished" else f"  ({s['state']})"
            print(f"{prefix:30s} {seed:4d} {s['step']:9d} {fmt(s['win'], True)} {fmt(s['loss'], True)} "
                  f"{fmt(s['timeout'], True):>8s} {fmt(s['win_2nd_half'], True):>12s} {fmt(s['potential']):>9s} "
                  f"{int(s['games'] or 0):7d}{note}")
        means = {k: [s[k] for _, s in found if s[k] is not None] for k in ("win", "loss", "win_2nd_half", "potential")}
        print(f"{prefix + ' (mean)':30s} {len(found):4d} {'':9s} "
              f"{fmt(st.mean(means['win']) if means['win'] else None, True)} "
              f"{fmt(st.mean(means['loss']) if means['loss'] else None, True)} {'':8s} "
              f"{fmt(st.mean(means['win_2nd_half']) if means['win_2nd_half'] else None, True):>12s} "
              f"{fmt(st.mean(means['potential']) if means['potential'] else None):>9s}")


if __name__ == "__main__":
    main()
