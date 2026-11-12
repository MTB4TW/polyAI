#!/usr/bin/env python
"""Log a finished Phase 1 evaluation to W&B as a read-only copy of the results.

Git stays the source of truth for results; this makes them visible next to the
checkpoint and training runs in W&B. It creates one W&B run (job type
"evaluation", named after the evaluation folder) with:

  - summary metrics per policy (mean SPT, median, 95% CI) and the built-in
    paired comparisons against the baselines,
  - optionally, the paired difference against a reference evaluation,
  - a per-map table of every policy's SPT,
  - the evaluation folder itself as an artifact of type "polyvision-evaluation",
  - optionally, a lineage link to the checkpoint artifact that was evaluated.

Usage:
    python tools/wandb_log_eval.py EVAL [--reference REF] [--checkpoint-artifact NAME[:VERSION]]

EVAL and REF are folder paths or folder names under outputs/evaluations/.
Project and entity come from --project/--entity, else WANDB_PROJECT/WANDB_ENTITY,
else project "polyAI" under your default W&B entity.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_evals import ComparisonError, paired_comparison, resolve  # noqa: E402

DEFAULT_PROJECT = "polyAI"
CHECKPOINT_ARTIFACT_TYPE = "polyvision-checkpoint"
EVALUATION_ARTIFACT_TYPE = "polyvision-evaluation"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def summary_metrics(summary: dict) -> dict:
    metrics = {}
    for policy, stats in summary.get("policies", {}).items():
        headline = stats[stats["headline_level"]]
        low, high = headline["mean_ci95"]
        metrics.update({
            f"{policy}/mean_spt": headline["mean"],
            f"{policy}/median_spt": headline["median"],
            f"{policy}/ci95_low": low,
            f"{policy}/ci95_high": high,
            f"{policy}/maps": stats["maps"],
            f"{policy}/episodes": stats["episodes"],
        })
    for name, paired in (summary.get("paired") or {}).items():
        if not paired:
            continue
        low, high = paired["mean_difference_ci95"]
        metrics.update({
            f"paired/{name}/mean_difference": paired["mean_difference"],
            f"paired/{name}/ci95_low": low,
            f"paired/{name}/ci95_high": high,
            f"paired/{name}/wins": paired["wins"],
            f"paired/{name}/ties": paired["ties"],
            f"paired/{name}/losses": paired["losses"],
        })
    return metrics


def per_map_rows(folder: Path) -> tuple[list[str], list[list]]:
    by_map: dict[str, dict] = {}
    policies: list[str] = []
    with (folder / "per_map.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["policy"] not in policies:
                policies.append(row["policy"])
            by_map.setdefault(row["map_filename"], {})[row["policy"]] = float(row["mean_spt"])
    columns = ["map"] + [f"{p}_spt" for p in policies]
    rows = [[m] + [scores.get(p) for p in policies] for m, scores in sorted(by_map.items())]
    return columns, rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("evaluation")
    p.add_argument("--reference", help="evaluation to compare against (paired, ppo_argmax)")
    p.add_argument("--checkpoint-artifact", help="checkpoint artifact that was evaluated, e.g. NAME or NAME:v0")
    p.add_argument("--project")
    p.add_argument("--entity")
    args = p.parse_args()

    try:
        import wandb
    except ModuleNotFoundError:
        sys.exit("wandb is not installed in this environment: pip install wandb")

    try:
        folder = resolve(args.evaluation)
    except ComparisonError as e:
        sys.exit(str(e))
    config = load_json(folder / "config.json")
    summary = load_json(folder / "summary.json")
    if not summary:
        sys.exit(f"{folder}: summary.json missing; is the evaluation finished?")

    checkpoint_meta = config.get("checkpoint_metadata") or {}
    canonical = bool(config.get("canonical"))
    project = args.project or os.environ.get("WANDB_PROJECT") or DEFAULT_PROJECT
    entity = args.entity or os.environ.get("WANDB_ENTITY") or None

    run = wandb.init(
        project=project,
        entity=entity,
        job_type="evaluation",
        name=folder.name,
        tags=[config.get("pool", "unknown-pool"), "canonical" if canonical else "noncanonical",
              checkpoint_meta.get("network_arch", "mlp")],
        config={
            "evaluation_id": folder.name,
            "pool": config.get("pool"),
            "canonical": canonical,
            "classification": config.get("classification"),
            "evaluation_seed": config.get("evaluation_seed"),
            "repeats_per_map": config.get("repeats_per_stochastic_map"),
            "evaluated_map_count": config.get("evaluated_map_count"),
            "checkpoint_sha256": config.get("checkpoint_sha256"),
            "network_arch": checkpoint_meta.get("network_arch", "mlp"),
            "actor_mode": checkpoint_meta.get("actor_mode"),
            "environment_version": checkpoint_meta.get("phase1_environment_version"),
            "git_commit": (config.get("runtime") or {}).get("git_commit"),
            "git_dirty": (config.get("runtime") or {}).get("git_dirty"),
        },
    )
    warnings = []

    if args.checkpoint_artifact:
        ref = args.checkpoint_artifact if ":" in args.checkpoint_artifact else f"{args.checkpoint_artifact}:latest"
        try:
            checkpoint = run.use_artifact(ref, type=CHECKPOINT_ARTIFACT_TYPE)
            expected = (checkpoint.metadata or {}).get("model_sha256")
            if expected and expected != config.get("checkpoint_sha256"):
                warnings.append(f"checkpoint artifact {ref} has sha256 {expected[:16]}..., but the evaluation "
                                f"used {str(config.get('checkpoint_sha256'))[:16]}...; lineage link may be wrong")
        except Exception as e:  # missing artifact should not lose the evaluation log
            warnings.append(f"could not link checkpoint artifact {ref}: {e}")

    metrics = summary_metrics(summary)
    if args.reference:
        try:
            comparison = paired_comparison(args.reference, folder)
            low, high = comparison["difference_ci95"]
            metrics.update({
                "vs_reference/evaluation": comparison["baseline"]["evaluation"],
                "vs_reference/reference_mean_spt": comparison["baseline"]["mean_spt"],
                "vs_reference/mean_difference": comparison["mean_difference"],
                "vs_reference/ci95_low": low,
                "vs_reference/ci95_high": high,
                "vs_reference/wins": comparison["wins"],
                "vs_reference/ties": comparison["ties"],
                "vs_reference/losses": comparison["losses"],
            })
            warnings.extend(comparison["warnings"])
        except ComparisonError as e:
            warnings.append(f"reference comparison skipped: {e}")
            metrics["vs_reference/error"] = str(e)
    run.summary.update(metrics)

    columns, rows = per_map_rows(folder)
    run.log({"per_map": wandb.Table(columns=columns, data=rows)})

    results = wandb.Artifact(folder.name, type=EVALUATION_ARTIFACT_TYPE,
                             metadata={"canonical": canonical, "pool": config.get("pool"),
                                       "checkpoint_sha256": config.get("checkpoint_sha256")})
    results.add_dir(str(folder))
    run.log_artifact(results)

    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    headline = metrics.get("ppo_argmax/mean_spt")
    print(f"logged {folder.name} to W&B: {run.url}" + (f"  (ppo_argmax {headline:.2f} SPT)" if headline is not None else ""))
    run.finish()


if __name__ == "__main__":
    main()
