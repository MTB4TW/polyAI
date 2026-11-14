#!/usr/bin/env python
"""Diagnostic comparison of PPO and feature heuristics, with recorded human references.

The historical filename is retained for CLI compatibility. This is not a
superhuman certification or a substitute for the canonical evaluation suite.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
jdks = sorted((REPO_ROOT / ".runtime").glob("jdk-*/Contents/Home"))
if jdks:
    os.environ["JAVA_HOME"] = str(jdks[-1])
    os.environ["PATH"] = str(jdks[-1] / "bin") + os.pathsep + os.environ.get("PATH", "")

from pol_env.Tribes.py.register_env import TribesGymWrapper
from pol_env.Tribes.py.environment_contract import (
    read_checkpoint_metadata, environment_compatibility_metadata, validate_checkpoint_compatibility,
)
from py_rl.cleanrl.cleanrl.agent_factory import make_agent
from py_rl.cleanrl.cleanrl.obs_normalization import observation_normalization_from_metadata
from tools.phase1_eval_core import PPOPolicy, load_verified_pool, build_schedule, sha256_file
from tools.evaluate_phase1 import run_episode, runtime_provenance
from tools.human_benchmark import load_attempts
from tools.lookahead_policy import TacticalLookaheadPolicy


def human_reference(attempts, record, seed):
    matches = [a for a in attempts if a.get("participant_kind") == "human"
               and a.get("status") == "completed" and a.get("is_first_completed_attempt")
               and a.get("map", {}).get("csv_sha256") == record.csv_sha256
               and a.get("configuration", {}).get("episode_seed") == seed]
    if len(matches) > 1:
        raise ValueError(f"Multiple human first attempts match {record.filename}")
    if not matches:
        return None
    attempt = matches[0]
    return {"attempt_id": attempt["attempt_id"],
            "spt": attempt["result"]["final_t10_spt"],
            "comparison_status": "historical_reference_environment_parity_unverified"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--pool", choices=["human_benchmark", "validation"], default="validation")
    p.add_argument("--max-maps", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    p.add_argument("--spt-weight", type=float, default=3.0)
    p.add_argument("--pop-weight", type=float, default=1.5)
    p.add_argument("--reveal-weight", type=float, default=0.5)
    p.add_argument("--human-results", type=Path, default=REPO_ROOT / "outputs/human_benchmark")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not args.model_path.is_file():
        p.error("--model-path must identify an existing checkpoint file")
    if args.output.exists():
        p.error("Output exists; choose a new report path")
    if args.max_maps is not None and args.max_maps < 1:
        p.error("--max-maps must be positive")
    torch.set_num_threads(1)
    sidecar = read_checkpoint_metadata(str(args.model_path))
    manifest, maps, _ = load_verified_pool(REPO_ROOT, args.pool, max_maps=args.max_maps)
    attempts = load_attempts(args.human_results)
    os.environ.update(POLYVISION_PHASE="1", POLYVISION_COMBAT_MODE="0",
                      POLYVISION_SOLO_NO_OPPONENT_MODE="1", POLYVISION_INFO_MODE="fast",
                      POLYVISION_MAX_LEGAL_ACTIONS=str(sidecar["max_legal_actions"]),
                      POLYVISION_LEVEL_POOL_GLOB=f"levels/phase1_pool_bardur_real/{args.pool}/*.csv")
    device = torch.device(args.device)
    env = TribesGymWrapper()
    rows = []
    try:
        meta = environment_compatibility_metadata(env, actor_mode=sidecar["actor_mode"])
        validate_checkpoint_compatibility(sidecar, meta)
        adapter = SimpleNamespace(single_observation_space=env.observation_space, single_action_space=env.action_space)
        agent = make_agent(adapter, network_arch=sidecar.get("network_arch", "mlp"),
                           actor_mode=sidecar["actor_mode"], max_legal_actions=sidecar["max_legal_actions"],
                           legal_action_feature_dim=sidecar["legal_action_feature_dim"],
                           observation_normalization=observation_normalization_from_metadata(sidecar)).to(device)
        agent.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        agent.eval()
        policies = {"ppo_argmax": PPOPolicy(agent, device, False, None),
                    "tactical_heuristic": TacticalLookaheadPolicy(agent, device, spt_weight=args.spt_weight,
                        pop_weight=args.pop_weight, reveal_weight=args.reveal_weight)}
        schedule = build_schedule(maps, ("ppo_argmax",), 1, args.seed)
        for m, record in zip(maps, schedule):
            # Use the requested exact seed for historical human reference matching.
            from dataclasses import replace
            record = replace(record, episode_seed=args.seed)
            row = {"map": m.filename, "map_sha256": m.csv_sha256, "episode_seed": args.seed,
                   "human_reference": human_reference(attempts, m, args.seed), "policies": {}}
            for name, policy in policies.items():
                row["policies"][name] = run_episode(env, policy, replace(record, policy=name), 300)
            rows.append(row)
            print(m.filename, {k: v["final_spt_t10"] for k, v in row["policies"].items()}, flush=True)
    finally:
        env.close()
    report = {"classification": "diagnostic_noncanonical", "checkpoint_sha256": sha256_file(args.model_path),
              "checkpoint_metadata": sidecar, "environment_metadata": meta, "runtime": runtime_provenance(),
              "pool_identity": manifest["pool_identities"][args.pool],
              "heuristic_weights": {"spt": args.spt_weight, "population": args.pop_weight, "reveal": args.reveal_weight},
              "human_comparison_note": "Historical references only; environment parity must be established before human-level claims.",
              "episodes": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2)
    print(f"Diagnostic report: {args.output}")

if __name__ == "__main__":
    main()
