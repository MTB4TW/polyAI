#!/usr/bin/env python
"""Canonical PolyVision combat (Phase 2) evaluator: win/loss/timeout against the scripted opponent.

    python tools/evaluate_combat.py --model-path runs/.../NAME.cleanrl_model

plays every map of the `eval_validation` pool (120 maps never used for training) 5 times per policy
(`ppo_argmax`, `ppo_sampled`, `random_legal`) with seed 42, against SimpleAgent in CAPITALS mode, and writes
`outputs/evaluations/<id>/` (config, episodes, per-map rates, summary). The rules and the turn limit are read
from the checkpoint's environment version, and the checkpoint is checked against the environment before any
game is played. Compare two evaluations map by map with `tools/compare_combat_evals.py`.

`eval_test` is held out: it needs `--pool eval_test --confirm-test` and must only be used when a person asks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.combat_eval_core import (  # noqa: E402
    ALL_POLICIES, MANIFEST_REL, OPTIONAL_POLICIES, OUTCOMES, SCHEMA_VERSION, TEST_POOLS, aggregate, build_schedule, is_canonical,
    load_verified_pool, paired_difference, per_map_rates,
)
from tools.phase1_eval_core import PPOPolicy, RandomLegalPolicy, json_safe, sha256_file, write_csv  # noqa: E402

HORIZON_RE = re.compile(r"horizon_(\d+)")
DIAGNOSTICS = {  # episode field <- end-of-game info key (engine-side facts, not used by the policy)
    "enemy_capitals_held": "combat_end_enemy_capitals_held",
    "min_unit_dist_to_enemy_capital": "combat_end_min_unit_dist_to_enemy_capital",
    "own_cities": "combat_end_own_cities",
    "enemy_cities": "combat_end_enemy_cities",
    "first_contact_tick": "combat_first_contact_tick",
    "units_ours": "combat_end_raw_units_ours",
    "units_theirs": "combat_end_raw_units_theirs",
    "kills_ours": "combat_end_raw_nKills_ours",
    "kills_theirs": "combat_end_raw_nKills_theirs",
    "riders_ours": "combat_end_raw_units_RIDER_ours",
    "warriors_ours": "combat_end_raw_units_WARRIOR_ours",
    "swordsmen_ours": "combat_end_raw_units_SWORDMAN_ours",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def environment_settings(checkpoint_meta, rules_arg, max_turns_arg):
    """(rules, max_turns, max_legal_actions) for the run: from the checkpoint when there is one."""
    if checkpoint_meta is None:
        return rules_arg, int(max_turns_arg), 512
    from py_rl.cleanrl.cleanrl.init_from import rules_of_version

    version = str(checkpoint_meta.get("phase1_environment_version", ""))
    if not version.startswith("combat_"):
        raise SystemExit(f"Not a combat checkpoint (environment version {version!r})")
    m = HORIZON_RE.search(version)
    if not m:
        raise SystemExit(f"Cannot read the turn limit from environment version {version!r}")
    return rules_of_version(version), int(m.group(1)), int(checkpoint_meta.get("max_legal_actions", 512))


class SearchAssistedPolicy:
    """ppo_search: the combat tactical search (CombatTactics) when it finds a tactical sequence that beats stopping
    now, otherwise the policy's highest-logit action. Measures what search adds at play time."""

    name = "ppo_search"

    def __init__(self, agent, device, env, episode_seed):
        self.ppo = PPOPolicy(agent, device, False, None)
        self.env, self.episode_seed, self.decision = env, int(episode_seed), 0
        self.search_moves = 0

    def choose_action(self, obs, info):
        from pol_env.Tribes.py import search_teacher

        ids = np.asarray(info["legal_global_ids_padded"]).reshape(-1)
        valid = np.asarray(info["legal_action_valid_mask"], dtype=bool).reshape(-1)
        seed = search_teacher.decision_seed(self.episode_seed, self.decision)
        self.decision += 1
        slot = search_teacher.tactical_slot(self.env.tribes_env._env, self.env.tribes_env._gateway, ids, valid,
                                            self.env._current_legal_id_to_raw_index, seed)
        if slot >= 0:
            self.search_moves += 1
            return int(ids[slot]), int(slot)
        return self.ppo.choose_action(obs, info)


def make_policy(name, agent, device, policy_seed, env=None, episode_seed=0):
    if name == "ppo_search":
        return SearchAssistedPolicy(agent, device, env, episode_seed)
    if name == "ppo_argmax":
        return PPOPolicy(agent, device, False, None)
    if name == "ppo_sampled":
        return PPOPolicy(agent, device, True, policy_seed)
    if name == "random_legal":
        return RandomLegalPolicy(policy_seed)
    raise ValueError(name)


def run_game(env, policy, game, max_decisions: int) -> dict:
    # The runner pins the map; the policy only ever sees observations and legal actions.
    env._level_pool = [game.map.path]
    env._level_pool_size = 1
    started = time.perf_counter()
    obs, info = env.reset(seed=game.episode_seed)
    actual = str(Path(env._current_level_file).resolve())
    if actual != game.map.path or sha256_file(Path(actual)) != game.map.csv_sha256:
        raise RuntimeError(f"Scheduled map mismatch: expected {game.map.path}, got {actual}")
    decisions = 0
    terminated = truncated = False
    while not (terminated or truncated):
        ids = np.asarray(info["legal_global_ids_padded"], dtype=np.int64).reshape(-1)
        valid = np.asarray(info["legal_action_valid_mask"], dtype=bool).reshape(-1)
        gid, slot = policy.choose_action(obs, info)
        if slot < 0 or slot >= ids.size or not valid[slot] or int(ids[slot]) != int(gid):
            raise RuntimeError("Policy selected an invalid legal slot")
        obs, _reward, terminated, truncated, info = env.step(int(gid))
        decisions += 1
        if info.get("fallback_to_end_turn") or info.get("illegal_sampled_global_id"):
            raise RuntimeError("Unexpected illegal-action fallback during evaluation")
        if decisions > max_decisions:
            raise RuntimeError(f"Game exceeded the safety cap of {max_decisions} decisions")
    outcome = str(info.get("combat_outcome"))
    if outcome not in OUTCOMES:
        raise RuntimeError(f"Game ended without a combat outcome: {outcome!r}")
    row = {
        "schema_version": SCHEMA_VERSION, "policy": game.policy, "map_filename": game.map.filename,
        "map_csv_sha256": game.map.csv_sha256, "map_seed": game.map.map_seed, "our_tribe": game.map.our_tribe, "opponent_tribe": game.map.opponent_tribe,
        "replicate_index": game.replicate_index, "episode_seed": game.episode_seed, "policy_seed": game.policy_seed,
        "outcome": outcome, "win": int(outcome == "win"), "loss": int(outcome == "loss"),
        "timeout": int(outcome == "timeout"), "end_turn": int(info.get("turn_count", info.get("turn", -1))),
        "decisions": decisions, "duration_seconds": float(time.perf_counter() - started),
    }
    for field, key in DIAGNOSTICS.items():
        row[field] = info.get(key)
    return json_safe(row)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", type=Path, help="Combat checkpoint (needed for the ppo_* policies)")
    p.add_argument("--pool", choices=["eval_validation", "eval_test"], default="eval_validation")
    p.add_argument("--confirm-test", action="store_true", help="Required to evaluate the held-out eval_test pool")
    p.add_argument("--policy", action="append", choices=ALL_POLICIES + OPTIONAL_POLICIES,
                   help="Repeat for a narrower policy set; ppo_search (search at play time) is opt-in")
    p.add_argument("--repeats-per-map", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-maps", type=int)
    p.add_argument("--rules", default="tribes_2020", help="Without --model-path: POLYVISION_RULES to play")
    p.add_argument("--max-turns", type=int, default=20, help="Without --model-path: turn limit")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-decisions-per-game", type=int, default=5000)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs/evaluations")
    p.add_argument("--evaluation-id")
    return p.parse_args()


def print_summary(pool, n_maps, summary, paired, out_dir):
    print("\n" + "=" * 72 + f"\nPOLYVISION COMBAT — {pool.upper()} EVALUATION ({n_maps} maps)\n" + "=" * 72)
    print(f"{'Policy':14s} {'Games':>6s} {'Win':>7s} {'95% CI':>16s} {'Loss':>7s} {'Timeout':>8s} {'Win turn':>9s}")
    for policy, v in summary.items():
        ci = v["win_rate_ci95"]
        turn = f"{v['mean_win_turn']:.1f}" if v["mean_win_turn"] is not None else "-"
        print(f"{policy:14s} {v['games']:6d} {v['win_rate']:7.1%} [{ci[0]:6.1%}, {ci[1]:6.1%}] "
              f"{v['loss_rate']:7.1%} {v['timeout_rate']:8.1%} {turn:>9s}")
    for key, v in paired.items():
        if v:
            print(f"{key}: win rate {v['mean_difference']:+.1%} [{v['ci95'][0]:+.1%}, {v['ci95'][1]:+.1%}], "
                  f"{v['maps_better']} better / {v['maps_equal']} equal / {v['maps_worse']} worse maps")
    print(f"\nResults: {out_dir}")


def main():
    args = parse_args()
    if args.pool in TEST_POOLS and not args.confirm_test:
        raise SystemExit("Refusing the held-out eval_test pool without --confirm-test (evaluate it only when asked)")
    if args.repeats_per_map < 1 or (args.max_maps is not None and args.max_maps < 1):
        raise SystemExit("Replicate and map counts must be positive")
    policies = tuple(dict.fromkeys(args.policy or (ALL_POLICIES if args.model_path else ("random_legal",))))
    if any(p.startswith("ppo_") for p in policies) and not args.model_path:
        raise SystemExit("--model-path is required for the ppo_* policies")

    from pol_env.Tribes.py.environment_contract import (
        environment_compatibility_metadata, read_checkpoint_metadata, validate_checkpoint_compatibility,
    )

    manifest, maps, manifest_path = load_verified_pool(REPO_ROOT, args.pool, args.max_maps)
    canonical = is_canonical(args.pool, len(maps), args.repeats_per_map, args.seed, policies, args.max_maps)
    label = "canonical" if canonical else "partial-diagnostic-noncanonical"
    evaluation_id = args.evaluation_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_combat_{args.pool}_{label}"
    out_dir = args.output_root / evaluation_id
    if out_dir.exists():
        raise SystemExit(f"Output directory already exists: {out_dir}")
    schedule = build_schedule(maps, policies, args.repeats_per_map, args.seed)

    model_path = args.model_path.resolve() if args.model_path else None
    checkpoint_meta = read_checkpoint_metadata(str(model_path)) if model_path else None
    sidecar = Path(str(model_path) + ".action_interface.json") if model_path else None
    rules, max_turns, max_legal = environment_settings(checkpoint_meta, args.rules, args.max_turns)
    from pol_env.Tribes.py.environment_contract import fog_memory_of_version
    fog_memory = (fog_memory_of_version(checkpoint_meta.get("phase1_environment_version")) if checkpoint_meta
                  else os.environ.get("POLYVISION_FOG_MEMORY", "0").strip() == "1")
    for key in [k for k in os.environ if k.startswith("POLYVISION_COMBAT_REWARD_")]:
        del os.environ[key]  # outcomes do not depend on reward weights; keep the run's config clean
    os.environ.update({
        "POLYVISION_PHASE": "2", "POLYVISION_COMBAT_MODE": "1", "POLYVISION_SOLO_NO_OPPONENT_MODE": "0",
        "POLYVISION_LEVEL_POOL_GLOB": f"levels/phase2_pool_1v1/{args.pool}/*.csv",
        "POLYVISION_LEVEL_SELECTION_MODE": "round_robin", "POLYVISION_INFO_MODE": "fast",
        "POLYVISION_MAX_TURNS": str(max_turns), "POLYVISION_RULES": rules,
        # Combat fog memory follows the checkpoint (it changes the observation); without one, the caller's setting.
        "POLYVISION_FOG_MEMORY": "1" if fog_memory else "0",
        "POLYVISION_MAX_LEGAL_ACTIONS": str(max_legal),
    })

    from pol_env.Tribes.py.register_env import TribesGymWrapper
    from py_rl.cleanrl.cleanrl.agent_factory import make_agent
    from py_rl.cleanrl.cleanrl.obs_normalization import observation_normalization_from_metadata
    from tools.evaluate_phase1 import runtime_provenance

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    env = TribesGymWrapper()
    agent, env_meta = None, None
    try:
        if checkpoint_meta:
            env_meta = environment_compatibility_metadata(env, actor_mode=checkpoint_meta["actor_mode"],
                                                          max_legal_actions=checkpoint_meta["max_legal_actions"])
            validate_checkpoint_compatibility(checkpoint_meta, env_meta)
            adapter = SimpleNamespace(single_observation_space=env.observation_space, single_action_space=env.action_space)
            agent = make_agent(adapter, network_arch=checkpoint_meta.get("network_arch", "mlp"),
                               actor_mode=checkpoint_meta["actor_mode"], max_legal_actions=checkpoint_meta["max_legal_actions"],
                               legal_action_feature_dim=checkpoint_meta["legal_action_feature_dim"],
                               observation_normalization=observation_normalization_from_metadata(checkpoint_meta)).to(device)
            agent.load_state_dict(torch.load(model_path, map_location=device))
            agent.eval()
        started = utc_now()
        episodes = []
        for i, game in enumerate(schedule, 1):
            policy = make_policy(game.policy, agent, device, game.policy_seed, env, game.episode_seed)
            row = run_game(env, policy, game, args.max_decisions_per_game)
            if isinstance(policy, SearchAssistedPolicy):
                row["search_moves"] = policy.search_moves
            episodes.append(row)
            if args.progress_every and (i % args.progress_every == 0 or i == len(schedule)):
                print(f"{game.policy}: {i}/{len(schedule)} games", flush=True)
        summary, per_map = aggregate(episodes, args.seed)
        rates = per_map_rates(episodes)
        paired = {}
        if "ppo_argmax" in rates and "random_legal" in rates:
            paired["ppo_argmax_vs_random_legal"] = paired_difference(rates["ppo_argmax"], rates["random_legal"], "win", args.seed)
        if "ppo_search" in rates and "ppo_argmax" in rates:
            paired["ppo_search_vs_ppo_argmax"] = paired_difference(rates["ppo_search"], rates["ppo_argmax"], "win", args.seed)
        if "ppo_sampled" in rates and "random_legal" in rates:
            paired["ppo_sampled_vs_random_legal"] = paired_difference(rates["ppo_sampled"], rates["random_legal"], "win", args.seed)
        config = {
            "evaluation_schema_version": SCHEMA_VERSION, "evaluation_id": evaluation_id, "status": "complete",
            "canonical": canonical, "classification": label, "started_utc": started, "ended_utc": utc_now(),
            "pool": args.pool, "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
            "manifest_sha256": sha256_file(manifest_path), "evaluated_map_count": len(maps),
            "ordered_maps": [m.__dict__ for m in maps], "evaluation_seed": args.seed,
            "repeats_per_map": args.repeats_per_map, "policies": list(policies), "opponent": "SimpleAgent (seeded by the episode seed)",
            "game_mode": "CAPITALS", "rules": rules, "fog_memory": bool(fog_memory), "max_turns": max_turns,
            "schedule_rules": "episode seed depends on map and replicate only, so every policy and evaluation with the same seed meets the same games; policy seed is policy-specific",
            "policy_semantics": {"ppo_argmax": "highest-logit valid slot", "ppo_sampled": "sampled from the policy with a per-game generator",
                                 "random_legal": "uniform over valid slots",
                                 "ppo_search": "combat tactical search (depth 3, beam 8, fogged) where it beats stopping, else highest-logit slot"},
            "model_path": str(model_path) if model_path else None,
            "checkpoint_sha256": sha256_file(model_path) if model_path else None,
            "sidecar_sha256": sha256_file(sidecar) if sidecar else None,
            "checkpoint_metadata": checkpoint_meta, "environment_interface_metadata": env_meta,
            "runtime": runtime_provenance(),
            "polyvision_environment": {k: v for k, v in os.environ.items() if k.startswith("POLYVISION_")},
        }
        full_summary = {"evaluation_schema_version": SCHEMA_VERSION, "evaluation_id": evaluation_id, "canonical": canonical,
                        "pool": args.pool, "rules": rules, "max_turns": max_turns,
                        "primary_metric": "win rate against SimpleAgent (per-map mean over replicates, mean over maps)",
                        "bootstrap": {"samples": 5000, "unit": "map"}, "policies": summary, "paired": paired}
        out_dir.mkdir(parents=True, exist_ok=False)
        (out_dir / "config.json").write_text(json.dumps(json_safe(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with (out_dir / "episodes.jsonl").open("w", encoding="utf-8") as fh:
            for row in episodes:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        write_csv(out_dir / "per_map.csv", per_map)
        (out_dir / "summary.json").write_text(json.dumps(json_safe(full_summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_csv(out_dir / "summary.csv", [{"policy": p, **{k: v for k, v in s.items() if not isinstance(v, dict)}}
                                            for p, s in summary.items()])
        print_summary(args.pool, len(maps), summary, paired, out_dir)
    finally:
        env.close()


if __name__ == "__main__":
    main()
