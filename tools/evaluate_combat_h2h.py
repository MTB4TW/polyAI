#!/usr/bin/env python
"""Head-to-head combat (Phase 2) evaluation: checkpoint A against checkpoint B, both seats.

    python tools/evaluate_combat_h2h.py --model-a runs/.../A.cleanrl_model --model-b runs/.../B.cleanrl_model

Against SimpleAgent the strongest combat models are near the ceiling (95%+ wins), so this measures strength
against another trained model instead. Every map of the `eval_validation` pool is played `--repeats-per-map`
times in each seat order: A as tribe 0 (moving first) with B as tribe 1, and B as tribe 0 with A as tribe 1.
Both orders use the same episode seeds as the canonical evaluator, so seat effects cancel map by map. Results
are from A's point of view: A's win / loss / timeout rates (B's wins are A's losses), per seat and overall,
and the margin (A's win rate minus B's) with bootstrap CIs over maps.

The tribe-1 player is driven exactly as a self-play league snapshot (`pol_env/Tribes/py/combat_selfplay.py`):
the Java engine leaves tribe 1's turn to Python, and its fogged observation and legal actions are relabelled so
it sees itself as tribe 0, encoded with the same code as the learner's inputs.

`--mode sampled` (default) samples both players' actions with per-game seeds; `--mode argmax` plays the highest-
logit action (then the engine's randomness is all that differs between replicates). A full run (120 maps,
5 replicates, 2 seats, seed 42, sampled) is labelled `h2h-canonical`; anything smaller is a diagnostic.

Both checkpoints must have the same environment version. Rules and the turn limit come from A's checkpoint.
`eval_test` is held out: it needs `--pool eval_test --confirm-test` and must only be used when a person asks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.combat_eval_core import (  # noqa: E402
    OUTCOMES, TEST_POOLS, CombatMap, aggregate, load_verified_pool, paired_difference, per_map_rates,
)
from tools.phase1_eval_core import json_safe, sha256_file, stable_seed, write_csv  # noqa: E402

SCHEMA_VERSION = "polyvision-combat-h2h-v1"
SEATS = ("a_tribe0", "a_tribe1")  # a_tribe0: A plays tribe 0 (moves first), B tribe 1; a_tribe1: the reverse
MODES = ("sampled", "argmax")
CANONICAL_H2H = {"pool": "eval_validation", "maps": 120, "repeats_per_map": 5, "seed": 42, "mode": "sampled"}
FLIP = {"win": "loss", "loss": "win", "timeout": "timeout"}


@dataclass(frozen=True)
class H2HGame:
    seat: str
    map: CombatMap
    replicate_index: int
    episode_seed: int
    seed_tribe0: int
    seed_tribe1: int


def build_h2h_schedule(maps: Sequence[CombatMap], repeats: int, seed: int) -> list[H2HGame]:
    """Every map, `repeats` times, in both seat orders. The episode seed is the canonical evaluator's (map and
    replicate only), so both seat orders, and every other evaluation with this seed, meet the same games."""
    games = []
    for seat in SEATS:
        for m in maps:
            for rep in range(int(repeats)):
                episode_seed = stable_seed(seed, "combat_episode", m.csv_sha256, rep)
                games.append(H2HGame(seat, m, rep, episode_seed,
                                     stable_seed(seed, "h2h_policy", seat, 0, m.csv_sha256, rep),
                                     stable_seed(seed, "h2h_policy", seat, 1, m.csv_sha256, rep)))
    return games


def outcome_for_a(seat: str, tribe0_outcome: str) -> str:
    """The engine reports the outcome for tribe 0; turn it into A's outcome."""
    if tribe0_outcome not in OUTCOMES:
        raise ValueError(f"Unknown outcome {tribe0_outcome!r}")
    return tribe0_outcome if seat == "a_tribe0" else FLIP[tribe0_outcome]


def is_h2h_canonical(pool: str, n_maps: int, repeats: int, seed: int, mode: str, max_maps) -> bool:
    c = CANONICAL_H2H
    return (pool == c["pool"] and n_maps == c["maps"] and max_maps is None and repeats == c["repeats_per_map"]
            and seed == c["seed"] and mode == c["mode"])


def summarize_h2h(episodes: Sequence[Mapping], seed: int) -> tuple[dict, list[dict]]:
    """A's rates per seat and overall (per-map mean over replicates and seats, CI over maps), and the margin
    (A's win rate minus B's) per map. Episode rows carry `policy` = seat and `outcome` from A's view."""
    by_seat, rows = aggregate(episodes, seed)
    overall_eps = [{**e, "policy": "overall"} for e in episodes]
    overall, overall_rows = aggregate(overall_eps, seed)
    rates = per_map_rates(overall_eps)["overall"]
    a_win = {m: {"rate": c["win"]} for m, c in rates.items()}
    b_win = {m: {"rate": c["loss"]} for m, c in rates.items()}
    margin = paired_difference(a_win, b_win, "rate", seed)
    seat_gap = None
    seat_rates = per_map_rates(episodes)
    if all(s in seat_rates for s in SEATS):
        # How much moving first is worth to A (positive: A wins more as tribe 0).
        seat_gap = paired_difference(seat_rates["a_tribe0"], seat_rates["a_tribe1"], "win", seed)
    summary = {"overall": overall["overall"], "by_seat": by_seat, "margin_a_minus_b_win": margin,
               "seat_gap_a_win_tribe0_minus_tribe1": seat_gap}
    return summary, overall_rows + rows


def environment_settings(meta) -> tuple[str, int, int]:
    from tools.evaluate_combat import environment_settings as canonical_settings

    return canonical_settings(meta, None, 0)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-a", type=Path, required=True, help="Checkpoint A (results are from A's point of view)")
    p.add_argument("--model-b", type=Path, required=True, help="Checkpoint B (the reference opponent)")
    p.add_argument("--pool", choices=["eval_validation", "eval_test"], default="eval_validation")
    p.add_argument("--confirm-test", action="store_true", help="Required to evaluate the held-out eval_test pool")
    p.add_argument("--repeats-per-map", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-maps", type=int)
    p.add_argument("--mode", choices=MODES, default="sampled")
    p.add_argument("--max-decisions-per-game", type=int, default=5000, help="Safety cap on tribe-0 decisions")
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs/evaluations")
    p.add_argument("--evaluation-id")
    return p.parse_args(argv)


def play_game(env, tribe0, tribe1, game: H2HGame, sampled: bool, max_decisions: int) -> tuple[str, dict, int]:
    """One game: `tribe0` (a SnapshotOpponent, driven through the env's own observation) against `tribe1`
    (a SnapshotOpponent playing tribe 1's turns through the opponent hook). Returns (tribe-0 outcome, info, n)."""
    from pol_env.Tribes.py.combat_selfplay import EXTERNAL_OPPONENT, OpponentView, play_external_turn
    from tools.phase1_eval_core import PPOPolicy

    env._level_pool = [game.map.path]
    env._level_pool_size = 1
    tribe1.deterministic = not sampled
    tribe1.reseed(game.seed_tribe1)
    state = {}

    def hook():
        if "view" not in state:  # built at tribe 1's first turn, after this game's reset bookkeeping
            state["view"] = OpponentView(env)
        play_external_turn(env, tribe1, state["view"])

    env.tribes_env._env.setOpponentAgent(EXTERNAL_OPPONENT, 0)
    env.tribes_env.opponent_turn_hook = hook
    policy = PPOPolicy(tribe0.agent, torch.device("cpu"), sampled, game.seed_tribe0 if sampled else None)
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
            raise RuntimeError("Tribe-0 policy selected an invalid legal slot")
        obs, _r, terminated, truncated, info = env.step(int(gid))
        decisions += 1
        if info.get("fallback_to_end_turn") or info.get("illegal_sampled_global_id"):
            raise RuntimeError("Unexpected illegal-action fallback during evaluation")
        if decisions > max_decisions:
            raise RuntimeError(f"Game exceeded the safety cap of {max_decisions} decisions")
    outcome = str(info.get("combat_outcome"))
    if outcome not in OUTCOMES:
        raise RuntimeError(f"Game ended without a combat outcome: {outcome!r}")
    return outcome, info, decisions


def print_summary(pool, n_maps, summary, out_dir):
    print("\n" + "=" * 72 + f"\nPOLYVISION COMBAT HEAD-TO-HEAD — {pool.upper()} ({n_maps} maps), A's point of view\n" + "=" * 72)
    print(f"{'':10s} {'Games':>6s} {'A wins':>7s} {'95% CI':>16s} {'B wins':>7s} {'Timeout':>8s}")
    rows = [("overall", summary["overall"])] + list(summary["by_seat"].items())
    for name, v in rows:
        ci = v["win_rate_ci95"]
        print(f"{name:10s} {v['games']:6d} {v['win_rate']:7.1%} [{ci[0]:6.1%}, {ci[1]:6.1%}] "
              f"{v['loss_rate']:7.1%} {v['timeout_rate']:8.1%}")
    m = summary["margin_a_minus_b_win"]
    if m:
        print(f"margin (A wins - B wins): {m['mean_difference']:+.1%} [{m['ci95'][0]:+.1%}, {m['ci95'][1]:+.1%}], "
              f"{m['maps_better']} maps A ahead / {m['maps_equal']} level / {m['maps_worse']} B ahead")
    print(f"\nResults: {out_dir}")


def main(argv=None):
    args = parse_args(argv)
    if args.pool in TEST_POOLS and not args.confirm_test:
        raise SystemExit("Refusing the held-out eval_test pool without --confirm-test (evaluate it only when asked)")
    if args.repeats_per_map < 1 or (args.max_maps is not None and args.max_maps < 1):
        raise SystemExit("Replicate and map counts must be positive")

    from pol_env.Tribes.py.environment_contract import read_checkpoint_metadata

    path_a, path_b = args.model_a.resolve(), args.model_b.resolve()
    meta_a, meta_b = read_checkpoint_metadata(str(path_a)), read_checkpoint_metadata(str(path_b))
    if meta_a.get("phase1_environment_version") != meta_b.get("phase1_environment_version"):
        raise SystemExit("A and B have different environment versions: "
                         f"{meta_a.get('phase1_environment_version')!r} vs {meta_b.get('phase1_environment_version')!r}")
    manifest, maps, manifest_path = load_verified_pool(REPO_ROOT, args.pool, args.max_maps)
    canonical = is_h2h_canonical(args.pool, len(maps), args.repeats_per_map, args.seed, args.mode, args.max_maps)
    label = "h2h-canonical" if canonical else "h2h-partial-diagnostic-noncanonical"
    evaluation_id = args.evaluation_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_combat_h2h_{args.pool}_{label}"
    out_dir = args.output_root / evaluation_id
    if out_dir.exists():
        raise SystemExit(f"Output directory already exists: {out_dir}")
    schedule = build_h2h_schedule(maps, args.repeats_per_map, args.seed)

    rules, max_turns, max_legal = environment_settings(meta_a)
    from pol_env.Tribes.py.environment_contract import fog_memory_of_version
    fog_memory = fog_memory_of_version(meta_a.get("phase1_environment_version"))
    for key in [k for k in os.environ if k.startswith(("POLYVISION_COMBAT_REWARD_", "POLYVISION_COMBAT_LEAGUE",
                                                       "POLYVISION_PRIVILEGED_OBS"))]:
        del os.environ[key]  # no league (the hook below sets the opponent), no reward weights, no critic input
    os.environ.update({
        "POLYVISION_PHASE": "2", "POLYVISION_COMBAT_MODE": "1", "POLYVISION_SOLO_NO_OPPONENT_MODE": "0",
        "POLYVISION_LEVEL_POOL_GLOB": f"levels/phase2_pool_1v1/{args.pool}/*.csv",
        "POLYVISION_LEVEL_SELECTION_MODE": "round_robin", "POLYVISION_INFO_MODE": "fast",
        "POLYVISION_MAX_TURNS": str(max_turns), "POLYVISION_RULES": rules,
        "POLYVISION_FOG_MEMORY": "1" if fog_memory else "0", "POLYVISION_MAX_LEGAL_ACTIONS": str(max_legal),
    })

    from pol_env.Tribes.py.combat_selfplay import SnapshotOpponent
    from pol_env.Tribes.py.register_env import TribesGymWrapper
    from tools.evaluate_phase1 import runtime_provenance

    env = TribesGymWrapper()
    try:
        player_a = SnapshotOpponent(str(path_a), env)  # validates each checkpoint against this environment
        player_b = SnapshotOpponent(str(path_b), env)
        sampled = args.mode == "sampled"
        started = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        episodes = []
        for i, game in enumerate(schedule, 1):
            tribe0, tribe1 = (player_a, player_b) if game.seat == "a_tribe0" else (player_b, player_a)
            t0 = time.perf_counter()
            outcome, info, decisions = play_game(env, tribe0, tribe1, game, sampled, args.max_decisions_per_game)
            a_out = outcome_for_a(game.seat, outcome)
            episodes.append(json_safe({
                "schema_version": SCHEMA_VERSION, "policy": game.seat, "map_filename": game.map.filename,
                "map_csv_sha256": game.map.csv_sha256, "map_seed": game.map.map_seed,
                "our_tribe": game.map.our_tribe, "opponent_tribe": game.map.opponent_tribe,
                "replicate_index": game.replicate_index, "episode_seed": game.episode_seed,
                "seed_tribe0": game.seed_tribe0, "seed_tribe1": game.seed_tribe1,
                "tribe0_outcome": outcome, "outcome": a_out, "win": int(a_out == "win"), "loss": int(a_out == "loss"),
                "timeout": int(a_out == "timeout"), "end_turn": int(info.get("turn_count", info.get("turn", -1))),
                "tribe0_decisions": decisions, "duration_seconds": float(time.perf_counter() - t0),
            }))
            if args.progress_every and (i % args.progress_every == 0 or i == len(schedule)):
                print(f"{game.seat}: {i}/{len(schedule)} games", flush=True)
        env.tribes_env.opponent_turn_hook = None
        summary, per_map = summarize_h2h(episodes, args.seed)
        config = {
            "evaluation_schema_version": SCHEMA_VERSION, "evaluation_id": evaluation_id, "status": "complete",
            "canonical": canonical, "classification": label, "started_utc": started,
            "ended_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "pool": args.pool, "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
            "manifest_sha256": sha256_file(manifest_path), "evaluated_map_count": len(maps),
            "ordered_maps": [m.__dict__ for m in maps], "evaluation_seed": args.seed,
            "repeats_per_map": args.repeats_per_map, "seats": list(SEATS), "mode": args.mode,
            "game_mode": "CAPITALS", "rules": rules, "fog_memory": bool(fog_memory), "max_turns": max_turns,
            "schedule_rules": "episode seed depends on map and replicate only (as in tools/evaluate_combat.py); both seat orders play every (map, replicate); per-player policy seeds depend on seat, tribe, map and replicate",
            "point_of_view": "A: win = A won, loss = B won",
            "model_a": str(path_a), "model_a_sha256": sha256_file(path_a), "model_a_metadata": meta_a,
            "model_b": str(path_b), "model_b_sha256": sha256_file(path_b), "model_b_metadata": meta_b,
            "runtime": runtime_provenance(),
            "polyvision_environment": {k: v for k, v in os.environ.items() if k.startswith("POLYVISION_")},
        }
        full_summary = {"evaluation_schema_version": SCHEMA_VERSION, "evaluation_id": evaluation_id,
                        "canonical": canonical, "pool": args.pool, "mode": args.mode, "rules": rules,
                        "max_turns": max_turns, "model_a": str(path_a), "model_b": str(path_b),
                        "primary_metric": "A's win rate against B over both seats (per-map mean, mean over maps)",
                        "bootstrap": {"samples": 5000, "unit": "map"}, **summary}
        out_dir.mkdir(parents=True, exist_ok=False)
        (out_dir / "config.json").write_text(json.dumps(json_safe(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with (out_dir / "episodes.jsonl").open("w", encoding="utf-8") as fh:
            for row in episodes:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        write_csv(out_dir / "per_map.csv", per_map)
        (out_dir / "summary.json").write_text(json.dumps(json_safe(full_summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print_summary(args.pool, len(maps), summary, out_dir)
        return out_dir
    finally:
        env.close()


if __name__ == "__main__":
    main()
