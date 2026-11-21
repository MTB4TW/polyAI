#!/usr/bin/env python
"""Head-to-head combat games between search variants of a checkpoint and frozen checkpoints (no SimpleAgent).

    python tools/evaluate_search_h2h.py --player-model A.cleanrl_model --mode order \
        --opponent-model B.cleanrl_model --evaluation-id NAME [--max-maps 4]
    python tools/evaluate_search_h2h.py --summarize outputs/evaluations/<prefix>*   # paired table

Each map of eval_validation is played `--repeats-per-map` times with the canonical episode seeds (seed 42), once
with the player in each seat. The opponent always plays the policy alone (argmax). Seat 0 goes through the
training wrapper, as in evaluation. Seat 1 goes through the external opponent hook, the way league snapshots
play (`combat_selfplay.OpponentView`). Both players face the same two paths, so the comparison is fair. The
OpponentView limitation (its "move back to the previous tile" feature is always 0) applies to whoever is in
seat 1.

A game is scored for the player: win 1, timeout 0.5, loss 0. `--summarize` pairs every run with the `none`
(policy-only) run against the same opponent, game by game, and bootstraps the difference over maps. These are
diagnostics, not canonical SimpleAgent evaluations; only eval_validation is used.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import play_vs_agent as P  # noqa: E402  (also sets up the JDK)
from tools.search_variants import MODES, SearchPlayer  # noqa: E402

SCHEMA = "polyai-search-h2h-v1"


def play_game(env, p0, p1, game_map, episode_seed: int) -> dict:
    """p0 plays tribe 0 through the wrapper, p1 tribe 1 through the external hook. Returns tribe 0's result."""
    from pol_env.Tribes.py.combat_selfplay import EXTERNAL_OPPONENT, OpponentView, play_deferred_village_captures, swap_tribes

    te = env.tribes_env
    java = te._env
    state = {}

    def tribe1_turn():
        if "view" not in state:
            state["view"] = OpponentView(env)
        v = state["view"].view
        for _ in range(2000):
            if bool(java.isDone()) or int(java.getActiveTribeID()) != 1:
                return
            obs, actions = te.get_observation(), te.list_actions()
            s_obs, s_actions = swap_tribes(obs, actions, 0, 1)
            v._turn_count = max(0, int(getattr(env, "_turn_count", 0)) - 1)  # turns it has finished
            mask, id_to_raw, _ = v._build_action_mask_and_mapping(s_actions, obs=s_obs)
            ids, valid, _ = v._build_legal_slot_tensors(mask)
            feats = v._build_legal_action_features_padded(ids, valid, id_to_raw, s_actions, s_obs)
            gid = p1.act(v._dict_to_array(s_obs), np.asarray(ids), np.asarray(valid, dtype=bool),
                         np.asarray(feats, dtype=np.float32), raw_map=id_to_raw)
            raw = int(id_to_raw[int(gid)])
            if str(s_actions[raw].get("type", "")).upper() == "END_TURN" and v._queued_village_capture_unit_ids:
                play_deferred_village_captures(te, state["view"], 1)  # CHANGELOG 089
                raw = next(i for i, a in enumerate(te.list_actions()) if a.get("type") == "END_TURN")
            java.stepByIndex(raw)
        raise RuntimeError("Tribe 1 exceeded its action budget")

    def select_opponent():
        java.setOpponentAgent(EXTERNAL_OPPONENT, 0)
        te.opponent_turn_hook = tribe1_turn

    env._select_combat_opponent = select_opponent
    env._level_pool, env._level_pool_size = [str(game_map.path)], 1
    for p in (p0, p1):
        p.reseed(episode_seed * 31 + 7)
    started = time.perf_counter()
    try:
        obs, info = env.reset(seed=episode_seed)
        done, decisions = False, 0
        while not done:
            obs, _r, term, trunc, info = env.step(int(p0.choose(obs, info)))
            done, decisions = term or trunc, decisions + 1
            if decisions > 5000:
                raise RuntimeError("Decision cap exceeded")
    finally:
        te.opponent_turn_hook = None
    return {"tribe0_outcome": str(info.get("combat_outcome")), "turns": int(getattr(env, "_turn_count", 0)),
            "tribe0_decisions": decisions, "seconds": round(time.perf_counter() - started, 2)}


def run(args):
    from tools.phase1_eval_core import stable_seed

    player_info = P.find_models([str(args.player_model)])[0]
    opp_info = P.find_models([str(args.opponent_model)])[0]
    if (player_info.rules, player_info.max_turns) != (opp_info.rules, opp_info.max_turns):
        raise SystemExit("The two checkpoints play different rules or turn limits")
    env, settings = P.make_env(player_info.path, "eval_validation", player_info.rules, player_info.max_turns)
    player = SearchPlayer(player_info, env, args.mode)
    opponent = SearchPlayer(opp_info, env, "none")
    maps = P.load_maps("eval_validation")[: args.max_maps or None]
    out = args.output_root / args.evaluation_id
    out.mkdir(parents=True, exist_ok=True)
    meta = {"schema": SCHEMA, "evaluation_id": args.evaluation_id, "mode": args.mode,
            "player_model": player_info.public(with_sha=True), "opponent_model": opp_info.public(with_sha=True),
            "opponent_mode": "none", "settings": settings, "pool": "eval_validation", "seed": 42,
            "repeats_per_map": args.repeats_per_map, "maps": len(maps), "diagnostic": True}
    (out / "config.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    rows = []
    with (out / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for k, m in enumerate(maps):
            for rep in range(args.repeats_per_map):
                seed = stable_seed(42, "combat_episode", m.csv_sha256, rep)
                for seat in (0, 1):
                    before = dict(player.stats)
                    p0, p1 = (player, opponent) if seat == 0 else (opponent, player)
                    g = play_game(env, p0, p1, m, seed)
                    t0 = g["tribe0_outcome"]
                    res = t0 if seat == 0 else {"win": "loss", "loss": "win"}.get(t0, t0)
                    row = {"map": m.filename, "replicate": rep, "episode_seed": seed, "player_seat": seat, "result": res,
                           "score": {"win": 1.0, "timeout": 0.5}.get(res, 0.0), **g,
                           "player_decisions": player.stats["decisions"] - before.get("decisions", 0),
                           "player_search_moves": player.stats["search"] - before.get("search", 0)}
                    rows.append(row)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
            if (k + 1) % 10 == 0:
                sc = np.mean([r["score"] for r in rows])
                print(f"{args.mode}: {k + 1}/{len(maps)} maps, {len(rows)} games, score {sc:.3f}", flush=True)
    env.close()
    summary = dict(meta, games=len(rows), **rates(rows), search_stats=dict(player.stats))
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("mode", "games", "score", "win", "loss", "timeout")}))


def rates(rows):
    n = max(1, len(rows))
    out = {k: sum(r["result"] == k for r in rows) / n for k in ("win", "loss", "timeout")}
    out["score"] = float(np.mean([r["score"] for r in rows])) if rows else 0.0
    for seat in (0, 1):
        s = [r["score"] for r in rows if r["player_seat"] == seat]
        out[f"score_seat{seat}"] = float(np.mean(s)) if s else None
    dec = sum(r["player_decisions"] for r in rows)
    out["search_share"] = sum(r["player_search_moves"] for r in rows) / dec if dec else 0.0
    return out


def summarize(dirs):
    runs = []
    for d in dirs:
        d = Path(d)
        if not (d / "summary.json").is_file():
            continue
        cfg = json.loads((d / "config.json").read_text())
        rows = [json.loads(l) for l in (d / "episodes.jsonl").read_text().splitlines() if l.strip()]
        opp = cfg["opponent_model"]
        runs.append({"dir": d.name, "mode": cfg["mode"], "opp": f"{opp['run']} {opp['step'] or 'final'}", "rows": rows})
    rng = np.random.default_rng(42)
    print(f"{'opponent':48s} {'mode':15s} {'games':>5s} {'score':>6s} {'seat0':>6s} {'seat1':>6s} {'W/T/L %':>14s} {'search%':>7s}  vs none (paired, 95% CI)")
    for opp in sorted({r["opp"] for r in runs}):
        group = {r["mode"]: r for r in runs if r["opp"] == opp}
        base = group.get("none")
        key = lambda r: (r["map"], r["replicate"], r["player_seat"])
        for mode in MODES:
            if mode not in group:
                continue
            rows = group[mode]["rows"]
            x = rates(rows)
            wtl = f"{100 * x['win']:.0f}/{100 * x['timeout']:.0f}/{100 * x['loss']:.0f}"
            line = (f"{opp:48s} {mode:15s} {len(rows):5d} {x['score']:6.3f} {x['score_seat0'] or 0:6.3f} {x['score_seat1'] or 0:6.3f} "
                    f"{wtl:>14s} {100 * x['search_share']:6.1f}%")
            if base is not None and mode != "none":
                b = {key(r): r["score"] for r in base["rows"]}
                by_map = defaultdict(list)
                for r in rows:
                    if key(r) in b:
                        by_map[r["map"]].append(r["score"] - b[key(r)])
                means = np.array([np.mean(v) for v in by_map.values()])
                if len(means):
                    boots = [rng.choice(means, len(means)).mean() for _ in range(5000)]
                    lo, hi = np.percentile(boots, [2.5, 97.5])
                    better = int((means > 0).sum()); worse = int((means < 0).sum())
                    line += f"  {means.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  maps +{better}/={len(means) - better - worse}/-{worse}"
            print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--player-model", type=Path)
    ap.add_argument("--mode", choices=MODES, default="none")
    ap.add_argument("--opponent-model", type=Path)
    ap.add_argument("--repeats-per-map", type=int, default=2)
    ap.add_argument("--max-maps", type=int)
    ap.add_argument("--evaluation-id")
    ap.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs" / "evaluations")
    ap.add_argument("--summarize", nargs="+")
    args = ap.parse_args()
    if args.summarize:
        return summarize(args.summarize)
    if not (args.player_model and args.opponent_model and args.evaluation_id):
        ap.error("--player-model, --opponent-model and --evaluation-id are required")
    run(args)


if __name__ == "__main__":
    main()
