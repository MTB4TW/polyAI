#!/usr/bin/env python
"""Interactive HTML replay of one combat (Phase 2) game: our policy against SimpleAgent.

    python tools/export_combat_replay.py --model-path runs/.../NAME.cleanrl_model \
        --map map_2p_030048.csv --episode-seed 1028470471 --output-html replays/game.html

The game is set up exactly like `tools/evaluate_combat.py` (rules, turn limit and fog memory from the checkpoint's
environment version, CAPITALS mode, SimpleAgent opponent), so an (map, episode seed) pair taken from an evaluation's
`episodes.jsonl` replays that game. `--replicate N` derives the episode seed of replicate N of the canonical schedule
(seed 42) instead. The policy plays argmax by default (`--sampled` samples).

After every decision the exporter records the true board (full visibility) and the board our policy sees (fogged).
The opponent plays inside our END_TURN step, so its moves, attacks, kills and captures are reconstructed by comparing
the true board before and after. Only the validation pool is accepted (never eval_test).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The engine needs the bundled JDK 11 (in this checkout's .runtime, or the main checkout's when run from a git
# worktree under .claude/worktrees/), not a newer system Java.
_runtimes = [REPO_ROOT / ".runtime"] + ([REPO_ROOT.parents[2] / ".runtime"] if REPO_ROOT.parent.name == "worktrees" else [])
_jdks = [j for r in _runtimes for j in sorted(r.glob("jdk-*/Contents/Home"))]
if _jdks:
    os.environ["JAVA_HOME"] = str(_jdks[-1])
    os.environ["PATH"] = str(_jdks[-1] / "bin") + os.pathsep + os.environ.get("PATH", "")

from tools.combat_eval_core import POOL_ROOT_REL, load_verified_pool  # noqa: E402
from tools.phase1_eval_core import PPOPolicy, stable_seed  # noqa: E402

UNIT_NAMES = ["Warrior", "Rider", "Defender", "Swordsman", "Archer", "Catapult", "Knight", "Mind Bender",
              "Boat", "Ship", "Battleship", "Giant", "Cloak", "Dagger"]
UNIT_LETTERS = ["W", "R", "D", "S", "A", "C", "K", "M", "B", "Sh", "Bs", "G", "Cl", "Dg"]
TERRAIN_NAMES = ["plain", "shallow", "deep", "mountain", "village", "city", "forest", "fog"]
FOG = 7
TRIBE_NAMES = {0: "Xin-xi", 1: "Imperius", 2: "Bardur", 3: "Oumaji"}


def unit_name(t: int) -> str:
    return UNIT_NAMES[t] if 0 <= t < len(UNIT_NAMES) else f"unit{t}"


def configure_environment(checkpoint_meta: dict, pool: str) -> dict:
    """Same env vars as tools/evaluate_combat.py, derived from the checkpoint's environment version."""
    from pol_env.Tribes.py.environment_contract import fog_memory_of_version
    from tools.evaluate_combat import environment_settings

    rules, max_turns, max_legal = environment_settings(checkpoint_meta, None, 20)
    fog_memory = fog_memory_of_version(checkpoint_meta.get("phase1_environment_version"))
    for key in [k for k in os.environ if k.startswith("POLYVISION_COMBAT_REWARD_")]:
        del os.environ[key]
    settings = {
        "POLYVISION_PHASE": "2", "POLYVISION_COMBAT_MODE": "1", "POLYVISION_SOLO_NO_OPPONENT_MODE": "0",
        "POLYVISION_LEVEL_POOL_GLOB": f"levels/phase2_pool_1v1/{pool}/*.csv",
        "POLYVISION_LEVEL_SELECTION_MODE": "round_robin", "POLYVISION_INFO_MODE": "fast",
        "POLYVISION_MAX_TURNS": str(max_turns), "POLYVISION_RULES": rules,
        "POLYVISION_FOG_MEMORY": "1" if fog_memory else "0", "POLYVISION_MAX_LEGAL_ACTIONS": str(max_legal),
    }
    os.environ.update(settings)
    return {"rules": rules, "max_turns": max_turns, "fog_memory": bool(fog_memory)}


def snapshot(obs: dict) -> dict:
    """Compact board state from an engine observation (board arrays are indexed [x][y])."""
    board = obs.get("board", {})
    units = {}
    for uid, u in (obs.get("unit") or {}).items():
        units[str(uid)] = {"x": int(u["x"]), "y": int(u["y"]), "t": int(u["type"]), "o": int(u["tribeId"]),
                           "hp": int(u["currentHP"]), "vet": bool(u.get("isVeteran", False)), "k": int(u.get("kill", 0))}
    cities = {}
    for cid, c in (obs.get("city") or {}).items():
        cities[str(cid)] = {"x": int(c["x"]), "y": int(c["y"]), "o": int(c["tribeID"]), "cap": bool(c.get("isCapital")),
                            "lvl": int(c.get("level", 1)), "walls": bool(c.get("hasWalls", False))}
    tribes = {}
    for tid, t in (obs.get("tribes") or {}).items():
        tribes[str(tid)] = {"stars": int(t.get("star", 0)), "kills": int(t.get("nKills", 0)), "score": int(t.get("score", 0)),
                            "cities": len(t.get("citiesID", []))}
    return {"terrain": board.get("terrain", []), "units": units, "cities": cities, "tribes": tribes,
            "tick": int(obs.get("tick", 0))}


def describe_action(action: dict, before: dict, events: list[dict]) -> str:
    """Readable text for one of our actions, from the action and the board changes it caused."""
    kind = str(action.get("type", "?"))
    rep = str(action.get("repr", kind))
    moves = [e for e in events if e["kind"] == "move" and e["side"] == 0]
    if kind == "END_TURN":
        return "End turn"
    if kind == "MOVE":
        return f"Move {moves[0]['text']}" if moves else rep
    if kind == "ATTACK":
        m = re.search(r"by unit (\d+)", rep)
        attacker = before["units"].get(m.group(1)) if m else None
        hits = "; ".join(e["text"] for e in events if e["kind"] in ("damaged", "killed"))
        who = f"{unit_name(attacker['t'])} ({attacker['x']},{attacker['y']})" if attacker else "Unit"
        return f"{who} attacks — {hits}" if hits else rep
    if kind == "CAPTURE":
        caps = [e for e in events if e["kind"] == "capture"]
        return caps[0]["text"] if caps else rep
    if kind == "RESEARCH_TECH":
        return f"Research {str(action.get('tech_type', '?')).replace('_', ' ').title()}"
    if kind == "SPAWN":
        trained = [e for e in events if e["kind"] == "train"]
        return f"Train {trained[0]['text'].replace(' appears at', ' at')}" if trained else rep
    if kind == "RESOURCE_GATHERING":
        return "Harvest " + rep.split(":")[-1].strip().lower()
    if kind == "LEVEL_UP":
        return "City level-up bonus: " + rep.split("bonus")[-1].strip().replace("_", " ").lower()
    return rep


def diff_events(before: dict, after: dict) -> list[dict]:
    """Events between two true-board snapshots. `actor` is the tribe that acted (0 = us, 1 = opponent)."""
    ev = []
    b_units, a_units = before["units"], after["units"]
    for uid, u in a_units.items():
        if uid not in b_units:
            ev.append({"side": u["o"], "kind": "train", "text": f"{unit_name(u['t'])} appears at ({u['x']},{u['y']})",
                       "at": [u["x"], u["y"]]})
    for uid, u in b_units.items():
        if uid not in a_units:
            ev.append({"side": u["o"], "kind": "killed", "text": f"{unit_name(u['t'])} at ({u['x']},{u['y']}) killed",
                       "at": [u["x"], u["y"]]})
            continue
        v = a_units[uid]
        if (v["x"], v["y"]) != (u["x"], u["y"]):
            ev.append({"side": u["o"], "kind": "move", "text": f"{unit_name(u['t'])} ({u['x']},{u['y']}) → ({v['x']},{v['y']})",
                       "from": [u["x"], u["y"]], "to": [v["x"], v["y"]]})
        if v["hp"] < u["hp"]:
            ev.append({"side": u["o"], "kind": "damaged", "text": f"{unit_name(u['t'])} at ({v['x']},{v['y']}) hit: HP {u['hp']} → {v['hp']}",
                       "at": [v["x"], v["y"]]})
        elif v["hp"] > u["hp"]:
            ev.append({"side": u["o"], "kind": "heal", "text": f"{unit_name(u['t'])} healed: HP {u['hp']} → {v['hp']}",
                       "at": [v["x"], v["y"]]})
        if v["vet"] and not u["vet"]:
            ev.append({"side": u["o"], "kind": "veteran", "text": f"{unit_name(u['t'])} became a veteran", "at": [v["x"], v["y"]]})
    for cid, c in after["cities"].items():
        old = before["cities"].get(cid)
        if old is None:
            ev.append({"side": c["o"], "kind": "capture", "text": f"Village at ({c['x']},{c['y']}) captured (new city)",
                       "at": [c["x"], c["y"]]})
        elif old["o"] != c["o"]:
            what = "CAPITAL" if (c["cap"] or old["cap"]) else "City"
            ev.append({"side": c["o"], "kind": "capture", "text": f"{what} at ({c['x']},{c['y']}) captured",
                       "at": [c["x"], c["y"]]})
        elif c["lvl"] > old["lvl"]:
            ev.append({"side": c["o"], "kind": "grow", "text": f"City at ({c['x']},{c['y']}) grew to level {c['lvl']}",
                       "at": [c["x"], c["y"]]})
    return ev


def visible_mask(fog_obs: dict) -> list[list[int]]:
    return [[0 if int(t) == FOG else 1 for t in row] for row in fog_obs.get("board", {}).get("terrain", [])]


def resolve_map(map_arg: str, pool: str) -> tuple:
    _, maps, _ = load_verified_pool(REPO_ROOT, pool)
    name = Path(map_arg).name
    for m in maps:
        if m.filename == name:
            return m
    raise SystemExit(f"{name} is not in the {pool} pool ({POOL_ROOT_REL}/{pool})")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--map", required=True, help="Map file name in the validation pool, e.g. map_2p_030048.csv")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--episode-seed", type=int, help="Episode seed (from an evaluation's episodes.jsonl)")
    g.add_argument("--replicate", type=int, help="Replicate index of the canonical schedule (seed 42)")
    p.add_argument("--sampled", action="store_true", help="Sample actions instead of argmax")
    p.add_argument("--policy-seed", type=int, default=0)
    p.add_argument("--title", default=None)
    p.add_argument("--output-html", type=Path, required=True)
    p.add_argument("--output-json", type=Path, help="Also write the raw replay data")
    p.add_argument("--max-decisions", type=int, default=5000)
    return p.parse_args()


def main():
    args = parse_args()
    pool = "eval_validation"
    from pol_env.Tribes.py.environment_contract import (
        environment_compatibility_metadata, read_checkpoint_metadata, validate_checkpoint_compatibility,
    )

    model_path = args.model_path.resolve()
    meta = read_checkpoint_metadata(str(model_path))
    settings = configure_environment(meta, pool)
    game_map = resolve_map(args.map, pool)
    episode_seed = (args.episode_seed if args.episode_seed is not None
                    else stable_seed(42, "combat_episode", game_map.csv_sha256, int(args.replicate)))

    from pol_env.Tribes.py.register_env import TribesGymWrapper
    from py_rl.cleanrl.cleanrl.agent_factory import make_agent
    from py_rl.cleanrl.cleanrl.obs_normalization import observation_normalization_from_metadata

    device = torch.device("cpu")
    env = TribesGymWrapper()
    try:
        env_meta = environment_compatibility_metadata(env, actor_mode=meta["actor_mode"], max_legal_actions=meta["max_legal_actions"])
        validate_checkpoint_compatibility(meta, env_meta)
        adapter = SimpleNamespace(single_observation_space=env.observation_space, single_action_space=env.action_space)
        agent = make_agent(adapter, network_arch=meta.get("network_arch", "mlp"), actor_mode=meta["actor_mode"],
                           max_legal_actions=meta["max_legal_actions"], legal_action_feature_dim=meta["legal_action_feature_dim"],
                           observation_normalization=observation_normalization_from_metadata(meta)).to(device)
        agent.load_state_dict(torch.load(model_path, map_location=device))
        agent.eval()
        policy = PPOPolicy(agent, device, bool(args.sampled), args.policy_seed if args.sampled else None)

        env._level_pool = [game_map.path]
        env._level_pool_size = 1
        obs, info = env.reset(seed=episode_seed)
        te = env.tribes_env

        def capture():
            full = snapshot(te.get_observation(full_visibility=True))
            seen_obs = te.get_observation(full_visibility=False)
            seen = snapshot(seen_obs)
            return full, {"units": sorted(seen["units"]), "cities": sorted(seen["cities"]), "visible": visible_mask(seen_obs)}

        full, seen = capture()
        frames = [{"turn": int(info.get("turn_count", 0)), "label": "Start", "events": [], "full": full, "seen": seen}]
        terminated = truncated = False
        decisions = 0
        while not (terminated or truncated):
            gid, _slot = policy.choose_action(obs, info)
            raw = env._current_legal_actions[env._current_legal_id_to_raw_index[int(gid)]]
            before = frames[-1]["full"]
            turn_before = int(info.get("turn_count", 0))
            obs, _r, terminated, truncated, info = env.step(int(gid))
            decisions += 1
            if decisions > args.max_decisions:
                raise RuntimeError("Decision cap exceeded")
            full, seen = capture()
            if raw.get("type") == "END_TURN":
                # Our END_TURN also runs the opponent's whole turn; split it out as its own frame.
                events = diff_events(before, full)
                for e in events:  # the wrapper runs queued village captures (fewer than 2 cities) inside END_TURN
                    if e["kind"] == "capture" and e["side"] == 0:
                        e["text"] += " — queued capture, applied as we ended the turn"
                frames.append({"turn": turn_before, "label": "End turn", "events": [],
                               "full": before, "seen": frames[-1]["seen"], "action": raw.get("type")})
                frames.append({"turn": turn_before, "label": "Opponent turn", "events": events, "full": full, "seen": seen,
                               "opponent": True})
            else:
                events = diff_events(before, full)
                text = describe_action(raw, before, events)
                frames.append({"turn": int(info.get("turn_count", turn_before)), "label": text, "events": events,
                               "full": full, "seen": seen, "action": raw.get("type"), "raw": raw.get("repr")})
        outcome = str(info.get("combat_outcome"))
        end_turn = int(info.get("turn_count", -1))
    finally:
        env.close()

    tribes = (game_map.our_tribe, game_map.opponent_tribe)
    payload = {
        "title": args.title or f"{game_map.filename} — {outcome} on turn {end_turn}",
        "map": game_map.filename, "episode_seed": episode_seed, "pool": pool,
        "policy": "ppo_sampled" if args.sampled else "ppo_argmax", "model": model_path.name,
        "environment_version": meta.get("phase1_environment_version"), "settings": settings,
        "tribes": list(tribes), "outcome": outcome, "end_turn": end_turn, "decisions": decisions,
        "unit_names": UNIT_NAMES, "unit_letters": UNIT_LETTERS, "terrain_names": TERRAIN_NAMES, "frames": frames,
    }
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    template = (REPO_ROOT / "tools/combat_replay_template.html").read_text(encoding="utf-8")
    args.output_html.write_text(template.replace("/*REPLAY_DATA*/null", json.dumps(payload, separators=(",", ":"))), encoding="utf-8")
    if args.output_json:
        args.output_json.write_text(json.dumps(payload), encoding="utf-8")
    print(f"{game_map.filename} seed {episode_seed}: {outcome} on turn {end_turn} after {decisions} decisions")
    print(f"Replay: {args.output_html.resolve()}")


if __name__ == "__main__":
    main()
