#!/usr/bin/env python
"""Play a combat (Phase 2) game against a trained agent in the browser.

    python tools/play_vs_agent.py --model-path runs/.../NAME.cleanrl_model
    # then open http://127.0.0.1:8765  (on Farmshare: ssh -L 8765:127.0.0.1:8765 <host>, then the same URL)

The game is the one the agent is evaluated on (`tools/evaluate_combat.py`): rules, turn limit and fog memory come
from the checkpoint's environment version, capitals mode, validation maps. The agent plays tribe 0 and moves first,
exactly as in evaluation; you play tribe 1, the seat SimpleAgent normally takes. You see only what tribe 1 sees (the
engine's fogged observation for the active tribe) and may take any action the engine allows tribe 1, the same action
set SimpleAgent chooses from. Win by taking the agent's capital; the agent wins by taking yours; at the turn limit
the game is a timeout (the agent did not win).

How it works: the Java engine's opponent is set to "external" (as for self-play snapshots, combat_selfplay.py), so
when the agent ends its turn the wrapper calls back into Python and the game thread waits for your moves from the
browser. Only the game thread talks to the engine; the web server only reads published state and queues commands.

Every finished game is written as JSON (map, episode seed, checkpoint, every action of both sides in order, outcome)
to --record-dir. The engine is deterministic given (map, seed, actions), so `--verify-record FILE` replays a record
and checks it reproduces the same outcome. Only the validation pool is offered; eval_test needs --confirm-test.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The engine needs the bundled JDK 11 (macOS layout under Contents/Home, Linux layout directly), not a newer Java.
_jdks = [h for d in sorted((REPO_ROOT / ".runtime").glob("jdk-*")) for h in (d / "Contents" / "Home", d)
         if (h / "bin" / "java").exists()]
if _jdks and not os.environ.get("JAVA_HOME"):
    os.environ["JAVA_HOME"] = str(_jdks[0])
    os.environ["PATH"] = str(_jdks[0] / "bin") + os.pathsep + os.environ.get("PATH", "")

RECORD_VERSION = "human_game_v1"
HUMAN_TRIBE = 1
AGENT_TRIBE = 0
# Diplomacy actions the engine lists but no combat player uses (the agent's catalog has none of them).
HIDDEN_ACTION_TYPES = {"SEND_STARS", "DECLARE_WAR"}
UNIT_NAMES = ["Warrior", "Rider", "Defender", "Swordsman", "Archer", "Catapult", "Knight", "Mind Bender",
              "Boat", "Ship", "Battleship", "Giant", "Cloak", "Dagger"]
FOG = 7


class GameAborted(Exception):
    """Raised inside the engine callback to abandon a game (new game, resign or server shutdown)."""


# --- environment set-up -------------------------------------------------------------------------------------

def configure_environment(checkpoint_meta: Optional[dict], pool: str, rules: str, max_turns: int) -> dict:
    """Same POLYVISION_* settings as tools/evaluate_combat.py (from the checkpoint when there is one)."""
    from tools.evaluate_combat import environment_settings

    rules, max_turns, max_legal = environment_settings(checkpoint_meta, rules, max_turns)
    fog_memory = False
    if checkpoint_meta is not None:
        from pol_env.Tribes.py.environment_contract import fog_memory_of_version
        fog_memory = bool(fog_memory_of_version(checkpoint_meta.get("phase1_environment_version")))
    for key in [k for k in os.environ if k.startswith("POLYVISION_COMBAT_REWARD_") or k.startswith("POLYVISION_COMBAT_LEAGUE")
                or k.startswith("POLYVISION_COMBAT_TEACHER") or k.startswith("POLYVISION_PRIVILEGED")]:
        del os.environ[key]
    settings = {
        "POLYVISION_PHASE": "2", "POLYVISION_COMBAT_MODE": "1", "POLYVISION_SOLO_NO_OPPONENT_MODE": "0",
        "POLYVISION_LEVEL_POOL_GLOB": f"levels/phase2_pool_1v1/{pool}/*.csv",
        "POLYVISION_LEVEL_SELECTION_MODE": "round_robin", "POLYVISION_INFO_MODE": "fast",
        "POLYVISION_MAX_TURNS": str(max_turns), "POLYVISION_RULES": rules,
        "POLYVISION_FOG_MEMORY": "1" if fog_memory else "0", "POLYVISION_MAX_LEGAL_ACTIONS": str(max_legal),
    }
    os.environ.update(settings)
    return {"rules": rules, "max_turns": int(max_turns), "fog_memory": fog_memory, "max_legal_actions": int(max_legal)}


class AgentPlayer:
    """The trained checkpoint (argmax or sampled), or a uniformly random legal action without one."""

    def __init__(self, model_path: Optional[Path], env, sampled: bool, seed: int):
        import torch

        from tools.phase1_eval_core import PPOPolicy, RandomLegalPolicy

        self.model = model_path.name if model_path else None
        if model_path is None:
            self.policy, self.name = RandomLegalPolicy(seed), "random_legal"
            return
        from pol_env.Tribes.py.environment_contract import (
            environment_compatibility_metadata, read_checkpoint_metadata, validate_checkpoint_compatibility,
        )
        from py_rl.cleanrl.cleanrl.agent_factory import make_agent
        from py_rl.cleanrl.cleanrl.obs_normalization import observation_normalization_from_metadata

        meta = read_checkpoint_metadata(str(model_path))
        env_meta = environment_compatibility_metadata(env, actor_mode=meta["actor_mode"], max_legal_actions=meta["max_legal_actions"])
        validate_checkpoint_compatibility(meta, env_meta)
        adapter = SimpleNamespace(single_observation_space=env.observation_space, single_action_space=env.action_space)
        agent = make_agent(adapter, network_arch=meta.get("network_arch", "mlp"), actor_mode=meta["actor_mode"],
                           max_legal_actions=meta["max_legal_actions"], legal_action_feature_dim=meta["legal_action_feature_dim"],
                           observation_normalization=observation_normalization_from_metadata(meta))
        agent.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        agent.eval()
        self.policy = PPOPolicy(agent, torch.device("cpu"), sampled, seed if sampled else None)
        self.name = self.policy.name

    def choose(self, obs, info) -> int:
        gid, _slot = self.policy.choose_action(obs, info)
        return int(gid)


# --- what the human sees ------------------------------------------------------------------------------------

def tech_names() -> List[str]:
    from pol_env.Tribes.py.environment_contract import TECHNOLOGY_ORDER
    return [str(t) for t in TECHNOLOGY_ORDER]


def board_view(obs: dict) -> dict:
    """Compact, JSON-ready view of a fogged observation (board arrays are indexed [x][y])."""
    board = obs.get("board", {})
    units = [{"id": int(uid), "x": int(u["x"]), "y": int(u["y"]), "t": int(u["type"]), "o": int(u["tribeId"]),
              "hp": int(u["currentHP"]), "maxhp": int(u.get("maxHP", u["currentHP"]) or u["currentHP"]),
              "vet": bool(u.get("isVeteran", False)), "kills": int(u.get("kill", 0))}
             for uid, u in (obs.get("unit") or {}).items()]
    cities = [{"id": int(cid), "x": int(c["x"]), "y": int(c["y"]), "o": int(c["tribeID"]), "cap": bool(c.get("isCapital")),
               "lvl": int(c.get("level", 1)), "pop": int(c.get("population", 0)), "need": int(c.get("population_need", 0)),
               "prod": int(c.get("production", 0)), "walls": bool(c.get("hasWalls", False))}
              for cid, c in (obs.get("city") or {}).items()]
    names = tech_names()
    tribes = {}
    for tid, t in (obs.get("tribes") or {}).items():
        researched = (t.get("technology") or t.get("techTree") or {}).get("researched") or []
        tribes[str(tid)] = {"stars": int(t.get("star", 0)), "kills": int(t.get("nKills", 0)), "score": int(t.get("score", 0)),
                            "cities": len(t.get("citiesID", [])),
                            "techs": [names[i] if i < len(names) else str(i) for i, r in enumerate(researched) if r]}
    return {"terrain": board.get("terrain", []), "resource": board.get("resource", []), "building": board.get("building", []),
            "units": units, "cities": cities, "tribes": tribes}


def legal_view(actions: List[dict]) -> List[dict]:
    out = []
    for i, a in enumerate(actions):
        if int(a.get("idx", i)) != i:
            raise RuntimeError("Engine action list is not indexed by position")
        if a.get("type") in HIDDEN_ACTION_TYPES:
            continue
        item = {"i": i, "type": a.get("type"), "text": describe(a), "cost": int(a.get("star_cost", 0) or 0)}
        if "src_x" in a:
            item["src"] = [int(a["src_x"]), int(a["src_y"])]
        if "target_x" in a:
            item["dst"] = [int(a["target_x"]), int(a["target_y"])]
        out.append(item)
    return out


def describe(a: dict) -> str:
    kind = str(a.get("type", "?"))
    pretty = lambda s: str(s).replace("_", " ").title()
    if kind == "END_TURN":
        return "End turn"
    if kind == "RESEARCH_TECH":
        return f"Research {pretty(a.get('tech_type'))}"
    if kind == "SPAWN":
        return f"Train {pretty(a.get('unit_type'))}"
    if kind == "MOVE":
        return f"Move to ({a.get('target_x')},{a.get('target_y')})"
    if kind == "ATTACK":
        return f"Attack ({a.get('target_x')},{a.get('target_y')})"
    if kind == "CAPTURE":
        return f"Capture {pretty(a.get('capture_type') or 'city')}"
    if kind == "RESOURCE_GATHERING":
        return f"Harvest {pretty(a.get('resource_type'))}"
    if kind == "BUILD":
        return f"Build {pretty(a.get('building_type'))}"
    if kind == "LEVEL_UP":
        return f"City reward: {pretty(a.get('levelup_choice'))}"
    return pretty(kind) if kind != "?" else str(a.get("repr", kind))


def seen_changes(before: Optional[dict], after: dict) -> List[str]:
    """What the human can see changed over the agent's turn (fogged views; units leaving sight are not deaths)."""
    if before is None:
        return []
    visible = lambda x, y: 0 <= x < len(after["terrain"]) and int(after["terrain"][x][y]) != FOG
    name = lambda u: UNIT_NAMES[u["t"]] if 0 <= u["t"] < len(UNIT_NAMES) else f"unit {u['t']}"
    whose = lambda u: "Your" if u["o"] == HUMAN_TRIBE else "Enemy"
    b_units = {u["id"]: u for u in before["units"]}
    a_units = {u["id"]: u for u in after["units"]}
    lines = []
    for uid, u in b_units.items():
        v = a_units.get(uid)
        if v is None:
            if u["o"] == HUMAN_TRIBE or visible(u["x"], u["y"]):
                lines.append(f"{whose(u)} {name(u)} at ({u['x']},{u['y']}) was killed")
            continue
        if v["hp"] < u["hp"]:
            lines.append(f"{whose(u)} {name(u)} at ({v['x']},{v['y']}) took {u['hp'] - v['hp']} damage (HP {v['hp']})")
        if (v["x"], v["y"]) != (u["x"], u["y"]) and v["o"] != HUMAN_TRIBE:
            lines.append(f"Enemy {name(v)} moved ({u['x']},{u['y']}) → ({v['x']},{v['y']})")
    for uid, v in a_units.items():
        if uid not in b_units and v["o"] != HUMAN_TRIBE:
            lines.append(f"Enemy {name(v)} seen at ({v['x']},{v['y']})")
    b_cities = {c["id"]: c for c in before["cities"]}
    for c in after["cities"]:
        old = b_cities.get(c["id"])
        if old is not None and old["o"] != c["o"]:
            what = "capital" if (c["cap"] or old["cap"]) else "city"
            lines.append(f"{'You lost' if old['o'] == HUMAN_TRIBE else 'Enemy lost'} the {what} at ({c['x']},{c['y']})")
        elif old is None and c["o"] != HUMAN_TRIBE:
            lines.append(f"Enemy city seen at ({c['x']},{c['y']})")
    return lines


# --- one game -----------------------------------------------------------------------------------------------

class GameSession:
    """Runs one game on a game thread. The web server reads `snapshot()` and calls `submit()`."""

    def __init__(self, env, agent: AgentPlayer, map_path: str, episode_seed: int, settings: dict,
                 record_dir: Optional[Path], player: str = "", human_policy: Optional[Callable] = None,
                 move_timeout_s: Optional[float] = None):
        self.env, self.agent, self.map_path, self.episode_seed = env, agent, map_path, int(episode_seed)
        self.settings, self.record_dir, self.player = settings, record_dir, player
        self.human_policy = human_policy           # tests: f(state) -> action index, instead of the browser
        self.move_timeout_s = move_timeout_s
        self.commands: "queue.Queue[dict]" = queue.Queue()
        self.lock = threading.Lock()
        self.state: Dict[str, Any] = {"version": 0, "phase": "starting", "map": Path(map_path).name,
                                      "episode_seed": self.episode_seed, "agent": agent.name, "model": agent.model,
                                      "settings": settings, "log": []}
        self.history: List[dict] = []
        self.agent_moves: List[dict] = []          # revealed only when the game is over (they include fogged moves)
        self.record_path: Optional[Path] = None
        self.error: Optional[str] = None
        self._abort = threading.Event()
        self._last_human_view: Optional[dict] = None
        self.thread = threading.Thread(target=self._run, name="game", daemon=True)

    # server side ----------------------------------------------------------------
    def start(self):
        self.thread.start()
        return self

    def snapshot(self) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.state))

    def submit(self, command: dict) -> Optional[str]:
        """Queue a command ({"act": i, "version": v} or {"resign": true}); returns an error string or None."""
        with self.lock:
            if self.state.get("phase") != "your_turn":
                return "It is not your turn"
            if "act" in command and int(command.get("version", -1)) != self.state["version"]:
                return "The board changed; pick again"
        self.commands.put(command)
        return None

    def abort(self):
        self._abort.set()
        self.commands.put({"abort": True})
        self.thread.join(timeout=30)

    # game thread ---------------------------------------------------------------
    def _publish(self, **fields):
        with self.lock:
            self.state.update(fields)
            self.state["version"] += 1

    def _log(self, side: str, text: str, turn: int):
        with self.lock:
            self.state["log"].append({"side": side, "text": text, "turn": turn})

    def _human_turn(self):
        te = self.env.tribes_env
        java = te._env
        changes_logged = False
        while not bool(java.isDone()) and int(java.getActiveTribeID()) != AGENT_TRIBE:
            if self._abort.is_set():
                raise GameAborted()
            obs = te.get_observation()
            actions = te.list_actions()
            view = board_view(obs)
            turn = int(getattr(self.env, "_turn_count", 0))
            if not changes_logged:
                for line in seen_changes(self._last_human_view, view) or ["(nothing you can see changed)"]:
                    self._log("agent", line, turn)
                changes_logged = True
            legal = legal_view(actions)
            self._publish(phase="your_turn", board=view, legal=legal, turn=turn, tribe=HUMAN_TRIBE)
            if self.human_policy is not None:
                command = {"act": int(self.human_policy(self.snapshot()))}
            else:
                try:
                    command = self.commands.get(timeout=self.move_timeout_s)
                except queue.Empty:
                    raise GameAborted("timed out waiting for a move")
            if command.get("abort"):
                raise GameAborted()
            if command.get("resign"):
                self.history.append({"side": HUMAN_TRIBE, "resign": True})
                raise GameAborted("resigned")
            idx = int(command["act"])
            if not any(item["i"] == idx for item in legal):
                self._publish(notice=f"Action {idx} is not legal now")
                continue
            raw = actions[idx]
            self.history.append({"side": HUMAN_TRIBE, "raw": idx, "repr": raw.get("repr"), "type": raw.get("type")})
            self._log("you", describe(raw) + (f" — {raw['repr']}" if raw.get("type") not in ("END_TURN", "RESEARCH_TECH") else ""), turn)
            java.stepByIndex(idx)
            if raw.get("type") == "END_TURN":
                self._last_human_view = view  # after stepping, the active tribe (and so the observation) is the agent's
        self._publish(phase="agent_turn", legal=[])

    def _run(self):
        env = self.env
        outcome, resigned = None, False
        try:
            from pol_env.Tribes.py.combat_selfplay import EXTERNAL_OPPONENT

            def select_opponent():  # replaces the league hook for this game: tribe 1 is played from Python
                env.tribes_env._env.setOpponentAgent(EXTERNAL_OPPONENT, 0)
                env.tribes_env.opponent_turn_hook = self._human_turn

            env._select_combat_opponent = select_opponent
            env._level_pool, env._level_pool_size = [self.map_path], 1
            obs, info = env.reset(seed=self.episode_seed)
            self._publish(phase="agent_turn")
            terminated = truncated = False
            while not (terminated or truncated):
                if self._abort.is_set():
                    raise GameAborted()
                gid = self.agent.choose(obs, info)
                raw_index = env._current_legal_id_to_raw_index[int(gid)]
                raw = env._current_legal_actions[raw_index]
                turn = int(info.get("turn_count", 0))
                self.history.append({"side": AGENT_TRIBE, "gid": int(gid), "type": raw.get("type"), "repr": raw.get("repr")})
                if raw.get("type") != "END_TURN":
                    self.agent_moves.append({"side": "agent", "turn": turn, "text": describe(raw) + f" — {raw.get('repr')}"})
                obs, _r, terminated, truncated, info = env.step(int(gid))
            outcome = str(info.get("combat_outcome"))
        except GameAborted as exc:
            resigned = str(exc) == "resigned"
            outcome = "human_resigned" if resigned else "aborted"
        except Exception:
            self.error = traceback.format_exc()
            outcome = "error"
        finally:
            env.tribes_env.opponent_turn_hook = None
        human_result = {"win": "loss", "loss": "win", "timeout": "timeout", "human_resigned": "loss"}.get(outcome, outcome)
        final = None
        try:
            if outcome not in ("aborted", "error"):
                final = board_view(env.tribes_env.get_observation(full_visibility=True))
        except Exception:
            pass
        self._save(outcome)
        self._publish(phase="over", agent_outcome=outcome, result=human_result, legal=[], error=self.error,
                      final_board=final, turn=int(getattr(env, "_turn_count", 0)),
                      record=str(self.record_path) if self.record_path else None, agent_moves=self.agent_moves)

    def _save(self, outcome: str):
        if self.record_dir is None or outcome in ("aborted", "error"):
            return
        self.record_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.record_path = self.record_dir / f"{stamp}_{Path(self.map_path).stem}_{self.episode_seed}.json"
        record = {"record_version": RECORD_VERSION, "created_utc": stamp, "player": self.player,
                  "map": Path(self.map_path).name, "map_path": self.map_path, "episode_seed": self.episode_seed,
                  "agent": self.agent.name, "model": self.agent.model, "agent_tribe": AGENT_TRIBE, "human_tribe": HUMAN_TRIBE,
                  "settings": self.settings, "agent_outcome": outcome, "turns": int(getattr(self.env, "_turn_count", 0)),
                  "actions": self.history}
        self.record_path.write_text(json.dumps(record, indent=1), encoding="utf-8")


def replay_record(env, record: dict) -> str:
    """Replay a saved game action by action; returns the agent's outcome (raises if an action no longer fits)."""
    actions = [a for a in record["actions"] if not a.get("resign")]
    human = iter([a for a in actions if a["side"] == HUMAN_TRIBE])
    agent = [a for a in actions if a["side"] == AGENT_TRIBE]
    picks = iter(agent)

    class Scripted:
        name, model = record.get("agent"), record.get("model")

        def choose(self, obs, info):
            return next(picks)["gid"]

    def human_policy(state):
        step = next(human, None)
        if step is None:  # the human resigned here
            raise GameAborted("resigned")
        match = [item for item in state["legal"] if item["i"] == step["raw"]]
        if not match or env.tribes_env.list_actions()[step["raw"]].get("repr") != step["repr"]:
            raise RuntimeError(f"Recorded human action {step['repr']!r} is not legal on replay")
        return step["raw"]

    session = GameSession(env, Scripted(), record["map_path"], record["episode_seed"], record.get("settings", {}),
                          None, human_policy=human_policy)
    session._run()
    if session.error:
        raise RuntimeError(session.error)
    if any(a.get("resign") for a in record["actions"]):
        return "human_resigned"
    return session.state["agent_outcome"]


# --- web server ---------------------------------------------------------------------------------------------

class Server:
    def __init__(self, env, agent, maps: List[Any], settings: dict, record_dir: Path, player: str, seed: Optional[int]):
        self.env, self.agent, self.maps, self.settings = env, agent, maps, settings
        self.record_dir, self.player = record_dir, player
        self.rng = random.Random(seed)
        self.session: Optional[GameSession] = None
        self.lock = threading.Lock()

    def new_game(self, map_name: Optional[str] = None, episode_seed: Optional[int] = None) -> GameSession:
        with self.lock:
            if self.session is not None:
                self.session.abort()
            if map_name:
                matches = [m for m in self.maps if m.filename == Path(map_name).name]
                if not matches:
                    raise ValueError(f"{map_name} is not in the pool")
                game_map = matches[0]
            else:
                game_map = self.rng.choice(self.maps)
            seed = int(episode_seed) if episode_seed is not None else self.rng.randrange(1, 2**31 - 1)
            self.session = GameSession(self.env, self.agent, str(game_map.path), seed, self.settings, self.record_dir,
                                       self.player)
            self.session.state["tribes"] = {"agent": game_map.our_tribe, "you": game_map.opponent_tribe}
            return self.session.start()

    def handler(self):
        server = self
        page = (REPO_ROOT / "tools/play_vs_agent.html").read_bytes()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code: int, body: bytes, ctype: str = "application/json"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _json(self, code: int, obj):
                self._send(code, json.dumps(obj).encode("utf-8"))

            def do_GET(self):
                path = urlparse(self.path).path
                if path in ("/", "/index.html"):
                    return self._send(200, page, "text/html; charset=utf-8")
                if path == "/api/state":
                    s = server.session
                    state = s.snapshot() if s else {"phase": "none", "version": 0}
                    state["maps"] = [m.filename for m in server.maps]
                    state["unit_names"], state["techs"] = UNIT_NAMES, tech_names()
                    return self._json(200, state)
                self._json(404, {"error": "not found"})

            def do_POST(self):
                path = urlparse(self.path).path
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(length) or b"{}")
                except (ValueError, json.JSONDecodeError):
                    return self._json(400, {"error": "bad request"})
                if path == "/api/new":
                    try:
                        server.new_game(body.get("map") or None, body.get("seed"))
                    except (ValueError, TypeError) as exc:
                        return self._json(400, {"error": str(exc)})
                    return self._json(200, {"ok": True})
                if path in ("/api/act", "/api/resign"):
                    s = server.session
                    if s is None:
                        return self._json(409, {"error": "No game"})
                    command = {"resign": True} if path == "/api/resign" else {"act": body.get("i"), "version": body.get("version")}
                    if "act" in command and not isinstance(command["act"], int):
                        return self._json(400, {"error": "i must be an action index"})
                    err = s.submit(command)
                    return self._json(409 if err else 200, {"error": err} if err else {"ok": True})
                self._json(404, {"error": "not found"})

        return Handler


def load_maps(pool: str):
    from tools.combat_eval_core import load_verified_pool
    _, maps, _ = load_verified_pool(REPO_ROOT, pool)
    return maps


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", type=Path, help="Combat checkpoint to play against (omit: a random-legal opponent)")
    p.add_argument("--sampled", action="store_true", help="The agent samples its actions instead of playing argmax")
    p.add_argument("--policy-seed", type=int, default=0)
    p.add_argument("--map", help="Map file name to start with (default: a random validation map)")
    p.add_argument("--episode-seed", type=int)
    p.add_argument("--seed", type=int, help="Seed for choosing maps and episode seeds")
    p.add_argument("--pool", default="eval_validation", choices=["eval_validation", "eval_test"])
    p.add_argument("--confirm-test", action="store_true", help="Required with --pool eval_test (human protocol only)")
    p.add_argument("--rules", default="tribes_2020", help="Without --model-path: rules to play")
    p.add_argument("--max-turns", type=int, default=20, help="Without --model-path: turn limit")
    p.add_argument("--player", default="", help="Alias stored in the game records (no real names needed)")
    p.add_argument("--record-dir", type=Path, default=REPO_ROOT / "runs" / "human_games")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--verify-record", type=Path, help="Replay a saved game and check its outcome, then exit")
    return p.parse_args()


def make_env(args, recorded: Optional[dict] = None):
    from pol_env.Tribes.py.environment_contract import read_checkpoint_metadata

    meta = read_checkpoint_metadata(str(args.model_path)) if args.model_path else None
    settings = configure_environment(meta, args.pool, args.rules, args.max_turns)
    if recorded:  # replaying a record: play under exactly the recorded settings
        os.environ["POLYVISION_FOG_MEMORY"] = "1" if recorded.get("fog_memory") else "0"
        os.environ["POLYVISION_MAX_LEGAL_ACTIONS"] = str(recorded.get("max_legal_actions", settings["max_legal_actions"]))
        settings.update(recorded)
    from pol_env.Tribes.py.register_env import TribesGymWrapper
    env = TribesGymWrapper()
    return env, settings


def main():
    args = parse_args()
    if args.pool == "eval_test" and not args.confirm_test:
        raise SystemExit("eval_test maps are for the human protocol only: pass --confirm-test when a human asked for it")

    if args.verify_record:
        record = json.loads(args.verify_record.read_text(encoding="utf-8"))
        s = record.get("settings", {})
        args.rules, args.max_turns = s.get("rules", args.rules), s.get("max_turns", args.max_turns)
        args.model_path = None  # the agent's recorded moves are replayed, not recomputed
        env, _ = make_env(args, recorded=s)
        try:
            outcome = replay_record(env, record)
        finally:
            env.close()
        ok = outcome == record["agent_outcome"]
        print(f"{args.verify_record.name}: recorded {record['agent_outcome']}, replayed {outcome} — {'OK' if ok else 'MISMATCH'}")
        sys.exit(0 if ok else 1)

    env, settings = make_env(args)
    agent = AgentPlayer(args.model_path.resolve() if args.model_path else None, env, args.sampled, args.policy_seed)
    maps = load_maps(args.pool)
    server = Server(env, agent, maps, settings, args.record_dir, args.player, args.seed)
    server.new_game(args.map, args.episode_seed)
    try:
        httpd = ThreadingHTTPServer((args.host, args.port), server.handler())
    except OSError as exc:
        server.session.abort()
        env.close()
        raise SystemExit(f"Cannot listen on {args.host}:{args.port} ({exc.strerror}). Another program (or an "
                         f"`ssh -L {args.port}:...` tunnel started on this machine) is using it: pick another --port. "
                         "The ssh tunnel belongs on your laptop, not on the machine running this server.")
    print(f"Playing {agent.name}{' (' + agent.model + ')' if agent.model else ''}: {settings['rules']}, "
          f"{settings['max_turns']}-turn limit, {args.pool} maps. Records go to {args.record_dir}")
    print(f"Open http://{args.host}:{args.port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        if server.session is not None:
            server.session.abort()
        env.close()


if __name__ == "__main__":
    main()
