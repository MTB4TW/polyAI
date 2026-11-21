#!/usr/bin/env python
"""Play a combat (Phase 2) game against a trained agent in the browser, then watch the replay.

    python tools/play_vs_agent.py --models '/home/users/devinlai/polyAI-wt/*/runs/*/*.cleanrl_model'
    # then open http://127.0.0.1:8765   (from a laptop: ssh -L 8765:127.0.0.1:8765 <you>@<this host> first)

Choosing the opponent: the page lists every combat checkpoint found under this checkout's `runs/` and the --models
globs (or POLYAI_MODEL_GLOBS, ':'-separated), newest first and grouped by run. Each entry shows the run, the
training step, when it was saved, its environment version and network, and its canonical evaluation if this
checkout has one. The game shows the first 12 characters of the file's SHA-256, so it is always clear which
weights you are playing. --model-path picks the default.

The game: the setup is `tools/evaluate_combat.py`'s:
- validation maps;
- capitals mode;
- rules, turn limit and fog memory from the checkpoint's environment version.

You pick your seat:
- **Second (tribe 1, SimpleAgent's seat in evaluation).** The agent plays tribe 0 through the wrapper exactly as
  in evaluation. You play through the engine's "external" opponent hook and may take any action the engine
  allows tribe 1 (SimpleAgent's action set).
- **First (tribe 0).** You act through the wrapper, so you choose from the agent's own legal-action interface
  (the same moves it may choose from in that seat). The agent plays tribe 1 the way frozen league snapshots do
  (`combat_selfplay.OpponentView`: its own fogged view relabelled so it appears as tribe 0, with its turn count
  the number of turns it has finished, as in seat-swap training). Checkpoints trained without seat swap have
  never played this seat.

Fog and results:
- You see only your tribe's fogged observation.
- You win by taking the agent's capital; it wins by taking yours. At the turn limit the game is a timeout.

After the game:
- The replay opens in the combat replay viewer (`tools/combat_replay_template.html`, the same viewer as
  `tools/export_combat_replay.py`), with the true board and each player's own fogged view, step by step.
- Each player's view is rebuilt from the true board and the tiles that tribe had explored, which is exactly how
  the engine fogs an observation.

Records:
- Every finished game is saved to --record-dir as JSON (`human_game_v2`): map, seed, checkpoint and its SHA-256,
  settings, seats, every action of both sides in order, and the outcome. The replay HTML is saved next to it.
- `--verify-record FILE` replays a record and checks every action and the outcome.
- Only the validation pool is offered; eval_test needs --confirm-test.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import queue
import random
import re
import sys
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The engine needs the bundled JDK 11 (macOS layout under Contents/Home, Linux layout directly), not a newer Java.
_jdks = [h for d in sorted((REPO_ROOT / ".runtime").glob("jdk-*")) for h in (d / "Contents" / "Home", d)
         if (h / "bin" / "java").exists()]
if _jdks and not os.environ.get("JAVA_HOME"):
    os.environ["JAVA_HOME"] = str(_jdks[0])
    os.environ["PATH"] = str(_jdks[0] / "bin") + os.pathsep + os.environ.get("PATH", "")

import numpy as np  # noqa: E402

from tools.export_combat_replay import (  # noqa: E402  (shared with the replay exporter)
    TERRAIN_NAMES, UNIT_LETTERS, UNIT_NAMES, describe_action, diff_events, snapshot as full_snapshot,
)

RECORD_VERSION = "human_game_v2"
# Diplomacy actions the engine lists but no combat player uses (the agent's catalog has none of them).
HIDDEN_ACTION_TYPES = {"SEND_STARS", "DECLARE_WAR"}
FOG = 7
RUN_DIR_RE = re.compile(r"^Tribes-v0__(?P<name>.+?)__(?P<seed>\d+)__(?P<ts>\d+)$")
STEP_RE = re.compile(r"^model_checkpoint_(\d+)\.cleanrl_model$")


class GameAborted(Exception):
    """Raised on the game thread to abandon a game (new game, resign or server shutdown)."""


# --- the model catalogue ------------------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ModelInfo:
    path: Path
    run: str
    seed: Optional[int]
    step: Optional[int]          # None = the run's final model
    saved: str
    mtime: float
    env_version: str
    rules: str
    max_turns: int
    arch: str
    init_from: Optional[str] = None
    evaluation: Optional[dict] = None
    name: Optional[str] = None           # display name (labels.json next to the checkpoint)
    note: Optional[str] = None           # one-line description shown with it
    order: int = 0                       # position in the list (labels.json), before newest-first
    _sha: Optional[str] = field(default=None, repr=False)

    @property
    def key(self) -> str:
        return hashlib.sha1(str(self.path).encode()).hexdigest()[:12]

    @property
    def sha256(self) -> str:
        if self._sha is None:
            self._sha = _sha256(self.path)
        return self._sha

    def label(self) -> str:
        step = f"step {self.step:,}" if self.step is not None else "final"
        ev = f" · eval {self.evaluation['win']:.1%}" if self.evaluation and self.evaluation.get("win") is not None else ""
        if self.name:
            return f"{self.name} · {step}" + (f" · {self.note}" if self.note else "") + ev
        return f"{step} · saved {self.saved}{ev}"

    def public(self, with_sha: bool = False) -> dict:
        d = {"key": self.key, "run": self.run, "seed": self.seed, "step": self.step, "saved": self.saved,
             "file": self.path.name, "path": str(self.path), "env_version": self.env_version, "rules": self.rules,
             "max_turns": self.max_turns, "arch": self.arch, "init_from": self.init_from,
             "evaluation": self.evaluation, "label": self.label(), "name": self.name, "note": self.note}
        if with_sha:
            d["sha256"] = self.sha256
        return d


def load_evaluations(root: Path = REPO_ROOT) -> Dict[Tuple[str, str], dict]:
    """(run directory name, checkpoint file name) -> canonical combat evaluation in this checkout."""
    out: Dict[Tuple[str, str], dict] = {}
    for cfg in glob.glob(str(root / "outputs" / "evaluations" / "*" / "config.json")):
        try:
            config = json.loads(Path(cfg).read_text(encoding="utf-8"))
            summary = json.loads((Path(cfg).parent / "summary.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        model = config.get("model_path")
        if not model or "combat" not in str(summary.get("evaluation_schema_version", "")):
            continue
        argmax = (summary.get("policies") or {}).get("ppo_argmax") or {}
        p = Path(model)
        out[(p.parent.name, p.name)] = {"id": Path(cfg).parent.name, "win": argmax.get("win_rate"),
                                       "canonical": bool(summary.get("canonical")),
                                       "against": "SimpleAgent, agent in seat 0"}
    return out


def find_models(patterns: List[str]) -> List[ModelInfo]:
    """Combat checkpoints matching the globs (both checkpoint files present), newest first."""
    from py_rl.cleanrl.cleanrl.init_from import rules_of_version
    from tools.evaluate_combat import HORIZON_RE

    evaluations = load_evaluations()
    seen, models = set(), []
    for pattern in patterns:
        for name in glob.glob(os.path.expanduser(pattern), recursive=True):
            path = Path(name).resolve()
            sidecar = Path(str(path) + ".action_interface.json")
            if path in seen or not sidecar.is_file():
                continue
            seen.add(path)
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            version = str(meta.get("phase1_environment_version", ""))
            horizon = HORIZON_RE.search(version)
            if not version.startswith("combat_") or not horizon:
                continue
            m = RUN_DIR_RE.match(path.parent.name)
            step = STEP_RE.match(path.name)
            mtime = path.stat().st_mtime
            init = (meta.get("init_from") or {}).get("path")
            models.append(ModelInfo(
                path=path, run=m.group("name") if m else path.parent.name, seed=int(m.group("seed")) if m else None,
                step=int(step.group(1)) if step else None,
                saved=datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M"), mtime=mtime, env_version=version,
                rules=rules_of_version(version), max_turns=int(horizon.group(1)),
                arch=str(meta.get("network_arch", "mlp")), init_from=Path(init).parent.name if init else None,
                evaluation=evaluations.get((path.parent.name, path.name))))
            # Optional labels.json next to the checkpoint: {"<file>": {"name", "group", "step", "note", "order"}},
            # for curated copies whose file and folder names say nothing about the run (runs/play_models/).
            labels = _labels_for(path.parent).get(path.name)
            if labels:
                info = models[-1]
                info.name, info.note = labels.get("name"), labels.get("note")
                info.run = labels.get("group", info.run)
                info.step = int(labels["step"]) if labels.get("step") is not None else info.step
                info.order = int(labels.get("order", 0))
    models.sort(key=lambda x: (x.order, -x.mtime, x.run, -(x.step or 10**12)))
    return models


def _labels_for(folder: Path) -> dict:
    f = folder / "labels.json"
    try:
        return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
    except (OSError, json.JSONDecodeError):
        return {}


# --- environment set-up -------------------------------------------------------------------------------------

def configure_environment(checkpoint_meta: Optional[dict], pool: str, rules: str, max_turns: int) -> dict:
    """Same POLYVISION_* settings as tools/evaluate_combat.py (from the checkpoint when there is one)."""
    from tools.evaluate_combat import environment_settings

    rules, max_turns, max_legal = environment_settings(checkpoint_meta, rules, max_turns)
    from pol_env.Tribes.py.environment_contract import (
        extra_actions_of_version, fog_memory_of_version, rules_off_of_version, unit_status_of_version)
    version = (checkpoint_meta or {}).get("phase1_environment_version")
    # Everything that changes what the checkpoint sees or may do follows the checkpoint (CHANGELOG 087, 090).
    fog_memory = bool(fog_memory_of_version(version)) if checkpoint_meta is not None else False
    unit_status = bool(unit_status_of_version(version)) if checkpoint_meta is not None else False
    rules_off = sorted(rules_off_of_version(version)) if checkpoint_meta is not None else []
    extra_actions = list(extra_actions_of_version(version)) if checkpoint_meta is not None else []
    map_pool = os.environ.get("POLYVISION_COMBAT_POOL", "").strip() or "phase2_pool_1v1"
    for key in [k for k in os.environ if k.startswith(("POLYVISION_COMBAT_REWARD_", "POLYVISION_COMBAT_LEAGUE",
                                                        "POLYVISION_COMBAT_TEACHER", "POLYVISION_PRIVILEGED"))]:
        del os.environ[key]
    settings = {
        "POLYVISION_PHASE": "2", "POLYVISION_COMBAT_MODE": "1", "POLYVISION_SOLO_NO_OPPONENT_MODE": "0",
        "POLYVISION_LEVEL_POOL_GLOB": f"levels/{map_pool}/{pool}/*.csv",
        "POLYVISION_LEVEL_SELECTION_MODE": "round_robin", "POLYVISION_INFO_MODE": "fast",
        "POLYVISION_MAX_TURNS": str(max_turns), "POLYVISION_RULES": rules,
        "POLYVISION_FOG_MEMORY": "1" if fog_memory else "0", "POLYVISION_MAX_LEGAL_ACTIONS": str(max_legal),
        "POLYVISION_UNIT_STATUS": "1" if unit_status else "0",
        "POLYVISION_COMBAT_RULES_OFF": ",".join(rules_off), "POLYVISION_COMBAT_EXTRA_ACTIONS": ",".join(extra_actions),
    }
    os.environ.update(settings)
    return {"rules": rules, "max_turns": int(max_turns), "fog_memory": fog_memory, "max_legal_actions": int(max_legal),
            "unit_status": unit_status, "wrapper_rules_off": rules_off, "extra_actions": extra_actions, "map_pool": map_pool}


def make_env(model_path: Optional[Path], pool: str, rules: str, max_turns: int, recorded: Optional[dict] = None):
    from pol_env.Tribes.py.environment_contract import read_checkpoint_metadata

    meta = read_checkpoint_metadata(str(model_path)) if model_path else None
    settings = configure_environment(meta, pool, rules, max_turns)
    if recorded:  # replaying a record: play under exactly the recorded settings
        os.environ["POLYVISION_RULES"] = str(recorded.get("rules", settings["rules"]))
        os.environ["POLYVISION_MAX_TURNS"] = str(recorded.get("max_turns", settings["max_turns"]))
        os.environ["POLYVISION_FOG_MEMORY"] = "1" if recorded.get("fog_memory") else "0"
        os.environ["POLYVISION_MAX_LEGAL_ACTIONS"] = str(recorded.get("max_legal_actions", settings["max_legal_actions"]))
        # What the checkpoint saw and could do (CHANGELOG 087, 090): the human's legal actions depend on it too.
        os.environ["POLYVISION_UNIT_STATUS"] = "1" if recorded.get("unit_status") else "0"
        os.environ["POLYVISION_COMBAT_RULES_OFF"] = ",".join(recorded.get("wrapper_rules_off") or [])
        os.environ["POLYVISION_COMBAT_EXTRA_ACTIONS"] = ",".join(recorded.get("extra_actions") or [])
        if "map_pool" in recorded:
            os.environ["POLYVISION_LEVEL_POOL_GLOB"] = os.environ["POLYVISION_LEVEL_POOL_GLOB"].replace(
                settings["map_pool"], str(recorded["map_pool"]), 1)
        settings.update({k: recorded[k] for k in ("rules", "max_turns", "fog_memory", "max_legal_actions", "map_pool",
                                                  "unit_status", "wrapper_rules_off", "extra_actions") if k in recorded})
    from pol_env.Tribes.py.register_env import TribesGymWrapper
    env = TribesGymWrapper()
    # As in the game, a unit pushed off a tile moves normally on its next turn (CHANGELOG 093). Records made before
    # this replay without it.
    fresh = bool(recorded.get("pushed_units_start_fresh", False)) if recorded else True
    env.tribes_env._env.setPushedUnitsStartFresh(fresh)
    settings["pushed_units_start_fresh"] = fresh
    return env, settings


class AgentPlayer:
    """A checkpoint (argmax or sampled), or uniformly random legal actions without one. With `search`, the combat
    tactical search (CombatTactics, as `ppo_search` in tools/evaluate_combat.py) picks the move whenever it finds a
    tactical sequence that beats stopping now; otherwise the policy decides."""

    def __init__(self, info: Optional[ModelInfo], env, sampled: bool, seed: int, search: bool = False):
        import torch

        self.info, self.sampled, self._torch = info, bool(sampled), torch
        self.search, self.env = bool(search), env
        self.last_source = "policy"
        self._search_seed, self._decision, self.search_moves = int(seed), 0, 0
        self.generator = torch.Generator().manual_seed(int(seed))
        self.rng = np.random.default_rng(int(seed))
        self.agent = None
        if info is None:
            return
        from pol_env.Tribes.py.environment_contract import (
            environment_compatibility_metadata, read_checkpoint_metadata, validate_checkpoint_compatibility,
        )
        from py_rl.cleanrl.cleanrl.agent_factory import make_agent
        from py_rl.cleanrl.cleanrl.obs_normalization import observation_normalization_from_metadata

        meta = read_checkpoint_metadata(str(info.path))
        env_meta = environment_compatibility_metadata(env, actor_mode=meta["actor_mode"], max_legal_actions=meta["max_legal_actions"])
        validate_checkpoint_compatibility(meta, env_meta)
        adapter = SimpleNamespace(single_observation_space=env.observation_space, single_action_space=env.action_space)
        agent = make_agent(adapter, network_arch=meta.get("network_arch", "mlp"), actor_mode=meta["actor_mode"],
                           max_legal_actions=meta["max_legal_actions"], legal_action_feature_dim=meta["legal_action_feature_dim"],
                           observation_normalization=observation_normalization_from_metadata(meta))
        agent.load_state_dict(torch.load(info.path, map_location="cpu", weights_only=True))
        agent.eval()
        self.agent = agent

    @property
    def name(self) -> str:
        if self.agent is None:
            return "random_legal"
        if getattr(self, "search", False):
            return "ppo_search_sampled" if self.sampled else "ppo_search"
        return "ppo_sampled" if self.sampled else "ppo_argmax"

    def reseed(self, seed: int):
        self.generator.manual_seed(int(seed) % (2**31 - 1))
        self.rng = np.random.default_rng(int(seed))
        self._search_seed, self._decision, self.search_moves = int(seed), 0, 0

    def _search_pick(self, ids, valid, raw_map) -> Optional[int]:
        """The tactical search's move for the active tribe on its own fogged view, or None (the policy decides)."""
        if not getattr(self, "search", False) or raw_map is None:
            return None
        from pol_env.Tribes.py import search_teacher

        te = self.env.tribes_env
        seed = search_teacher.decision_seed(self._search_seed, self._decision)
        self._decision += 1
        slot = search_teacher.tactical_slot(te._env, te._gateway, ids, valid, raw_map, seed)
        if slot < 0:
            return None
        self.search_moves += 1
        return int(ids[slot])

    def act(self, vec, ids, valid, feats, raw_map=None) -> int:
        ids, valid = np.asarray(ids).reshape(-1), np.asarray(valid, dtype=bool).reshape(-1)
        picked = self._search_pick(ids, valid, raw_map)
        self.last_source = "search" if picked is not None else "policy"
        if picked is not None:
            return picked
        if self.agent is None:
            return int(ids[int(self.rng.choice(np.flatnonzero(valid)))])
        torch = self._torch
        with torch.inference_mode():
            dist = self.agent.get_action_distribution(
                torch.as_tensor(vec, dtype=torch.float32).reshape(1, -1),
                legal_global_ids=torch.as_tensor(ids, dtype=torch.long).reshape(1, -1),
                legal_action_valid_mask=torch.as_tensor(valid, dtype=torch.bool).reshape(1, -1),
                legal_action_features=torch.as_tensor(np.asarray(feats, dtype=np.float32)).reshape(1, ids.size, -1),
            )
            if self.sampled:
                slot = int(torch.multinomial(dist.probs[0], 1, generator=self.generator).item())
            else:
                slot = int(torch.argmax(dist.logits[0]).item())
        if not valid[slot]:
            raise RuntimeError("The agent picked an invalid legal slot")
        return int(ids[slot])

    def choose(self, obs, info) -> int:
        from tools.phase1_eval_core import legal_tensors
        ids, valid, feats = legal_tensors(info)
        return self.act(obs, ids, valid, feats, raw_map=self.env._current_legal_id_to_raw_index)


# --- what the human sees ------------------------------------------------------------------------------------

def tech_names() -> List[str]:
    from pol_env.Tribes.py.environment_contract import TECHNOLOGY_ORDER
    return [str(t) for t in TECHNOLOGY_ORDER]


def board_view(obs: dict, max_hp: Optional[Dict[str, int]] = None) -> dict:
    """Compact, JSON-ready view of a fogged observation (board arrays are indexed [x][y])."""
    board = obs.get("board", {})
    max_hp = max_hp or {}
    units = [{"id": int(uid), "x": int(u["x"]), "y": int(u["y"]), "t": int(u["type"]), "o": int(u["tribeId"]),
              "hp": int(u["currentHP"]), "maxhp": int(max_hp.get(str(uid), u["currentHP"])),
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


def legal_view(pairs) -> List[dict]:
    """Browser list of (index the server accepts, engine action dict) pairs, minus unused diplomacy actions."""
    out = []
    for i, a in pairs:
        if a.get("type") in HIDDEN_ACTION_TYPES:
            continue
        item = {"i": int(i), "type": a.get("type"), "text": describe(a), "cost": int(a.get("star_cost", 0) or 0)}
        if "src_x" in a:
            item["src"] = [int(a["src_x"]), int(a["src_y"])]
        if "target_x" in a:
            item["dst"] = [int(a["target_x"]), int(a["target_y"])]
        out.append(item)
    return out


def raw_pairs(actions: List[dict]):
    for i, a in enumerate(actions):
        if int(a.get("idx", i)) != i:
            raise RuntimeError("Engine action list is not indexed by position")
        yield i, a


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


def seen_changes(before: Optional[dict], after: dict, me: int) -> List[str]:
    """What tribe `me` can see changed over the other side's turn (units leaving sight are not deaths)."""
    if before is None:
        return []
    visible = lambda x, y: 0 <= x < len(after["terrain"]) and int(after["terrain"][x][y]) != FOG
    name = lambda u: UNIT_NAMES[u["t"]] if 0 <= u["t"] < len(UNIT_NAMES) else f"unit {u['t']}"
    whose = lambda u: "Your" if u["o"] == me else "Enemy"
    b_units = {u["id"]: u for u in before["units"]}
    a_units = {u["id"]: u for u in after["units"]}
    lines = []
    for uid, u in b_units.items():
        v = a_units.get(uid)
        if v is None:
            if u["o"] == me or visible(u["x"], u["y"]):
                lines.append(f"{whose(u)} {name(u)} at ({u['x']},{u['y']}) was killed")
            continue
        if v["hp"] < u["hp"]:
            lines.append(f"{whose(u)} {name(u)} at ({v['x']},{v['y']}) took {u['hp'] - v['hp']} damage (HP {v['hp']})")
        if (v["x"], v["y"]) != (u["x"], u["y"]) and v["o"] != me:
            lines.append(f"Enemy {name(v)} moved ({u['x']},{u['y']}) → ({v['x']},{v['y']})")
    for uid, v in a_units.items():
        if uid not in b_units and v["o"] != me:
            lines.append(f"Enemy {name(v)} seen at ({v['x']},{v['y']})")
    b_cities = {c["id"]: c for c in before["cities"]}
    for c in after["cities"]:
        old = b_cities.get(c["id"])
        if old is not None and old["o"] != c["o"]:
            what = "capital" if (c["cap"] or old["cap"]) else "city"
            lines.append(f"{'You lost' if old['o'] == me else 'Enemy lost'} the {what} at ({c['x']},{c['y']})")
        elif old is None and c["o"] != me:
            lines.append(f"Enemy city seen at ({c['x']},{c['y']})")
    return lines


def full_board(te) -> dict:
    """True-board snapshot for replays: the exporter's snapshot plus resources, buildings and each unit's max HP."""
    obs = te.get_observation(full_visibility=True)
    snap = full_snapshot(obs)
    board = obs.get("board", {})
    snap["resource"], snap["building"] = board.get("resource", []), board.get("building", [])
    for cid, c in (obs.get("city") or {}).items():  # population toward the next level, for the replay
        if str(cid) in snap["cities"]:
            snap["cities"][str(cid)].update({"pop": int(c.get("population", 0)), "need": int(c.get("population_need", 0))})
    for uid, mh in max_hp_by_unit(te).items():
        if uid in snap["units"]:
            snap["units"][uid]["mh"] = mh
    return snap


def max_hp_by_unit(te) -> Dict[str, int]:
    """Unit id -> max HP from the engine (the observation JSON has only current HP). Max HP follows from the unit's
    type, veteran status and rules, so it is public for any unit you can see."""
    try:
        return {str(u["id"]): int(u["maxHp"]) for u in json.loads(te._env.combatStateJson()).get("units", [])}
    except Exception:
        return {}


def lighthouses(size: int) -> List[List[int]]:
    """2.15 rules: a lighthouse stands on each corner; a tribe finds one by revealing that tile."""
    return [[0, 0], [0, size - 1], [size - 1, 0], [size - 1, size - 1]]


def fogged_board(full: dict, mask: Optional[List[List[int]]], me: int, tribes: dict) -> dict:
    """Tribe `me`'s view of a true-board snapshot in board_view's format (what the engine would show it)."""
    n = len(full["terrain"])
    mask = mask or [[0] * n for _ in range(n)]
    on = lambda x, y: bool(mask[x][y])
    hide = lambda grid, blank: [[grid[x][y] if on(x, y) else blank for y in range(n)] for x in range(n)] if grid else []
    units = [{"id": int(uid), "x": u["x"], "y": u["y"], "t": u["t"], "o": u["o"], "hp": u["hp"], "maxhp": u.get("mh", u["hp"]),
              "vet": u["vet"], "kills": u.get("k", 0)}
             for uid, u in full["units"].items() if u["o"] == me or on(u["x"], u["y"])]
    cities = [{"id": int(cid), "x": c["x"], "y": c["y"], "o": c["o"], "cap": c["cap"], "lvl": c["lvl"], "walls": c["walls"]}
              for cid, c in full["cities"].items() if c["o"] == me or on(c["x"], c["y"])]
    return {"terrain": hide(full["terrain"], FOG), "resource": hide(full.get("resource"), -1),
            "building": hide(full.get("building"), -1), "units": units, "cities": cities, "tribes": tribes}


def visible_mask(obs: dict) -> List[List[int]]:
    return [[0 if int(t) == FOG else 1 for t in row] for row in obs.get("board", {}).get("terrain", [])]


def view_of(full: dict, mask: Optional[List[List[int]]], tribe: int) -> dict:
    """A tribe's fogged view of a true-board snapshot: what the engine shows is every actor on an explored tile."""
    if mask is None:
        mask = [[0] * len(row) for row in full["terrain"]]
    on = lambda x, y: 0 <= x < len(mask) and 0 <= y < len(mask[x]) and bool(mask[x][y])
    return {"units": sorted(uid for uid, u in full["units"].items() if u["o"] == tribe or on(u["x"], u["y"])),
            "cities": sorted(cid for cid, c in full["cities"].items() if c["o"] == tribe or on(c["x"], c["y"])),
            "visible": mask}


# --- one game -----------------------------------------------------------------------------------------------

class GameSession:
    """Runs one game on a game thread. The web server reads `snapshot()` and calls `submit()`."""

    def __init__(self, env, agent: AgentPlayer, map_info, episode_seed: int, settings: dict,
                 record_dir: Optional[Path], player: str = "", human_tribe: int = 1,
                 human_policy: Optional[Callable] = None, move_timeout_s: Optional[float] = None):
        self.env, self.agent, self.map_info, self.episode_seed = env, agent, map_info, int(episode_seed)
        self.map_path = str(map_info.path)
        settings = dict(settings or {})
        # Games saved before CHANGELOG 089 replay without the agent's end-of-turn village captures (tribe 1).
        settings.setdefault("deferred_village_captures", True)
        self.settings, self.record_dir, self.player = settings, record_dir, player
        self.human, self.agent_tribe = int(human_tribe), 1 - int(human_tribe)
        self.human_policy = human_policy           # tests and record replay: f(state) -> index, instead of the browser
        self.move_timeout_s = move_timeout_s
        self.commands: "queue.Queue[dict]" = queue.Queue()
        self.lock = threading.Lock()
        tribes = [map_info.our_tribe, map_info.opponent_tribe]
        model = agent.info.public(with_sha=True) if agent.info else None
        self.state: Dict[str, Any] = {
            "version": 0, "phase": "starting", "map": Path(self.map_path).name, "episode_seed": self.episode_seed,
            "agent": agent.name, "search": bool(getattr(agent, "search", False)), "model": model, "settings": settings,
            "log": [], "human_tribe": self.human,
            "tribes": tribes, "you": tribes[self.human], "opponent": tribes[self.agent_tribe],
            "you_move": "first" if self.human == 0 else "second"}
        self.history: List[dict] = []
        self.frames: List[dict] = []
        self.masks: Dict[int, Optional[List[List[int]]]] = {0: None, 1: None}
        self.record_path: Optional[Path] = None
        self.replay_path: Optional[Path] = None
        self.error: Optional[str] = None
        self._abort = threading.Event()
        self._last_human_view: Optional[dict] = None
        self._need_changes = False
        self._steps_from: Optional[int] = 0     # frames from here on are shown to the human at their next turn
        self._step_seq = 0
        self._opp_view = None
        self._end_frame_done = False
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
        if self.thread.is_alive():
            self.thread.join(timeout=60)

    # game thread: bookkeeping ----------------------------------------------------
    def _publish(self, **fields):
        with self.lock:
            self.state.update(fields)
            self.state["version"] += 1

    def _log(self, side: str, text: str, turn: int):
        with self.lock:
            self.state["log"].append({"side": side, "text": text, "turn": turn})

    def _round(self) -> int:
        """1-based round number. The wrapper counts tribe 0's finished turns (bumped before its END_TURN runs)."""
        n = int(getattr(self.env, "_turn_count", 0))
        java = self.env.tribes_env._env
        try:
            if bool(java.isDone()) or int(java.getActiveTribeID()) != 0:
                return max(1, n)
        except Exception:
            return max(1, n)
        return n + 1

    def _refresh_mask(self, tribe: int) -> dict:
        """Observation of the active tribe (must be `tribe`); records the tiles it has explored."""
        obs = self.env.tribes_env.get_observation()
        mask = visible_mask(obs)
        if self.masks[tribe] is None:  # back-fill earlier frames: a tribe's explored area only grows on its own turn
            for f in self.frames:
                f["_masks"][tribe] = mask
        self.masks[tribe] = mask
        return obs

    def _capture(self, actor: Optional[int], raw: Optional[dict] = None, label: Optional[str] = None):
        full = full_board(self.env.tribes_env)
        before = self.frames[-1]["full"] if self.frames else None
        events = diff_events(before, full) if before is not None else []
        if raw is not None:
            label = describe_action(raw, before, events) if before is not None else str(raw.get("repr"))
        frame = {"turn": self._round(), "label": label or "", "events": events, "full": full,
                 "actor": actor, "action": raw.get("type") if raw else None,
                 "raw": raw.get("repr") if raw else None, "_masks": dict(self.masks), "hits": []}
        if raw is not None and "src_x" in raw:
            frame["src"] = [int(raw["src_x"]), int(raw["src_y"])]
        if raw is not None and "target_x" in raw:
            frame["dst"] = [int(raw["target_x"]), int(raw["target_y"])]
        if before is not None:  # HP changes: damage to the target and retaliation on the attacker
            for uid, u in before["units"].items():
                v = full["units"].get(uid)
                if v is None:
                    frame["hits"].append({"at": [u["x"], u["y"]], "d": -u["hp"], "killed": True, "o": u["o"]})
                elif v["hp"] != u["hp"]:
                    frame["hits"].append({"at": [v["x"], v["y"]], "d": v["hp"] - u["hp"], "killed": False, "o": u["o"]})
        self.frames.append(frame)

    def _mark_search(self):
        """Label the agent's last frame when the tactical search (not the policy) chose that move."""
        if self.frames and getattr(self.agent, "last_source", "policy") == "search" and self.frames[-1]["actor"] == self.agent_tribe:
            self.frames[-1]["label"] += " [search]"
            self.frames[-1]["by"] = "search"

    def _check_abort(self):
        if self._abort.is_set():
            raise GameAborted()

    # game thread: the human -------------------------------------------------------
    def _agent_steps(self, mask, tribes) -> List[dict]:
        """The agent's actions since the human last ended a turn, as the human could see them (for the live
        animation): each with the human's fogged board after it and what to highlight."""
        steps = []
        on = lambda p: p is not None and 0 <= p[0] < len(mask) and 0 <= p[1] < len(mask) and bool(mask[p[0]][p[1]])
        for f in self.frames[self._steps_from:]:
            if f["actor"] != self.agent_tribe or f["action"] in (None, "END_TURN", "RESEARCH_TECH"):
                continue
            src, dst = f.get("src"), f.get("dst")
            hits = [h for h in f["hits"] if on(h["at"]) or h["o"] == self.human]
            if not (on(src) or on(dst) or hits):
                continue
            text = f["label"]
            unit = next((u for u in f["full"]["units"].values() if dst and [u["x"], u["y"]] == dst), None)
            name = UNIT_NAMES[unit["t"]] if unit and 0 <= unit["t"] < len(UNIT_NAMES) else "unit"
            if f["action"] == "MOVE" and not (on(src) and on(dst)):
                text = f"An enemy {name} moved into view at ({dst[0]},{dst[1]})" if on(dst) else \
                       f"An enemy unit moved out of sight from ({src[0]},{src[1]})"
            elif f["action"] == "ATTACK" and not on(src):
                text = "An enemy unit attacked from the fog: " + "; ".join(e["text"] for e in f["events"] if e["kind"] in ("damaged", "killed"))
                src = None
            steps.append({"text": text, "action": f["action"], "src": src if on(src) else None, "dst": dst if on(dst) else None,
                          "hits": hits, "board": fogged_board(f["full"], mask, self.human, tribes)})
        return steps

    def _wait_for_human(self, obs: dict, legal: List[dict]) -> dict:
        view = board_view(obs, max_hp_by_unit(self.env.tribes_env))
        if self.settings.get("rules") == "polytopia_2_15":
            n = len(view["terrain"])
            agent_mask = self.masks[self.agent_tribe]
            view["lighthouses"] = []
            for p in lighthouses(n):
                found = int(view["terrain"][p[0]][p[1]]) != FOG
                # As in the game: on a lighthouse you have found, you also see whether the opponent has found it.
                by = [self.human] if found else []
                if found and agent_mask is not None and agent_mask[p[0]][p[1]]:
                    by.append(self.agent_tribe)
                view["lighthouses"].append({"at": p, "found": found, "found_by": by})
        turn = self._round()
        if self._steps_from is not None:  # a new turn: hand the agent's visible moves to the browser to animate
            mask = self.masks[self.human]
            self._step_seq += 1
            with self.lock:
                self.state["agent_steps"] = {"id": self._step_seq, "steps": self._agent_steps(mask, view["tribes"])}
            self._steps_from = None
        if self._need_changes:
            for line in seen_changes(self._last_human_view, view, self.human) or ["(nothing you can see changed)"]:
                self._log("agent", line, turn)
            self._need_changes = False
        self._current_view = view
        self._publish(phase="your_turn", board=view, legal=legal, turn=turn, notice=None)
        if self.human_policy is not None:
            return {"act": int(self.human_policy(self.snapshot()))}
        try:
            command = self.commands.get(timeout=self.move_timeout_s)
        except queue.Empty:
            raise GameAborted("timed out waiting for a move")
        if command.get("abort"):
            raise GameAborted()
        if command.get("resign"):
            self.history.append({"side": self.human, "resign": True})
            raise GameAborted("resigned")
        if not any(item["i"] == int(command["act"]) for item in legal):
            self._publish(notice=f"Action {command['act']} is not legal now")
            return {}
        return command

    def _human_turn_raw(self):
        """Human as tribe 1: any engine action, stepped directly (the agent's turn resumes when this returns)."""
        java = self.env.tribes_env._env
        while not bool(java.isDone()) and int(java.getActiveTribeID()) == self.human:
            self._check_abort()
            obs = self._refresh_mask(self.human)
            actions = self.env.tribes_env.list_actions()
            command = self._wait_for_human(obs, legal_view(raw_pairs(actions)))
            if "act" not in command:
                continue
            idx = int(command["act"])
            raw = actions[idx]
            self.history.append({"side": self.human, "raw": idx, "type": raw.get("type"), "repr": raw.get("repr")})
            self._log("you", describe(raw), self._round())
            java.stepByIndex(idx)
            if not bool(java.isDone()) and int(java.getActiveTribeID()) == self.human:
                self._refresh_mask(self.human)
            self._capture(self.human, raw)
            if raw.get("type") == "END_TURN":
                self._last_human_view, self._need_changes = self._current_view, True
                self._steps_from = len(self.frames)
        self._publish(phase="agent_turn", legal=[])

    def _human_pick_wrapper(self, obs_vec, info) -> int:
        """Human as tribe 0: one of the wrapper's legal actions (the agent's own action interface)."""
        while True:
            self._check_abort()
            obs = self._refresh_mask(self.human)
            id_to_raw = self.env._current_legal_id_to_raw_index
            actions = self.env._current_legal_actions
            command = self._wait_for_human(obs, legal_view((int(g), actions[r]) for g, r in id_to_raw.items()))
            if "act" in command:
                return int(command["act"])

    # game thread: the agent as tribe 1 ----------------------------------------------
    def _agent_turn_external(self):
        """The agent as tribe 1, the way league snapshots play (combat_selfplay.OpponentView)."""
        from pol_env.Tribes.py.combat_selfplay import OpponentView, play_deferred_village_captures, swap_tribes

        te = self.env.tribes_env
        java = te._env
        if self._opp_view is None:
            self._opp_view = OpponentView(self.env)
        v = self._opp_view.view
        while not bool(java.isDone()) and int(java.getActiveTribeID()) == self.agent_tribe:
            self._check_abort()
            obs = self._refresh_mask(self.agent_tribe)
            actions = te.list_actions()
            s_obs, s_actions = swap_tribes(obs, actions, 0, self.agent_tribe)
            # Its turn count is the number of turns it has finished (seat-swap training): one less than the
            # wrapper's, which already counts the turn tribe 0 just ended.
            v._turn_count = max(0, int(getattr(self.env, "_turn_count", 0)) - 1)
            mask, id_to_raw, _diag = v._build_action_mask_and_mapping(s_actions, obs=s_obs)
            ids, valid, _count = v._build_legal_slot_tensors(mask)
            feats = v._build_legal_action_features_padded(ids, valid, id_to_raw, s_actions, s_obs)
            vec = v._dict_to_array(s_obs)
            gid = self.agent.act(vec, np.asarray(ids), np.asarray(valid, dtype=bool), np.asarray(feats, dtype=np.float32),
                                 raw_map=id_to_raw)
            idx = id_to_raw.get(int(gid))
            if idx is None:
                raise RuntimeError(f"The agent chose global id {gid} with no engine action")
            raw = actions[int(idx)]
            if (str(raw.get("type", "")).upper() == "END_TURN" and v._queued_village_capture_unit_ids
                    and self.settings.get("deferred_village_captures")):
                # The wrapper's rule: with one city a unit on a village is frozen and captures as the turn ends
                # (the learner's step() does this; CHANGELOG 089). Replays redo it from the END_TURN choice.
                for cap in play_deferred_village_captures(te, self._opp_view, self.agent_tribe):
                    self._capture(self.agent_tribe, {"type": "CAPTURE", "repr": "village capture (end of turn)"})
                actions = te.list_actions()
                idx = next(i for i, a in enumerate(actions) if a.get("type") == "END_TURN")
                raw = actions[int(idx)]
            self.history.append({"side": self.agent_tribe, "gid": int(gid), "raw": int(idx), "type": raw.get("type"),
                                 "repr": raw.get("repr"), "by": getattr(self.agent, "last_source", "policy")})
            java.stepByIndex(int(idx))
            if not bool(java.isDone()) and int(java.getActiveTribeID()) == self.agent_tribe:
                self._refresh_mask(self.agent_tribe)
            self._capture(self.agent_tribe, raw)
            self._mark_search()

    def _tribe1_turn(self):
        """Engine callback after tribe 0 ends its turn: tribe 1 is played from Python."""
        self._capture(0, label="End turn")
        self._end_frame_done = True
        if self.human == 1:
            self._human_turn_raw()
        else:
            self._agent_turn_external()

    # game thread: main loop (tribe 0 through the wrapper) -----------------------------
    def _run(self):
        env = self.env
        outcome = None
        try:
            from pol_env.Tribes.py.combat_selfplay import EXTERNAL_OPPONENT

            def select_opponent():  # replaces the league hook for this game: tribe 1 is played from Python
                env.tribes_env._env.setOpponentAgent(EXTERNAL_OPPONENT, 0)
                env.tribes_env.opponent_turn_hook = self._tribe1_turn

            env._select_combat_opponent = select_opponent
            env._level_pool, env._level_pool_size = [self.map_path], 1
            self.agent.reseed(self.episode_seed * 31 + 7)
            obs, info = env.reset(seed=self.episode_seed)
            self._refresh_mask(0)
            self._capture(None, label="Start")
            self._publish(phase="agent_turn")
            terminated = truncated = False
            while not (terminated or truncated):
                self._check_abort()
                if self.agent_tribe == 0:
                    gid = self.agent.choose(obs, info)
                else:
                    gid = self._human_pick_wrapper(obs, info)
                raw = env._current_legal_actions[env._current_legal_id_to_raw_index[int(gid)]]
                entry = {"side": 0, "gid": int(gid), "type": raw.get("type"), "repr": raw.get("repr")}
                if self.agent_tribe == 0:
                    entry["by"] = getattr(self.agent, "last_source", "policy")
                self.history.append(entry)
                if self.human == 0:
                    self._log("you", describe(raw), self._round())
                    if raw.get("type") == "END_TURN":
                        self._last_human_view, self._need_changes = self._current_view, True
                        self._steps_from = len(self.frames) + 1  # after this END_TURN's own frame
                self._end_frame_done = False
                obs, _r, terminated, truncated, info = env.step(int(gid))
                if raw.get("type") != "END_TURN":
                    if not (terminated or truncated):
                        self._refresh_mask(0)
                    self._capture(0, raw)
                    if self.agent_tribe == 0:
                        self._mark_search()
                elif not self._end_frame_done:  # the game ended without tribe 1 moving again
                    self._capture(0, label="End turn")
            outcome = str(info.get("combat_outcome"))  # tribe 0's result
        except GameAborted as exc:
            outcome = "resigned" if str(exc) == "resigned" else "aborted"
        except Exception:
            self.error = traceback.format_exc()
            outcome = "error"
        finally:
            env.tribes_env.opponent_turn_hook = None
        if outcome == "resigned":
            human_result = "loss"
        elif outcome in ("win", "loss"):
            human_result = outcome if self.human == 0 else {"win": "loss", "loss": "win"}[outcome]
        else:
            human_result = outcome
        self.outcome, self.result = outcome, human_result
        final = None
        if outcome not in ("aborted", "error"):
            try:
                final = board_view(env.tribes_env.get_observation(full_visibility=True), max_hp_by_unit(env.tribes_env))
            except Exception:
                pass
        self._save()
        self._publish(phase="over", tribe0_outcome=outcome, result=human_result, legal=[], error=self.error,
                      final_board=final, turn=self._round(), has_replay=bool(self.frames) and outcome != "error",
                      record=str(self.record_path) if self.record_path else None,
                      replay=str(self.replay_path) if self.replay_path else None)

    # replay and record ------------------------------------------------------------
    def replay_payload(self) -> dict:
        frames = []
        for f in self.frames:
            g = {k: v for k, v in f.items() if k != "_masks"}
            g["views"] = {str(t): view_of(f["full"], f["_masks"].get(t) or self.masks.get(t), t) for t in (0, 1)}
            frames.append(g)
        tribes = self.state["tribes"]
        who = self.player or "Human"
        model = self.state.get("model") or {}
        model_name = f"{model.get('run')} {('step ' + format(model['step'], ',')) if model.get('step') else 'final'}" if model else "random legal"
        players = {str(self.human): f"{who} ({tribes[self.human]})", str(self.agent_tribe): f"Agent ({tribes[self.agent_tribe]})"}
        word = {"win": f"{who} won", "loss": "Agent won", "timeout": "Timeout", "aborted": "Abandoned"}.get(self.result, self.result)
        if self.outcome == "resigned":
            word = f"{who} resigned"
        return {
            "title": f"{who} vs {model_name}", "page_title": f"Replay {Path(self.map_path).name}",
            "map": Path(self.map_path).name, "episode_seed": self.episode_seed, "pool": "eval_validation",
            "policy": self.agent.name, "model": f"{model.get('file')} (sha256 {model.get('sha256', '')[:12]})" if model else "random legal",
            "environment_version": model.get("env_version") if model else None, "settings": self.settings,
            "tribes": tribes, "players": players,
            "short_names": {str(self.human): "You" if not self.player else self.player, str(self.agent_tribe): "Agent"},
            "outcome": self.result if self.result in ("win", "loss", "timeout") else "loss",
            "outcome_text": word, "end_turn": self._round(), "decisions": len(self.history),
            "mark_pov": self.human, "blue_tribe": self.human,
            "lighthouses": str(self.settings.get("rules")) == "polytopia_2_15",
            "unit_names": UNIT_NAMES, "unit_letters": UNIT_LETTERS, "terrain_names": TERRAIN_NAMES, "frames": frames,
        }

    def replay_html(self) -> str:
        template = (REPO_ROOT / "tools/combat_replay_template.html").read_text(encoding="utf-8")
        data = json.dumps(self.replay_payload(), separators=(",", ":")).replace("</", "<\\/")
        return template.replace("/*REPLAY_DATA*/null", data)

    def _save(self):
        if self.record_dir is None or self.outcome == "error":
            return
        if self.outcome == "aborted" and not any(a.get("side") == self.human for a in self.history):
            return  # abandoned before the human moved: nothing worth keeping
        self.record_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = self.record_dir / f"{stamp}_{Path(self.map_path).stem}_{self.episode_seed}_human{self.human}"
        model = self.state.get("model")
        record = {"record_version": RECORD_VERSION, "created_utc": stamp, "player": self.player,
                  "map": Path(self.map_path).name, "map_path": self.map_path, "episode_seed": self.episode_seed,
                  "agent": self.agent.name, "agent_search": bool(getattr(self.agent, "search", False)),
                  "agent_search_moves": int(getattr(self.agent, "search_moves", 0)),
                  "model": model, "agent_tribe": self.agent_tribe, "human_tribe": self.human,
                  "tribes": self.state.get("tribes"), "settings": self.settings, "tribe0_outcome": "abandoned" if self.outcome == "aborted" else self.outcome,
                  "human_result": "abandoned" if self.outcome == "aborted" else self.result,
                  "turns": self._round(), "actions": self.history}
        self.record_path = base.with_suffix(".json")
        self.record_path.write_text(json.dumps(record, indent=1), encoding="utf-8")
        self.replay_path = base.with_suffix(".html")
        self.replay_path.write_text(self.replay_html(), encoding="utf-8")


def replay_record(env, record: dict, map_info):
    """Replay a saved game action by action. Returns (tribe 0's outcome, replayed history, session)."""
    actions = [a for a in record["actions"] if not a.get("resign")]
    human_t = int(record["human_tribe"])
    human = iter([a for a in actions if a["side"] == human_t])
    agent = iter([a for a in actions if a["side"] != human_t])

    class Scripted(AgentPlayer):
        def __init__(self):
            self.info, self.agent, self.sampled = None, None, False

        def reseed(self, seed):
            pass

        def act(self, *args, **kwargs):
            return self.choose(None, None)

        def choose(self, obs, info):
            step = next(agent, None)
            if step is None:
                raise GameAborted("abandoned")
            return int(step["gid"])

    abandoned = record.get("tribe0_outcome") == "abandoned"

    def human_policy(state):
        step = next(human, None)
        if step is None:  # the human resigned (or left) here
            raise GameAborted("abandoned" if abandoned else "resigned")
        return int(step["gid"] if human_t == 0 else step["raw"])

    settings = dict(record.get("settings", {}))
    settings.setdefault("deferred_village_captures", False)  # recorded before CHANGELOG 089
    session = GameSession(env, Scripted(), map_info, record["episode_seed"], settings, None,
                          human_tribe=human_t, human_policy=human_policy)
    if isinstance(record.get("model"), dict):
        session.state["model"] = record["model"]
    elif record.get("model"):  # human_game_v1 stored only the file name
        session.state["model"] = {"run": record["model"], "file": record["model"], "step": None, "sha256": ""}
    session._run()
    if session.error:
        raise RuntimeError(session.error)
    return ("abandoned" if session.outcome == "aborted" else session.outcome), session.history, session


# --- web server ---------------------------------------------------------------------------------------------

class Server:
    def __init__(self, models: List[ModelInfo], maps: List[Any], pool: str, record_dir: Path, player: str,
                 seed: Optional[int], rules: str, max_turns: int, policy_seed: int = 0):
        self.models, self.maps, self.pool = models, maps, pool
        self.by_key = {m.key: m for m in models}
        self.record_dir, self.player, self.rules, self.max_turns = record_dir, player, rules, max_turns
        self.policy_seed = policy_seed
        self.rng = random.Random(seed)
        self.env = None
        self.env_settings: Optional[dict] = None
        self.agent: Optional[AgentPlayer] = None
        self.agent_id: Optional[tuple] = None
        self.session: Optional[GameSession] = None
        self.lock = threading.Lock()

    def close(self):
        if self.session is not None:
            self.session.abort()
        if self.env is not None:
            self.env.close()
            self.env = None

    def _prepare(self, info: Optional[ModelInfo], sampled: bool, search: bool = False):
        """An environment set up for this checkpoint (rebuilt only when its settings differ) and the agent."""
        from pol_env.Tribes.py.environment_contract import read_checkpoint_metadata
        from tools.evaluate_combat import environment_settings

        meta = read_checkpoint_metadata(str(info.path)) if info else None
        wanted = environment_settings(meta, self.rules, self.max_turns) + (info.env_version if info else None,)
        if self.env is None or self.env_settings.get("_wanted") != wanted:
            if self.env is not None:
                self.env.close()
                self.env, self.agent, self.agent_id = None, None, None
            self.env, settings = make_env(info.path if info else None, self.pool, self.rules, self.max_turns)
            self.env_settings = dict(settings, _wanted=wanted)
        agent_id = (info.key if info else None, bool(sampled), bool(search) and info is not None)
        if self.agent is None or self.agent_id != agent_id:
            self.agent = AgentPlayer(info, self.env, sampled, self.policy_seed, search=bool(search) and info is not None)
            self.agent_id = agent_id
        return {k: v for k, v in self.env_settings.items() if k != "_wanted"}

    def new_game(self, model_key: Optional[str] = None, map_name: Optional[str] = None,
                 episode_seed: Optional[int] = None, seat: str = "second", sampled: bool = False,
                 search: bool = False, your_tribe: Optional[str] = None, agent_tribe: Optional[str] = None) -> GameSession:
        with self.lock:
            if self.session is not None:
                self.session.abort()
            if model_key in (None, "", "random"):
                info = None if model_key == "random" or not self.models else self.models[0]
            else:
                info = self.by_key.get(model_key)
                if info is None:
                    raise ValueError("Unknown model")
            if seat == "random":
                seat = self.rng.choice(["first", "second"])
            if seat not in ("first", "second"):
                raise ValueError("seat must be first, second or random")
            your_tribe, agent_tribe = _tribe_choice(your_tribe), _tribe_choice(agent_tribe)
            if your_tribe and your_tribe == agent_tribe:
                raise ValueError("Choose two different tribes: the engine cannot play a tribe against itself")
            if map_name:
                matches = [m for m in self.maps if m.filename == Path(map_name).name]
                if not matches:
                    raise ValueError(f"{map_name} is not in the pool")
                game_map = matches[0]
            elif your_tribe or agent_tribe:
                # Tribe 0 moves first. A pool map with this pairing if there is one, else a new map generated for it
                # (the tribes shape the terrain and resources, so the map has to be made for them).
                want0, want1 = (your_tribe, agent_tribe) if seat == "first" else (agent_tribe, your_tribe)
                fits = [m for m in self.maps if (want0 is None or m.our_tribe == want0)
                        and (want1 is None or m.opponent_tribe == want1)]
                if fits:
                    game_map = self.rng.choice(fits)
                else:
                    t0 = want0 or self.rng.choice([t for t in PLAYABLE_TRIBES if t != want1])
                    t1 = want1 or self.rng.choice([t for t in PLAYABLE_TRIBES if t != t0])
                    game_map = generate_map(t0, t1, self.rng.randrange(100000, 999999))
            else:
                game_map = self.rng.choice(self.maps)
            seed = int(episode_seed) if episode_seed is not None else self.rng.randrange(1, 2**31 - 1)
            settings = self._prepare(info, sampled, search)
            self.session = GameSession(self.env, self.agent, game_map, seed, settings, self.record_dir, self.player,
                                       human_tribe=0 if seat == "first" else 1)
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
                    return self._json(200, s.snapshot() if s else {"phase": "none", "version": 0})
                if path == "/api/catalog":
                    return self._json(200, {"models": [m.public() for m in server.models],
                                            "maps": [m.filename for m in server.maps],
                                            "unit_names": UNIT_NAMES, "player": server.player})
                if path == "/api/games":
                    return self._json(200, {"games": list_games(server.record_dir), "dir": str(server.record_dir)})
                if path.startswith("/saved/"):
                    name = Path(path[len("/saved/"):]).name  # no directories: only files in the record folder
                    f = server.record_dir / name
                    if not name.endswith(".html") or not f.is_file():
                        return self._send(404, b"No such saved replay.", "text/plain")
                    return self._send(200, f.read_bytes(), "text/html; charset=utf-8")
                if path == "/replay":
                    s = server.session
                    if s is None or s.state.get("phase") != "over" or not s.state.get("has_replay"):
                        return self._send(409, b"The replay is available once the game is over.", "text/plain")
                    return self._send(200, s.replay_html().encode("utf-8"), "text/html; charset=utf-8")
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
                        server.new_game(body.get("model") or None, body.get("map") or None, body.get("seed"),
                                        str(body.get("seat") or "second"), bool(body.get("sampled")),
                                        bool(body.get("search")), body.get("your_tribe"), body.get("agent_tribe"))
                    except (ValueError, TypeError) as exc:
                        return self._json(400, {"error": str(exc)})
                    except Exception as exc:  # e.g. a checkpoint that does not fit this environment
                        return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
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


def list_games(record_dir: Path) -> List[dict]:
    """Saved games, newest first: what the page's "Past games" list shows."""
    games = []
    for path in sorted(record_dir.glob("*.json"), reverse=True):
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not str(r.get("record_version", "")).startswith("human_game_"):
            continue
        model = r.get("model")
        if isinstance(model, dict):
            name = f"{model.get('run')} " + (f"step {model['step']:,}" if model.get("step") else "final")
        else:
            name = str(model or "random legal")
        human = r.get("human_result")
        if human is None:  # human_game_v1: the agent was tribe 0 and its outcome is recorded
            human = {"win": "loss", "loss": "win"}.get(r.get("agent_outcome"), r.get("agent_outcome"))
        html = path.with_suffix(".html")
        games.append({"file": path.name, "replay": html.name if html.is_file() else None, "created": r.get("created_utc"),
                      "map": r.get("map"), "seed": r.get("episode_seed"), "player": r.get("player"), "model": name,
                      "you_moved": "first" if r.get("human_tribe") == 0 else "second", "result": human,
                      "turns": r.get("turns")})
    return games


def load_maps(pool: str):
    from tools.combat_eval_core import load_verified_pool
    _, maps, _ = load_verified_pool(REPO_ROOT, pool)
    return maps


PLAYABLE_TRIBES = ("BARDUR", "IMPERIUS", "XIN_XI", "OUMAJI")
GENERATED_MAP_DIR = REPO_ROOT / "runs" / "generated_maps"


def _tribe_choice(name: Optional[str]) -> Optional[str]:
    """A tribe name from the page ("any" or empty: no preference)."""
    t = str(name or "").strip().upper().replace("-", "_")
    if t in ("", "ANY"):
        return None
    if t not in PLAYABLE_TRIBES:
        raise ValueError(f"tribe must be one of {', '.join(PLAYABLE_TRIBES)} or any")
    return t


def generate_map(tribe0: str, tribe1: str, seed: int):
    """A new 11x11 Drylands map with tribe0 moving first, made by the same generator as the pools
    (core.levelgen.GenerateLevelCli, as tools/generate_phase2_maps.py). Kept in runs/generated_maps/ so records
    replay."""
    import shutil
    import subprocess

    from tools.combat_eval_core import CombatMap
    from tools.phase1_eval_core import sha256_file

    GENERATED_MAP_DIR.mkdir(parents=True, exist_ok=True)
    out = GENERATED_MAP_DIR / f"map_2p_gen_{seed:06d}_{tribe0.lower()}_vs_{tribe1.lower()}.csv"
    jdks = sorted((REPO_ROOT / ".runtime").glob("jdk-*"))
    java = str(jdks[-1] / "bin" / "java") if jdks else (shutil.which("java") or "java")
    tribes_dir = REPO_ROOT / "pol_env" / "Tribes"
    res = subprocess.run([java, "-cp", "out:lib/json.jar", "core.levelgen.GenerateLevelCli", str(seed), "11",
                          str(out), "1.0", "DRYLANDS", tribe0, tribe1], cwd=tribes_dir, capture_output=True, text=True)
    if res.returncode != 0 or not out.is_file():
        raise RuntimeError(f"Map generation failed: {res.stderr.strip()[-300:]}")
    return CombatMap(out.name, str(out.resolve()), sha256_file(out), int(seed), tribe0, tribe1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", action="append", default=[],
                   help="Glob of checkpoints to offer (repeatable; quote it). This checkout's runs/ is always searched.")
    p.add_argument("--only-models", action="store_true",
                   help="Offer only the --models globs (do not also search this checkout's runs/)")
    p.add_argument("--map-pool", default=None,
                   help="Map pool version under pol_env/Tribes/levels (default: POLYVISION_COMBAT_POOL, else "
                        "phase2_pool_1v1; the league runs since CHANGELOG 088 train on phase2_pool_1v1_wiki)")
    p.add_argument("--model-path", type=Path, help="Checkpoint for the first game (default: the newest found)")
    p.add_argument("--seat", default="second", choices=["first", "second", "random"], help="Your seat in the first game")
    p.add_argument("--sampled", action="store_true", help="The agent samples its actions instead of playing argmax")
    p.add_argument("--search", action="store_true", help="The agent plays with the tactical search (ppo_search) in the first game")
    p.add_argument("--policy-seed", type=int, default=0)
    p.add_argument("--map", help="Map file name for the first game (default: a random validation map)")
    p.add_argument("--episode-seed", type=int)
    p.add_argument("--seed", type=int, help="Seed for choosing maps, seeds and random seats")
    p.add_argument("--pool", default="eval_validation", choices=["eval_validation", "eval_test"])
    p.add_argument("--confirm-test", action="store_true", help="Required with --pool eval_test (human protocol only)")
    p.add_argument("--rules", default="tribes_2020", help="Random opponent only: rules to play")
    p.add_argument("--max-turns", type=int, default=20, help="Random opponent only: turn limit")
    p.add_argument("--player", default="", help="Alias stored in the game records (no real names needed)")
    p.add_argument("--record-dir", type=Path, default=REPO_ROOT / "runs" / "human_games")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--verify-record", type=Path, help="Replay a saved game and check every action and the outcome")
    p.add_argument("--rebuild-replay", type=Path, nargs="+", help="Replay saved games and (re)write their replay HTML")
    return p.parse_args()


def verify(record_path: Path, pool: str, rebuild: bool = False) -> bool:
    """Replay a record and check its actions and outcome; with rebuild, (re)write its replay HTML next to it."""
    record = json.loads(record_path.read_text(encoding="utf-8"))
    s = record.get("settings", {})
    env, _ = make_env(None, pool, s.get("rules", "tribes_2020"), int(s.get("max_turns", 20)), recorded=s)
    try:
        found = [m for m in load_maps(pool) if m.filename == record["map"]]
        if found:
            game_map = found[0]
        else:  # a map generated for a tribe choice: its file is kept next to the records' runs/ folder
            from tools.combat_eval_core import CombatMap
            from tools.phase1_eval_core import sha256_file
            path = Path(record["map_path"])
            t = record.get("tribes") or [None, None]
            game_map = CombatMap(path.name, str(path), sha256_file(path), 0, t[0], t[1])
        outcome, history, session = replay_record(env, record, game_map)
        session.player = record.get("player", "")
        if rebuild:
            html = record_path.with_suffix(".html")
            html.write_text(session.replay_html(), encoding="utf-8")
            print(f"wrote {html}")
    finally:
        env.close()
    strip = lambda h: [(a.get("side"), a.get("gid"), a.get("raw"), a.get("repr")) for a in h if not a.get("resign")]
    same_moves = strip(history) == strip(record["actions"])
    recorded = record.get("tribe0_outcome", record.get("agent_outcome"))  # human_game_v1 stored the agent's (tribe 0's)
    ok = same_moves and outcome == recorded
    print(f"{record_path.name}: recorded {recorded}, replayed {outcome}, "
          f"{'same' if same_moves else 'DIFFERENT'} actions — {'OK' if ok else 'MISMATCH'}")
    return ok


def main():
    args = parse_args()
    if args.map_pool:  # before tools.combat_eval_core is imported: it reads the pool version once
        os.environ["POLYVISION_COMBAT_POOL"] = args.map_pool
    if args.pool == "eval_test" and not args.confirm_test:
        raise SystemExit("eval_test maps are for the human protocol only: pass --confirm-test when a human asked for it")
    if args.verify_record:
        sys.exit(0 if verify(args.verify_record, args.pool) else 1)
    if args.rebuild_replay:
        sys.exit(0 if all([verify(p, args.pool, rebuild=True) for p in args.rebuild_replay]) else 1)

    patterns = ([] if args.only_models else [str(REPO_ROOT / "runs" / "**" / "*.cleanrl_model")]) + args.models
    patterns += [g for g in os.environ.get("POLYAI_MODEL_GLOBS", "").split(":") if g]
    if args.model_path:
        patterns.append(str(args.model_path))
    models = find_models(patterns)
    first = None
    if args.model_path:
        first = next((m for m in models if m.path == args.model_path.resolve()), None)
        if first is None:
            raise SystemExit(f"{args.model_path} is not a combat checkpoint with its .action_interface.json sidecar")
    elif models:
        first = models[0]
    print(f"Found {len(models)} combat checkpoints" + ("" if models else " — the opponent will play random legal moves"))

    server = Server(models, load_maps(args.pool), args.pool, args.record_dir, args.player, args.seed,
                    args.rules, args.max_turns, args.policy_seed)
    server.new_game(first.key if first else "random", args.map, args.episode_seed, args.seat, args.sampled, args.search)
    try:
        httpd = ThreadingHTTPServer((args.host, args.port), server.handler())
    except OSError as exc:
        server.close()
        raise SystemExit(f"Cannot listen on {args.host}:{args.port} ({exc.strerror}). Another program (or an "
                         f"`ssh -L {args.port}:...` tunnel started on this machine) is using it: pick another --port. "
                         "The ssh tunnel belongs on your laptop, not on the machine running this server.")
    if first:
        print(f"First game against {first.run} ({first.label()}), {first.path}")
    print(f"Records go to {args.record_dir}")
    print(f"Open http://{args.host}:{args.port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        server.close()


if __name__ == "__main__":
    main()
