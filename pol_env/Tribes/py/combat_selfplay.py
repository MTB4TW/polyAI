"""Self-play opponents for combat (Phase 2): frozen policy snapshots play the learner's opponent instead of SimpleAgent.

How it works
- Java: `PythonEnv.setOpponentAgent("external", ...)` leaves tribe 1's turn to Python (no Java agent runs), and
  `stepByIndex`, `observationJson` and `listActionsJson*` act for whichever tribe is active.
- The wrapper's observation and legal-action code assume the controlled tribe is tribe 0. Rather than changing
  that code, the opponent's (fogged, own-POV) observation and legal actions are relabelled so tribe 1 appears as
  tribe 0 (`swap_tribes`), and an `OpponentView` (a copy of the wrapper with its own per-episode state) encodes
  them exactly as the learner's own inputs are encoded. So a snapshot of the learner plays tribe 1 with the same
  network, observation and action interface it was trained with.
- `CombatLeague` picks the opponent of each game: SimpleAgent with probability `simple_prob`, otherwise a
  snapshot chosen by prioritised fictitious self-play (snapshots the learner beats less often are picked more).

Settings (environment variables, read by the wrapper in combat mode):
  POLYVISION_COMBAT_LEAGUE=1                  enable
  POLYVISION_COMBAT_LEAGUE_SELF_DIR           folder of the learner's own checkpoints (ppo.py sets it when
                                              POLYVISION_COMBAT_LEAGUE_SELF=1)
  POLYVISION_COMBAT_LEAGUE_GLOB               extra snapshot globs, separated by ':' (e.g. other runs)
  POLYVISION_COMBAT_LEAGUE_SIMPLE_PROB        share of games against SimpleAgent (default 0.5)
  POLYVISION_COMBAT_LEAGUE_MAX_SNAPSHOTS      newest snapshots kept in the pool (default 8)
  POLYVISION_COMBAT_LEAGUE_PFSP_POWER         p in (1 - learner win rate)^p (default 2)
  POLYVISION_COMBAT_LEAGUE_MAX_SELF           if > 0: keep every GLOB opponent and the newest N own checkpoints
                                              (instead of the newest MAX_SNAPSHOTS of all; default 0)
  POLYVISION_COMBAT_LEAGUE_SWAP_PROB          share of games where the learner plays tribe 1 (moves second, the
                                              opponent's usual tribe and start) and the opponent tribe 0 (default 0)

Seat swap: in a swapped game Java's controlled tribe is 1 (`PythonEnv.setControlledTribe`), the opponent moves
first, and `TribesGymEnv` relabels every observation, legal action, score and combat state so the learner still
appears as tribe 0 (same encoder, mask, reward and outcome code). Each side plays MAX_TURNS turns in either seat.
Evaluation never enables the league, so it always has the learner as tribe 0.
"""

from __future__ import annotations

import copy
import glob
import os
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import numpy as np

EXTERNAL_OPPONENT = "external"
SIMPLE_OPPONENT = "simple"
# Wrapper attributes shared (not copied) by the opponent view: the Java bridge, the catalog and large constants.
_SHARED_ATTRS = {"tribes_env", "_catalog", "observation_space", "action_space", "_level_pool"}


# --- relabelling tribes ---------------------------------------------------------------------------------------

def _swap_id(value, a: int, b: int):
    try:
        v = int(value)
    except (TypeError, ValueError):
        return value
    return b if v == a else a if v == b else v


def swap_tribes(obs: dict, actions: Sequence[dict], a: int = 0, b: int = 1, inplace: bool = False):
    """Copies of an observation and its legal actions with tribe ids a and b exchanged everywhere they appear
    (inplace=True edits `obs` and the action dicts themselves; for freshly parsed JSON nobody else holds)."""
    if a == b:
        return obs, list(actions)
    if not inplace:
        obs = copy.deepcopy(obs)
    tribes = obs.get("tribes")
    if isinstance(tribes, dict):
        swapped = {}
        for key, info in tribes.items():
            new_key = str(_swap_id(key, a, b))
            if isinstance(info, dict) and isinstance(info.get("diplomacy"), dict):
                for name, values in info["diplomacy"].items():
                    if isinstance(values, list) and max(a, b) < len(values):
                        values[a], values[b] = values[b], values[a]
            swapped[new_key] = info
        obs["tribes"] = swapped
    for unit in (obs.get("unit") or {}).values():
        if isinstance(unit, dict) and "tribeId" in unit:
            unit["tribeId"] = _swap_id(unit["tribeId"], a, b)
    for city in (obs.get("city") or {}).values():
        if isinstance(city, dict) and "tribeID" in city:
            city["tribeID"] = _swap_id(city["tribeID"], a, b)
    if "activeTribeID" in obs:
        obs["activeTribeID"] = _swap_id(obs["activeTribeID"], a, b)
    out_actions = []
    for act in actions:
        act = act if inplace else dict(act)
        for field in ("tribe_id", "target_tribe_id"):
            if field in act:
                act[field] = _swap_id(act[field], a, b)
        out_actions.append(act)
    return obs, out_actions


def swap_combat_state(state: dict, a: int = 0, b: int = 1) -> dict:
    """`PythonEnv.combatStateJson` output with tribe ids a and b exchanged (in place; returns `state`)."""
    if a == b or not isinstance(state, dict):
        return state
    tribes = state.get("tribes")
    if isinstance(tribes, dict):
        state["tribes"] = {str(_swap_id(k, a, b)): v for k, v in tribes.items()}
    for key in ("units", "cities"):
        for item in state.get(key) or []:
            if isinstance(item, dict) and "tribe" in item:
                item["tribe"] = _swap_id(item["tribe"], a, b)
    return state


# --- the opponent's view -------------------------------------------------------------------------------------

class OpponentView:
    """Encodes the opponent's turn with the wrapper's own code, on a copy of the wrapper with separate state."""

    def __init__(self, wrapper):
        view = copy.copy(wrapper)
        for name, value in vars(wrapper).items():
            if name in _SHARED_ATTRS:
                continue
            if isinstance(value, (dict, list, set)):
                setattr(view, name, copy.deepcopy(value))
            elif isinstance(value, np.ndarray):
                setattr(view, name, value.copy())
        view._controlled_tribe_id = 0
        self.view = view
        self.wrapper = wrapper

    def encode(self, obs: dict, legal_actions: List[dict], opponent_tribe: int = 1):
        """(obs vector, legal ids, valid mask, features, global id -> raw action index) for a relabelled view.

        The view's turn count is the number of turns the opponent has finished, as the learner's own count is:
        moving after the learner (tribe 1) that is one less than the learner's finished turns, moving first equal."""
        v = self.view
        turns = int(getattr(self.wrapper, "_turn_count", 0))
        learner = int(getattr(getattr(self.wrapper, "tribes_env", None), "learner_seat", 0))
        v._turn_count = max(0, turns - 1) if int(opponent_tribe) > learner else turns
        action_mask, id_to_raw, _diag = v._build_action_mask_and_mapping(legal_actions, obs=obs)
        ids, valid, _count = v._build_legal_slot_tensors(action_mask)
        feats = v._build_legal_action_features_padded(ids, valid, id_to_raw, legal_actions, obs)
        vec = v._dict_to_array(obs)
        return vec, np.asarray(ids), np.asarray(valid, dtype=bool), np.asarray(feats, dtype=np.float32), id_to_raw


# --- snapshot opponents --------------------------------------------------------------------------------------

class SnapshotOpponent:
    """A frozen checkpoint that samples its actions (CPU)."""

    def __init__(self, path: str, wrapper):
        import torch

        from pol_env.Tribes.py.environment_contract import (
            environment_compatibility_metadata, read_checkpoint_metadata, validate_checkpoint_compatibility,
        )
        from py_rl.cleanrl.cleanrl.agent_factory import make_agent
        from py_rl.cleanrl.cleanrl.obs_normalization import observation_normalization_from_metadata

        self.path = path
        meta = read_checkpoint_metadata(path)
        env_meta = environment_compatibility_metadata(wrapper, actor_mode=meta["actor_mode"],
                                                      max_legal_actions=meta["max_legal_actions"])
        observation_space = wrapper.observation_space
        self.strip = None  # (unit-status slice of the observation, number of base action features)
        old_contract = without_unit_status(wrapper, meta, env_meta)
        if old_contract is not None:
            env_meta, observation_space, self.strip = old_contract
        validate_checkpoint_compatibility(meta, env_meta)  # raises if the snapshot does not fit this environment
        adapter = SimpleNamespace(single_observation_space=observation_space,
                                  single_action_space=wrapper.action_space)
        previous = os.environ.get("POLYVISION_UNIT_STATUS")
        if self.strip is not None:
            os.environ["POLYVISION_UNIT_STATUS"] = "0"  # the network sizes itself from the snapshot's layout
        try:
            self.agent = make_agent(adapter, network_arch=meta.get("network_arch", "mlp"), actor_mode=meta["actor_mode"],
                                    max_legal_actions=meta["max_legal_actions"],
                                    legal_action_feature_dim=meta["legal_action_feature_dim"],
                                    observation_normalization=observation_normalization_from_metadata(meta))
        finally:
            if self.strip is not None:
                if previous is None:
                    os.environ.pop("POLYVISION_UNIT_STATUS", None)
                else:
                    os.environ["POLYVISION_UNIT_STATUS"] = previous
        self.agent.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.agent.eval()
        self._torch = torch
        self.generator = torch.Generator().manual_seed(0)
        self.deterministic = False  # True: highest-logit slot (head-to-head evaluation only; the league samples)

    def reseed(self, seed: int):
        self.generator.manual_seed(int(seed) % (2**31 - 1))

    def act(self, vec, ids, valid, feats) -> int:
        torch = self._torch
        if self.strip is not None:  # an older snapshot: drop the unit-status planes and attack-outcome features
            (start, end), base = self.strip
            vec = np.concatenate([np.asarray(vec).reshape(-1)[:start], np.asarray(vec).reshape(-1)[end:]])
            feats = np.asarray(feats)[:, :base]
        with torch.inference_mode():
            dist = self.agent.get_action_distribution(
                torch.as_tensor(vec, dtype=torch.float32).reshape(1, -1),
                legal_global_ids=torch.as_tensor(ids, dtype=torch.long).reshape(1, -1),
                legal_action_valid_mask=torch.as_tensor(valid, dtype=torch.bool).reshape(1, -1),
                legal_action_features=torch.as_tensor(feats, dtype=torch.float32).unsqueeze(0),
            )
            if self.deterministic:
                slot = int(torch.argmax(dist.logits[0]).item())
            else:
                slot = int(torch.multinomial(dist.probs[0], 1, generator=self.generator).item())
        if not valid[slot]:
            raise RuntimeError("Snapshot opponent picked an invalid slot")
        return int(ids[slot])


def without_unit_status(wrapper, meta, env_meta):
    """For a snapshot trained without unit status in a run that has it: (env metadata as that snapshot expects it,
    its observation space, ((start, end) of the unit-status block, base feature count)). None otherwise."""
    from pol_env.Tribes.py.environment_contract import (
        UNIT_STATUS_FEATURE_VERSION_SUFFIX, UNIT_STATUS_VERSION_SUFFIX, observation_layout, unit_status_of_version,
    )
    if not getattr(wrapper, "UNIT_STATUS", False) or unit_status_of_version(meta.get("phase1_environment_version")):
        return None
    import gymnasium as gym

    layout = observation_layout(int(meta.get("map_width", 11)), int(meta.get("map_height", 11)),
                                getattr(wrapper, "RULES_VERSION", None), getattr(wrapper, "FOG_MEMORY", None), True)
    width = layout.unit_status_end - layout.unit_status_start
    base = len(type(wrapper).LEGAL_ACTION_FEATURE_NAMES)
    env_meta = dict(env_meta)
    env_meta["phase1_environment_version"] = str(env_meta["phase1_environment_version"]).replace(UNIT_STATUS_VERSION_SUFFIX, "")
    env_meta["observation_dim"] = int(env_meta["observation_dim"]) - width
    env_meta["legal_action_feature_dim"] = base
    env_meta["legal_action_feature_version"] = str(env_meta["legal_action_feature_version"]).replace(
        UNIT_STATUS_FEATURE_VERSION_SUFFIX, "")
    space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(int(env_meta["observation_dim"]),), dtype=np.float32)
    return env_meta, space, ((layout.unit_status_start, layout.unit_status_end), base)


def play_external_turn(wrapper, opponent: SnapshotOpponent, view: OpponentView, max_actions: int = 2000):
    """Play the opponent's whole turn (any tribe but the learner's seat) with `opponent`, through the Java bridge.
    Uses the engine's own tribe ids (relabel=False) and relabels them so the opponent appears as tribe 0."""
    env = wrapper.tribes_env
    java = env._env
    learner = int(getattr(env, "learner_seat", 0))
    for _ in range(int(max_actions)):
        if bool(java.isDone()) or int(java.getActiveTribeID()) == learner:
            return
        active = int(java.getActiveTribeID())
        obs = env.get_observation(relabel=False)
        actions = env.list_actions(relabel=False)
        s_obs, s_actions = swap_tribes(obs, actions, 0, active)
        vec, ids, valid, feats, id_to_raw = view.encode(s_obs, s_actions, opponent_tribe=active)
        gid = opponent.act(vec, ids, valid, feats)
        raw = id_to_raw.get(int(gid))
        if raw is None:
            raise RuntimeError(f"Snapshot opponent chose global id {gid} with no raw action")
        java.stepByIndex(int(raw))
    raise RuntimeError("Snapshot opponent exceeded its action budget without ending its turn")


# --- the league ----------------------------------------------------------------------------------------------

class CombatLeague:
    """Chooses each game's opponent and tracks the learner's results against every snapshot (per env worker)."""

    def __init__(self, environ=None, rng: Optional[np.random.Generator] = None):
        environ = os.environ if environ is None else environ
        self.self_dir = str(environ.get("POLYVISION_COMBAT_LEAGUE_SELF_DIR", "") or "")
        self.globs = [g for g in str(environ.get("POLYVISION_COMBAT_LEAGUE_GLOB", "") or "").split(":") if g]
        self.simple_prob = float(environ.get("POLYVISION_COMBAT_LEAGUE_SIMPLE_PROB", "0.5"))
        self.max_snapshots = int(environ.get("POLYVISION_COMBAT_LEAGUE_MAX_SNAPSHOTS", "8"))
        # > 0: every GLOB opponent stays in the pool and only the newest max_self own checkpoints are kept
        # (MAX_SNAPSHOTS is then ignored). 0 (default): the newest MAX_SNAPSHOTS of all candidates, by mtime.
        self.max_self = int(environ.get("POLYVISION_COMBAT_LEAGUE_MAX_SELF", "0"))
        self.power = float(environ.get("POLYVISION_COMBAT_LEAGUE_PFSP_POWER", "2"))
        self.refresh_every = int(environ.get("POLYVISION_COMBAT_LEAGUE_REFRESH_EVERY", "20"))
        self.swap_prob = float(environ.get("POLYVISION_COMBAT_LEAGUE_SWAP_PROB", "0"))
        if not 0.0 <= self.simple_prob <= 1.0:
            raise ValueError("POLYVISION_COMBAT_LEAGUE_SIMPLE_PROB must be between 0 and 1")
        if not 0.0 <= self.swap_prob <= 1.0:
            raise ValueError("POLYVISION_COMBAT_LEAGUE_SWAP_PROB must be between 0 and 1")
        self.rng = rng if rng is not None else np.random.default_rng()
        self.stats: Dict[str, List[int]] = {}   # path -> [games, learner wins]
        self.loaded: Dict[str, SnapshotOpponent] = {}
        self.rejected: Dict[str, str] = {}
        self.pool: List[str] = []
        self._games_since_refresh = 10**9

    def candidates(self) -> List[str]:
        def usable(paths):
            paths = sorted({p for p in paths if os.path.isfile(p + ".action_interface.json") and p not in self.rejected},
                           key=lambda p: (os.path.getmtime(p), p))
            return paths

        own = usable(glob.glob(os.path.join(self.self_dir, "model_checkpoint_*.cleanrl_model"))) if self.self_dir else []
        fixed = usable(p for pattern in self.globs for p in glob.glob(pattern))
        if self.max_self > 0:
            own_set = set(own)
            return [p for p in fixed if p not in own_set] + own[-self.max_self:]
        paths = usable(own + fixed)
        return paths[-self.max_snapshots:] if self.max_snapshots > 0 else paths

    def refresh(self):
        self.pool = self.candidates()
        self._games_since_refresh = 0

    def weights(self, paths: Sequence[str]) -> np.ndarray:
        """PFSP 'hard' weighting: (1 - smoothed learner win rate)^p."""
        w = []
        for p in paths:
            games, wins = self.stats.get(p, [0, 0])
            w.append((1.0 - (wins + 1.0) / (games + 2.0)) ** self.power)
        w = np.asarray(w, dtype=np.float64)
        return w / w.sum() if w.sum() > 0 else np.full(len(paths), 1.0 / max(1, len(paths)))

    def choose(self, wrapper) -> Optional[SnapshotOpponent]:
        """None = SimpleAgent; otherwise a loaded snapshot for this game."""
        if self._games_since_refresh >= self.refresh_every:
            self.refresh()
        self._games_since_refresh += 1
        if not self.pool or self.rng.random() < self.simple_prob:
            return None
        while self.pool:
            path = self.pool[int(self.rng.choice(len(self.pool), p=self.weights(self.pool)))]
            if path in self.loaded:
                return self.loaded[path]
            try:
                self.loaded[path] = SnapshotOpponent(path, wrapper)
                for old in [p for p in self.loaded if p not in self.pool]:
                    del self.loaded[old]
                return self.loaded[path]
            except Exception as exc:  # incompatible or unreadable snapshot: never offer it again
                self.rejected[path] = f"{type(exc).__name__}: {exc}"
                self.pool.remove(path)
        return None

    def choose_seat(self) -> int:
        """The learner's tribe this game: 1 (second to move, the opponent's usual seat) with probability
        swap_prob, else 0. With swap_prob 0 the random stream is not touched (unchanged league games)."""
        if self.swap_prob <= 0.0:
            return 0
        return int(self.rng.random() < self.swap_prob)

    def label(self, path: Optional[str]) -> str:
        """Short, stable name of an opponent for logging: 'simple', 'self/<checkpoint>' or '<run dir>/<file>'."""
        if path is None:
            return "simple"
        stem = os.path.basename(path)
        if stem.endswith(".cleanrl_model"):
            stem = stem[: -len(".cleanrl_model")]
        parent = os.path.dirname(os.path.abspath(path))
        if self.self_dir and os.path.abspath(self.self_dir) == parent:
            return f"self/{stem}"
        run = os.path.basename(parent)
        # Run folders are Tribes-v0__<exp name>__<seed>__<time>; the experiment name is what people recognise.
        parts = run.split("__")
        if len(parts) >= 3 and parts[0].startswith("Tribes"):
            run = parts[1]
        return f"{run}/{stem}" if stem != run and not stem.startswith(run) else stem

    def record(self, path: Optional[str], learner_won: bool):
        if path is None:
            return
        cell = self.stats.setdefault(path, [0, 0])
        cell[0] += 1
        cell[1] += int(bool(learner_won))


# --- league statistics across env workers (trainer side) ------------------------------------------------------

class LeagueStatsTracker:
    """Per-opponent picks and learner results, aggregated over every env worker from end-of-game infos.

    Each worker's CombatLeague keeps its own counts (they drive its PFSP sampling); this collects the
    `combat_league_opponent` / `combat_league_learner_won` / `combat_end_raw_learner_seat` fields the wrapper
    adds to every finished league game, so the trainer can log how often each opponent is picked and how
    often the learner beats it, over all games and over the last `window` games."""

    def __init__(self, window: int = 1000, power: float = 2.0):
        from collections import deque

        self.total: Dict[str, List[int]] = {}        # label -> [games, learner wins]
        self.recent = deque(maxlen=int(window))      # (label, seat, won)
        self.power = float(power)

    def add(self, row: dict) -> bool:
        if not isinstance(row, dict) or "combat_league_opponent" not in row:
            return False
        label = str(row["combat_league_opponent"])
        won = int(float(row.get("combat_league_learner_won", 0.0)) > 0.5)
        seat = int(float(row.get("combat_end_raw_learner_seat", 0.0)))
        cell = self.total.setdefault(label, [0, 0])
        cell[0] += 1
        cell[1] += won
        self.recent.append((label, seat, won))
        return True

    def scalars(self) -> Dict[str, float]:
        """W&B scalars: league/pick_share/<opp> and league/win_rate/<opp> over the recent window, league/games/<opp>
        cumulative, league/pfsp_weight/<opp> ((1 - smoothed win rate)^p on the recent window, normalised over
        snapshots, as the workers' sampling would weight them), and win rate by seat."""
        out: Dict[str, float] = {}
        n = len(self.recent)
        if n == 0:
            return out
        games: Dict[str, List[int]] = {}
        seats: Dict[int, List[int]] = {}
        for label, seat, won in self.recent:
            g = games.setdefault(label, [0, 0])
            g[0] += 1
            g[1] += won
            s = seats.setdefault(seat, [0, 0])
            s[0] += 1
            s[1] += won
        snap_games = sum(g for lab, (g, _) in games.items() if lab != "simple")
        weights = {lab: (1.0 - (w + 1.0) / (g + 2.0)) ** self.power for lab, (g, w) in games.items() if lab != "simple"}
        wsum = sum(weights.values())
        for label, (g, w) in games.items():
            key = label.replace(" ", "_")
            out[f"league/pick_share/{key}"] = g / n
            out[f"league/win_rate/{key}"] = w / g
            if label != "simple" and snap_games:
                out[f"league/snapshot_pick_share/{key}"] = g / snap_games
                if wsum > 0:
                    out[f"league/pfsp_weight/{key}"] = weights[label] / wsum
        for label, (g, _w) in self.total.items():
            out[f"league/games/{label.replace(' ', '_')}"] = float(g)
        for seat, (g, w) in seats.items():
            out[f"league/win_rate_seat{seat}"] = w / g
            out[f"league/seat{seat}_share"] = g / n
        out["league/opponents_seen"] = float(len(self.total))
        return out

    def table(self) -> List[dict]:
        """Cumulative rows (opponent, games, learner wins, win rate), most-played first, for a JSON dump."""
        rows = [{"opponent": k, "games": g, "learner_wins": w, "win_rate": (w / g if g else None)}
                for k, (g, w) in self.total.items()]
        return sorted(rows, key=lambda r: -r["games"])
