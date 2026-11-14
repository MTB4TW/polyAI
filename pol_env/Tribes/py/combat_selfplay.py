"""Self-play opponents for combat (Phase 2): frozen policy snapshots play tribe 1 instead of SimpleAgent.

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


def swap_tribes(obs: dict, actions: Sequence[dict], a: int = 0, b: int = 1):
    """Copies of an observation and its legal actions with tribe ids a and b exchanged everywhere they appear."""
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
        act = dict(act)
        for field in ("tribe_id", "target_tribe_id"):
            if field in act:
                act[field] = _swap_id(act[field], a, b)
        out_actions.append(act)
    return obs, out_actions


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

    def encode(self, obs: dict, legal_actions: List[dict]):
        """(obs vector, legal ids, valid mask, features, global id -> raw action index) for a relabelled view."""
        v = self.view
        v._turn_count = getattr(self.wrapper, "_turn_count", 0)
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
        validate_checkpoint_compatibility(meta, env_meta)  # raises if the snapshot does not fit this environment
        adapter = SimpleNamespace(single_observation_space=wrapper.observation_space,
                                  single_action_space=wrapper.action_space)
        self.agent = make_agent(adapter, network_arch=meta.get("network_arch", "mlp"), actor_mode=meta["actor_mode"],
                                max_legal_actions=meta["max_legal_actions"],
                                legal_action_feature_dim=meta["legal_action_feature_dim"],
                                observation_normalization=observation_normalization_from_metadata(meta))
        self.agent.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.agent.eval()
        self._torch = torch
        self.generator = torch.Generator().manual_seed(0)

    def reseed(self, seed: int):
        self.generator.manual_seed(int(seed) % (2**31 - 1))

    def act(self, vec, ids, valid, feats) -> int:
        torch = self._torch
        with torch.inference_mode():
            dist = self.agent.get_action_distribution(
                torch.as_tensor(vec, dtype=torch.float32).reshape(1, -1),
                legal_global_ids=torch.as_tensor(ids, dtype=torch.long).reshape(1, -1),
                legal_action_valid_mask=torch.as_tensor(valid, dtype=torch.bool).reshape(1, -1),
                legal_action_features=torch.as_tensor(feats, dtype=torch.float32).unsqueeze(0),
            )
            slot = int(torch.multinomial(dist.probs[0], 1, generator=self.generator).item())
        if not valid[slot]:
            raise RuntimeError("Snapshot opponent picked an invalid slot")
        return int(ids[slot])


def play_external_turn(wrapper, opponent: SnapshotOpponent, view: OpponentView, max_actions: int = 2000):
    """Play the active non-zero tribe's whole turn with `opponent`, through the Java bridge."""
    env = wrapper.tribes_env
    java = env._env
    for _ in range(int(max_actions)):
        if bool(java.isDone()) or int(java.getActiveTribeID()) == 0:
            return
        active = int(java.getActiveTribeID())
        obs = env.get_observation()
        actions = env.list_actions()
        s_obs, s_actions = swap_tribes(obs, actions, 0, active)
        vec, ids, valid, feats, id_to_raw = view.encode(s_obs, s_actions)
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
        self.power = float(environ.get("POLYVISION_COMBAT_LEAGUE_PFSP_POWER", "2"))
        self.refresh_every = int(environ.get("POLYVISION_COMBAT_LEAGUE_REFRESH_EVERY", "20"))
        if not 0.0 <= self.simple_prob <= 1.0:
            raise ValueError("POLYVISION_COMBAT_LEAGUE_SIMPLE_PROB must be between 0 and 1")
        self.rng = rng if rng is not None else np.random.default_rng()
        self.stats: Dict[str, List[int]] = {}   # path -> [games, learner wins]
        self.loaded: Dict[str, SnapshotOpponent] = {}
        self.rejected: Dict[str, str] = {}
        self.pool: List[str] = []
        self._games_since_refresh = 10**9

    def candidates(self) -> List[str]:
        paths = []
        if self.self_dir:
            paths += glob.glob(os.path.join(self.self_dir, "model_checkpoint_*.cleanrl_model"))
        for pattern in self.globs:
            paths += glob.glob(pattern)
        paths = [p for p in paths if os.path.isfile(p + ".action_interface.json") and p not in self.rejected]
        paths.sort(key=lambda p: (os.path.getmtime(p), p))
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

    def record(self, path: Optional[str], learner_won: bool):
        if path is None:
            return
        cell = self.stats.setdefault(path, [0, 0])
        cell[0] += 1
        cell[1] += int(bool(learner_won))
