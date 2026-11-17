"""`GameReplayEngine`: the one object the trainer talks to.

It wires a selector (what to keep), a `GameStore` (how much to keep), and a learner (what to learn)
into three trainer calls:

    engine.note_infos(step, infos)          # in the rollout loop; cheap, only reads final_info
    engine.ingest(...rollout buffers...)    # once per iteration, after the rollout
    metrics = engine.update(agent, optimizer, global_step)   # after the PPO update

Everything is off unless `--game-store` is passed, so the default trainer is byte-for-byte unchanged.
The replay update uses its own minibatches and gradient steps after PPO's, with the same optimizer
and gradient clipping, so it can be switched on or off without touching the PPO maths.
"""

from __future__ import annotations

import dataclasses
import os
import time
from typing import Any, Dict, Optional

import numpy as np

from .recorder import EpisodeRecorder
from .registry import make_learner, make_selector
from .store import GameStore


@dataclasses.dataclass
class GameStoreConfig:
    selector: str = "all"  # see registry.py, e.g. "win|surprise"
    learner: str = "sil"  # sil | awr | bc | value | replay_ppo | vtrace
    payload: str = "full"  # full | actions
    budget_mb: float = 1024.0
    max_games: int = 100_000
    alpha: float = 1.0  # sampling priority exponent (0 = uniform)
    half_life: Optional[float] = None  # priority half-life, in games added
    coef: float = 0.1  # weight of the replay loss
    updates: int = 4  # replay gradient steps per iteration
    batch_size: int = 512  # steps per replay minibatch (step learners)
    games_per_batch: int = 16  # games per minibatch
    min_games: int = 20  # wait until the store holds this many games with payload
    start_step: int = 0  # no replay updates before this global step
    obs_dtype: str = "float16"
    codec: str = "xor_zlib"
    save_dir: Optional[str] = None  # persist the store here at the end (and every save_every iterations)
    save_every: int = 0
    load_dir: Optional[str] = None  # start from a store saved by an earlier run
    seed: int = 0


class GameReplayEngine:
    def __init__(self, config: GameStoreConfig, num_envs: int, gamma: float, per_turn: bool, device,
                 max_grad_norm: float = 0.5, extra_meta: Optional[Dict[str, Any]] = None):
        self.cfg = config
        self.gamma = float(gamma)
        self.per_turn = bool(per_turn)
        self.device = device
        self.max_grad_norm = float(max_grad_norm)
        self.selector = make_selector(config.selector)
        self.learner = make_learner(config.learner)
        self.store = GameStore(budget_mb=config.budget_mb, max_games=config.max_games, alpha=config.alpha,
                               half_life=config.half_life, seed=config.seed)
        if config.load_dir:
            self.store.load(config.load_dir)
        self.recorder = EpisodeRecorder(
            num_envs=num_envs,
            sink=self.store.add,
            selector=self.selector,
            payload=config.payload,
            codec=config.codec,
            obs_dtype=np.dtype(config.obs_dtype),
            feature_dtype=np.dtype(config.obs_dtype),
            gamma=gamma,
            extra_meta=extra_meta,
        )
        self.rng = np.random.default_rng(config.seed + 1)
        self._iterations = 0
        self.timing = {"ingest_s": 0.0, "update_s": 0.0}

    def note_infos(self, step, infos):
        self.recorder.note_infos(step, infos)

    def ingest(self, **buffers) -> int:
        t0 = time.perf_counter()
        n = self.recorder.ingest(**buffers)
        self.timing["ingest_s"] = time.perf_counter() - t0
        self._iterations += 1
        if self.cfg.save_dir and self.cfg.save_every > 0 and self._iterations % self.cfg.save_every == 0:
            self.save()
        return n

    def ready(self, global_step: int) -> bool:
        if self.cfg.payload != "full" or self.cfg.updates <= 0 or self.cfg.coef == 0.0:
            return False
        if global_step < self.cfg.start_step:
            return False
        return sum(1 for r in self.store.records() if r.has_payload) >= self.cfg.min_games

    def update(self, agent, optimizer, global_step: int, reward_scale_now: float = 1.0) -> Dict[str, float]:
        """Run the replay gradient steps; returns metrics averaged over them (empty if not ready)."""
        import torch
        from .learners import batch_to_torch

        if not self.ready(global_step):
            return {}
        t0 = time.perf_counter()
        ctx = {"device": self.device, "gamma": self.gamma, "per_turn": self.per_turn,
               "reward_scale_now": reward_scale_now, "rng": self.rng}
        sums: Dict[str, float] = {}
        done = 0
        for _ in range(self.cfg.updates):
            if self.learner.kind == "games":
                batch = self.store.sample_games(self.cfg.games_per_batch)
                if not batch:
                    break
            else:
                raw = self.store.sample_steps(self.cfg.batch_size, self.cfg.games_per_batch, self.gamma,
                                              self.per_turn, reward_scale_now)
                if raw is None:
                    break
                batch = batch_to_torch(raw, self.device)
            loss, metrics = self.learner.loss(agent, batch, ctx)
            if loss is None:
                break
            optimizer.zero_grad()
            (self.cfg.coef * loss).backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), self.max_grad_norm)
            optimizer.step()
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            done += 1
        self.timing["update_s"] = time.perf_counter() - t0
        return {k: v / max(1, done) for k, v in sums.items()}

    def log_metrics(self) -> Dict[str, float]:
        out = {f"store_{k}": v for k, v in self.store.stats().items()}
        out["recorder_games_seen"] = float(self.recorder.games_seen)
        out["recorder_games_kept"] = float(self.recorder.games_kept)
        out.update({f"time_{k}": v for k, v in self.timing.items()})
        return out

    def save(self):
        if self.cfg.save_dir:
            os.makedirs(self.cfg.save_dir, exist_ok=True)
            return self.store.save(self.cfg.save_dir)
        return []
