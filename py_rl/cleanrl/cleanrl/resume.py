"""Stop a training run cleanly and continue it later exactly where it stopped (ppo.py --resume).

What a checkpoint holds:
- Every checkpoint ppo.py saves (`X.cleanrl_model`, plus `X.privileged_critic.pt`) now also gets
  `X.cleanrl_model.resume.pt`.
- The bundle holds the optimizer state (Adam moments), the completed iteration and global step (so the
  learning-rate schedule, the teacher decay and the checkpoint numbering continue), the return-normalisation
  state, the Python/NumPy/Torch random states, the privileged-critic guard state, and the run's arguments.
- Weights and the observation-normalisation statistics stay in the checkpoint itself, as before.
- `runs/<run>/LATEST_RESUME.json` points at the newest bundle, so `--resume runs/<run>` works.

How to stop a run:
- `touch runs/<run>/STOP`, or `scancel --signal=USR1 --batch <job>`: the run saves a checkpoint and its bundle
  at the end of the current iteration, then exits.
- The training sbatch scripts ask SLURM for SIGUSR1 ten minutes before the time limit (`--signal=B:USR1@600`), so
  a run that hits the 48 h limit stops the same way. With AUTO_RESUME=1 the job resubmits itself with --resume.
- A plain `scancel` (SIGTERM) saves immediately instead, without waiting for the iteration to finish. The
  weights and optimizer are consistent with each other, but may be part-way through that iteration's update.

What --resume needs:
- The run's structural settings must be unchanged (network, action interface, num_envs, num_steps, ...).
- Schedule settings (learning rate, total timesteps) may change; a warning is printed.
- Environment episodes restart: games in progress when the run stopped are not continued.
- Self-play league statistics live in the env workers and restart.
"""

from __future__ import annotations

import copy
import glob
import json
import os
import random
import re
import signal
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import torch

RESUME_VERSION = "resume_bundle_v1"
RESUME_SUFFIX = ".resume.pt"
STOP_FILE = "STOP"
LATEST_FILE = "LATEST_RESUME.json"

# Settings that change what the network is or how a rollout is shaped: resuming with different values is refused.
STRUCTURAL_ARGS = ("env_id", "network_arch", "actor_mode", "num_envs", "num_steps", "max_legal_actions",
                   "legal_action_feature_dim", "norm_obs", "norm_returns", "privileged_critic",
                   "privileged_critic_value_norm", "gamma", "gamma_per_turn")
# Settings that only change the schedule: allowed, with a warning.
SCHEDULE_ARGS = ("learning_rate", "total_timesteps", "anneal_lr", "teacher", "teacher_coef", "teacher_decay_frac",
                 "num_minibatches", "update_epochs", "ent_coef", "vf_coef", "clip_coef", "target_kl")


def resume_path(checkpoint_path: str) -> str:
    return str(checkpoint_path) + RESUME_SUFFIX


def _plain_args(args) -> Dict[str, Any]:
    out = {}
    for k, v in vars(args).items():
        try:
            json.dumps(v)
            out[k] = v
        except TypeError:
            out[k] = str(v)
    return out


def save_resume_bundle(checkpoint_path: str, *, global_step: int, iteration: int, optimizer, args,
                       return_normalizer=None, extras: Optional[dict] = None, reason: str = "periodic") -> str:
    """Write X.resume.pt next to checkpoint X (atomically) and point LATEST_RESUME.json at it."""
    state = {
        "version": RESUME_VERSION,
        "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reason": reason,
        "checkpoint": os.path.basename(str(checkpoint_path)),
        "global_step": int(global_step),
        "iteration": int(iteration),
        "optimizer": optimizer.state_dict(),
        "return_normalizer": copy.deepcopy(vars(return_normalizer)) if return_normalizer is not None else None,
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        "extras": extras or {},
        "args": _plain_args(args),
    }
    path = resume_path(checkpoint_path)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)
    run_dir = os.path.dirname(os.path.abspath(str(checkpoint_path)))
    latest = {"checkpoint": os.path.abspath(str(checkpoint_path)), "global_step": int(global_step),
              "iteration": int(iteration), "reason": reason, "saved_utc": state["saved_utc"]}
    tmp = os.path.join(run_dir, LATEST_FILE + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(latest, f, indent=1)
    os.replace(tmp, os.path.join(run_dir, LATEST_FILE))
    return path


def find_resume_checkpoint(path: str) -> str:
    """A checkpoint with a resume bundle: the file itself, or a run directory's newest one."""
    path = str(path)
    if os.path.isdir(path):
        latest = os.path.join(path, LATEST_FILE)
        if os.path.isfile(latest):
            with open(latest, encoding="utf-8") as f:
                ckpt = json.load(f)["checkpoint"]
            if os.path.isfile(ckpt) and os.path.isfile(resume_path(ckpt)):
                return ckpt
        found = []
        for b in glob.glob(os.path.join(path, "*.cleanrl_model" + RESUME_SUFFIX)):
            ckpt = b[: -len(RESUME_SUFFIX)]
            m = re.search(r"model_checkpoint_(\d+)\.cleanrl_model$", ckpt)
            if os.path.isfile(ckpt):
                found.append((int(m.group(1)) if m else -1, os.path.getmtime(b), ckpt))
        if not found:
            raise FileNotFoundError(f"No checkpoint with a resume bundle in {path}")
        return max(found)[2]
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if not os.path.isfile(resume_path(path)):
        raise FileNotFoundError(f"{path} has no resume bundle ({RESUME_SUFFIX}); use --init-from for a warm start")
    return path


def load_resume_bundle(checkpoint_path: str) -> dict:
    state = torch.load(resume_path(checkpoint_path), map_location="cpu", weights_only=False)
    if state.get("version") != RESUME_VERSION:
        raise RuntimeError(f"Unsupported resume bundle version {state.get('version')!r}")
    return state


def check_resume_args(saved: Dict[str, Any], args) -> list:
    """Raise on structural differences; return warnings for schedule changes."""
    now = _plain_args(args)
    bad = [f"{k}: run {saved.get(k)!r} -> now {now.get(k)!r}" for k in STRUCTURAL_ARGS
           if k in saved and k in now and saved.get(k) != now.get(k)]
    if bad:
        raise SystemExit("--resume: these settings differ from the run being resumed:\n  " + "\n  ".join(bad))
    return [f"{k}: {saved.get(k)!r} -> {now.get(k)!r}" for k in SCHEDULE_ARGS
            if k in saved and k in now and saved.get(k) != now.get(k)]


def restore_rng(rng: dict):
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])


def restore_return_normalizer(normalizer, saved: Optional[dict]):
    if normalizer is not None and saved is not None:
        normalizer.__dict__.update(copy.deepcopy(saved))


class StopRequests:
    """SIGUSR1 or a STOP file: stop at the end of the iteration. SIGTERM: call `on_terminate` now, then exit."""

    def __init__(self, run_dir: str, on_terminate=None):
        self.run_dir = run_dir
        self.on_terminate = on_terminate
        self.signal_received: Optional[str] = None
        self._saving = False
        self._installed = []
        for sig, handler in ((getattr(signal, "SIGUSR1", None), self._graceful), (signal.SIGTERM, self._terminate)):
            if sig is None:
                continue
            try:
                self._installed.append((sig, signal.signal(sig, handler)))
            except ValueError:  # not the main thread (tests): file-based stop still works
                pass

    def _graceful(self, signum, frame):
        self.signal_received = signal.Signals(signum).name
        print(f"[RESUME] {self.signal_received} received: saving and stopping at the end of this iteration", flush=True)

    def _terminate(self, signum, frame):
        if self._saving:
            return
        self._saving = True
        print("[RESUME] SIGTERM received: saving now", flush=True)
        try:
            if self.on_terminate is not None:
                self.on_terminate()
        finally:
            raise SystemExit(143)

    def stop_file(self) -> str:
        return os.path.join(self.run_dir, STOP_FILE)

    def requested(self) -> Optional[str]:
        if self.signal_received:
            return self.signal_received
        if os.path.exists(self.stop_file()):
            return "STOP file"
        return None

    def restore(self):
        for sig, old in self._installed:
            try:
                signal.signal(sig, old)
            except (ValueError, TypeError):
                pass
