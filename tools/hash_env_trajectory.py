#!/usr/bin/env python3
"""Hash a fixed-seed random-legal-play trajectory of the Phase 1 environment.

Used to prove that performance changes leave environment outputs unchanged: run it before and after a
change with the same arguments and compare the printed digests. Hashed per step: the flattened
observation, reward, terminated/truncated, legal global ids, valid mask and count, the padded legal-action
feature tensor (raw float32 bytes), and every other info value except timing/profile fields.

    python tools/hash_env_trajectory.py --steps 1500 --seed 7

Requires the compiled Java classes (pol_env/Tribes/out). Environment variables are set to the Phase 1
training defaults unless already present.
"""

import argparse
import hashlib
import os
import random
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PHASE1_DEFAULTS = {
    "POLYVISION_LEVEL_POOL_GLOB": "levels/phase1_pool_bardur_real/train/*.csv",
    "POLYVISION_SOLO_NO_OPPONENT_MODE": "1",
    "POLYVISION_INFO_MODE": "fast",
}

# Values that legitimately differ between runs (wall-clock timings and profiler counters).
_SKIP_INFO_PREFIXES = ("profile_",)


def _update_value(h, value):
    if isinstance(value, np.ndarray):
        h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode())
        h.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, dict):
        h.update(b"{")
        for k in sorted(value, key=str):
            h.update(repr(k).encode())
            _update_value(h, value[k])
        h.update(b"}")
    elif isinstance(value, (list, tuple)):
        h.update(b"[")
        for v in value:
            _update_value(h, v)
        h.update(b"]")
    elif isinstance(value, float):
        h.update(np.float64(value).tobytes())
    else:
        h.update(repr(value).encode())


def hash_info(h, info, skip_prefixes=_SKIP_INFO_PREFIXES):
    for key in sorted(info):
        if any(str(key).startswith(p) for p in skip_prefixes):
            continue
        h.update(repr(key).encode())
        _update_value(h, info[key])


def legal_ids_from_info(info):
    ids = np.asarray(info["legal_global_ids_padded"], dtype=np.int64).reshape(-1)
    valid = np.asarray(info["legal_action_valid_mask"], dtype=bool).reshape(-1)
    return ids[valid]


def run(steps, seed, env=None, extra_skip_prefixes=()):
    """Return (combined digest, features-only digest, per-step seconds list).

    extra_skip_prefixes: further info keys to leave out, e.g. ("teacher_",) to check that POLYVISION_TEACHER only
    adds its own keys and leaves every other output unchanged."""
    skip = tuple(_SKIP_INFO_PREFIXES) + tuple(extra_skip_prefixes)
    for k, v in PHASE1_DEFAULTS.items():
        os.environ.setdefault(k, v)
    from pol_env.Tribes.py.register_env import TribesGymWrapper

    own_env = env is None
    if own_env:
        env = TribesGymWrapper()
    rng = random.Random(seed)
    h_all = hashlib.sha256()
    h_feat = hashlib.sha256()
    step_seconds = []
    episode = 0
    try:
        obs, info = env.reset(seed=seed)
        _update_value(h_all, obs)
        hash_info(h_all, info, skip)
        _update_value(h_feat, info["legal_action_features_padded"])
        for _ in range(int(steps)):
            legal = legal_ids_from_info(info)
            action = int(legal[rng.randrange(len(legal))])
            t0 = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            step_seconds.append(time.perf_counter() - t0)
            for value in (action, obs, float(reward), bool(terminated), bool(truncated)):
                _update_value(h_all, value)
            hash_info(h_all, info, skip)
            _update_value(h_feat, info["legal_action_features_padded"])
            if terminated or truncated:
                episode += 1
                obs, info = env.reset(seed=seed + episode)
                _update_value(h_all, obs)
                hash_info(h_all, info, skip)
                _update_value(h_feat, info["legal_action_features_padded"])
    finally:
        if own_env:
            env.close()
    return h_all.hexdigest(), h_feat.hexdigest(), step_seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ignore-info-prefix", action="append", default=[],
                        help="also leave out info keys with this prefix (repeatable), e.g. teacher_")
    args = parser.parse_args()
    digest, feat_digest, step_seconds = run(args.steps, args.seed, extra_skip_prefixes=args.ignore_info_prefix)
    s = np.asarray(step_seconds)
    print(f"trajectory_sha256={digest}")
    print(f"features_sha256={feat_digest}")
    print(f"steps={len(s)} mean_step_ms={1000 * s.mean():.3f} median_step_ms={1000 * np.median(s):.3f}")


if __name__ == "__main__":
    main()
