#!/usr/bin/env python3
"""Break down where a training vector-env step spends its time (diagnostic, not a benchmark).

Runs the training vector env (same make_env / make_training_vector_env as ppo.py: compact legal IPC,
SAME_STEP autoreset, spawn workers) with an untrained policy choosing the actions, and wraps every
worker env in a timer. Per vector step it records each worker's busy time (env.step plus the autoreset
env.reset when the episode ended), the slowest worker, and the remainder of the vector wall time
(vector wall - slowest worker busy), which is dispatch + pickling + pipe + result batching overhead.
It also classifies the slowest worker's step (ordinary / end turn / episode end with reset).

    python tools/profile_vector_step.py --num-envs 8 --steps 400

perf_counter is system-wide monotonic on macOS and Linux, so worker timestamps are comparable with the
trainer's. Numbers on a shared machine are noisy; ratios are the useful output.
"""

import argparse
import json
import os
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
    "OMP_NUM_THREADS": "1",
}
for _k, _v in PHASE1_DEFAULTS.items():
    os.environ.setdefault(_k, _v)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402


class StepTimingWrapper(gym.Wrapper):
    """Stamps absolute perf_counter times of env.step / env.reset into the returned info."""

    def reset(self, **kwargs):
        t0 = time.perf_counter()
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["diag_reset_t0"] = t0
        info["diag_reset_t1"] = time.perf_counter()
        return obs, info

    def step(self, action):
        t0 = time.perf_counter()
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["diag_step_t0"] = t0
        info["diag_step_t1"] = time.perf_counter()
        catalog = getattr(self.env.unwrapped, "_catalog", None)
        info["diag_end_turn"] = bool(catalog is not None and int(action) == int(catalog.id_end_turn()))
        return obs, reward, terminated, truncated, info


def timed_env_fn(env_id, idx):
    from py_rl.cleanrl.cleanrl.ppo import make_env

    inner = make_env(env_id, idx, False, "profile_vector_step", 0.1, 1.0)

    def thunk():
        return StepTimingWrapper(inner())

    return thunk


def _field(infos, key, i):
    if key in infos and bool(np.asarray(infos.get("_" + key, [True] * (i + 1)))[i]):
        return infos[key][i]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--network-arch", default="spatial_resnet")
    parser.add_argument("--actor-mode", default="legal_features")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    from py_rl.cleanrl.cleanrl.ppo import (
        Args, make_training_vector_env, _normalize_final_infos, _extract_vector_legal_tensors,
        _extract_vector_legal_feature_tensors,
    )
    from py_rl.cleanrl.cleanrl.agent_factory import make_agent

    n = args.num_envs
    defaults = Args()
    max_legal, feat_dim = int(defaults.max_legal_actions), int(defaults.legal_action_feature_dim)
    envs = make_training_vector_env([timed_env_fn("Tribes-v0", i) for i in range(n)])
    device = torch.device("cpu")
    agent = make_agent(envs, network_arch=args.network_arch, actor_mode=args.actor_mode,
                       max_legal_actions=max_legal, legal_action_feature_dim=feat_dim)
    rows = []
    try:
        obs, infos = envs.reset(seed=args.seed)
        for step in range(args.warmup + args.steps):
            t_policy0 = time.perf_counter()
            ids, valid = _extract_vector_legal_tensors(infos, n, max_legal, envs.single_action_space.n, device)
            feats = (_extract_vector_legal_feature_tensors(infos, n, max_legal, feat_dim, device)
                     if args.actor_mode == "legal_features" else None)
            with torch.no_grad():
                action = agent.get_action_and_value(torch.Tensor(obs), legal_global_ids=ids,
                                                    legal_action_valid_mask=valid, legal_action_features=feats)[0]
            action_np = action.cpu().numpy()
            t_send = time.perf_counter()
            envs.step_async(action_np)
            obs, _reward, term, trunc, infos = envs.step_wait()
            t_recv = time.perf_counter()
            infos = _normalize_final_infos(infos, n)
            if step < args.warmup:
                continue
            finals = infos.get("final_info", [None] * n)
            busy, kinds, starts, ends = [], [], [], []
            for i in range(n):
                fin = finals[i] if i < len(finals) else None
                src = fin if isinstance(fin, dict) else {k: _field(infos, k, i) for k in
                                                         ("diag_step_t0", "diag_step_t1", "diag_end_turn")}
                s0, s1 = float(src["diag_step_t0"]), float(src["diag_step_t1"])
                b, end = s1 - s0, s1
                kind = "end_turn" if bool(src.get("diag_end_turn")) else "ordinary"
                if isinstance(fin, dict):
                    r0, r1 = float(_field(infos, "diag_reset_t0", i)), float(_field(infos, "diag_reset_t1", i))
                    b += r1 - r0
                    end = r1
                    kind = "episode_end"
                busy.append(b)
                kinds.append(kind)
                starts.append(s0)
                ends.append(end)
            busy = np.asarray(busy)
            slow = int(np.argmax(busy))
            rows.append({
                "policy_s": t_send - t_policy0,
                "vector_wall_s": t_recv - t_send,
                "max_busy_s": float(busy.max()),
                "mean_busy_s": float(busy.mean()),
                "dispatch_s": float(min(starts) - t_send),
                "return_s": float(t_recv - max(ends)),
                "slowest_kind": kinds[slow],
                "kinds": kinds,
                "busy": busy.tolist(),
            })
    finally:
        envs.close()

    def ms(key):
        return 1000.0 * float(np.mean([r[key] for r in rows]))

    wall, mx, mean = ms("vector_wall_s"), ms("max_busy_s"), ms("mean_busy_s")
    all_busy = {k: [] for k in ("ordinary", "end_turn", "episode_end")}
    for r in rows:
        for k, b in zip(r["kinds"], r["busy"]):
            all_busy[k].append(b)
    n_worker_steps = sum(len(v) for v in all_busy.values())
    slow_counts = {k: sum(r["slowest_kind"] == k for r in rows) for k in all_busy}
    summary = {
        "num_envs": n,
        "steps": len(rows),
        "policy_forward_ms": ms("policy_s"),
        "vector_step_wall_ms": wall,
        "slowest_worker_busy_ms": mx,
        "mean_worker_busy_ms": mean,
        "straggler_wait_ms": mx - mean,
        "ipc_overhead_ms": wall - mx,
        "dispatch_ms": ms("dispatch_s"),
        "return_ms": ms("return_s"),
        "worker_busy_ms_by_kind": {k: {"mean": 1000 * float(np.mean(v)) if v else None,
                                       "share_of_worker_steps": len(v) / max(1, n_worker_steps)}
                                   for k, v in all_busy.items()},
        "slowest_worker_kind_share": {k: c / max(1, len(rows)) for k, c in slow_counts.items()},
        "worker_utilisation_during_vector_step": mean / wall if wall else None,
        "worker_utilisation_including_policy": mean / (wall + ms("policy_s")),
    }
    print(json.dumps(summary, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"summary": summary, "rows": rows}, f)


if __name__ == "__main__":
    main()
