"""Double-buffered rollout stepping (opt-in, ``--overlap-rollout``).

The N training envs are split into two halves, A = envs[:ceil(N/2)] and B = the rest, each its own
vector env. Instead of "policy for all N, then step all N, then wait for the slowest", a rollout step runs

    policy(B, t) -> send B(t)              [A(t) is already stepping]
    wait A(t) -> policy(A, t+1) -> send    [B(t) is stepping meanwhile]
    wait B(t)

so the policy forward of one half overlaps the env step of the other, and a slow worker (episode end with
reset, end turn) delays only its own half. A(t+1) is never sent on the last step of a rollout, because the
PPO update changes the policy before the next rollout starts.

The rollout buffers keep the usual (step, env index) layout: half A writes env columns [0, nA), B writes
[nA, N), and the merged step result (observations, rewards, flags and Gymnasium infos) is exactly what one
vector env over all N envs would have returned for the same actions. Reset seeds follow the same rule
(env i gets seed + i). Only the sampling differs from the single-batch path: actions are drawn in two
batches per step, so the policy RNG is consumed in a different order and a flag-on run does not reproduce
a flag-off run with the same seed; it is still deterministic for a fixed seed.
"""

import time

import numpy as np
import torch
from gymnasium.vector.utils import batch_space


def split_sizes(num_envs):
    if int(num_envs) < 2:
        raise ValueError("--overlap-rollout needs at least 2 envs (two halves).")
    first = (int(num_envs) + 1) // 2
    return first, int(num_envs) - first


def _fill_like(arr, n):
    if arr.dtype == object:
        return np.full((n, *arr.shape[1:]), None, dtype=object)
    return np.zeros((n, *arr.shape[1:]), dtype=arr.dtype)


def merge_vector_infos(info_a, n_a, info_b, n_b):
    """Merge the batched infos of two vector envs (A = envs [0, n_a), B = [n_a, n_a + n_b)) into the dict
    Gymnasium's VectorEnv._add_info builds for one vector env over all n_a + n_b envs: same key order, same
    `_key` masks, the array dtype of the first env that has the key, and defaults (0 / None) where absent.

    One corner differs from Gymnasium: if A and B batched a key with different dtypes, B's rows are cast
    from B's array (already cast once by B) rather than from the original value. This only happens when
    envs report the same key with different types, which the Phase 1 env does not do (see the tests)."""
    out = {}
    keys = list(info_a.keys()) + [k for k in info_b.keys() if k not in info_a]
    for key in keys:
        if isinstance(key, str) and key.startswith("_") and key[1:] in out:
            continue  # masks are rebuilt right after their key
        in_a, in_b = key in info_a, key in info_b
        va, vb = info_a.get(key), info_b.get(key)
        mask_a = np.asarray(info_a.get("_" + key, np.zeros(n_a, dtype=np.bool_)), dtype=np.bool_)
        mask_b = np.asarray(info_b.get("_" + key, np.zeros(n_b, dtype=np.bool_)), dtype=np.bool_)
        if isinstance(va, dict) or isinstance(vb, dict):
            merged = merge_vector_infos(va if in_a else {}, n_a, vb if in_b else {}, n_b)
        elif not in_a:
            merged = np.concatenate([_fill_like(vb, n_a), vb])
        elif not in_b:
            merged = np.concatenate([va, _fill_like(va, n_b)])
        elif va.dtype == vb.dtype and va.shape[1:] == vb.shape[1:]:
            merged = np.concatenate([va, vb])  # B's absent rows already hold the same defaults
        else:
            merged = np.concatenate([va, _fill_like(va, n_b)])
            for j in np.flatnonzero(mask_b):
                merged[n_a + j] = vb[j]
        out[key] = merged
        out["_" + key] = np.concatenate([mask_a, mask_b])
    return out


class OverlappedVectorEnv:
    """Two vector envs presented as one: reset/step/close behave like a single vector env over all N envs;
    step_async_half/step_wait_half drive the halves separately for the double-buffered rollout."""

    def __init__(self, env_fns, make_vector_env):
        self.num_envs = len(env_fns)
        self.sizes = split_sizes(self.num_envs)
        self.slices = (slice(0, self.sizes[0]), slice(self.sizes[0], self.num_envs))
        self.halves = []
        try:
            for sl in self.slices:
                self.halves.append(make_vector_env(list(env_fns[sl])))
        except Exception:
            self.close()
            raise
        self.single_observation_space = self.halves[0].single_observation_space
        self.single_action_space = self.halves[0].single_action_space
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.metadata = getattr(self.halves[0], "metadata", {})

    def reset(self, seed=None, options=None):
        seeds = [None, None]
        if seed is not None:
            seeds = [int(seed), int(seed) + self.sizes[0]]
        for h, s in zip(self.halves, seeds):
            h.reset_async(seed=s, options=options)
        (obs_a, info_a), (obs_b, info_b) = [h.reset_wait() for h in self.halves]
        return np.concatenate([obs_a, obs_b]), merge_vector_infos(info_a, self.sizes[0], info_b, self.sizes[1])

    def step_async_half(self, half, actions):
        self.halves[half].step_async(actions)

    def step_wait_half(self, half):
        return self.halves[half].step_wait()

    def merge_step(self, result_a, result_b):
        obs_a, rew_a, term_a, trunc_a, info_a = result_a
        obs_b, rew_b, term_b, trunc_b, info_b = result_b
        return (
            np.concatenate([obs_a, obs_b]),
            np.concatenate([rew_a, rew_b]),
            np.concatenate([term_a, term_b]),
            np.concatenate([trunc_a, trunc_b]),
            merge_vector_infos(info_a, self.sizes[0], info_b, self.sizes[1]),
        )

    def step(self, actions):
        actions = np.asarray(actions)
        for h, sl in enumerate(self.slices):
            self.step_async_half(h, actions[sl])
        return self.merge_step(self.step_wait_half(0), self.step_wait_half(1))

    def close(self, **kwargs):
        for h in self.halves:
            try:
                h.close(**kwargs)
            except Exception:
                pass


class OverlappedRolloutStepper:
    """Runs one double-buffered rollout step at a time (see module docstring).

    prepare_inputs(obs_np, infos, n) -> dict of policy-input tensors for n envs, built exactly as the
        trainer builds its next_* tensors from a step result.
    act(step, env_slice, inputs) -> action tensor for that slice; it samples the policy and writes the
        rollout-buffer rows [step, env_slice].
    """

    def __init__(self, envs, prepare_inputs, act, profiler=None):
        self.envs = envs
        self.prepare_inputs = prepare_inputs
        self.act = act
        self.profiler = profiler
        self._prefetched = None  # (step, inputs, action tensor) of half A sent ahead of its rollout step

    def _time(self, key, t0):
        if self.profiler is not None:
            self.profiler.add(key, time.perf_counter() - t0)

    def _act(self, step, half, inputs):
        t0 = time.perf_counter()
        action = self.act(step, self.envs.slices[half], inputs)
        self._time("trainer/policy_forward_action_selection", t0)
        self.envs.step_async_half(half, action.cpu().numpy())
        return action

    def step(self, step, inputs, last_step):
        """inputs: the trainer's next_* tensors for all N envs at `step`. Returns the merged step result
        and the action tensor for all N envs."""
        sl_a, sl_b = self.envs.slices
        if self._prefetched is None:
            action_a = self._act(step, 0, {k: v[sl_a] for k, v in inputs.items()})
        else:
            pre_step, pre_inputs, action_a = self._prefetched
            self._prefetched = None
            if pre_step != step:
                raise RuntimeError(f"overlap: half A was sent for step {pre_step}, trainer is at step {step}")
            for k, v in pre_inputs.items():
                if not torch.equal(v, inputs[k][sl_a]):
                    raise RuntimeError(f"overlap: prefetched policy input {k!r} differs from the trainer's")
        action_b = self._act(step, 1, {k: v[sl_b] for k, v in inputs.items()})
        t0 = time.perf_counter()
        result_a = self.envs.step_wait_half(0)
        self._time("trainer/envs_step_wall", t0)
        if not last_step:
            next_inputs_a = self.prepare_inputs(result_a[0], result_a[4], self.envs.sizes[0])
            self._prefetched = (step + 1, next_inputs_a, self._act(step + 1, 0, next_inputs_a))
        t0 = time.perf_counter()
        result_b = self.envs.step_wait_half(1)
        self._time("trainer/envs_step_wall", t0)
        return self.envs.merge_step(result_a, result_b), torch.cat([action_a, action_b])
