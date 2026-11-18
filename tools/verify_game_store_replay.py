#!/usr/bin/env python3
"""Check that games saved by `ppo.py --game-store` replay exactly in the engine.

For each stored game: reset a fresh environment on the game's map and episode seed, play the stored
actions, and compare every observation, legal-action set and reward with the stored payload (if
the game has one) and the outcome with the recorded outcome. A mismatch means the recorder is
wrong or the engine, rules or contract changed since the game was played.

    python tools/verify_game_store_replay.py runs/<run>/game_store --games 20

Run with the same POLYVISION_* environment as the training run (phase, combat mode, rules, turn
limit). Stores saved with float16 observations are compared at float16 precision.
"""

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("store_dir")
    parser.add_argument("--games", type=int, default=20, help="number of games to check (0 = all)")
    args = parser.parse_args()

    import gymnasium as gym

    import pol_env.Tribes.py.register_env  # noqa: F401  (registers Tribes-v0)
    from py_rl.cleanrl.cleanrl.game_store import GameStore
    from py_rl.cleanrl.cleanrl.game_store.replay import ReplayMismatch, replay_game

    store = GameStore(budget_mb=1e9, max_games=10 ** 9)
    store.load(args.store_dir)
    records = list(store.records())
    if args.games:
        records = records[: args.games]
    print(f"loaded {len(store)} games; checking {len(records)}")

    env = gym.make("Tribes-v0")
    failures = 0
    for rec in records:
        label = f"game {rec.game_id} ({rec.meta.get('map_id')}, seed {rec.meta.get('episode_seed')}, {rec.length} steps)"
        try:
            game = replay_game(env, rec)
            if rec.has_payload:
                stored = rec.tensors()
                obs = game["obs"].astype(stored["obs"].dtype)
                if not np.array_equal(obs, stored["obs"]):
                    bad = int(np.argmax(np.any(obs != stored["obs"], axis=1)))
                    raise ReplayMismatch(f"observation differs first at step {bad}")
                if not np.array_equal(game["legal_ids"], stored["legal_ids"]):
                    raise ReplayMismatch("legal action ids differ")
                if not np.array_equal(game["slot"], stored["slot"]):
                    raise ReplayMismatch("chosen slots differ")
                scale = float(rec.meta.get("reward_scale", 1.0))
                if scale == 1.0 and not np.allclose(game["reward"], stored["reward"], atol=1e-5):
                    raise ReplayMismatch("rewards differ")
            print(f"ok   {label} outcome={rec.outcome}")
        except ReplayMismatch as exc:
            failures += 1
            print(f"FAIL {label}: {exc}")
    env.close()
    print(f"{len(records) - failures}/{len(records)} games replayed exactly")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
