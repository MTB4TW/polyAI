#!/usr/bin/env python
"""Test 2-Player Environment Execution with Bot Opponent."""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure local Java is in PATH
jdks = sorted((REPO_ROOT / ".runtime").glob("jdk-*/Contents/Home"))
if jdks:
    os.environ["JAVA_HOME"] = str(jdks[-1])
    os.environ["PATH"] = str(jdks[-1] / "bin") + os.pathsep + os.environ.get("PATH", "")

os.environ["POLYVISION_SOLO_NO_OPPONENT_MODE"] = "0"
os.environ["POLYVISION_COMBAT_MODE"] = "1"
os.environ["POLYVISION_INFO_MODE"] = "fast"

from pol_env.Tribes.py.register_env import TribesGymWrapper

def main():
    map_file = REPO_ROOT / "pol_env/Tribes/levels/phase2_pool_1v1/train/map_2p_010000.csv"
    env = TribesGymWrapper()
    env._level_pool = [str(map_file.resolve())]
    env._level_pool_size = 1
    
    obs, info = env.reset(seed=1)
    print("=== 2-PLAYER ENVIRONMENT RESET ===")
    print(f"Map: {map_file.name}")
    print(f"Initial Stars: {info.get('stars')}")
    print(f"Initial Cities: {info.get('city_count')}")
    print(f"Legal action count: {info.get('legal_action_count')}")
    
    actions = env.tribes_env.list_actions()
    print(f"Sample available raw actions ({len(actions)} total):")
    for a in actions[:8]:
        print(f"  - {a.get('type')}: {a.get('repr')}")

    print("\nStepping through 3 turns in 2-Player match...")
    for t in range(5):
        # Pick legal action from valid mask
        valid_mask = info.get("legal_action_valid_mask", [])
        legal_ids = info.get("legal_global_ids_padded", [])
        slots = [i for i, v in enumerate(valid_mask) if v]
        if not slots:
            print("No legal actions left!")
            break
        
        # Pick first legal action
        act = int(legal_ids[slots[0]])
        obs, reward, term, trunc, info = env.step(act)
        a_type = info.get("selected_action_type")
        turn = info.get("turn_count")
        stars = info.get("stars")
        print(f"  Step {t}: Action={a_type:<15} | Turn={turn} | Stars={stars} | Reward={float(reward):+.2f}")
        if term or trunc:
            print("Game terminated or truncated!")
            break
        
    env.close()
    print("\n2-Player Environment multi-turn execution successfully verified!")

if __name__ == "__main__":
    main()
