#!/usr/bin/env python
"""Replay a saved human game (tools/play_vs_agent.py) with its checkpoint and show what the agent was thinking.

    python tools/analyze_human_game.py runs/human_games/<file>.json --turns 9-11

For every agent decision in the chosen turns it prints:
- the action it played and its probability;
- the policy's top alternatives;
- every move onto a village or an enemy city, with its probability;
- any CAPTURE the engine allowed.

Turn headers also list the true board (units with HP, cities). The policy is recomputed from the checkpoint
on the replayed position, so this is the network's view at that moment, not a guess. Agent in seat 0 (the human
moved second) only for now.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import play_vs_agent as P  # noqa: E402  (also sets up the JDK)


def parse_turns(text: str):
    if not text:
        return None
    lo, _, hi = text.partition("-")
    return set(range(int(lo), int(hi or lo) + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record", type=Path)
    ap.add_argument("--turns", default="", help="e.g. 10 or 9-11 (default: all)")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    import numpy as np
    import torch
    from tools.phase1_eval_core import legal_tensors

    rec = json.loads(args.record.read_text(encoding="utf-8"))
    if rec.get("agent_tribe", 0) != 0:
        raise SystemExit("Only games where the agent moved first (human second) are supported for now")
    model = rec["model"]["path"] if isinstance(rec.get("model"), dict) else None
    if not model:
        raise SystemExit("The record does not name its checkpoint path (human_game_v1): nothing to analyse with")
    info = P.find_models([model])[0]
    s = rec["settings"]
    env, _ = P.make_env(info.path, "eval_validation", s["rules"], s["max_turns"])
    net = P.AgentPlayer(info, env, False, 0).agent
    game_map = [m for m in P.load_maps("eval_validation") if m.filename == rec["map"]][0]
    turns = parse_turns(args.turns)
    human = iter([a for a in rec["actions"] if a["side"] == 1 and not a.get("resign")])
    agent_moves = iter([a for a in rec["actions"] if a["side"] == 0])
    last_turn = [None]

    class Analyst(P.AgentPlayer):
        def __init__(self):
            self.info, self.agent, self.sampled = info, net, False

        def reseed(self, seed):
            pass

        def choose(self, obs, info_):
            step = next(agent_moves, None)
            if step is None:
                raise P.GameAborted("abandoned")
            gid = int(step["gid"])
            turn = int(env._turn_count) + 1
            if turns is not None and turn not in turns:
                return gid
            ids, valid, feats = legal_tensors(info_)
            with torch.inference_mode():
                dist = net.get_action_distribution(
                    torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1),
                    legal_global_ids=torch.as_tensor(ids).reshape(1, -1),
                    legal_action_valid_mask=torch.as_tensor(valid).reshape(1, -1),
                    legal_action_features=torch.as_tensor(feats).unsqueeze(0))
            probs = dist.probs[0].numpy()
            acts, to_raw = env._current_legal_actions, env._current_legal_id_to_raw_index
            full = env.tribes_env.get_observation(full_visibility=True)
            cities = {(int(c["x"]), int(c["y"])): (int(c["tribeID"]), bool(c.get("isCapital"))) for c in full["city"].values()}
            terrain = full["board"]["terrain"]
            if turn != last_turn[0]:
                last_turn[0] = turn
                units = sorted((int(u["tribeId"]), P.UNIT_NAMES[int(u["type"])], int(u["x"]), int(u["y"]), int(u["currentHP"]))
                               for u in full["unit"].values())
                print(f"\n===== agent turn {turn}: {full['tribes']['0']['star']} stars")
                print("  cities (x,y) -> (owner, capital):", cities)
                print("  units (owner, type, x, y, hp):", units)
            rows = sorted(((float(probs[k]), acts[to_raw[int(ids[k])]]) for k in np.flatnonzero(valid)), key=lambda r: -r[0])
            slot = [int(i) for i in ids].index(gid)
            chosen = (float(probs[slot]), acts[to_raw[gid]])
            print(f"  chose {chosen[1].get('repr')} (p={chosen[0]:.3f}); top: " +
                  ", ".join(f"{a.get('repr')} {p:.3f}" for p, a in rows[:args.top]))
            onto = []
            for p, a in rows:
                if a.get("type") != "MOVE":
                    continue
                dst = (int(a.get("target_x", -1)), int(a.get("target_y", -1)))
                if dst in cities and cities[dst][0] != 0:
                    onto.append(f"{a.get('repr')} [{'ENEMY CAPITAL' if cities[dst][1] else 'enemy city'}] {p:.3f}")
                elif 0 <= dst[0] < len(terrain) and int(terrain[dst[0]][dst[1]]) == 4:
                    onto.append(f"{a.get('repr')} [village] {p:.3f}")
            if onto:
                print("     onto cities/villages: " + "; ".join(onto))
            caps = [a.get("repr") for a in env.tribes_env.list_actions() if a.get("type") == "CAPTURE"]
            if caps:
                print("     engine allows: " + "; ".join(caps))
            return gid

    def human_policy(state):
        step = next(human, None)
        if step is None:
            raise P.GameAborted("abandoned")
        return step["raw"]

    session = P.GameSession(env, Analyst(), game_map, rec["episode_seed"], s, None, human_tribe=1, human_policy=human_policy)
    session._run()
    env.close()
    print(f"\nreplayed to: {session.outcome} (recorded {rec.get('tribe0_outcome', rec.get('agent_outcome'))})")
    if session.error:
        print(session.error)


if __name__ == "__main__":
    main()
