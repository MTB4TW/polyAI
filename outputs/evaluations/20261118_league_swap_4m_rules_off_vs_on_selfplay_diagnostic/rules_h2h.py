"""The same checkpoint on both sides, both sampling: one side through the wrapper (as the learner plays), the other
through the league hook (as snapshot opponents play). Equal paths => the wrapper side wins ~half its decided games.
FIX=1 also records the hook side's moves in its view's _unit_previous_tiles (the is_immediate_backtrack feature)."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.getcwd())
CKPT, N, FIX = sys.argv[1], int(sys.argv[2]), os.environ.get("FIX", "0") == "1"
os.environ.update(dict(POLYVISION_PHASE="2", POLYVISION_COMBAT_MODE="1", POLYVISION_SOLO_NO_OPPONENT_MODE="0",
    POLYVISION_MAX_TURNS="20", POLYVISION_INFO_MODE="fast", POLYVISION_RULES="polytopia_2_15",
    POLYVISION_LEVEL_POOL_GLOB="levels/phase2_pool_1v1/train/*.csv", POLYVISION_MAX_LEGAL_ACTIONS="512",
    POLYVISION_UNIT_STATUS="0", POLYVISION_COMBAT_RULES_OFF=os.environ.get("WRAP_RULES_OFF", "")))
from pol_env.Tribes.py.register_env import TribesGymWrapper
from pol_env.Tribes.py import combat_selfplay as cs

w = TribesGymWrapper()
me, opp = cs.SnapshotOpponent(CKPT, w), cs.SnapshotOpponent(CKPT, w)

def external_turn(wrapper, opponent, view, max_actions=2000):
    env = wrapper.tribes_env; java = env._env; learner = int(env.learner_seat)
    for _ in range(max_actions):
        if bool(java.isDone()) or int(java.getActiveTribeID()) == learner: return
        active = int(java.getActiveTribeID())
        obs = env.get_observation(relabel=False); actions = env.list_actions(relabel=False)
        s_obs, s_actions = cs.swap_tribes(obs, actions, 0, active)
        vec, ids, valid, feats, id_to_raw = view.encode(s_obs, s_actions, opponent_tribe=active)
        raw = id_to_raw[int(opponent.act(vec, ids, valid, feats))]
        a = s_actions[raw]
        if FIX and str(a.get("type", "")).upper() == "MOVE":
            uid, sx, sy, _dx, _dy = view.view._extract_move_components(a, s_obs)
            if uid is not None and sx is not None:
                view.view._unit_previous_tiles[int(uid)] = (int(sx), int(sy))
        java.stepByIndex(int(raw))
    raise RuntimeError("budget")

res = {0: [], 1: []}
OFF = int(os.environ.get('SHARD', '0')) * N
import torch; torch.set_num_threads(1)
for g in range(OFF, OFF + N):
    seat = g % 2
    w.tribes_env.learner_seat = seat
    w.tribes_env._env.setOpponentAgent(cs.EXTERNAL_OPPONENT, 0)
    state = {}
    def hook():
        if "view" not in state:
            state["view"] = cs.OpponentView(w)
            from pol_env.Tribes.py.environment_contract import resolve_combat_rules_off
            state["view"].view.RULES_OFF = resolve_combat_rules_off({"POLYVISION_COMBAT_RULES_OFF": os.environ.get("HOOK_RULES_OFF", ""), "POLYVISION_COMBAT_MODE": "1"})
        cs.play_external_turn(w, opp, state["view"])
    w.tribes_env.opponent_turn_hook = hook
    me.reseed(1000 + g); opp.reseed(5000 + g)
    obs, info = w.reset(seed=100 + g)
    done = False
    while not done:
        ids = np.asarray(info["legal_global_ids_padded"]).reshape(-1)
        valid = np.asarray(info["legal_action_valid_mask"], bool).reshape(-1)
        feats = np.asarray(info["legal_action_features_padded"], np.float32).reshape(ids.size, -1)
        obs, r, term, trunc, info = w.step(int(me.act(np.asarray(obs), ids, valid, feats)))
        done = term or trunc
    outcome = info.get("combat_outcome") or ("win" if info.get("combat_win") else "other")
    res[seat].append(1.0 if info.get("combat_win") else (0.0 if info.get("combat_loss", 0) or outcome == "loss" else 0.5))
    if g % 10 == 9:
        print(g + 1, {s: (len(v), round(float(np.mean(v)), 3)) for s, v in res.items() if v}, flush=True)
print(json.dumps({"wrap_rules_off": os.environ.get("WRAP_RULES_OFF", ""), "hook_rules_off": os.environ.get("HOOK_RULES_OFF", ""), "fix": FIX, "wrapper_side_score_by_seat": {s: [len(v), float(np.mean(v))] for s, v in res.items()}}))
