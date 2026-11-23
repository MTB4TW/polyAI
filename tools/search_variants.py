"""Ways to combine the policy with the combat tactical search at play time (no training).

Every variant plays the policy's highest-logit move unless its rule lets the search override it:

| mode | when the search overrides the policy |
|---|---|
| `none` | never (the policy alone) |
| `search` | whenever the search's best line beats ending the turn now (`ppo_search` in tools/evaluate_combat.py) |
| `order` | only on the order of the policy's own plan: when the policy attacks target T, any attack on T may go first; an attack that kills is always allowed; when the policy picks a passive move (RECOVER / END_TURN) while it rates an attack at >= `want_p`, that attack may go first |
| `decisive` | when the best line wins, captures a city, puts a unit on an enemy city (a capture next turn) or removes a threat to our capital |
| `decisive_kills` | as `decisive`, and also when the line kills an enemy unit |
| `plan` | when the best line beats the line that starts with the policy's own move by `plan_margin`. Both are searched deeper (depth 4) and scored one turn ahead: after the opponent's tactical reply |

The search always works on the acting tribe's own fogged view (`PythonEnv.tacticalChooseIndex` /
`tacticalAnalyze`), so it works in either seat. Each player counts how often the search decided (`stats`).
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from tools.play_vs_agent import AgentPlayer

MODES = ("none", "search", "order", "decisive", "decisive_kills", "plan")
PASSIVE = {"RECOVER", "END_TURN"}
# CombatTactics.analyze output layout
BEST, BEST_VALUE, STOP_VALUE, FORCED_VALUE, EFFECTS = 0, 1, 2, 3, 4
WON, CITIES, SIEGES, CAPITAL_SIEGES, KILLS, LOST, THREATS_REMOVED = range(7)


class SearchPlayer(AgentPlayer):
    def __init__(self, info, env, mode: str, seed: int = 0, want_p: float = 0.05, plan_margin: float = 0.5,
                 plan_depth: int = 4, plan_beam: int = 8, reply_actions: int = 6):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if info is None:
            raise ValueError("SearchPlayer needs a checkpoint")
        super().__init__(info, env, False, seed, search=False)
        self.mode, self.want_p, self.plan_margin = mode, float(want_p), float(plan_margin)
        self.plan_depth, self.plan_beam, self.reply_actions = int(plan_depth), int(plan_beam), int(reply_actions)
        self.stats: Counter = Counter()

    @property
    def name(self) -> str:
        return f"ppo_{self.mode}" if self.mode != "none" else "ppo_argmax"

    def _policy(self, vec, ids, feats, valid):
        torch = self._torch
        with torch.inference_mode():
            dist = self.agent.get_action_distribution(
                torch.as_tensor(vec, dtype=torch.float32).reshape(1, -1),
                legal_global_ids=torch.as_tensor(ids, dtype=torch.long).reshape(1, -1),
                legal_action_valid_mask=torch.as_tensor(valid, dtype=torch.bool).reshape(1, -1),
                legal_action_features=torch.as_tensor(np.asarray(feats, dtype=np.float32)).reshape(1, ids.size, -1))
        return dist.probs[0].numpy(), int(torch.argmax(dist.logits[0]).item())

    def act(self, vec, ids, valid, feats, raw_map=None) -> int:
        ids, valid = np.asarray(ids).reshape(-1), np.asarray(valid, dtype=bool).reshape(-1)
        probs, slot = self._policy(vec, ids, feats, valid)
        if not valid[slot]:
            raise RuntimeError("The policy picked an invalid legal slot")
        gid_p = int(ids[slot])
        self.last_source = "policy"
        self.stats["decisions"] += 1
        if self.mode == "none" or raw_map is None:
            return gid_p
        picked = self._override(probs, ids, valid, raw_map, gid_p)
        if picked is not None and picked != gid_p:
            self.last_source = "search"
            self.search_moves += 1
            self.stats["search"] += 1
            return picked
        return gid_p

    def _override(self, probs, ids, valid, raw_map, gid_p) -> Optional[int]:
        from pol_env.Tribes.py import search_teacher

        te = self.env.tribes_env
        java, gw = te._env, te._gateway
        seed = search_teacher.decision_seed(self._search_seed, self._decision)
        self._decision += 1
        slots = [int(s) for s in np.flatnonzero(valid)]
        gid_of_raw, prob_of_raw = {}, {}
        for s in slots:
            r = raw_map.get(int(ids[s]))
            if r is not None:
                gid_of_raw[int(r)] = int(ids[s])
                prob_of_raw[int(r)] = float(probs[s])
        raw_p = raw_map.get(gid_p)
        if raw_p is None or not gid_of_raw:
            return None
        legal = sorted(gid_of_raw)
        if self.mode == "search":
            raw = search_teacher.tactical_choose_raw(java, gw, legal, seed)
            return gid_of_raw.get(raw) if raw >= 0 else None

        if self.mode == "order":
            actions = te.list_actions()
            prev = list(java.attackPreviews())
            is_attack = lambda r: actions[r].get("type") == "ATTACK"
            target = lambda r: prev[r * 5 + 4]
            allowed = {raw_p}
            if is_attack(raw_p):  # the same target, any attacker: the order of the policy's attack
                allowed |= {r for r in legal if is_attack(r) and target(r) == target(raw_p)}
            allowed |= {r for r in legal if is_attack(r) and prev[r * 5 + 2] == 1}  # finishing blows
            if actions[raw_p].get("type") in PASSIVE:  # attacks the policy wants before a passive move
                allowed |= {r for r in legal if is_attack(r) and prob_of_raw.get(r, 0.0) >= self.want_p}
            if allowed == {raw_p}:
                return None
            raw = search_teacher.tactical_choose_raw(java, gw, sorted(allowed), seed)
            if raw >= 0 and raw != raw_p:
                self.stats["order_" + ("kill" if prev[raw * 5 + 2] == 1 else "reorder")] += 1
            return gid_of_raw.get(raw) if raw >= 0 else None

        allowed_arr = gw.new_array(gw.jvm.int, len(legal))
        for i, r in enumerate(legal):
            allowed_arr[i] = int(r)
        cfg = search_teacher.COMBAT_TEACHER_MODES["tactical_search"]
        if self.mode in ("decisive", "decisive_kills"):
            res = list(java.tacticalAnalyze(None, allowed_arr, -1, int(cfg["depth"]), int(cfg["beam"]), int(cfg["radius"]),
                                            int(seed), 0))
            best = int(res[BEST])
            if best < 0 or best == raw_p:
                return None
            e = res[EFFECTS:EFFECTS + 7]
            decisive = e[WON] > 0 or e[CITIES] > 0 or e[SIEGES] > 0 or e[CAPITAL_SIEGES] > 0 or e[THREATS_REMOVED] > 0
            if self.mode == "decisive_kills":
                decisive = decisive or e[KILLS] > 0
            if decisive:
                self.stats["decisive"] += 1
                return gid_of_raw.get(best)
            return None

        if self.mode == "plan":
            res = list(java.tacticalAnalyze(None, allowed_arr, int(raw_p), self.plan_depth, self.plan_beam, int(cfg["radius"]),
                                            int(seed), self.reply_actions))
            best, best_v, stop_v, forced_v = int(res[BEST]), res[BEST_VALUE], res[STOP_VALUE], res[FORCED_VALUE]
            if best < 0 or best == raw_p:
                return None
            # The policy's plan: its move followed by the best continuation; END_TURN (not searched) = stopping now.
            baseline = stop_v if (forced_v != forced_v or not np.isfinite(forced_v)) else max(forced_v, stop_v)
            if best_v > baseline + self.plan_margin:
                return gid_of_raw.get(best)
            return None
        return None
