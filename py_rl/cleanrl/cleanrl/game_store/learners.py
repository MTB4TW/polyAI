"""What to learn from stored games.

A learner turns a sampled batch into a loss on the live agent:

    loss, metrics = learner.loss(agent, batch, ctx)

* step learners (`kind = "steps"`) get a padded batch of single steps from `GameStore.sample_steps`;
* game learners (`kind = "games"`) get whole games from `GameStore.sample_games`.

Every learner calls the agent through the same `get_action_and_value(obs, legal_global_ids=...,
legal_action_valid_mask=..., legal_action_features=..., selected_slot=...)` that PPO uses, so each one
works with the MLP, ResNet and composed networks without changes. `ctx` carries `device`, `gamma`
and `per_turn`.

Add a learner by subclassing `ReplayLearner` and registering it with `@register_learner("name")`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

LEARNERS: Dict[str, type] = {}


def register_learner(name: str):
    def deco(cls):
        LEARNERS[name] = cls
        cls.registry_name = name
        return cls

    return deco


def batch_to_torch(batch: Dict[str, np.ndarray], device) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in batch.items():
        if isinstance(value, np.ndarray):
            out[key] = torch.as_tensor(value, device=device)
    return out


def evaluate(agent, b: Dict[str, torch.Tensor]):
    """(log pi(a|s), entropy, value) of the stored actions under the current agent."""
    _, _, logprob, entropy, value = agent.get_action_and_value(
        b["obs"].float(),
        legal_global_ids=b["legal_ids"],
        legal_action_valid_mask=b["legal_valid"],
        legal_action_features=b.get("legal_features"),
        selected_slot=b["slot"],
    )
    return logprob, entropy, value.view(-1)


class ReplayLearner:
    registry_name = "base"
    kind = "steps"

    def loss(self, agent, batch, ctx) -> Tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError

    def describe(self) -> str:
        return self.registry_name


@register_learner("sil")
class SelfImitation(ReplayLearner):
    """Self-imitation learning (Oh et al., 2018): imitate past actions only where they turned out
    better than the critic now expects, and pull the critic up to those returns.

    policy: -log pi(a|s) * max(R - V(s), 0);   value: 0.5 * max(R - V(s), 0)^2.
    Games that went worse than expected contribute nothing, so this is safe to run on every stored
    game; pair it with the `surprise` or `win` selector to spend memory on the useful ones.
    """

    def __init__(self, value_coef: float = 0.01, max_adv: float = 10.0):
        self.value_coef = float(value_coef)
        self.max_adv = float(max_adv)

    def loss(self, agent, batch, ctx):
        logprob, entropy, value = evaluate(agent, batch)
        gap = (batch["return"] - value).clamp(min=0.0, max=self.max_adv)
        pg = -(logprob * gap.detach()).mean()
        vl = 0.5 * (gap ** 2).mean()
        active = float((gap > 0).float().mean().item())
        return pg + self.value_coef * vl, {"policy_loss": pg.item(), "value_loss": vl.item(), "active_frac": active,
                                           "entropy": entropy.mean().item()}


@register_learner("awr")
class AdvantageWeighted(ReplayLearner):
    """Advantage-weighted regression (Peng et al., 2019): weight log pi(a|s) by exp(A / beta).

    A = R - V(s). Unlike self-imitation it also *down-weights* (never pushes away from) bad
    actions, and it fits the critic to every stored return. beta sets how sharply it prefers good
    actions; weights are capped at max_weight.
    """

    def __init__(self, beta: float = 1.0, max_weight: float = 20.0, value_coef: float = 0.5):
        self.beta = float(beta)
        self.max_weight = float(max_weight)
        self.value_coef = float(value_coef)

    def loss(self, agent, batch, ctx):
        logprob, entropy, value = evaluate(agent, batch)
        adv = (batch["return"] - value).detach()
        weight = torch.exp(adv / self.beta).clamp(max=self.max_weight)
        pg = -(logprob * weight).mean()
        vl = 0.5 * ((batch["return"] - value) ** 2).mean()
        return pg + self.value_coef * vl, {"policy_loss": pg.item(), "value_loss": vl.item(),
                                           "mean_weight": weight.mean().item(), "entropy": entropy.mean().item()}


@register_learner("bc")
class BehaviourCloning(ReplayLearner):
    """Plain imitation of the stored actions: -log pi(a|s).

    Meant for games chosen by the selector as worth copying outright: wins against strong
    opponents, search or teacher games, or imported human games.
    """

    def __init__(self, value_coef: float = 0.0):
        self.value_coef = float(value_coef)

    def loss(self, agent, batch, ctx):
        logprob, entropy, value = evaluate(agent, batch)
        pg = -logprob.mean()
        vl = 0.5 * ((batch["return"] - value) ** 2).mean()
        return pg + self.value_coef * vl, {"policy_loss": pg.item(), "value_loss": vl.item(),
                                           "entropy": entropy.mean().item()}


@register_learner("value")
class ValueReplay(ReplayLearner):
    """Fit only the critic to stored Monte Carlo returns (more value data at no rollout cost).

    The policy is untouched. Useful when explained variance is low or the critic collapses.
    """

    def loss(self, agent, batch, ctx):
        _, _, value = evaluate(agent, batch)
        vl = 0.5 * ((batch["return"] - value) ** 2).mean()
        return vl, {"value_loss": vl.item()}


@register_learner("replay_ppo")
class ReplayPPO(ReplayLearner):
    """A PPO step on old games, corrected for the policy having moved since they were played.

    ratio = pi(a|s) / mu(a|s) with mu the stored behaviour probability, advantage = R - V(s)
    (normalised), and the usual clipped objective. Steps where the ratio has drifted past
    `max_ratio` (or below its inverse) are dropped: those games are too off-policy to correct.
    """

    def __init__(self, clip: float = 0.2, max_ratio: float = 3.0, value_coef: float = 0.5, norm_adv: bool = True):
        self.clip = float(clip)
        self.max_ratio = float(max_ratio)
        self.value_coef = float(value_coef)
        self.norm_adv = bool(norm_adv)

    def loss(self, agent, batch, ctx):
        logprob, entropy, value = evaluate(agent, batch)
        ratio = torch.exp(logprob - batch["logprob"])
        keep = ((ratio < self.max_ratio) & (ratio > 1.0 / self.max_ratio)).float().detach()
        adv = (batch["return"] - value).detach()
        if self.norm_adv and adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg1 = -adv * ratio
        pg2 = -adv * ratio.clamp(1 - self.clip, 1 + self.clip)
        pg = (torch.max(pg1, pg2) * keep).sum() / keep.sum().clamp(min=1.0)
        vl = 0.5 * ((batch["return"] - value) ** 2).mean()
        return pg + self.value_coef * vl, {"policy_loss": pg.item(), "value_loss": vl.item(),
                                           "kept_frac": keep.mean().item(), "mean_ratio": ratio.mean().item()}


@register_learner("vtrace")
class VTrace(ReplayLearner):
    """Off-policy actor-critic on whole stored games with V-trace targets (Espeholt et al., 2018).

    For each game: rho_t = min(rho_bar, pi/mu), c_t = lambda * min(c_bar, pi/mu),
    v_s = V(s) + sum_t gamma^(t-s) (prod c) rho_t delta_t, and the policy gradient uses
    rho_s * (r_s + gamma v_{s+1} - V(s)). Unlike the step learners this bootstraps from the critic,
    so it does not need the Monte Carlo return and is correct for games of any age.
    `max_steps` caps the steps evaluated per game (a random window) to bound memory.
    """

    kind = "games"

    def __init__(self, rho_bar: float = 1.0, c_bar: float = 1.0, lam: float = 0.95, value_coef: float = 0.5,
                 ent_coef: float = 0.0, max_steps: int = 256):
        self.rho_bar = float(rho_bar)
        self.c_bar = float(c_bar)
        self.lam = float(lam)
        self.value_coef = float(value_coef)
        self.ent_coef = float(ent_coef)
        self.max_steps = int(max_steps)

    def loss(self, agent, games: List[Dict[str, Any]], ctx):
        from .store import collate_steps

        device, gamma, per_turn = ctx["device"], float(ctx["gamma"]), bool(ctx.get("per_turn", False))
        rng = ctx.get("rng") or np.random.default_rng()
        total_pg, total_vl, total_ent, n_steps, rhos = 0.0, 0.0, 0.0, 0, []
        losses = []
        for game in games:
            length = int(game["slot"].shape[0])
            start = 0 if length <= self.max_steps else int(rng.integers(0, length - self.max_steps + 1))
            end = min(length, start + self.max_steps)
            rows = [(game["game_id"], t, game, 0.0) for t in range(start, end)]
            b = batch_to_torch(collate_steps(rows), device)
            logprob, entropy, value = evaluate(agent, b)
            with torch.no_grad():
                rho = torch.exp(logprob - b["logprob"])
                reward = b["reward"] * float(game["meta"].get("reward_scale", 1.0)) / max(float(ctx.get("reward_scale_now", 1.0)), 1e-8)
                end_turn = torch.as_tensor(game["end_turn"][start:end], device=device)
                disc = torch.where(end_turn, gamma, 1.0) if per_turn else torch.full_like(reward, gamma)
                # The window's end bootstraps from V unless it is the game's real end.
                bootstrap = 0.0 if end == length else float(value[-1].item())
                vs, pg_adv = vtrace_targets(reward, value.detach(), disc, rho, self.rho_bar, self.c_bar, self.lam, bootstrap)
                v = value.detach()
            pg = -(logprob * pg_adv).sum()
            vl = 0.5 * ((vs - value) ** 2).sum()
            ent = entropy.sum()
            losses.append(pg + self.value_coef * vl - self.ent_coef * ent)
            total_pg += pg.item()
            total_vl += vl.item()
            total_ent += ent.item()
            n_steps += int(v.shape[0])
            rhos.append(rho.mean().item())
        if not losses:
            return None, {}
        loss = torch.stack(losses).sum() / max(1, n_steps)
        return loss, {"policy_loss": total_pg / max(1, n_steps), "value_loss": total_vl / max(1, n_steps),
                      "entropy": total_ent / max(1, n_steps), "mean_rho": float(np.mean(rhos))}


def vtrace_targets(rewards, values, discounts, rhos, rho_bar=1.0, c_bar=1.0, lam=1.0, bootstrap=0.0):
    """V-trace value targets vs and policy-gradient advantages for one trajectory (1-D tensors).

    discounts[t] is the discount of the transition after step t; `bootstrap` is V of the state after
    the last step (0 at a real game end). With rhos == 1 and lam == 1, vs is the n-step Monte Carlo
    return bootstrapped from `bootstrap`.
    """
    rho_c = rhos.clamp(max=rho_bar)
    cs = lam * rhos.clamp(max=c_bar)
    last = torch.full((1,), float(bootstrap), dtype=values.dtype, device=values.device)
    next_v = torch.cat([values[1:], last])
    deltas = rho_c * (rewards + discounts * next_v - values)
    vs_minus_v = torch.zeros_like(values)
    acc = torch.zeros((), dtype=values.dtype, device=values.device)
    for t in range(values.shape[0] - 1, -1, -1):
        acc = deltas[t] + discounts[t] * cs[t] * acc
        vs_minus_v[t] = acc
    vs = values + vs_minus_v
    next_vs = torch.cat([vs[1:], last])
    return vs, rho_c * (rewards + discounts * next_vs - values)
