"""Generalized advantage estimation, with optional per-step discounts.

Kept free of torch imports so it can be unit-tested with numpy; it works on torch tensors or numpy
arrays of shape [num_steps, num_envs].
"""

END_TURN_GLOBAL_ID = 0  # stable global action ID of END_TURN in the flat-v1 catalog


def compute_gae(rewards, values, dones, next_value, next_done, gamma, gae_lambda, step_discounts=None):
    """Return (advantages, returns).

    dones[t] marks that the observation at step t starts a new episode (CleanRL convention);
    next_value/next_done describe the state after the last stored step.
    step_discounts[t], if given, replaces gamma for the transition after step t. With
    per-turn discounting it is gamma when step t ended the turn and 1.0 otherwise.
    """
    num_steps = rewards.shape[0]
    advantages = rewards * 0
    lastgaelam = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        discount = gamma if step_discounts is None else step_discounts[t]
        delta = rewards[t] + discount * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + discount * gae_lambda * nextnonterminal * lastgaelam
    return advantages, advantages + values


def per_turn_discounts(actions, gamma, end_turn_id=END_TURN_GLOBAL_ID):
    """gamma where the selected global action is END_TURN, else 1.0 (same type as `actions`)."""
    ended_turn = (actions == end_turn_id)
    return ended_turn * gamma + (1 - ended_turn * 1.0) * 1.0
