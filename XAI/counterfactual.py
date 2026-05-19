"""Counterfactual "what-if" trajectories and walkers over a saved agent index."""

import numpy as np


def counterfactual_rollout(s0, a, alt_first_step, optimal_successor, max_steps=60):
    s1 = alt_first_step.get((s0, a))
    if s1 is None:
        return [s0]
    traj = [s0, s1]
    cur = s1
    for _ in range(max_steps - 1):
        nxt = optimal_successor.get(cur)
        if nxt is None or nxt == cur:
            break
        traj.append(nxt)
        cur = nxt
    return traj


def fan_trajectories(s0, agent, max_steps=60):
    """Yield (action, advantage, trajectory) for each alternative stored at s0."""
    actions = agent["alt_actions"].get(s0, [])
    adv     = agent["advantage"].get(s0, {})
    for a in actions:
        traj = counterfactual_rollout(
            s0, a, agent["alt_first_step"], agent["optimal_successor"], max_steps
        )
        yield a, adv.get(a, 0.0), traj


def walk(s0, agent, max_steps=60, deviation_day=None, deviation_action=None):
    """Yield (day, state, action, next_state) along the optimal chain from s0.

    If deviation_day is given, replace the action at that step with
    deviation_action; otherwise the walk is fully optimal.
    """
    cur = s0
    for day in range(max_steps):
        a_list = agent["alt_actions"].get(cur)
        if not a_list:
            return
        if day == deviation_day:
            action = deviation_action
            nxt = agent["alt_first_step"].get((cur, action))
        else:
            action = a_list[0]
            nxt = agent["optimal_successor"].get(cur)
        if nxt is None or nxt == cur:
            return
        yield day, cur, action, nxt
        cur = nxt


def trajectory_and_actions(s0, agent, max_steps=60, deviation_day=None, deviation_action=None):
    """Return (states, actions). states has len = actions+1; last action emits no successor."""
    states  = [s0]
    actions = []
    for _, _, a, ns in walk(s0, agent, max_steps, deviation_day, deviation_action):
        actions.append(a)
        states.append(ns)
    return states, actions


def cumulative_reward(states, weight_idx=0):
    w = np.array([s[weight_idx] for s in states], dtype=float)
    return w - w[0]
