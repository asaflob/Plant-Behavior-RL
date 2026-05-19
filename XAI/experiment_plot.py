"""Three-panel experiment plot: cumulative reward, action, state — over T days.

The optimal trajectory uses π* every day. The 2nd-best / worst trajectories
follow π* for [0, deviation_day), deviate at deviation_day, and return to π*.
"""

import matplotlib.pyplot as plt
import numpy as np

from .counterfactual import trajectory_and_actions, cumulative_reward


COLORS = {
    "optimal": "#2e7d32",
    "second":  "#ef6c00",
    "worst":   "#c62828",
    "plant":   "#000000",
}


def _alternative_actions(agent, opt_states, deviation_day):
    """Return (second_action, worst_action) usable at deviation_day, or (None, None)."""
    if deviation_day >= len(opt_states):
        return None, None
    dev_state = opt_states[deviation_day]
    a_list = agent["alt_actions"].get(dev_state, [])
    second = a_list[1] if len(a_list) >= 2 else None
    worst  = a_list[-1] if a_list else None
    if worst == a_list[0] if a_list else False:
        worst = None  # only one observed action; nothing meaningful to compare
    return second, worst


def _pad_actions(actions, length):
    out = np.full(length, np.nan, dtype=float)
    for i, a in enumerate(actions):
        if a is not None:
            out[i] = a
    return out


def plot_experiment(s0, agent, deviation_day, T=60, weight_idx=0,
                    plant_states=None, plant_actions=None,
                    state_names=None):
    """Render reward / action / state panels for π*, 2nd-best, worst, and plant."""
    opt_states, opt_actions = trajectory_and_actions(s0, agent, T)
    second_a, worst_a = _alternative_actions(agent, opt_states, deviation_day)

    runs = [("optimal", opt_states, opt_actions)]
    if second_a is not None:
        s2, a2 = trajectory_and_actions(s0, agent, T, deviation_day, second_a)
        runs.append(("second", s2, a2))
    if worst_a is not None:
        sw, aw = trajectory_and_actions(s0, agent, T, deviation_day, worst_a)
        runs.append(("worst", sw, aw))
    if plant_states is not None:
        runs.append(("plant", list(plant_states),
                     list(plant_actions) if plant_actions is not None else []))

    state_dim = len(s0)
    state_names = state_names or [f"state[{i}]" for i in range(state_dim)]

    fig, axes = plt.subplots(2 + state_dim, 1,
                             figsize=(10, 4 + 2.5 * (1 + state_dim)),
                             sharex=True)
    ax_reward, ax_action = axes[0], axes[1]
    state_axes = axes[2:]

    for label, states, actions in runs:
        color = COLORS[label]
        ls = "--" if label == "plant" else "-"

        y = cumulative_reward(states, weight_idx)
        ax_reward.plot(range(len(y)), y, label=label, color=color, linestyle=ls, lw=2)

        a_arr = _pad_actions(actions, len(states) - 1)
        ax_action.plot(range(len(a_arr)), a_arr, label=label,
                       color=color, linestyle=ls, marker="o", ms=3)

        for dim, ax in enumerate(state_axes):
            vals = [s[dim] for s in states]
            ax.plot(range(len(vals)), vals, label=label,
                    color=color, linestyle=ls, lw=1.5)

    for ax in axes:
        ax.axvline(deviation_day, color="grey", alpha=0.35, linestyle=":")

    ax_reward.set_ylabel("cumulative reward")
    ax_reward.set_title(f"Counterfactual deviation at day {deviation_day}")
    ax_reward.legend(loc="best")

    ax_action.set_ylabel("action")

    for ax, name in zip(state_axes, state_names):
        ax.set_ylabel(name)

    axes[-1].set_xlabel("day")
    fig.tight_layout()
    return fig, axes
