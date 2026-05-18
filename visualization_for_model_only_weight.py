"""Plot the agent's average stomatal action vs growth stage for sand and soil."""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from agent_io import load_agent

TOTAL_ACTIONS = 50


def load_aggregated_policy(filename: str) -> tuple[Optional[list[float]], Optional[list[float]]]:
    """Average the agent's chosen action across all states sharing each weight."""
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return None, None

    agent = load_agent(filename)
    actions_by_weight: dict[float, list[int]] = defaultdict(list)
    # Weight is the first dimension of the state tuple.
    for state, action in agent.optimal_policy.items():
        actions_by_weight[state[0]].append(action)

    sorted_weights = sorted(actions_by_weight.keys())
    weights = list(sorted_weights)
    avg_actions = [float(np.mean(actions_by_weight[w])) for w in sorted_weights]

    w_min, w_max = min(weights), max(weights)
    if w_max == w_min:
        normalized_weights = [0.0 for _ in weights]
    else:
        normalized_weights = [(w - w_min) / (w_max - w_min) * 100 for w in weights]

    return normalized_weights, avg_actions


def plot_normalized_comparison() -> None:
    file_soil = "q_agent_soil_w5_t2_h10_actions_50.pkl"
    file_sand = "q_agent_sand_w5_t2_h10_actions_50.pkl"

    norm_w_soil, act_soil = load_aggregated_policy(file_soil)
    norm_w_sand, act_sand = load_aggregated_policy(file_sand)

    plt.figure(figsize=(12, 7))
    if norm_w_soil:
        plt.plot(norm_w_soil, act_soil, label="Soil Strategy (Avg)",
                 color="brown", linewidth=3)
    if norm_w_sand:
        plt.plot(norm_w_sand, act_sand, label="Sand Strategy (Avg)",
                 color="orange", linewidth=3, linestyle="--")

    plt.title(f"Strategy Comparison: Soil vs. Sand ({TOTAL_ACTIONS} Actions)", fontsize=16)
    plt.xlabel("Relative Growth Stage (0% = Seedling, 100% = Mature)", fontsize=12)
    plt.ylabel(f"Avg Stomatal Opening (Action 0–{TOTAL_ACTIONS - 1})", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")

    # Bottom 20% / top 30% bands annotate the policy as conservative vs aggressive.
    conservative_limit = TOTAL_ACTIONS * 0.2
    aggressive_start = TOTAL_ACTIONS * 0.7
    plt.axhspan(0, conservative_limit, color="red", alpha=0.1, label="Conservative Zone")
    plt.axhspan(aggressive_start, TOTAL_ACTIONS, color="green", alpha=0.1, label="Aggressive Zone")
    plt.ylim(-0.5, TOTAL_ACTIONS + 0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_normalized_comparison()
