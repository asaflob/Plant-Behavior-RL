"""Tabular Q-learning loop used by all trainers in this project."""
from __future__ import annotations

from typing import Callable, Hashable, Tuple

import numpy as np

# A training episode runs for ~2 months of simulated daily steps. 60 was chosen
# to match the typical length of the real-world greenhouse experiments the agent
# is meant to mimic.
EPISODE_LENGTH_DAYS = 60

# When picking an episode start state we filter to "young plants" so the agent
# learns full growth trajectories rather than only end-of-experiment states.
# Weight is the first dimension of the state tuple (start_weight in grams).
SEEDLING_WEIGHT_THRESHOLD_GRAMS = 250

# Returned by env_step_func when the current (state, action) pair was never
# observed in the data. The negative reward discourages the agent from drifting
# into unsupported regions of the policy.
UNKNOWN_ACTION_PENALTY = -10

PROGRESS_LOG_EVERY = 5000

State = Hashable
EnvStep = Callable[[State, int], Tuple[State, float]]


def q_learning(
    mdp_model,
    env_step_func: EnvStep,
    num_actions: int,
    max_iterations: int = 100_000,
    alpha: float = 0.1,
    gamma: float = 0.95,
    init_epsilon: float = 1.0,
    epsilon_min: float = 0.01,
    decay: float = 0.9995,
    threshold: float = 1e-4,
    consecutive_iterations: int = 10,
):
    """Run tabular Q-learning until convergence.

    Returns (q_table, convergence_history). The q_table maps each state in
    ``mdp_model.states`` to a dict of ``{action: q_value}``. The history is a
    list of the largest |ΔQ| observed in each iteration, useful for plotting
    convergence.
    """
    print("Starting Q-Learning Algorithm...")

    actions_list = list(range(num_actions))
    q_table = {s: {a: 0.0 for a in actions_list} for s in mdp_model.states}

    epsilon = init_epsilon
    converged_count = 0
    convergence_history: list[float] = []

    for iteration in range(max_iterations):
        # Pick a start state that represents a young plant we have data for.
        possible_starts = [
            s for s in mdp_model.states
            if s[0] < SEEDLING_WEIGHT_THRESHOLD_GRAMS and s in mdp_model.transitions
        ]
        if not possible_starts:
            possible_starts = list(mdp_model.transitions.keys())

        current_state = possible_starts[np.random.choice(len(possible_starts))]
        max_change_this_iteration = 0.0

        # Anneal the learning rate slowly across iterations.
        current_alpha = max(0.01, alpha * (0.9999 ** iteration))

        for _ in range(EPISODE_LENGTH_DAYS):
            # Epsilon-greedy action selection.
            if np.random.random() < epsilon:
                action = np.random.choice(actions_list)
            else:
                action = max(q_table[current_state], key=q_table[current_state].get)

            next_state, reward = env_step_func(current_state, action)

            # Bellman update: Q ← Q + α·(r + γ·max Q' − Q).
            max_future_q = max(q_table[next_state].values()) if next_state in q_table else 0
            current_q = q_table[current_state][action]
            new_q = current_q + current_alpha * (reward + gamma * max_future_q - current_q)

            change = abs(new_q - current_q)
            if change > max_change_this_iteration:
                max_change_this_iteration = change

            q_table[current_state][action] = new_q
            current_state = next_state

        convergence_history.append(max_change_this_iteration)

        if epsilon > epsilon_min:
            epsilon *= decay

        if max_change_this_iteration < threshold:
            converged_count += 1
        else:
            converged_count = 0

        if converged_count >= consecutive_iterations:
            print(f"Algorithm converged successfully at iteration {iteration}!")
            break

        if iteration % PROGRESS_LOG_EVERY == 0:
            print(
                f"Iteration {iteration}/{max_iterations} | "
                f"Epsilon: {epsilon:.4f} | Max Change: {max_change_this_iteration:.6f}"
            )

    print("Q-Learning Training Finished.")
    return q_table, convergence_history
