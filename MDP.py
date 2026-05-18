"""Legacy grid-based MDP. Kept for the old PlantGrowthTrainer pipeline.

Newer code should use ``PlantMDPCluster`` in MDP_cluster.py.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from itertools import product
from typing import Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

DataSource = Union[str, os.PathLike, pd.DataFrame]


class PlantMDP:
    """Build an MDP whose state space is a manually-binned grid.

    Parameters
    ----------
    data : path or DataFrame
        Source data (one row per plant-day).
    state_map : dict
        Maps each state column to ``{"bounds": (lo, hi), "granularity": step}``.
        These define the grid axes.
    action_col : str
        Column holding the discrete action.
    weight_col : str
        Name of a state column (also bounded in ``state_map``) used to compute
        reward as Δweight between consecutive days.
    """

    def __init__(
        self,
        data: DataSource,
        state_map: dict,
        action_col: str,
        weight_col: str = "start_weight",
    ):
        self.data: DataSource = data
        self.state_map = state_map
        self.action_col = action_col
        self.weight_col_name = weight_col

        self.state_cols = list(state_map.keys())

        self.states: list[tuple] = self._build_states()
        self.state_to_idx = {state: i for i, state in enumerate(self.states)}

        self.observations: list[tuple] = []
        self.expected_rewards: dict[tuple, float] = {}
        self.transitions: defaultdict = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )

    def _build_states(self) -> list[tuple]:
        axis_values = []
        for config in self.state_map.values():
            lo, hi = config["bounds"]
            step = config["granularity"]
            vals = np.arange(lo, hi + step, step)
            # Trim values that overshoot the upper bound due to float arithmetic.
            vals = vals[vals <= hi + (step * 0.0001)]
            axis_values.append(vals)
        return list(product(*axis_values))

    def process_data(self) -> None:
        """Walk per-plant trajectories to populate transitions and rewards."""
        df = (
            self.data
            if isinstance(self.data, pd.DataFrame)
            else pd.read_parquet(self.data)
        )

        if "unique_id" in df.columns and "date" in df.columns:
            df = df.sort_values(["unique_id", "date"])

        reward_accumulator: defaultdict = defaultdict(list)

        # Iterate per-plant so we never cross trajectories between different
        # plants when recording transitions.
        for _, plant_df in df.groupby("unique_id"):
            states_data = plant_df[self.state_cols].values
            actions_data = plant_df[self.action_col].values

            try:
                weight_idx = self.state_cols.index(self.weight_col_name)
            except ValueError as exc:
                raise ValueError(
                    f"Weight column '{self.weight_col_name}' must be one of the "
                    f"state keys defined in state_map!"
                ) from exc

            for i in range(len(plant_df) - 1):
                curr_s = tuple(states_data[i])
                next_s = tuple(states_data[i + 1])
                action = actions_data[i]

                if pd.isna(action):
                    continue

                self.observations.append((curr_s, action, next_s))
                self.transitions[curr_s][action][next_s] += 1

                gain = next_s[weight_idx] - curr_s[weight_idx]
                reward_accumulator[(curr_s, action)].append(gain)

        for key, gains in reward_accumulator.items():
            self.expected_rewards[key] = sum(gains) / len(gains)

    def to_sparse_matrix(self) -> csr_matrix:
        """Return a SciPy CSR matrix of raw transition counts (state × state)."""
        row_idx, col_idx, data = [], [], []
        for s, actions in self.transitions.items():
            for next_states in actions.values():
                for ns, count in next_states.items():
                    row_idx.append(self.state_to_idx[s])
                    col_idx.append(self.state_to_idx[ns])
                    data.append(count)
        size = len(self.states)
        return csr_matrix((data, (row_idx, col_idx)), shape=(size, size))

    def save(self, folder: str) -> None:
        os.makedirs(folder, exist_ok=True)
        pd.DataFrame(self.states, columns=self.state_cols).to_parquet(
            f"{folder}/states.parquet"
        )
        metadata = {
            "state_map": self.state_map,
            "action_column": self.action_col,
            "weight_column": self.weight_col_name,
            "expected_rewards": {str(k): v for k, v in self.expected_rewards.items()},
        }
        with open(f"{folder}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        if self.observations:
            obs_df = pd.DataFrame(
                self.observations, columns=["curr_s", "action", "next_s"]
            )
            obs_df.to_parquet(f"{folder}/observations.parquet")

    def print_occupancy_stats(self) -> None:
        theoretical_bins = {}
        total_theoretical = 1
        for col, config in self.state_map.items():
            lo, hi, step = config["bounds"][0], config["bounds"][1], config["granularity"]
            bins = len(np.arange(lo, hi + step, step))
            theoretical_bins[col] = bins
            total_theoretical *= bins

        observed_states = set(self.transitions.keys())
        filled_bins_per_col = {}
        for i, col_name in enumerate(self.state_cols):
            unique_values_visited = {state[i] for state in observed_states}
            filled_bins_per_col[col_name] = len(unique_values_visited)

        print("--- Bin Occupancy ---")
        for col in self.state_cols:
            filled = filled_bins_per_col[col]
            total = theoretical_bins[col]
            print(f"{col}: {filled}/{total} bins filled ({filled / total:.1%})")

        print("\n--- Total State Occupancy ---")
        visited = len(observed_states)
        print(f"Total States Visited: {visited:,}")
        print(f"Total States Possible: {total_theoretical:,}")
        print(f"Space Sparsity: {visited / total_theoretical:.4%}")
