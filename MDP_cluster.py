"""MDP whose states come from clustering (GMM / K-Means), not a grid."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Sequence, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

DataSource = Union[str, os.PathLike, pd.DataFrame]


class PlantMDPCluster:
    """Build transitions and expected rewards from a clustered dataset.

    Parameters
    ----------
    data : path or DataFrame
        Either a parquet path or a preloaded DataFrame.
    state_cols : list of column names
        These columns form the state tuple, in order.
    action_col : str
        Column holding the discrete action.
    weight_col : str
        Original (unbinned) weight column, used to compute reward (Δweight).
    """

    def __init__(
        self,
        data: DataSource,
        state_cols: Sequence[str],
        action_col: str,
        weight_col: str = "start_weight",
    ):
        self.state_cols = list(state_cols)
        self.action_col = action_col
        self.weight_col_name = weight_col

        self.df = data if isinstance(data, pd.DataFrame) else pd.read_parquet(data)

        self.states: list[tuple] = self._extract_unique_states()
        self.state_to_idx = {state: i for i, state in enumerate(self.states)}

        self.observations: list[tuple] = []
        self.expected_rewards: dict[tuple, float] = {}
        self.transitions: defaultdict = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )

    def _extract_unique_states(self) -> list[tuple]:
        unique_states_df = self.df[self.state_cols].drop_duplicates()
        return [tuple(x) for x in unique_states_df.to_numpy()]

    def process_data(self) -> None:
        """Walk per-plant trajectories to populate transitions and rewards."""
        if "unique_id" in self.df.columns and "date" in self.df.columns:
            self.df = self.df.sort_values(["unique_id", "date"])

        reward_accumulator: defaultdict = defaultdict(list)

        for _, plant_df in self.df.groupby("unique_id"):
            states_data = plant_df[self.state_cols].values
            actions_data = plant_df[self.action_col].values
            weights_data = plant_df[self.weight_col_name].values

            for i in range(len(plant_df) - 1):
                curr_state = tuple(states_data[i])
                next_state = tuple(states_data[i + 1])
                action = actions_data[i]

                if pd.isna(action):
                    continue

                self.observations.append((curr_state, action, next_state))
                self.transitions[curr_state][action][next_state] += 1

                # Reward = weight gain between consecutive days.
                gain = weights_data[i + 1] - weights_data[i]
                reward_accumulator[(curr_state, action)].append(gain)

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
        """Persist states, observations, and metadata to ``folder``."""
        os.makedirs(folder, exist_ok=True)
        pd.DataFrame(self.states, columns=self.state_cols).to_parquet(
            f"{folder}/states.parquet"
        )
        metadata = {
            "state_cols": self.state_cols,
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
        observed_states = set(self.transitions.keys())
        total_unique = len(self.states)
        visited = len(observed_states)

        print("\n--- State Occupancy (Clustered Data X Weight) ---")
        print(f"Total Unique States Available: {total_unique:,}")
        print(f"Total States Actually Visited by Agent: {visited:,}")

        print("\n--- Feature Diversity ---")
        for i, col_name in enumerate(self.state_cols):
            unique_values_visited = {state[i] for state in observed_states}
            print(
                f"{col_name} has {len(unique_values_visited)} unique values in visited states."
            )
