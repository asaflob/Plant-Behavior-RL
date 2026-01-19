import pandas as pd
import numpy as np
import json
import os
from itertools import product
from collections import defaultdict
from scipy.sparse import csr_matrix


class PlantMDP:
    def __init__(self, data_path, state_map, action_col):
        self.data_path = data_path
        self.state_map = state_map
        self.action_col = action_col
        self.state_cols = list(state_map.keys())

        # 1. Generate State Space
        self.states = self._build_states()
        self.state_to_idx = {state: i for i, state in enumerate(self.states)}

        # 2. Storage for observations
        self.observations = []  # List of (s, a, s_next)
        self.expected_rewards = {}  # (s, a) -> mean_reward
        self.transitions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    def _build_states(self):
        axis_values = []
        for col, config in self.state_map.items():
            l, u = config['bounds']
            s = config['granularity']
            vals = np.arange(l, u + s, s)
            axis_values.append(vals[vals <= u + (s * 0.0001)])
        return list(product(*axis_values))

    def process_data(self):
        """Streams the parquet data and populates transitions/rewards."""
        df = pd.read_parquet(self.data_path)

        # Temporary storage to handle the 'different gains' for same state-action
        reward_accumulator = defaultdict(list)

        for i in range(len(df) - 1):
            # Extract states as tuples
            curr_s = tuple(df.iloc[i][self.state_cols])
            next_s = tuple(df.iloc[i + 1][self.state_cols])
            action = df.iloc[i][self.action_col]

            # 1. Record transition
            self.observations.append((curr_s, action, next_s))
            self.transitions[curr_s][action][next_s] += 1

            # 2. Record reward (Weight Gain)
            # In your case: next_weight - curr_weight
            gain = next_s[0] - curr_s[0]
            reward_accumulator[(curr_s, action)].append(gain)

        # 3. Calculate Expected Rewards (Averages)
        for key, gains in reward_accumulator.items():
            self.expected_rewards[str(key)] = sum(gains) / len(gains)

    def to_sparse_matrix(self):
        """Returns a sparse matrix of transition counts."""
        row_idx, col_idx, data = [], [], []
        for s, actions in self.transitions.items():
            for a, next_states in actions.items():
                for ns, count in next_states.items():
                    row_idx.append(self.state_to_idx[s])
                    col_idx.append(self.state_to_idx[ns])
                    data.append(count)

        size = len(self.states)
        return csr_matrix((data, (row_idx, col_idx)), shape=(size, size))

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        # Save states and metadata
        pd.DataFrame(self.states, columns=self.state_cols).to_parquet(f"{folder}/states.parquet")

        metadata = {
            "state_map": self.state_map,
            "action_column": self.action_col,
            "expected_rewards": self.expected_rewards
        }
        with open(f"{folder}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        # Save observations
        obs_df = pd.DataFrame([
            {'curr_s': o[0], 'action': o[1], 'next_s': o[2]}
            for o in self.observations
        ])
        obs_df.to_parquet(f"{folder}/observations.parquet")

    def print_occupancy_stats(self):
        """
        Prints how many bins are used and how many total states were actually visited.
        """
        # 1. Theoretical calculations
        theoretical_bins = {}
        total_theoretical = 1
        for col, config in self.state_map.items():
            l, u, s = config['bounds'][0], config['bounds'][1], config['granularity']
            bins = len(np.arange(l, u + s, s))
            theoretical_bins[col] = bins
            total_theoretical *= bins

        # 2. Count observed (filled) unique states
        # self.transitions.keys() contains every state that appeared as 'curr_s'
        observed_states = set(self.transitions.keys())

        # 3. Count unique bins filled per column
        # We use a set comprehension to find unique values in each dimension of observed states
        filled_bins_per_col = {}
        for i, col_name in enumerate(self.state_cols):
            unique_values_visited = set(state[i] for state in observed_states)
            filled_bins_per_col[col_name] = len(unique_values_visited)

        # --- Print Results ---
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
