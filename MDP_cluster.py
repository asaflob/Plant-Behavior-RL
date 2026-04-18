import pandas as pd
import numpy as np
import json
import os
from collections import defaultdict
from scipy.sparse import csr_matrix

class PlantMDPCluster:
    def __init__(self, data_path, state_cols, action_col, weight_col='start_weight'):
        self.data_path = data_path
        self.state_cols = state_cols
        self.action_col = action_col
        self.weight_col_name = weight_col

        self.df = pd.read_parquet(self.data_path)
        self.states = self._extract_unique_states()
        self.state_to_idx = {state: i for i, state in enumerate(self.states)}

        self.observations = []
        self.expected_rewards = {}
        self.transitions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    def _extract_unique_states(self):
        unique_states_df = self.df[self.state_cols].drop_duplicates()
        unique_states = [tuple(x) for x in unique_states_df.to_numpy()]
        return unique_states

    def process_data(self):
        if 'unique_id' in self.df.columns and 'date' in self.df.columns:
            self.df = self.df.sort_values(['unique_id', 'date'])

        reward_accumulator = defaultdict(list)

        for plant_id, plant_df in self.df.groupby('unique_id'):
            states_data = plant_df[self.state_cols].values
            actions_data = plant_df[self.action_col].values
            weights_data = plant_df[self.weight_col_name].values

            for i in range(len(plant_df) - 1):
                # New State = [cluster-state] X Weight
                new_state = tuple(states_data[i])
                next_new_state = tuple(states_data[i + 1])
                action = actions_data[i]

                if pd.isna(action):
                    continue

                self.observations.append((new_state, action, next_new_state))
                self.transitions[new_state][action][next_new_state] += 1

                # Reward = Weight at beginning of day i+1 - Weight at the beginning of day i
                gain = weights_data[i + 1] - weights_data[i]

                # S \in NewState: אוספים את כל נקודות המידע שחולקות את אותו New State בדיוק
                reward_accumulator[(new_state, action)].append(gain)

        # Reward (Newstate, action) = AVG_ s \in NewState Reward (s, action)
        for key, gains in reward_accumulator.items():
            self.expected_rewards[key] = sum(gains) / len(gains)

    def to_sparse_matrix(self):
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
        pd.DataFrame(self.states, columns=self.state_cols).to_parquet(f"{folder}/states.parquet")
        metadata = {
            "state_cols": self.state_cols,
            "action_column": self.action_col,
            "weight_column": self.weight_col_name,
            "expected_rewards": {str(k): v for k, v in self.expected_rewards.items()}
        }
        with open(f"{folder}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        if self.observations:
            obs_df = pd.DataFrame(self.observations, columns=['curr_s', 'action', 'next_s'])
            obs_df.to_parquet(f"{folder}/observations.parquet")

    def print_occupancy_stats(self):
        observed_states = set(self.transitions.keys())
        total_unique = len(self.states)
        visited = len(observed_states)

        print("\n--- State Occupancy (Clustered Data X Weight) ---")
        print(f"Total Unique States Available: {total_unique:,}")
        print(f"Total States Actually Visited by Agent: {visited:,}")

        print("\n--- Feature Diversity ---")
        for i, col_name in enumerate(self.state_cols):
            unique_values_visited = set(state[i] for state in observed_states)
            print(f"{col_name} has {len(unique_values_visited)} unique values in visited states.")