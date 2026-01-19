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


# ======================================== Plant Instance ======================================== #

    # --- 1. Define your Configuration ---

    # State is JUST weight.
    # We want to track weights from 50g to 1000g in 10g increments.
    # plant_state_map = {
    #     'plant_weight': {'bounds': (50, 1000), 'granularity': 10}
    # }
    #
    # # The column in your parquet that represents Water Loss
    # action_column = 'water_loss'
    #
    # # --- 2. Initialize and Run ---
    #
    # # Assumes you have a 'plant_data.parquet' with columns: ['plant_weight', 'water_loss']
    # plant_mdp = PlantMDP(
    #     data_path="plant_data.parquet",
    #     state_map=plant_state_map,
    #     action_col=action_column
    # )
    #
    # # Process the large table
    # plant_mdp.process_data()
    #
    # # --- 3. Analyze the Results ---
    #
    # # Let's see how it handled the variation you mentioned:
    # print(f"Total States explored: {len(plant_mdp.transitions)}")
    #
    # # Example: Inspecting a specific weight and water loss
    # sample_s = (500.0,)  # Plant weighs 500g
    # sample_a = 30.0  # It lost 30g of water
    # expected_g = plant_mdp.expected_rewards.get(str((sample_s, sample_a)), 0)
    #
    # print(f"For weight {sample_s} and water loss {sample_a}:")
    # print(f"Average growth (Expected Reward): {expected_g}g")
    #
    # # Save the model
    # plant_mdp.save("plant_growth_model")