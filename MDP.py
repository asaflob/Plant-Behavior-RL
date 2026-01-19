import pandas as pd
import numpy as np
import json
import os
from itertools import product
from collections import defaultdict
from scipy.sparse import csr_matrix


# class PlantMDP:
#     def __init__(self, data_path, state_map, action_col):
#         self.data_path = data_path
#         self.state_map = state_map
#         self.action_col = action_col
#         self.state_cols = list(state_map.keys())
#
#         # 1. Generate State Space
#         self.states = self._build_states()
#         self.state_to_idx = {state: i for i, state in enumerate(self.states)}
#
#         # 2. Storage for observations
#         self.observations = []  # List of (s, a, s_next)
#         self.expected_rewards = {}  # (s, a) -> mean_reward
#         self.transitions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
#
#     def _build_states(self):
#         axis_values = []
#         for col, config in self.state_map.items():
#             l, u = config['bounds']
#             s = config['granularity']
#             vals = np.arange(l, u + s, s)
#             axis_values.append(vals[vals <= u + (s * 0.0001)])
#         return list(product(*axis_values))
#
#     def process_data(self):
#         """Streams the parquet data and populates transitions/rewards."""
#         df = pd.read_parquet(self.data_path)
#
#         # Temporary storage to handle the 'different gains' for same state-action
#         reward_accumulator = defaultdict(list)
#
#         for i in range(len(df) - 1):
#             # Extract states as tuples
#             curr_s = tuple(df.iloc[i][self.state_cols])
#             next_s = tuple(df.iloc[i + 1][self.state_cols])
#             action = df.iloc[i][self.action_col]
#
#             # 1. Record transition
#             self.observations.append((curr_s, action, next_s))
#             self.transitions[curr_s][action][next_s] += 1
#
#             # 2. Record reward (Weight Gain)
#             # In your case: next_weight - curr_weight
#             gain = next_s[0] - curr_s[0]
#             reward_accumulator[(curr_s, action)].append(gain)
#
#         # 3. Calculate Expected Rewards (Averages)
#         for key, gains in reward_accumulator.items():
#             self.expected_rewards[str(key)] = sum(gains) / len(gains)
#
#     def to_sparse_matrix(self):
#         """Returns a sparse matrix of transition counts."""
#         row_idx, col_idx, data = [], [], []
#         for s, actions in self.transitions.items():
#             for a, next_states in actions.items():
#                 for ns, count in next_states.items():
#                     row_idx.append(self.state_to_idx[s])
#                     col_idx.append(self.state_to_idx[ns])
#                     data.append(count)
#
#         size = len(self.states)
#         return csr_matrix((data, (row_idx, col_idx)), shape=(size, size))
#
#     def save(self, folder):
#         os.makedirs(folder, exist_ok=True)
#         # Save states and metadata
#         pd.DataFrame(self.states, columns=self.state_cols).to_parquet(f"{folder}/states.parquet")
#
#         metadata = {
#             "state_map": self.state_map,
#             "action_column": self.action_col,
#             "expected_rewards": self.expected_rewards
#         }
#         with open(f"{folder}/metadata.json", "w") as f:
#             json.dump(metadata, f, indent=4)
#
#         # Save observations
#         obs_df = pd.DataFrame([
#             {'curr_s': o[0], 'action': o[1], 'next_s': o[2]}
#             for o in self.observations
#         ])
#         obs_df.to_parquet(f"{folder}/observations.parquet")
#
#     def print_occupancy_stats(self):
#         """
#         Prints how many bins are used and how many total states were actually visited.
#         """
#         # 1. Theoretical calculations
#         theoretical_bins = {}
#         total_theoretical = 1
#         for col, config in self.state_map.items():
#             l, u, s = config['bounds'][0], config['bounds'][1], config['granularity']
#             bins = len(np.arange(l, u + s, s))
#             theoretical_bins[col] = bins
#             total_theoretical *= bins
#
#         # 2. Count observed (filled) unique states
#         # self.transitions.keys() contains every state that appeared as 'curr_s'
#         observed_states = set(self.transitions.keys())
#
#         # 3. Count unique bins filled per column
#         # We use a set comprehension to find unique values in each dimension of observed states
#         filled_bins_per_col = {}
#         for i, col_name in enumerate(self.state_cols):
#             unique_values_visited = set(state[i] for state in observed_states)
#             filled_bins_per_col[col_name] = len(unique_values_visited)
#
#         # --- Print Results ---
#         print("--- Bin Occupancy ---")
#         for col in self.state_cols:
#             filled = filled_bins_per_col[col]
#             total = theoretical_bins[col]
#             print(f"{col}: {filled}/{total} bins filled ({filled / total:.1%})")
#
#         print("\n--- Total State Occupancy ---")
#         visited = len(observed_states)
#         print(f"Total States Visited: {visited:,}")
#         print(f"Total States Possible: {total_theoretical:,}")
#         print(f"Space Sparsity: {visited / total_theoretical:.4%}")



class PlantMDP:
    def __init__(self, data_path, state_map, action_col, weight_col='start_weight'):
        self.data_path = data_path
        self.state_map = state_map
        self.action_col = action_col
        self.weight_col_name = weight_col  # השם של העמודה בדאטה המקורי שמייצגת משקל

        # המפתחות של המילון state_map הם שמות העמודות שאנחנו רוצים ב-State
        self.state_cols = list(state_map.keys())

        # 1. Generate State Space
        self.states = self._build_states()
        self.state_to_idx = {state: i for i, state in enumerate(self.states)}

        # 2. Storage
        self.observations = []
        self.expected_rewards = {}
        self.transitions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    def _build_states(self):
        axis_values = []
        for col, config in self.state_map.items():
            l, u = config['bounds']
            s = config['granularity']
            # יצירת טווחים
            vals = np.arange(l, u + s, s)
            # תיקון קטן לבעיות נקודה צפה
            vals = vals[vals <= u + (s * 0.0001)]
            axis_values.append(vals)
        return list(product(*axis_values))

    def process_data(self):
        """Streams the parquet data and populates transitions/rewards."""
        df = pd.read_parquet(self.data_path)

        # מיון כדי לוודא שהימים מסודרים
        if 'unique_id' in df.columns and 'date' in df.columns:
            df = df.sort_values(['unique_id', 'date'])

        reward_accumulator = defaultdict(list)

        # === שינוי קריטי: מעבר על כל צמח בנפרד ===
        # זה מונע מעבר שגוי בין סוף צמח אחד להתחלה של צמח אחר
        for plant_id, plant_df in df.groupby('unique_id'):

            # המרה ל-Numpy Array לביצועים מהירים יותר
            # אנחנו שולפים רק את העמודות הרלוונטיות ל-State + העמודה של הפעולה
            # אבל אנחנו צריכים גם את המשקל המקורי כדי לחשב Reward מדויק

            # 1. מכינים את ה-States (מעוגלים כבר מהקובץ)
            states_data = plant_df[self.state_cols].values
            actions_data = plant_df[self.action_col].values

            # אנחנו מניחים שהשמות ב-state_cols תואמים לעמודות ב-DF (כמו start_weight)
            # נאתר איפה נמצא המשקל בתוך ה-State כדי לחשב Reward
            try:
                weight_idx = self.state_cols.index(self.weight_col_name)
            except ValueError:
                raise ValueError(
                    f"Weight column '{self.weight_col_name}' must be one of the state keys defined in state_map!")

            # לולאה על ימי הצמח (עד יום אחד לפני הסוף)
            for i in range(len(plant_df) - 1):
                curr_s = tuple(states_data[i])
                next_s = tuple(states_data[i + 1])
                action = actions_data[i]

                # בדיקת תקינות: אם הפעולה היא NaN (למשל יום ללא דאטה), מדלגים
                if pd.isna(action):
                    continue

                # 1. Record transition
                self.observations.append((curr_s, action, next_s))
                self.transitions[curr_s][action][next_s] += 1

                # 2. Record reward (Weight Gain)
                # אנחנו משתמשים באינדקס שמצאנו כדי לשלוף את המשקל מתוך ה-State
                gain = next_s[weight_idx] - curr_s[weight_idx]
                reward_accumulator[(curr_s, action)].append(gain)

        # 3. Calculate Expected Rewards
        for key, gains in reward_accumulator.items():
            self.expected_rewards[str(key)] = sum(gains) / len(gains)

    # שאר הפונקציות נשארות זהות...
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
            "state_map": self.state_map,
            "action_column": self.action_col,
            "weight_column": self.weight_col_name,  # שומרים גם את זה
            "expected_rewards": self.expected_rewards
        }
        with open(f"{folder}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        # המרה מהירה לדאטה-פריים בלי לולאת פייתון איטית אם הרשימה ענקית
        if self.observations:
            obs_df = pd.DataFrame(self.observations, columns=['curr_s', 'action', 'next_s'])
            obs_df.to_parquet(f"{folder}/observations.parquet")

    def print_occupancy_stats(self):
        theoretical_bins = {}
        total_theoretical = 1
        for col, config in self.state_map.items():
            l, u, s = config['bounds'][0], config['bounds'][1], config['granularity']
            bins = len(np.arange(l, u + s, s))
            theoretical_bins[col] = bins
            total_theoretical *= bins

        observed_states = set(self.transitions.keys())
        filled_bins_per_col = {}
        for i, col_name in enumerate(self.state_cols):
            unique_values_visited = set(state[i] for state in observed_states)
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