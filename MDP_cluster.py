import pandas as pd
import numpy as np
import json
import os
from collections import defaultdict
from scipy.sparse import csr_matrix


class PlantMDPCluster:
    def __init__(self, data_path, state_cols, action_col, weight_col='start_weight'):
        self.data_path = data_path
        self.state_cols = state_cols  # מקבל רשימה פשוטה של עמודות במקום מילון מסובך
        self.action_col = action_col
        self.weight_col_name = weight_col

        # 1. קריאת הנתונים וזיהוי כל המצבים הקיימים
        self.df = pd.read_parquet(self.data_path)
        self.states = self._extract_unique_states()
        self.state_to_idx = {state: i for i, state in enumerate(self.states)}

        # 2. Storage
        self.observations = []
        self.expected_rewards = {}
        self.transitions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    def _extract_unique_states(self):
        """שולף את כל המצבים הייחודיים ישירות מתוך הדאטה (ללא רשת/Granularity)"""
        # מוציאים את העמודות הרלוונטיות, מסירים כפילויות, והופכים לרשימה של טאפלים
        unique_states_df = self.df[self.state_cols].drop_duplicates()
        unique_states = [tuple(x) for x in unique_states_df.to_numpy()]
        return unique_states

    def process_data(self):
        """Streams the parquet data and populates transitions/rewards."""

        # מיון כדי לוודא שהימים מסודרים
        if 'unique_id' in self.df.columns and 'date' in self.df.columns:
            self.df = self.df.sort_values(['unique_id', 'date'])

        reward_accumulator = defaultdict(list)

        # מעבר על כל צמח בנפרד
        for plant_id, plant_df in self.df.groupby('unique_id'):

            # 1. מכינים את ה-States
            states_data = plant_df[self.state_cols].values
            actions_data = plant_df[self.action_col].values

            # נאתר איפה נמצא המשקל בתוך ה-State כדי לחשב Reward
            weights_data = plant_df[self.weight_col_name].values

            # לולאה על ימי הצמח (עד יום אחד לפני הסוף)
            for i in range(len(plant_df) - 1):
                curr_s = tuple(states_data[i])
                next_s = tuple(states_data[i + 1])
                action = actions_data[i]

                # בדיקת תקינות: אם הפעולה היא NaN מדלגים
                if pd.isna(action):
                    continue

                # 1. Record transition
                self.observations.append((curr_s, action, next_s))
                self.transitions[curr_s][action][next_s] += 1

                # 2. Record reward (Weight Gain)
                gain = weights_data[i + 1] - weights_data[i]
                reward_accumulator[(curr_s, action)].append(gain)
                ########################## todo
                # current_weight = weights_data[i]
                # next_weight = weights_data[i + 1]
                #
                # # מונעים חלוקה באפס למקרה שיש תקלות חיישן
                # if current_weight > 0:
                #     gain_pct = ((next_weight - current_weight) / current_weight) * 100
                # else:
                #     gain_pct = 0
                #
                # reward_accumulator[(curr_s, action)].append(gain_pct)
                ########################### todo check if it ok only with the pnw

        # 3. Calculate Expected Rewards
        for key, gains in reward_accumulator.items():
            self.expected_rewards[str(key)] = sum(gains) / len(gains)

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
            "expected_rewards": self.expected_rewards
        }
        with open(f"{folder}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        if self.observations:
            obs_df = pd.DataFrame(self.observations, columns=['curr_s', 'action', 'next_s'])
            obs_df.to_parquet(f"{folder}/observations.parquet")

    def print_occupancy_stats(self):
        """הדפסה מעודכנת שמתאימה לשימוש במצבים המבוססים על קלאסטרים"""
        observed_states = set(self.transitions.keys())
        total_unique = len(self.states)
        visited = len(observed_states)

        print("\n--- State Occupancy (Clustered Data) ---")
        print(f"Total Unique States Available (From Clustering): {total_unique:,}")
        print(f"Total States Actually Visited by Agent: {visited:,}")

        # הדפסת הפיצול לכל פיצ'ר למטרות בקרה
        print("\n--- Feature Diversity ---")
        for i, col_name in enumerate(self.state_cols):
            unique_values_visited = set(state[i] for state in observed_states)
            print(f"{col_name} has {len(unique_values_visited)} unique values in visited states.")