import pandas as pd
import numpy as np
from itertools import product
from collections import defaultdict


class PlantMDP:
    def __init__(self, data_path, state_map, action_col):
        self.data_path = data_path
        self.state_map = state_map
        self.action_col = action_col
        self.state_cols = list(state_map.keys())

        # 1. Generate State Space with rounding to fix precision errors
        self.states = self._build_states()
        self.state_to_idx = {state: i for i, state in enumerate(self.states)}

        self.observations = []
        self.transitions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.filled_states = set()

    def _build_states(self):
        axis_values = []
        for col, config in self.state_map.items():
            l, u, s = config['bounds'][0], config['bounds'][1], config['granularity']
            # We round the arange results to handle float precision issues
            vals = np.round(np.arange(l, u + s, s), 4)
            axis_values.append(vals[vals <= u + (s * 0.0001)].tolist())
        return [tuple(map(float, s)) for s in product(*axis_values)]

    def process_data(self):
        df = pd.read_parquet(self.data_path)
        for i in range(len(df) - 1):
            # Force rounding on every state extracted from data
            curr_s = tuple(np.round(df.iloc[i][self.state_cols].values.astype(float), 4))
            next_s = tuple(np.round(df.iloc[i + 1][self.state_cols].values.astype(float), 4))
            action = int(df.iloc[i][self.action_col])

            self.transitions[curr_s][action][next_s] += 1
            self.filled_states.add(curr_s)
            self.filled_states.add(next_s)

    def print_occupancy_stats(self):
        print("\n--- Bin Occupancy ---")
        theoretical_total = 1
        for i, (col, config) in enumerate(self.state_map.items()):
            l, u, s = config['bounds'][0], config['bounds'][1], config['granularity']
            bins = len(np.arange(l, u + s, s))
            theoretical_total *= bins
            unique_visited = set(st[i] for st in self.filled_states)
            print(f"{col}: {len(unique_visited)}/{bins} bins filled")

        print(f"Total States Visited: {len(self.filled_states)}")
        print(f"Total States Possible: {len(self.states)}")