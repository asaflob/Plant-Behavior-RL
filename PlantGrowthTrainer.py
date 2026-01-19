import pandas as pd
import numpy as np
import os
import pickle
import json
from MDP import PlantMDP


class PlantGrowthTrainer:
    def __init__(self, config_json_path):
        with open(config_json_path, 'r') as f:
            self.config = json.load(f)

        self.input_file = self.config['input_file']
        self.output_path = self.config['output_path']
        self.state_config = self.config.get('state_config', {})
        self.num_actions = self.config.get('num_actions', 10)
        self.episodes = self.config.get('episodes', 10000)
        self.soil_types = self.config.get('soil_types', ['sand', 'soil'])

        self.alpha = self.config.get('alpha', 0.1)
        self.gamma = self.config.get('gamma', 0.95)
        self.epsilon_start = self.config.get('epsilon_start', 1.0)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.9995)

        self.agents = {}

    def _resolve_bounds(self, df, col, bounds_config):
        lower, upper = bounds_config
        if lower == "min": lower = df[col].min()
        if upper == "max": upper = df[col].max()
        return (float(lower), float(upper))

    def _preprocess_data(self, soil_type):
        df = pd.read_parquet(self.input_file)
        if 'soil_type' in df.columns:
            df = df[df['soil_type'].astype(str).str.strip() == soil_type]

        if df.empty: return None, None

        state_cols = list(self.state_config.keys())
        df = df.dropna(subset=state_cols + ['dt', 'end_weight'])

        min_dt, max_dt = df['dt'].min(), df['dt'].max()
        df['stomatal_opening'] = (df['dt'] - min_dt) / (max_dt - min_dt)
        df['action_discrete'] = pd.cut(df['stomatal_opening'], bins=self.num_actions, labels=False)

        resolved_config = {}
        for col, settings in self.state_config.items():
            gran = settings['granularity']
            bounds = self._resolve_bounds(df, col, settings['bounds'])
            df[col] = np.round((df[col] / gran).round() * gran, 4)
            resolved_config[col] = {'bounds': bounds, 'granularity': gran}

        # ONLY create directory when we are ready to write
        os.makedirs(self.output_path, exist_ok=True)
        temp_path = os.path.join(self.output_path, f"temp_{soil_type}.parquet")
        df.to_parquet(temp_path)
        return temp_path, resolved_config

    def _get_env_step(self, mdp_model, state, action):
        state = tuple(np.round(state, 4))
        if state not in mdp_model.transitions or action not in mdp_model.transitions[state]:
            return state, -10

        next_dict = mdp_model.transitions[state][action]
        candidates = list(next_dict.keys())
        probs = np.array(list(next_dict.values())) / sum(next_dict.values())

        next_state = candidates[np.random.choice(len(candidates), p=probs)]
        return tuple(np.round(next_state, 4)), (next_state[0] - state[0])

    def train_soil_agent(self, soil_type):
        temp_file, resolved_config = self._preprocess_data(soil_type)
        if not temp_file: return

        mdp = PlantMDP(temp_file, resolved_config, 'action_discrete')
        mdp.process_data()
        mdp.print_occupancy_stats()

        ACTIONS = list(range(self.num_actions))
        # Ensure theoretical Q_table keys match the rounded data keys
        Q_table = {s: {a: 0.0 for a in ACTIONS} for s in mdp.states}
        epsilon = self.epsilon_start

        print(f"Training {soil_type}...")
        for ep in range(self.episodes):
            possible_starts = [s for s in Q_table.keys() if s[0] < 250]
            if not possible_starts: possible_starts = list(mdp.transitions.keys())

            current_state = possible_starts[np.random.choice(len(possible_starts))]

            for _ in range(60):
                # Extra safety check for key existence
                if current_state not in Q_table:
                    # Fallback to nearest neighbor or skip
                    break

                if np.random.random() < epsilon:
                    action = np.random.choice(ACTIONS)
                else:
                    action = max(Q_table[current_state], key=Q_table[current_state].get)

                next_state, reward = self._get_env_step(mdp, current_state, action)

                # Use .get to prevent crash if env_step returns a state slightly off-grid
                max_future_q = max(Q_table.get(next_state, {a: 0 for a in ACTIONS}).values())

                Q_table[current_state][action] += self.alpha * (
                            reward + self.gamma * max_future_q - Q_table[current_state][action])
                current_state = next_state

            epsilon = max(0.01, epsilon * self.epsilon_decay)

        self.save_agent(soil_type, Q_table, mdp, resolved_config)
        os.remove(temp_file)

    def save_agent(self, soil_type, Q_table, mdp, config):
        # Convert defaultdict for pickle
        trans = {s: {a: dict(ns) for a, ns in acts.items()} for s, acts in mdp.transitions.items()}

        save_dict = {
            "q_table": Q_table,
            "policy": {s: max(v, key=v.get) for s, v in Q_table.items()},
            "soil_type": soil_type,
            "state_config": config,
            "transitions": trans
        }

        os.makedirs(self.output_path, exist_ok=True)
        path = os.path.join(self.output_path, f"q_agent_{soil_type}.pkl")
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)
        print(f"Saved: {path}")

    def run_all(self):
        for soil in self.soil_types:
            self.train_soil_agent(soil)


if __name__ == "__main__":
    trainer = PlantGrowthTrainer("config.json")
    trainer.run_all()