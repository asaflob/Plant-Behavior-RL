"""Legacy config-driven trainer using the grid-based PlantMDP.

Newer code should use ``train_agent.py`` with one of the clustering methods.
This script is kept for reproducibility of older results.
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from MDP import PlantMDP

EPISODE_LENGTH_DAYS = 60
SEEDLING_WEIGHT_THRESHOLD_GRAMS = 250
UNKNOWN_ACTION_PENALTY = -10


class PlantGrowthTrainer:
    """Train one Q-learning agent per soil type, driven by a JSON config."""

    def __init__(self, config_json_path: str | Path):
        with open(config_json_path, "r") as f:
            self.config = json.load(f)

        self.input_file = self.config["input_file"]
        self.output_path = self.config["output_path"]
        self.state_config = self.config.get("state_config", {})
        self.num_actions = self.config.get("num_actions", 10)
        self.episodes = self.config.get("episodes", 10000)
        self.soil_types = self.config.get("soil_types", ["sand", "soil"])

        self.alpha = self.config.get("alpha", 0.1)
        self.gamma = self.config.get("gamma", 0.95)
        self.epsilon_start = self.config.get("epsilon_start", 1.0)
        self.epsilon_decay = self.config.get("epsilon_decay", 0.9995)

        self.agents: dict = {}

    def _resolve_bounds(self, df: pd.DataFrame, col: str, bounds_config) -> Tuple[float, float]:
        lower, upper = bounds_config
        if lower == "min":
            lower = df[col].min()
        if upper == "max":
            upper = df[col].max()
        return (float(lower), float(upper))

    def _preprocess_data(self, soil_type: str) -> Tuple[Optional[str], Optional[dict]]:
        df = pd.read_parquet(self.input_file)
        if "soil_type" in df.columns:
            df = df[df["soil_type"].astype(str).str.strip() == soil_type]
        if df.empty:
            return None, None

        state_cols = list(self.state_config.keys())
        df = df.dropna(subset=state_cols + ["dt", "end_weight"])

        min_dt, max_dt = df["dt"].min(), df["dt"].max()
        df["stomatal_opening"] = (df["dt"] - min_dt) / (max_dt - min_dt)
        df["action_discrete"] = pd.cut(df["stomatal_opening"], bins=self.num_actions, labels=False)

        resolved_config: dict = {}
        for col, settings in self.state_config.items():
            gran = settings["granularity"]
            bounds = self._resolve_bounds(df, col, settings["bounds"])
            df[col] = np.round((df[col] / gran).round() * gran, 4)
            resolved_config[col] = {"bounds": bounds, "granularity": gran}

        os.makedirs(self.output_path, exist_ok=True)
        temp_path = os.path.join(self.output_path, f"temp_{soil_type}.parquet")
        df.to_parquet(temp_path)
        return temp_path, resolved_config

    def _get_env_step(self, mdp_model: PlantMDP, state, action):
        state = tuple(np.round(state, 4))
        if state not in mdp_model.transitions or action not in mdp_model.transitions[state]:
            return state, UNKNOWN_ACTION_PENALTY

        next_dict = mdp_model.transitions[state][action]
        candidates = list(next_dict.keys())
        probs = np.array(list(next_dict.values())) / sum(next_dict.values())
        next_state = candidates[np.random.choice(len(candidates), p=probs)]
        return tuple(np.round(next_state, 4)), next_state[0] - state[0]

    def train_soil_agent(self, soil_type: str) -> None:
        temp_file, resolved_config = self._preprocess_data(soil_type)
        if not temp_file:
            return

        mdp = PlantMDP(temp_file, resolved_config, "action_discrete")
        mdp.process_data()
        mdp.print_occupancy_stats()

        actions = list(range(self.num_actions))
        q_table = {s: {a: 0.0 for a in actions} for s in mdp.states}
        epsilon = self.epsilon_start

        print(f"Training {soil_type}...")
        for _ in range(self.episodes):
            possible_starts = [
                s for s in q_table.keys() if s[0] < SEEDLING_WEIGHT_THRESHOLD_GRAMS
            ]
            if not possible_starts:
                possible_starts = list(mdp.transitions.keys())

            current_state = possible_starts[np.random.choice(len(possible_starts))]

            for _ in range(EPISODE_LENGTH_DAYS):
                if current_state not in q_table:
                    break

                if np.random.random() < epsilon:
                    action = np.random.choice(actions)
                else:
                    action = max(q_table[current_state], key=q_table[current_state].get)

                next_state, reward = self._get_env_step(mdp, current_state, action)

                # Fall back to a zero-q stub for states slightly off-grid so the
                # update doesn't crash on unexpected keys.
                max_future_q = max(q_table.get(next_state, {a: 0 for a in actions}).values())

                q_table[current_state][action] += self.alpha * (
                    reward + self.gamma * max_future_q - q_table[current_state][action]
                )
                current_state = next_state

            epsilon = max(0.01, epsilon * self.epsilon_decay)

        self.save_agent(soil_type, q_table, mdp, resolved_config)
        os.remove(temp_file)

    def save_agent(self, soil_type: str, q_table: dict, mdp: PlantMDP, config: dict) -> None:
        trans = {s: {a: dict(ns) for a, ns in acts.items()} for s, acts in mdp.transitions.items()}
        save_dict = {
            "q_table": q_table,
            "policy": {s: max(v, key=v.get) for s, v in q_table.items()},
            "soil_type": soil_type,
            "state_config": config,
            "transitions": trans,
        }
        os.makedirs(self.output_path, exist_ok=True)
        path = os.path.join(self.output_path, soil_type)
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)
        print(f"Saved: {path}")

    def run_all(self) -> None:
        for soil in self.soil_types:
            self.train_soil_agent(soil)


if __name__ == "__main__":
    trainer = PlantGrowthTrainer("config.json")
    trainer.run_all()
