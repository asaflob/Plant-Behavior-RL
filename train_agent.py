"""Unified training pipeline for the plant Q-learning agent.

Replaces the older ``train_agent_with_GMM.py`` and ``train_agent_knn.py`` —
both clustering methods now live in one place.

Example
-------
.. code-block:: bash

    python train_agent.py --soil sand --clustering gmm --num-states 500 \\
        --num-actions 50 --action-method EVAPORATION_PERCENTAGE
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from actions import discretize
from agent_io import build_saved_agent
from clustering import Clusterer, build_clusterer
from MDP_cluster import PlantMDPCluster
from q_learning_algo import UNKNOWN_ACTION_PENALTY, q_learning

DEFAULT_INPUT = Path("data") / "tomato_mdp_final_with_pnw_updated.parquet"
DEFAULT_STATE_COLS = ("avg_temp", "avg_humidity", "avg_par")
DEFAULT_REQUIRED_COLS = (
    "dt", "start_weight", "end_weight", "avg_temp", "avg_humidity", "avg_par",
)
WEIGHT_GRANULARITY_GRAMS = 1
CONVERGENCE_THRESHOLD = 1e-4


@dataclass
class TrainConfig:
    input_file: Path = DEFAULT_INPUT
    soil_type: str = "sand"
    num_actions: int = 50
    num_states: int = 500
    action_method: str = "EVAPORATION_PERCENTAGE"
    clustering_method: str = "gmm"
    state_cols: tuple[str, ...] = DEFAULT_STATE_COLS
    output_dir: Path = Path(".")


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def load_filtered_data(input_file: Path, soil_type: str) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}. Run the data preprocessing "
            "pipeline first."
        )
    df = pd.read_parquet(input_file)
    if "soil_type" in df.columns:
        df = df[df["soil_type"].astype(str).str.strip() == soil_type]
    if df.empty:
        raise ValueError(f"No data found for soil type {soil_type!r}.")
    return df.dropna(subset=list(DEFAULT_REQUIRED_COLS))


def cluster_states(
    df: pd.DataFrame, state_cols: tuple[str, ...], clusterer: Clusterer
) -> pd.DataFrame:
    """Normalise, cluster, and replace state columns with cluster centroids."""
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(
        scaler.fit_transform(df[list(state_cols)]), columns=list(state_cols)
    )
    cluster_labels = clusterer.fit_predict(df_normalized.values)
    centroids_real = scaler.inverse_transform(clusterer.centroids)

    out = df.copy()
    out[list(state_cols)] = centroids_real[cluster_labels]
    out["weight_state"] = (
        (out["start_weight"] / WEIGHT_GRANULARITY_GRAMS).round() * WEIGHT_GRANULARITY_GRAMS
    )
    return out


def make_env_step(mdp_model: PlantMDPCluster):
    """Return an env_step function for q_learning, sampling next-state from data."""

    def env_step(state, action):
        if state not in mdp_model.transitions or action not in mdp_model.transitions[state]:
            return state, UNKNOWN_ACTION_PENALTY

        next_states_dict = mdp_model.transitions[state][action]
        candidates = list(next_states_dict.keys())
        counts = list(next_states_dict.values())
        probs = np.array(counts) / sum(counts)
        next_state = candidates[np.random.choice(len(candidates), p=probs)]

        reward = mdp_model.expected_rewards.get((state, action), 0)
        return next_state, reward

    return env_step


def save_convergence_plot(
    history: list[float], output_path: Path, soil_type: str, clustering_method: str
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(history, color="blue", alpha=0.5, label="Max Q-value Change (Delta)")
    plt.axhline(
        y=CONVERGENCE_THRESHOLD, color="red", linestyle="--",
        label=f"Convergence Threshold ({CONVERGENCE_THRESHOLD})",
    )
    plt.title(
        f"Q-Learning Convergence for {soil_type} Soil ({clustering_method})", fontsize=14
    )
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Max Delta (Change) in Q-values", fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot saved as {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig) -> Path:
    """Run the full pipeline and return the path to the saved model."""
    print(f"Loading data from {cfg.input_file}...")
    df = load_filtered_data(cfg.input_file, cfg.soil_type)

    print(f"Calculating actions using method: {cfg.action_method}")
    df = discretize(cfg.action_method, df, cfg.num_actions)

    print(
        f"Running {cfg.clustering_method.upper()} clustering with {cfg.num_states} components..."
    )
    clusterer = build_clusterer(cfg.clustering_method, cfg.num_states)
    df = cluster_states(df, cfg.state_cols, clusterer)

    mdp_state_cols = list(cfg.state_cols) + ["weight_state"]
    mdp_model = PlantMDPCluster(
        data=df,
        state_cols=mdp_state_cols,
        action_col="action_discrete",
        weight_col="start_weight",
    )
    mdp_model.process_data()
    mdp_model.print_occupancy_stats()

    _assert_rewards_reachable(mdp_model)

    print(f"\nStarting Q-Learning Execution for {cfg.soil_type}...")
    q_table, history = q_learning(
        mdp_model=mdp_model,
        env_step_func=make_env_step(mdp_model),
        num_actions=cfg.num_actions,
        threshold=CONVERGENCE_THRESHOLD,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = cfg.output_dir / (
        f"convergence_plot_{cfg.soil_type}_{cfg.clustering_method}.png"
    )
    save_convergence_plot(history, plot_path, cfg.soil_type, clusterer.name)

    model_path = cfg.output_dir / (
        f"q_agent_{cfg.soil_type}_{cfg.clustering_method}_{cfg.num_states}_"
        f"act_{cfg.num_actions}_{cfg.action_method}.pkl"
    )
    agent = build_saved_agent(
        q_table=q_table,
        soil_type=cfg.soil_type,
        num_actions=cfg.num_actions,
        num_states=cfg.num_states,
        state_cols=mdp_state_cols,
        action_method=cfg.action_method,
        clustering_method=clusterer.name,
        expected_rewards=mdp_model.expected_rewards,
    )
    agent.save(model_path)
    print(f"Model saved to {model_path}")
    return model_path


def _assert_rewards_reachable(mdp_model: PlantMDPCluster) -> None:
    # Guards against silently regressing to the str()-keyed reward bug:
    # if we end up with a populated transitions dict but an empty reward map,
    # every Bellman update would learn from zeros.
    if mdp_model.transitions and not mdp_model.expected_rewards:
        raise RuntimeError(
            "MDP has transitions but no expected rewards — reward computation broke."
        )


def parse_args(argv: Optional[list[str]] = None) -> TrainConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--soil", default="sand", choices=("sand", "soil"))
    p.add_argument("--num-actions", type=int, default=50)
    p.add_argument("--num-states", type=int, default=500)
    p.add_argument(
        "--action-method", default="EVAPORATION_PERCENTAGE",
        choices=("DT_NORMALIZED", "EVAPORATION_PERCENTAGE", "DT_GRANULARITY"),
    )
    p.add_argument("--clustering", default="gmm", choices=("gmm", "kmeans"))
    p.add_argument("--output-dir", type=Path, default=Path("."))
    args = p.parse_args(argv)
    return TrainConfig(
        input_file=args.input,
        soil_type=args.soil,
        num_actions=args.num_actions,
        num_states=args.num_states,
        action_method=args.action_method,
        clustering_method=args.clustering,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    train(parse_args())
