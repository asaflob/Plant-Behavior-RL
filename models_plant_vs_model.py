"""Evaluate a trained agent against real per-experiment plant behaviour.

For each experiment in the dataset we build a "virtual average plant" (mean
trajectory across all plants in that experiment) and compare what the agent
would do at each daily state against what the plants actually did.

Run directly:

.. code-block:: bash

    python models_plant_vs_model.py path/to/q_agent_*.pkl

The historical multi-iteration version of this script lives in
``archive/models_plant_vs_model_legacy.py``.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from actions import discretize
from agent_io import SavedAgent, load_agent

DATA_FILE = Path("data") / "tomato_mdp_final_with_pnw.parquet"
RAW_DATA_FILE = Path("data") / "tomato_raw_data_v2.parquet"

# A daily prediction within ±TOLERANCE_ACTION_LEVELS of the real action is
# treated as a "match" for the relaxed-accuracy metric. Strict accuracy
# requires |diff| < 0.5.
TOLERANCE_ACTION_LEVELS = 2


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass
class EvaluationData:
    df: pd.DataFrame
    agent: SavedAgent


def load_data_and_agent(model_path: Path) -> EvaluationData:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

    agent = load_agent(model_path)
    print(
        f"Agent loaded -> Soil: {agent.soil_type}, Actions: {agent.num_actions}, "
        f"Action method: {agent.action_method}"
    )

    df = pd.read_parquet(DATA_FILE)
    df = df[df["soil_type"].astype(str).str.strip() == agent.soil_type]
    df = _attach_experiment_id(df)
    df = _attach_real_action(df, agent)
    return EvaluationData(df=df, agent=agent)


def _attach_experiment_id(df: pd.DataFrame) -> pd.DataFrame:
    if "exp_ID" in df.columns:
        return df
    if "exp_id" in df.columns:
        return df.rename(columns={"exp_id": "exp_ID"})
    if RAW_DATA_FILE.exists():
        raw = pd.read_parquet(RAW_DATA_FILE)
        col = "exp_ID" if "exp_ID" in raw.columns else "exp_id"
        raw = raw[["unique_id", col]].drop_duplicates()
        return df.merge(raw, on="unique_id", how="inner").rename(columns={col: "exp_ID"})
    raise RuntimeError(
        "Couldn't find exp_ID either in the dataset or the raw data file."
    )


def _attach_real_action(df: pd.DataFrame, agent: SavedAgent) -> pd.DataFrame:
    discretized = discretize(agent.action_method, df, agent.num_actions)
    discretized = discretized.rename(columns={"action_discrete": "real_action_discrete"})
    return discretized.dropna(subset=["real_action_discrete"])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _virtual_plant_daily(exp_df: pd.DataFrame) -> pd.DataFrame:
    """Average trajectory across all plants in one experiment."""
    exp_df = exp_df.sort_values(["unique_id", "day_num"]).copy()
    exp_df["Real_Reward_Daily"] = (
        exp_df.groupby("unique_id")["start_weight"].shift(-1) - exp_df["start_weight"]
    ).fillna(0)

    if "avg_par" not in exp_df.columns:
        exp_df["avg_par"] = 0

    return exp_df.groupby("day_num").agg({
        "start_weight": "mean",
        "avg_temp": "mean",
        "avg_humidity": "mean",
        "avg_par": "mean",
        "real_action_discrete": "mean",
        "Real_Reward_Daily": "mean",
    }).reset_index()


def _closest_known_state(climate: np.ndarray, weight: float, known_states: Iterable[tuple]) -> tuple:
    """Nearest cluster-state for a real climate × weight reading.

    First snaps to the nearest climate cluster (Euclidean on temp/humidity/par),
    then picks the weight bucket in that cluster closest to the real weight.
    """
    known_states = list(known_states)
    unique_climates = {(s[0], s[1], s[2]) for s in known_states}
    closest_climate = min(unique_climates, key=lambda c: np.sum((np.array(c) - climate) ** 2))
    matching = [s for s in known_states if (s[0], s[1], s[2]) == closest_climate]
    return min(matching, key=lambda s: abs(s[3] - weight))


def evaluate_virtual_plant(
    exp_df: pd.DataFrame, agent: SavedAgent
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], float, float, float]:
    avg_temp = exp_df["avg_temp"].mean()
    avg_hum = exp_df["avg_humidity"].mean()

    daily = _virtual_plant_daily(exp_df)
    known_states = list(agent.optimal_policy.keys())

    rows = []
    correct = 0
    total = 0
    for _, row in daily.iterrows():
        climate = np.array([row["avg_temp"], row["avg_humidity"], row["avg_par"]])
        state = _closest_known_state(climate, row["start_weight"], known_states)
        real_action = row["real_action_discrete"]

        if state in agent.optimal_policy:
            agent_action = agent.optimal_policy[state]
            total += 1
            if agent_action == round(real_action):
                correct += 1
            agent_reward = agent.expected_rewards.get((tuple(state), int(agent_action)), 0)
        else:
            agent_action = np.nan
            agent_reward = 0

        rows.append({
            "Day": int(row["day_num"]),
            "Real_Action": real_action,
            "Agent_Action": agent_action,
            "Diff": abs(real_action - agent_action) if not pd.isna(agent_action) else None,
            "Real_Reward": row["Real_Reward_Daily"],
            "Agent_Reward": agent_reward,
        })

    res_df = pd.DataFrame(rows)
    if res_df.empty:
        return None, None, 0.0, avg_temp, avg_hum

    res_df["Real_Accumulated"] = res_df["Real_Reward"].cumsum()
    res_df["Agent_Accumulated"] = res_df["Agent_Reward"].cumsum()

    valid = res_df.dropna(subset=["Agent_Action"])
    accuracy = correct / total if total else 0.0
    return res_df, valid, accuracy, avg_temp, avg_hum


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_experiment_results(
    exp_id_val,
    res_df: pd.DataFrame,
    valid: pd.DataFrame,
    num_actions: int,
    median_day_value: float,
    num_plants: int,
    accuracy: float,
) -> None:
    mae = valid["Diff"].mean() if not valid.empty else 0
    stats_text = (
        f"Experiment: {exp_id_val}\n"
        f"Virtual plant from {num_plants} actual plants\n"
        f"Total Days: {len(res_df)}\n"
        f"Accuracy (relaxed ±{TOLERANCE_ACTION_LEVELS}): {accuracy:.1%}\n"
        f"MAE: {mae:.2f}\n"
        f"Plant Gain: {res_df['Real_Accumulated'].iloc[-1]:.1f}g\n"
        f"Agent Gain: {res_df['Agent_Accumulated'].iloc[-1]:.1f}g"
    )

    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(res_df["Day"], res_df["Real_Action"], label="Avg Real Plant",
             color="blue", marker="o", alpha=0.6, linewidth=2)
    ax1.plot(res_df["Day"], res_df["Agent_Action"], label="Agent Policy",
             color="red", marker="x", linestyle="--", linewidth=2)
    ax1.axvline(x=median_day_value, color="black", linestyle="--", linewidth=2, alpha=0.7,
                label=f"Median day {median_day_value:g}")
    ax1.set_title(f"Experiment {exp_id_val}: Agent vs Virtual Avg Plant", fontsize=16)
    ax1.set_ylabel(f"Action Level (0–{num_actions - 1})")
    ax1.set_ylim(-0.5, num_actions - 0.5)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    if not valid.empty:
        ax2.bar(valid["Day"], valid["Diff"], color="purple", alpha=0.7)
    ax2.axvline(x=median_day_value, color="black", linestyle="--", linewidth=2, alpha=0.7)
    ax2.set_title(f"Experiment {exp_id_val}: Prediction Error per Day", fontsize=14)
    ax2.set_ylabel("Abs Diff (Error)")
    ax2.grid(axis="y", alpha=0.3)
    fig1.tight_layout()

    fig2, ax3 = plt.subplots(figsize=(14, 6))
    ax3.plot(res_df["Day"], res_df["Real_Accumulated"], label="Real Plant Growth",
             color="dodgerblue", marker="o", linewidth=3)
    ax3.plot(res_df["Day"], res_df["Agent_Accumulated"], label="Agent Expected Growth",
             color="forestgreen", marker="^", linewidth=3)
    ax3.axvline(x=median_day_value, color="black", linestyle="--", linewidth=2, alpha=0.7)
    ax3.set_title(f"Experiment {exp_id_val}: Accumulated Reward", fontsize=16)
    ax3.set_xlabel("Day in Experiment")
    ax3.set_ylabel("Accumulated Gain (grams)")
    ax3.legend(loc="upper left", fontsize=12)
    ax3.grid(True, alpha=0.4)
    fig2.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def analyze_per_experiment(model_path: Path) -> None:
    data = load_data_and_agent(model_path)
    global_errors: list[float] = []

    for exp_id_val, exp_group in data.df.groupby("exp_ID"):
        num_plants = exp_group["unique_id"].nunique()
        if num_plants == 0:
            continue

        plant_durations = exp_group.groupby("unique_id")["day_num"].nunique()
        median_days = plant_durations.median()

        print(f"\n{'=' * 50}")
        print(
            f"Processing Experiment: {exp_id_val} | "
            f"Valid Plants: {num_plants} | Median Days: {median_days:g}"
        )

        res_df, valid, _, avg_temp, avg_hum = evaluate_virtual_plant(exp_group, data.agent)
        if res_df is None or valid.empty:
            print(f"CRITICAL: Agent recognized NONE of the states in Experiment {exp_id_val}.")
            continue

        mae = valid["Diff"].mean()
        total_preds = len(valid)
        strict_acc = (valid["Diff"] < 0.5).mean() if total_preds else 0
        relaxed_acc = (valid["Diff"] <= TOLERANCE_ACTION_LEVELS).mean() if total_preds else 0
        global_errors.extend(valid["Diff"].tolist())

        print(f"Climate Averages -> Temp: {avg_temp:.1f}°C | Humidity: {avg_hum:.1f}%")
        print(f"--> Strict Accuracy (Exact): {strict_acc:.1%}")
        print(f"--> Relaxed Accuracy (±{TOLERANCE_ACTION_LEVELS}): {relaxed_acc:.1%}")
        print(f"--> MAE: {mae:.2f} action levels")

        plot_experiment_results(
            exp_id_val, res_df, valid, data.agent.num_actions,
            median_days, num_plants, relaxed_acc,
        )

    _print_global_summary(global_errors)


def _print_global_summary(errors: list[float]) -> None:
    print(f"\n{'#' * 50}")
    print("=== GLOBAL AGENT PERFORMANCE SUMMARY ===")
    if not errors:
        print("No valid predictions were made across any experiment.")
        print(f"{'#' * 50}\n")
        return
    mae = sum(errors) / len(errors)
    strict = sum(1 for e in errors if e < 0.5) / len(errors)
    relaxed = sum(1 for e in errors if e <= TOLERANCE_ACTION_LEVELS) / len(errors)
    print(f"Overall Strict Accuracy (Exact): {strict:.1%}")
    print(f"Overall Relaxed Accuracy (±{TOLERANCE_ACTION_LEVELS}): {relaxed:.1%}")
    print(f"Overall MAE: {mae:.2f} action levels")
    print(f"Total valid daily predictions: {len(errors)}")
    print(f"{'#' * 50}\n")


def check_state_action_coverage(model_path: Path) -> None:
    agent = load_agent(model_path)
    total_combos = agent.num_states * agent.num_actions
    valid_combos = len(agent.expected_rewards)
    active_states = sum(1 for actions in agent.q_table.values() if actions)

    print("\n" + "=" * 50)
    print("=== State-Action Coverage Analysis ===")
    print("=" * 50)
    print(f"Total States (Clusters × Weight): {agent.num_states}")
    print(f"Total Actions: {agent.num_actions}")
    print("-" * 50)
    print(f"Theoretical Max Combinations: {total_combos:,}")
    print(f"Actual Valid Combinations (with Reward): {valid_combos:,}")
    if total_combos:
        print(f"Matrix Coverage (Density): {valid_combos / total_combos:.2%}")
    print("-" * 50)
    print(
        f"States with at least one valid action: "
        f"{active_states} out of {agent.num_states}"
    )
    print("=" * 50 + "\n")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "model_path", type=Path,
        nargs="?",
        default=Path("q_agent_sand_gmm_500_act_50_DT_GRANULARITY_new_state.pkl"),
        help="Path to a SavedAgent .pkl file",
    )
    p.add_argument(
        "--coverage-only", action="store_true",
        help="Just print state/action coverage stats and exit (no plots).",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    if not args.model_path.exists():
        print(f"Error: model file not found: {args.model_path}", file=sys.stderr)
        sys.exit(1)
    if args.coverage_only:
        check_state_action_coverage(args.model_path)
    else:
        analyze_per_experiment(args.model_path)


if __name__ == "__main__":
    main()
