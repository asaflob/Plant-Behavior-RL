"""SHAP-based explanation of the trained Q-learning policy.

Fits a Random Forest surrogate model on the (state -> chosen action) mapping
from the agent's policy, then uses SHAP on the surrogate to attribute each
action choice to its input features.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Allow running this file directly from XAI/.
sys.path.append(str(Path(__file__).resolve().parent.parent))
from agent_io import load_agent  # noqa: E402

DEFAULT_FEATURE_NAMES = ["Temperature", "Humidity", "PAR", "Start_Weight"]
FIDELITY_R2_WARNING_THRESHOLD = 0.6


def run_shap_analysis(model_file: str | Path) -> None:
    model_file = Path(model_file)
    print(f"Loading Agent from: {model_file}...")
    if not model_file.exists():
        raise FileNotFoundError(f"Could not find model file at {model_file}")

    agent = load_agent(model_file)
    target_soil = agent.soil_type.upper()
    action_method = agent.action_method

    X_data = [list(state) for state in agent.optimal_policy.keys()]
    y_data = list(agent.optimal_policy.values())

    feature_names = agent.state_cols or DEFAULT_FEATURE_NAMES
    if X_data and len(X_data[0]) != len(feature_names):
        # Fall back to generic names when the saved schema doesn't match.
        feature_names = [f"feat_{i}" for i in range(len(X_data[0]))]

    X = pd.DataFrame(X_data, columns=feature_names)
    y = np.array(y_data)

    print(f"Extracted {len(X)} state-action pairs from the agent's policy.")

    print("Training Random Forest Surrogate Model...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X, y)

    predictions = rf_model.predict(X)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    print("\n--- Fidelity Metrics (Post-hoc Verification) ---")
    print(f"Surrogate MAE: {mae:.2f} action levels")
    print(f"Surrogate R^2: {r2:.2f}")
    if r2 < FIDELITY_R2_WARNING_THRESHOLD:
        print(
            "Warning: Surrogate fidelity is low — SHAP explanations may not be "
            "perfectly representative of the original policy."
        )

    print("\nCalculating SHAP values...")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X)

    print("Generating Global Summary Plot...")
    plt.figure(figsize=(10, 6))
    plt.title(
        f"Feature importance ({target_soil} | {action_method})\nSHAP Summary Plot",
        fontsize=16,
    )
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary_plot.png", dpi=300)
    plt.show()

    sample_idx = _pick_sample_index(X)
    sample_state = X.iloc[sample_idx]
    sample_action_rf = predictions[sample_idx]
    sample_action_q = y[sample_idx]
    print(f"\n--- Most critical state explanation (single decision) ---")
    print(f"State inputs:\n{sample_state.to_string()}")
    print(f"Original Agent Action: {sample_action_q}")
    print(f"Surrogate Predicted Action: {sample_action_rf:.1f}")

    base_value = (
        explainer.expected_value[0]
        if isinstance(explainer.expected_value, np.ndarray)
        else explainer.expected_value
    )
    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=base_value,
        data=sample_state.values,
        feature_names=feature_names,
    )

    plt.figure(figsize=(10, 6))
    plt.title(
        f"Most critical decision ({target_soil} | {action_method})\n"
        f"Chosen Action Level: {sample_action_q}",
        fontsize=16,
    )
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    plt.savefig("shap_waterfall_plot.png", dpi=300)
    plt.show()


def _pick_sample_index(X: pd.DataFrame) -> int:
    """Find a hand-picked critical state if present; otherwise pick at random."""
    if {"Start_Weight", "PAR"}.issubset(X.columns):
        condition = (X["Start_Weight"] == 7279.0) & (X["PAR"] > 336.7) & (X["PAR"] < 336.9)
        matches = X[condition].index
        if len(matches) > 0:
            print(f"Found Critical State at index {matches[0]}!")
            return int(matches[0])
    print("Falling back to a random state for the waterfall plot.")
    return int(np.random.randint(0, len(X)))


if __name__ == "__main__":
    target_model_path = os.path.join("..", "q_agent_sand_gmm_500_act_50_DT_GRANULARITY_new_state.pkl")
    run_shap_analysis(target_model_path)
