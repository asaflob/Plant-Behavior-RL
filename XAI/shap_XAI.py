import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os


def run_shap_analysis(model_file):
    # ==========================================
    # load the pickle
    # ==========================================
    print(f"Loading Agent from: {model_file}...")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Could not find model file at {model_file}")

    with open(model_file, 'rb') as f:
        agent_data = pickle.load(f)

    policy = agent_data['optimal_policy']

    target_soil = agent_data.get('soil_type', 'Unknown Soil').upper()
    action_method = agent_data.get('action_method', 'Unknown Action Method')

    X_data = []
    y_data = []

    for state, action in policy.items():
        # state הוא בדרך כלל טאפל של: (Temp, Humidity, PAR, Start_Weight)
        X_data.append(list(state))
        y_data.append(action)

    feature_names = ['Temperature', 'Humidity', 'PAR', 'Start_Weight']

    X = pd.DataFrame(X_data, columns=feature_names)
    y = np.array(y_data)

    print(f"Successfully extracted {len(X)} unique state-action pairs from the Agent's policy.")

    # ==========================================
    # random forest model
    # ==========================================
    print("Training Random Forest Surrogate Model...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X, y)

    # ==========================================
    # Fidelity
    # ==========================================
    predictions = rf_model.predict(X)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)

    print("\n--- Fidelity Metrics (Post-hoc Verification) ---")
    print(f"Surrogate MAE: {mae:.2f} action levels")
    print(f"Surrogate R^2: {r2:.2f}")
    print("------------------------------------------------")
    if r2 < 0.6:
        print(
            "Warning: Surrogate model fidelity is somewhat low. SHAP explanations might not be perfectly representative.")

    # ==========================================
    #  SHAP
    # ==========================================
    print("\nCalculating SHAP values... (this might take a few seconds)")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X)

    # --- Summary Plot ---
    # show which feature is the most important
    print("Generating Global Summary Plot...")
    plt.figure(figsize=(10, 6))

    plt.title(f"all features importance ({target_soil} | {action_method})\nSHAP Summary Plot", fontsize=16)

    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary_plot.png", dpi=300)
    plt.show()

    # --- Waterfall Plot ---
    # one example to show one decision
    # sample_idx = np.random.randint(0, len(X))  # random state from the policy
    print("\nLocating Critical State #1 in the dataset...")

    condition = (X['Start_Weight'] == 7279.0) & (X['PAR'] > 336.7) & (X['PAR'] < 336.9)

    matching_indices = X[condition].index
    if len(matching_indices) == 0:
        print("Could not find the exact state. Falling back to random...")
        sample_idx = np.random.randint(0, len(X))
    else:
        sample_idx = matching_indices[0]
        print(f"Found Critical State at index {sample_idx}!")

    sample_state = X.iloc[sample_idx]
    sample_action_rf = predictions[sample_idx]
    sample_action_q = y[sample_idx]

    print(f"\n--- most critical state explanation (single decision) ---")
    print(f"State inputs:\n{sample_state.to_string()}")
    print(f"Original Agent Action: {sample_action_q}")
    print(f"Surrogate Predicted Action: {sample_action_rf:.1f}")

    explanation = shap.Explanation(values=shap_values[sample_idx],
                                   base_values=explainer.expected_value[0] if isinstance(explainer.expected_value,
                                                                                         np.ndarray) else explainer.expected_value,
                                   data=sample_state.values,
                                   feature_names=feature_names)

    plt.figure(figsize=(10, 6))

    plt.title(f"most critical Decision Explanation ({target_soil} | {action_method})\nChosen Action Level: {sample_action_q}",
              fontsize=16)

    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    plt.savefig("shap_waterfall_plot.png", dpi=300)
    plt.show()


#q_agent_sand_gmm_500_act_50_DT_GRANULARITY_new_state.pkl
#q_agent_sand_gmm_500_act_50_EVAPORATION_PERCENTAGE_new_state.pkl

#q_agent_soil_gmm_116_act_50_DT_GRANULARITY_new_state.pkl
#q_agent_soil_gmm_116_act_50_EVAPORATION_PERCENTAGE_new_state.pkl

if __name__ == "__main__":
    target_model_path = os.path.join("..", "q_agent_sand_gmm_500_act_50_DT_GRANULARITY_new_state.pkl")
    run_shap_analysis(target_model_path)