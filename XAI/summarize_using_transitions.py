import pickle
import numpy as np
import os
from shap_XAI import *

def extract_critical_states(model_file, top_k=5):
    """
    Scans the Q-Table and extracts the most critical states based on the
    gap between the best and worst actions (State Importance ).
    """
    print(f"Loading Agent from: {model_file}...")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Could not find model file at {model_file}")

    with open(model_file, 'rb') as f:
        agent_data = pickle.load(f)

    # Note: We pull the full Q-table here, not just the policy,
    # as we need the values for all possible actions to calculate the importance gap.
    q_table = agent_data['q_table']
    target_soil = agent_data.get('soil_type', 'Unknown').upper()

    critical_states = []

    # Calculate State Importance I(s) for each state
    for state, actions_dict in q_table.items():
        if not actions_dict:
            continue

        q_values = list(actions_dict.values())
        if len(q_values) < 2:
            continue

        max_q = max(q_values)
        min_q = min(q_values)

        # Formula from Milani et al.: I(s) = max_Q - min_Q
        importance = max_q - min_q

        best_action = max(actions_dict, key=actions_dict.get)
        worst_action = min(actions_dict, key=actions_dict.get)

        critical_states.append({
            'State': state,
            'Importance': importance,
            'Best_Action': best_action,
            'Worst_Action': worst_action,
            'Max_Q': max_q,
            'Min_Q': min_q
        })

    # Sort states by importance (descending) and extract the top K
    critical_states = sorted(critical_states, key=lambda x: x['Importance'], reverse=True)
    top_states = critical_states[:top_k]

    # Pretty print for the report or presentation
    print(f"\n{'=' * 50}")
    print(f"Top {top_k} Most Critical States (Policy-Level Summarization )")
    print(f"Soil Type: {target_soil}")
    print(f"{'=' * 50}")

    for i, item in enumerate(top_states, 1):
        state_vals = item['State']
        # Format string based on the number of features (with or without PAR)
        if len(state_vals) == 4:
            temp, hum, par, weight = state_vals
            state_str = f"Temp: {temp:.1f}°C | Hum: {hum:.1f}% | PAR: {par:.1f} | Weight: {weight:.1f}g"
        elif len(state_vals) == 3:
            temp, hum, weight = state_vals
            state_str = f"Temp: {temp:.1f}°C | Hum: {hum:.1f}% | Weight: {weight:.1f}g"
        else:
            state_str = str(state_vals)

        print(f"\n--- Critical State #{i} ---")
        print(f"Environment: {state_str}")
        print(f"Dilemma Gap (Importance): {item['Importance']:.2f} expected reward points")
        print(f"Agent's Choice: Action {item['Best_Action']} (Expected Gain: {item['Max_Q']:.2f}g)")
        print(f"Worst Choice: Action {item['Worst_Action']} (Expected Gain: {item['Min_Q']:.2f}g)")

    return top_states


#q_agent_sand_gmm_500_act_50_DT_GRANULARITY_new_state.pkl
#q_agent_sand_gmm_500_act_50_EVAPORATION_PERCENTAGE_new_state.pkl

#q_agent_soil_gmm_116_act_50_DT_GRANULARITY_new_state.pkl
#q_agent_soil_gmm_116_act_50_EVAPORATION_PERCENTAGE_new_state.pkl


if __name__ == "__main__":
    # Update the path to your model file here as needed
    target_model_path = os.path.join("..", "q_agent_sand_gmm_500_act_50_DT_GRANULARITY_new_state.pkl")

    # Request to see the top 3 most critical moments for the agent
    extract_critical_states(target_model_path, top_k=3)