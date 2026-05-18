"""Policy-level summarisation: find the most "critical" states for the agent.

A state's importance is defined as ``max_a Q(s, a) - min_a Q(s, a)`` — the
larger the gap, the more the choice of action matters in that state. We
surface the top-K such states for further analysis (e.g. SHAP waterfalls).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running this file directly from XAI/.
sys.path.append(str(Path(__file__).resolve().parent.parent))
from agent_io import load_agent  # noqa: E402


def extract_critical_states(model_file: str | Path, top_k: int = 5) -> list[dict]:
    """Return the top-K states with the largest Q-value spread.

    Reference: Milani et al., "Policy summarization via state importance".
    """
    model_file = Path(model_file)
    print(f"Loading Agent from: {model_file}...")
    if not model_file.exists():
        raise FileNotFoundError(f"Could not find model file at {model_file}")

    agent = load_agent(model_file)
    target_soil = agent.soil_type.upper()

    critical_states: list[dict] = []
    for state, actions_dict in agent.q_table.items():
        if not actions_dict or len(actions_dict) < 2:
            continue
        q_values = list(actions_dict.values())
        max_q, min_q = max(q_values), min(q_values)
        critical_states.append({
            "State": state,
            "Importance": max_q - min_q,
            "Best_Action": max(actions_dict, key=actions_dict.get),
            "Worst_Action": min(actions_dict, key=actions_dict.get),
            "Max_Q": max_q,
            "Min_Q": min_q,
        })

    critical_states.sort(key=lambda x: x["Importance"], reverse=True)
    top_states = critical_states[:top_k]

    print(f"\n{'=' * 50}")
    print(f"Top {top_k} Most Critical States (Policy-Level Summarization)")
    print(f"Soil Type: {target_soil}")
    print(f"{'=' * 50}")

    for i, item in enumerate(top_states, 1):
        state_vals = item["State"]
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


if __name__ == "__main__":
    target_model_path = os.path.join("..", "q_agent_sand_gmm_500_act_50_DT_GRANULARITY_new_state.pkl")
    extract_critical_states(target_model_path, top_k=3)
