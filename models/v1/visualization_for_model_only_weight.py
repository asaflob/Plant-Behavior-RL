import pickle
import matplotlib.pyplot as plt
import os
import numpy as np


def load_normalized_policy(filename):
    """
    טוען את המודל ומנרמל את המשקלים לסקאלה של 0 עד 1
    """
    if not os.path.exists(filename):
        return None, None

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    policy = data['optimal_policy']

    # חילוץ המשקלים והפעולות
    weights = []
    actions = []

    sorted_items = sorted(policy.items())  # מיון לפי משקל

    for state, action in sorted_items:
        weights.append(state[0])
        actions.append(action)

    # --- נרמול (Normalization) ---
    # הופכים את הגרמים ל-0 עד 100 אחוז
    w_min = min(weights)
    w_max = max(weights)

    # מונע חלוקה באפס אם יש רק נקודה אחת
    if w_max == w_min:
        normalized_weights = [0 for _ in weights]
    else:
        normalized_weights = [(w - w_min) / (w_max - w_min) * 100 for w in weights]

    return normalized_weights, actions


def plot_normalized_comparison():
    # 1. טעינת הנתונים
    norm_w_soil, act_soil = load_normalized_policy("q_learning_agent_soil.pkl")
    norm_w_sand, act_sand = load_normalized_policy("q_learning_agent_sand.pkl")

    if norm_w_soil is None or norm_w_sand is None:
        print("Error: Missing model files.")
        return

    # 2. ציור הגרף
    plt.figure(figsize=(10, 6))

    # אדמה - קו רציף
    plt.plot(norm_w_soil, act_soil,
             label='Soil Strategy', color='brown', linewidth=2.5)

    # חול - קו מקווקו (כדי לראות חפיפה)
    plt.plot(norm_w_sand, act_sand,
             label='Sand Strategy', color='orange', linewidth=2.5, linestyle='--')

    # 3. עיצוב
    plt.title('Strategy Comparison: Soil vs. Sand (Normalized Growth)', fontsize=15)
    plt.xlabel('Relative Growth Stage (0% = Seedling, 100% = Mature)', fontsize=12)
    plt.ylabel('Stomatal Opening (Action 0-9)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # אזורי צבע להבנה
    plt.axhspan(0, 2, color='red', alpha=0.1, label='Conservative')
    plt.axhspan(6, 9, color='green', alpha=0.1, label='Aggressive')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_normalized_comparison()