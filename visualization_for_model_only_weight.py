# import pickle
# import matplotlib.pyplot as plt
# import os
# import numpy as np
# from collections import defaultdict
#
#
# def load_aggregated_policy(filename):
#     """
#     טוען את המודל, ומחשב את 'הפעולה הממוצעת' לכל משקל
#     (כדי לאחד את כל מצבי הטמפרטורה והלחות לקו אחד ברור)
#     """
#     if not os.path.exists(filename):
#         print(f"File not found: {filename}")
#         return None, None
#
#     with open(filename, 'rb') as f:
#         data = pickle.load(f)
#
#     policy = data['optimal_policy']
#
#     # 1. קיבוץ הפעולות לפי משקל
#     # המפתח הוא המשקל, הערך הוא רשימה של כל הפעולות שנבחרו במשקל הזה (בטמפרטורות שונות)
#     actions_by_weight = defaultdict(list)
#
#     for state, action in policy.items():
#         weight = state[0]  # אנחנו יודעים שהמשקל הוא האיבר הראשון
#         actions_by_weight[weight].append(action)
#
#     # 2. חישוב ממוצע ומיון
#     sorted_weights = sorted(actions_by_weight.keys())
#
#     avg_actions = []
#     weights = []
#
#     for w in sorted_weights:
#         # הממוצע מספר לנו על "הנטייה הכללית" של הצמח במשקל הזה
#         mean_action = np.mean(actions_by_weight[w])
#         weights.append(w)
#         avg_actions.append(mean_action)
#
#     # --- נרמול (Normalization) ---
#     w_min = min(weights)
#     w_max = max(weights)
#
#     if w_max == w_min:
#         normalized_weights = [0 for _ in weights]
#     else:
#         normalized_weights = [(w - w_min) / (w_max - w_min) * 100 for w in weights]
#
#     return normalized_weights, avg_actions
#
#
# def plot_normalized_comparison():
#     # שים לב לשמות הקבצים - עדכן אותם לשמות שיצרת בריצה האחרונה
#     file_soil = "q_agent_soil_w10_t2_h10_50.pkl"
#     file_sand = "q_agent_sand_w10_t2_h10_50.pkl"  # אם הרצת גם חול
#
#     # טעינת הנתונים הממוצעים
#     norm_w_soil, act_soil = load_aggregated_policy(file_soil)
#     norm_w_sand, act_sand = load_aggregated_policy(file_sand)  # יחזיר None אם לא קיים
#
#     plt.figure(figsize=(10, 6))
#
#     # אדמה
#     if norm_w_soil:
#         plt.plot(norm_w_soil, act_soil,
#                  label='Soil Strategy (Avg)', color='brown', linewidth=3)
#
#     # חול
#     if norm_w_sand:
#         plt.plot(norm_w_sand, act_sand,
#                  label='Sand Strategy (Avg)', color='orange', linewidth=3, linestyle='--')
#
#     plt.title('Strategy Comparison: Soil vs. Sand (Averaged over Weather)', fontsize=15)
#     plt.xlabel('Relative Growth Stage (0% = Seedling, 100% = Mature)', fontsize=12)
#     plt.ylabel('Avg Stomatal Opening (Action 0-9)', fontsize=12)
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#
#     plt.axhspan(0, 2, color='red', alpha=0.1, label='Conservative')
#     plt.axhspan(6, 9, color='green', alpha=0.1, label='Aggressive')
#
#     plt.tight_layout()
#     plt.show()
#
#
# if __name__ == "__main__":
#     plot_normalized_comparison()

import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict

# ==========================================
# הגדרות (Settings)
# ==========================================
# שנה את המספר הזה בהתאם למספר ה-Actions שהגדרת באימון!
TOTAL_ACTIONS = 50

def load_aggregated_policy(filename):
    """
    טוען את המודל, ומחשב את 'הפעולה הממוצעת' לכל משקל
    """
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return None, None

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    policy = data['optimal_policy']

    # 1. קיבוץ הפעולות לפי משקל
    actions_by_weight = defaultdict(list)

    for state, action in policy.items():
        weight = state[0]  # משקל הוא האיבר הראשון
        actions_by_weight[weight].append(action)

    # 2. חישוב ממוצע ומיון
    sorted_weights = sorted(actions_by_weight.keys())

    avg_actions = []
    weights = []

    for w in sorted_weights:
        mean_action = np.mean(actions_by_weight[w])
        weights.append(w)
        avg_actions.append(mean_action)

    # --- נרמול (Normalization) ---
    w_min = min(weights)
    w_max = max(weights)

    if w_max == w_min:
        normalized_weights = [0 for _ in weights]
    else:
        normalized_weights = [(w - w_min) / (w_max - w_min) * 100 for w in weights]

    return normalized_weights, avg_actions


def plot_normalized_comparison():
    # שמות הקבצים שלך (עם 50 פעולות)
    file_soil = "models/q_agent_soil_w10_t2_h10_50.pkl"
    file_sand = "models/q_agent_sand_w10_t2_h10_50.pkl"

    norm_w_soil, act_soil = load_aggregated_policy(file_soil)
    norm_w_sand, act_sand = load_aggregated_policy(file_sand)

    plt.figure(figsize=(12, 7))

    # אדמה
    if norm_w_soil:
        plt.plot(norm_w_soil, act_soil,
                 label='Soil Strategy (Avg)', color='brown', linewidth=3)

    # חול
    if norm_w_sand:
        plt.plot(norm_w_sand, act_sand,
                 label='Sand Strategy (Avg)', color='orange', linewidth=3, linestyle='--')

    # כותרות
    plt.title(f'Strategy Comparison: Soil vs. Sand ({TOTAL_ACTIONS} Actions)', fontsize=16)
    plt.xlabel('Relative Growth Stage (0% = Seedling, 100% = Mature)', fontsize=12)
    plt.ylabel(f'Avg Stomatal Opening (Action 0-{TOTAL_ACTIONS-1})', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')

    # --- חישוב דינמי של האזורים הצבעוניים ---
    # Conservative = 20% תחתונים
    conservative_limit = TOTAL_ACTIONS * 0.2

    # Aggressive = 30% עליונים (מ-70% ומעלה)
    aggressive_start = TOTAL_ACTIONS * 0.7

    plt.axhspan(0, conservative_limit, color='red', alpha=0.1, label='Conservative Zone')
    plt.axhspan(aggressive_start, TOTAL_ACTIONS, color='green', alpha=0.1, label='Aggressive Zone')

    # הגדרת גבולות ציר ה-Y שייראו טוב
    plt.ylim(-0.5, TOTAL_ACTIONS + 0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_normalized_comparison()