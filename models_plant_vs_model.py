import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import random
#q_agent_sand_w5_t2_h10_actions_10
#q_agent_soil_w5_t2_h10_actions_10


#q_agent_soil_w5_t2_h10_p100_actions_50
#q_agent_soil_w5_t2_h10_p100_actions_10
#q_agent_sand_w5_t2_h10_p100_actions_50
#q_agent_sand_w5_t2_h10_p100_actions_10

def normalize_real_action(val, min_val, max_val, num_actions):
    """ממיר את ה-dt האמיתי למספר פעולה בדיד (0-9 או 0-49)"""
    if val < min_val: val = min_val
    if val > max_val: val = max_val

    # נרמול ל-0 עד 1
    norm = (val - min_val) / (max_val - min_val)

    # המרה ל-Bin (למשל 0 עד 49)
    action = int(norm * num_actions)
    if action >= num_actions: action = num_actions - 1
    return action


def compare_real_vs_agent():
    # ==========================================
    # 1. הגדרות וטעינה
    # ==========================================
    # וודא שאתה טוען את המודל הנכון (עם 10 או 50 פעולות)
    model_path = "q_agent_sand_w5_t2_h10_p100_actions_10.pkl"  # <--- עדכן לשם הקובץ שיצרת
    data_path = os.path.join("data", "tomato_mdp_ready_with_temp_humidity.parquet")

    if not os.path.exists(model_path) or not os.path.exists(data_path):
        print("Error: Files not found.")
        return

    # טעינת המודל
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    policy = model_data['optimal_policy']
    gran = model_data['granularities']

    # אם לא שמרנו במודל את מספר הפעולות, ננחש לפי המקסימום בטבלה
    max_policy_action = max(policy.values())
    NUM_ACTIONS = max_policy_action + 1  # (אם המקסימום הוא 9, אז יש 10 פעולות)
    print(f"Model loaded. Granularities: {gran}, Actions detected: {NUM_ACTIONS}")

    # טעינת הדאטה
    df = pd.read_parquet(data_path)
    # סינון לפי סוג האדמה של המודל
    target_soil = model_data.get('soil_type', 'soil')
    df = df[df['soil_type'].astype(str).str.strip() == target_soil]

    # חישוב גבולות גלובליים (כדי לתרגם את ה-dt לפעולה)
    global_min_dt = df['dt'].min()
    global_max_dt = df['dt'].max()

    # ==========================================
    # 2. בחירת ניסוי (Experiment) להשוואה
    # ==========================================
    # נבחר את הצמח עם הכי הרבה ימים, שיהיה מעניין
    plant_counts = df['unique_id'].value_counts()
    best_plant_id = plant_counts.idxmax()

    print(f"Analyzing Plant ID: {best_plant_id} ({plant_counts.max()} days)")

    plant_data = df[df['unique_id'] == best_plant_id].sort_values('date')

    # ==========================================
    # 3. הרצת הסימולציה יום אחרי יום
    # ==========================================
    days = []
    real_actions = []
    agent_actions = []
    temps = []

    misses = 0  # כמה פעמים לסוכן לא הייתה תשובה

    for _, row in plant_data.iterrows():
        # א. נתונים גולמיים
        w_raw = row['start_weight']
        t_raw = row['avg_temp']
        h_raw = row['avg_humidity']
        dt_raw = row['dt']

        # ב. חישוב הפעולה שהצמח האמיתי עשה
        real_act = normalize_real_action(dt_raw, global_min_dt, global_max_dt, NUM_ACTIONS)

        # ג. הכנת המצב לסוכן (עיגול לפי הגרנולריות של המודל)
        w_grid = round(w_raw / gran['weight']) * gran['weight']
        t_grid = round(t_raw / gran['temp']) * gran['temp']
        h_grid = round(h_raw / gran['humid']) * gran['humid']

        state = (w_grid, t_grid, h_grid)

        # ד. שאלת הסוכן
        if state in policy:
            agent_act = policy[state]
        else:
            # אם המצב לא מוכר, נסמן כ-NaN (כדי שיופיע כחור בגרף)
            agent_act = None
            misses += 1

        days.append(row['day_num'])
        real_actions.append(real_act)
        agent_actions.append(agent_act)
        temps.append(t_raw)

    print(f"Simulation done. Agent didn't know what to do in {misses}/{len(days)} days.")

    # ==========================================
    # 4. ציור הגרף
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # ציר Y שמאלי - הפעולות
    ax1.set_xlabel('Day in Experiment')
    ax1.set_ylabel('Stomatal Opening (Action Level)', color='black')

    # קו כחול - צמח אמיתי
    line1, = ax1.plot(days, real_actions, label='Real Plant Behavior', color='blue', linewidth=2, marker='o', alpha=0.6)

    # קו אדום - סוכן
    line2, = ax1.plot(days, agent_actions, label='Agent Recommendation', color='red', linewidth=2, linestyle='--',
                      marker='x')

    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim(-0.5, NUM_ACTIONS + 0.5)
    ax1.grid(True, alpha=0.3)

    # (אופציונלי) ציר Y ימני - טמפרטורה, כדי להבין הקשר
    ax2 = ax1.twinx()
    ax2.set_ylabel('Temperature (°C)', color='gray')
    line3, = ax2.plot(days, temps, label='Temperature', color='gray', linestyle=':', alpha=0.5)
    ax2.tick_params(axis='y', labelcolor='gray')

    # כותרת ומקרא
    plt.title(f'Real Plant vs. Agent Strategy\n(Plant ID: {best_plant_id})', fontsize=16)

    # איחוד מקרא (Legend)
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.tight_layout()
    plt.show()

#################################

def get_real_action_from_dt(dt_val, min_dt, max_dt, num_actions):
    """ממיר את ה-dt האמיתי למספר פעולה בדיד (0-9)"""
    if pd.isna(dt_val): return None

    # נרמול ל-0 עד 1
    if max_dt == min_dt:
        normalized = 0
    else:
        normalized = (dt_val - min_dt) / (max_dt - min_dt)

    normalized = max(0.0, min(1.0, normalized))
    action = int(normalized * num_actions)
    if action >= num_actions: action = num_actions - 1
    return action


def analyze_experiment_prediction():
    # ==========================================
    # 1. הגדרות וטעינה
    # ==========================================
    model_file = "q_agent_soil_w5_t2_h10_p100_actions_50.pkl"
    data_file = os.path.join("data", "tomato_mdp_final_filtered.parquet")

    if not os.path.exists(model_file):
        print(f"Error: Model file '{model_file}' not found.")
        return
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found.")
        return

    # טעינת המודל
    print(f"Loading Agent: {model_file}...")
    with open(model_file, 'rb') as f:
        agent_data = pickle.load(f)

    policy = agent_data['optimal_policy']
    gran = agent_data['granularities']

    max_act_in_policy = max(policy.values())
    NUM_ACTIONS = max_act_in_policy + 1
    print(f"Agent Config -> Granularities: {gran}, Actions: {NUM_ACTIONS}")

    # טעינת הדאטה
    print("Loading Data...")
    df = pd.read_parquet(data_file)
    target_soil = agent_data.get('soil_type', 'soil')
    df = df[df['soil_type'].astype(str).str.strip() == target_soil]

    GLOBAL_MIN_DT = df['dt'].min()
    GLOBAL_MAX_DT = df['dt'].max()

    # ==========================================
    # 2. בחירת צמח (ניסוי) לבדיקה
    # ==========================================
    # כאן אפשר להחליף לבחירה רנדומלית אם רוצים
    best_plant = df['unique_id'].value_counts().idxmax()
    print(f"\nAnalyzing Prediction for Plant ID: {best_plant}")

    plant_df = df[df['unique_id'] == best_plant].sort_values('day_num')
    # import random
    #
    # # סופרים כמה ימים יש לכל צמח
    # plant_counts = df['unique_id'].value_counts()
    #
    # # מסננים רק צמחים שיש להם לפחות 15 יום (כדי לא ליפול על ניסוי שהופסק באמצע)
    # valid_plants = plant_counts[plant_counts > 15].index.tolist()
    #
    # if not valid_plants:
    #     print("No plants with enough data found.")
    #     return
    #
    # # בוחרים אחד רנדומלי מתוך הרשימה הטובה
    # best_plant = random.choice(valid_plants)
    #
    # print(f"\nAnalyzing RANDOM Plant ID: {best_plant} (Total days: {plant_counts[best_plant]})")
    #
    # plant_df = df[df['unique_id'] == best_plant].sort_values('day_num')

    # ==========================================
    # 3. הריצה: יום אחרי יום
    # ==========================================
    results = []
    correct_predictions = 0
    total_known_states = 0

    for _, row in plant_df.iterrows():
        w_real = row['start_weight']
        t_real = row['avg_temp']
        h_real = row['avg_humidity']
        dt_real = row['dt']

        real_action = get_real_action_from_dt(dt_real, GLOBAL_MIN_DT, GLOBAL_MAX_DT, NUM_ACTIONS)

        w_grid = round(w_real / gran['weight']) * gran['weight']
        t_grid = round(t_raw / gran['temp']) * gran['temp'] if 't_raw' in locals() else round(t_real / gran['temp']) * \
                                                                                        gran['temp']
        h_grid = round(h_real / gran['humid']) * gran['humid']

        state = (w_grid, t_grid, h_grid)

        if state in policy:
            agent_action = policy[state]
            status = "Hit"
            total_known_states += 1
            if agent_action == real_action:
                correct_predictions += 1
        else:
            agent_action = np.nan
            status = "Unknown"

        results.append({
            'Day': row['day_num'],
            'Real_Action': real_action,
            'Agent_Action': agent_action,
            'Diff': abs(real_action - agent_action) if not pd.isna(agent_action) else None
        })

    res_df = pd.DataFrame(results)

    # ==========================================
    # 4. חישוב סטטיסטיקה להצגה
    # ==========================================
    valid_rows = res_df.dropna(subset=['Agent_Action'])

    if len(valid_rows) == 0:
        print("CRITICAL: The Agent recognized NONE of the states.")
        return

    accuracy = correct_predictions / total_known_states
    mae = valid_rows['Diff'].mean()
    recognition_rate = len(valid_rows) / len(res_df)

    # יצירת הטקסט לתיבה בגרף
    stats_text = (
        f"Plant ID: {best_plant}\n"
        f"Total Days: {len(res_df)}\n"
        f"Recognized States: {recognition_rate:.1%}\n"
        f"Exact Accuracy: {accuracy:.1%}\n"
        f"Mean Error (MAE): {mae:.2f}"
    )

    print(f"\n=== RESULTS ===\n{stats_text}")

    # ==========================================
    # 5. ויזואליזציה מעודכנת
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})

    # גרף עליון
    ax1.plot(res_df['Day'], res_df['Real_Action'], label='Real Plant (Observed)',
             color='blue', marker='o', alpha=0.6, linewidth=2)
    ax1.plot(res_df['Day'], res_df['Agent_Action'], label='Agent Policy (Predicted)',
             color='red', marker='x', linestyle='--', linewidth=2)

    ax1.set_title(f'Generalization Test: Real Plant vs. Agent Strategy', fontsize=16)
    ax1.set_ylabel(f'Action Level (0-{NUM_ACTIONS - 1})', fontsize=12)
    ax1.set_ylim(-0.5, NUM_ACTIONS - 0.5)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # === הוספת תיבת הטקסט כאן ===
    # transform=ax1.transAxes אומר שהמיקום (0.02, 0.05) הוא יחסי לגודל הגרף (0 עד 1)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ax1.text(0.25, 0.95, stats_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=props)

    # גרף תחתון (שגיאות)
    ax2.bar(valid_rows['Day'], valid_rows['Diff'], color='purple', alpha=0.7)
    ax2.set_title('Prediction Error per Day (0 = Perfect Match)', fontsize=14)
    ax2.set_xlabel('Day in Experiment')
    ax2.set_ylabel('Absolute Difference')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

###########################

def analyze_experiment_prediction_final():
    # ==========================================
    # 1. הגדרות וטעינה
    # ==========================================
    # שים לב: וודא שזה השם המדויק של הקובץ שלך
    model_file = "q_agent_sand_w5_t2_h10_p100_actions_10.pkl"

    # שימוש בקובץ הדאטה החדש והמסונן
    data_file = os.path.join("data", "tomato_mdp_final_filtered.parquet")

    if not os.path.exists(model_file):
        print(f"Error: Model file '{model_file}' not found.")
        return
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found.")
        return

    # טעינת המודל
    print(f"Loading Agent: {model_file}...")
    with open(model_file, 'rb') as f:
        agent_data = pickle.load(f)

    policy = agent_data['optimal_policy']
    gran = agent_data['granularities']

    # בדיקה מהירה: האם המודל כולל PAR?
    has_par = 'par' in gran
    print(f"Agent Config -> Granularities: {gran}, Has PAR: {has_par}")

    max_act_in_policy = max(policy.values())
    NUM_ACTIONS = max_act_in_policy + 1

    # טעינת הדאטה
    print("Loading Data...")
    df = pd.read_parquet(data_file)
    target_soil = agent_data.get('soil_type', 'sand')  # ברירת מחדל sand
    df = df[df['soil_type'].astype(str).str.strip() == target_soil]

    GLOBAL_MIN_DT = df['dt'].min()
    GLOBAL_MAX_DT = df['dt'].max()

    # ==========================================
    # 2. בחירת צמח (ניסוי) לבדיקה
    # ==========================================
    plant_counts = df['unique_id'].value_counts()

    # מסננים רק צמחים שיש להם לפחות 10 ימים
    valid_plants = plant_counts[plant_counts >= 10].index.tolist()

    if not valid_plants:
        print("No plants with enough data (>10 days) found.")
        return
    #
    # # בחירה רנדומלית
    # chosen_plant_id = random.choice(valid_plants)
    # total_days = plant_counts[chosen_plant_id]
    #
    # print(f"\nAnalyzing Prediction for RANDOM Plant ID: {chosen_plant_id} (Total days: {total_days})")
    #
    # plant_df = df[df['unique_id'] == chosen_plant_id].sort_values('day_num')

    chosen_plant_id = df['unique_id'].value_counts().idxmax()
    print(f"\nAnalyzing Prediction for Plant ID: {chosen_plant_id}")

    plant_df = df[df['unique_id'] == chosen_plant_id].sort_values('day_num')


    # ==========================================
    # 3. הריצה: יום אחרי יום
    # ==========================================
    results = []
    correct_predictions = 0
    total_known_states = 0

    for _, row in plant_df.iterrows():
        # א. נתונים גולמיים
        w_real = row['start_weight']
        t_real = row['avg_temp']
        h_real = row['avg_humidity']
        dt_real = row['dt']

        # שליפת נתון ה-PAR
        p_real = row.get('avg_par', 0)

        # ב. המרה לפעולה אמיתית
        real_action = get_real_action_from_dt(dt_real, GLOBAL_MIN_DT, GLOBAL_MAX_DT, NUM_ACTIONS)

        # ג. יצירת ה-State (דיסקרטיזציה)
        w_grid = round(w_real / gran['weight']) * gran['weight']
        t_grid = round(t_real / gran['temp']) * gran['temp']
        h_grid = round(h_real / gran['humid']) * gran['humid']

        # בניית ה-State בהתאם למה שהמודל מכיר
        if has_par:
            p_grid = round(p_real / gran['par']) * gran['par']
            state = (w_grid, t_grid, h_grid, p_grid)
        else:
            state = (w_grid, t_grid, h_grid)

        # ד. בדיקה מול המדיניות
        if state in policy:
            agent_action = policy[state]
            total_known_states += 1
            if agent_action == real_action:
                correct_predictions += 1
        else:
            agent_action = np.nan

        results.append({
            'Day': row['day_num'],
            'Real_Action': real_action,
            'Agent_Action': agent_action,
            'Diff': abs(real_action - agent_action) if not pd.isna(agent_action) else None
        })

    res_df = pd.DataFrame(results)

    # ==========================================
    # 4. ויזואליזציה וסטטיסטיקה
    # ==========================================
    valid_rows = res_df.dropna(subset=['Agent_Action'])

    if len(valid_rows) == 0:
        print("CRITICAL: The Agent recognized NONE of the states. Check Granularity match!")
        return

    # --- חישובים סטטיסטיים ---
    accuracy = correct_predictions / total_known_states
    mae = valid_rows['Diff'].mean()  # Mean Absolute Error
    mse = (valid_rows['Diff'] ** 2).mean()  # <--- הוספנו: Mean Squared Error (Loss)
    recognition_rate = len(valid_rows) / len(res_df)

    # עדכון הטקסט בתיבה
    stats_text = (
        f"Plant ID: {chosen_plant_id}\n"
        f"Total Days: {len(res_df)}\n"
        f"Recognized States: {recognition_rate:.1%}\n"
        f"Exact Accuracy: {accuracy:.1%}\n"
        f"MAE (Avg Error): {mae:.2f}\n"
        f"MSE (Loss): {mse:.2f}"  # <--- הוספנו לתצוגה
    )

    print(f"\n=== RESULTS ===\n{stats_text}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})

    # גרף עליון
    ax1.plot(res_df['Day'], res_df['Real_Action'], label='Real Plant (Observed)',
             color='blue', marker='o', alpha=0.6, linewidth=2)
    ax1.plot(res_df['Day'], res_df['Agent_Action'], label='Agent Policy (Predicted)',
             color='red', marker='x', linestyle='--', linewidth=2)

    ax1.set_title(f'Generalization Test: Real Plant vs. Agent Strategy', fontsize=16)
    ax1.set_ylabel(f'Action Level (0-{NUM_ACTIONS - 1})', fontsize=12)
    ax1.set_ylim(-0.5, NUM_ACTIONS - 0.5)

    # --- תיקון מיקום ה-Legend והטקסט ---
    # Legend בצד ימין למעלה
    ax1.legend(loc='upper right')

    ax1.grid(True, alpha=0.3)

    # תיבת טקסט בצד שמאל למעלה (0.02, 0.95)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=props)

    # גרף תחתון (שגיאות)
    ax2.bar(valid_rows['Day'], valid_rows['Diff'], color='purple', alpha=0.7)
    ax2.set_title('Prediction Error per Day', fontsize=14)
    ax2.set_xlabel('Day in Experiment')
    ax2.set_ylabel('Diff (Abs Error)')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_experiment_prediction_final()

# def analyze_experiment_prediction_final():
#     # ==========================================
#     # 1. הגדרות וטעינה
#     # ==========================================
#     # שים לב: וודא שזה השם המדויק שנוצר לך באימון האחרון
#     # (שיניתי ל-sand כי בקוד האימון הקודם הגדרת target_soil='sand')
#     model_file = "q_agent_sand_w5_t2_h10_p100_actions_10.pkl"
#
#     # שימוש בקובץ הדאטה החדש והמסונן
#     data_file = os.path.join("data", "tomato_mdp_final_filtered.parquet")
#
#     if not os.path.exists(model_file):
#         print(f"Error: Model file '{model_file}' not found.")
#         return
#     if not os.path.exists(data_file):
#         print(f"Error: Data file '{data_file}' not found.")
#         return
#
#     # טעינת המודל
#     print(f"Loading Agent: {model_file}...")
#     with open(model_file, 'rb') as f:
#         agent_data = pickle.load(f)
#
#     policy = agent_data['optimal_policy']
#     gran = agent_data['granularities']
#
#     # בדיקה מהירה: האם המודל כולל PAR?
#     has_par = 'par' in gran
#     print(f"Agent Config -> Granularities: {gran}, Has PAR: {has_par}")
#
#     max_act_in_policy = max(policy.values())
#     NUM_ACTIONS = max_act_in_policy + 1
#
#     # טעינת הדאטה
#     print("Loading Data...")
#     df = pd.read_parquet(data_file)
#     target_soil = agent_data.get('soil_type', 'sand')  # ברירת מחדל sand
#     df = df[df['soil_type'].astype(str).str.strip() == target_soil]
#
#     GLOBAL_MIN_DT = df['dt'].min()
#     GLOBAL_MAX_DT = df['dt'].max()
#
#     # ==========================================
#     # 2. בחירת צמח (ניסוי) לבדיקה
#     # ==========================================
#
#     plant_counts = df['unique_id'].value_counts()
#
#     # מסננים רק צמחים שיש להם לפחות 10 ימים (כדי לא ליפול על ניסוי קצר ומשעמם)
#     valid_plants = plant_counts[plant_counts >= 10].index.tolist()
#
#     if not valid_plants:
#         print("No plants with enough data (>10 days) found.")
#         return
#
#     # בחירה רנדומלית
#     chosen_plant_id = random.choice(valid_plants)
#     total_days = plant_counts[chosen_plant_id]
#
#     print(f"\nAnalyzing Prediction for RANDOM Plant ID: {chosen_plant_id} (Total days: {total_days})")
#
#     plant_df = df[df['unique_id'] == chosen_plant_id].sort_values('day_num')
#
#
#     # # נבחר את הצמח עם הכי הרבה ימים
#     # best_plant = df['unique_id'].value_counts().idxmax()
#     # print(f"\nAnalyzing Prediction for Plant ID: {best_plant}")
#     #
#     # plant_df = df[df['unique_id'] == best_plant].sort_values('day_num')
#
#
#
#     # ==========================================
#     # 3. הריצה: יום אחרי יום
#     # ==========================================
#     results = []
#     correct_predictions = 0
#     total_known_states = 0
#
#     for _, row in plant_df.iterrows():
#         # א. נתונים גולמיים
#         w_real = row['start_weight']
#         t_real = row['avg_temp']
#         h_real = row['avg_humidity']
#         dt_real = row['dt']
#
#         # שליפת נתון ה-PAR (אם קיים בדאטה)
#         p_real = row.get('avg_par', 0)
#
#         # ב. המרה לפעולה אמיתית
#         real_action = get_real_action_from_dt(dt_real, GLOBAL_MIN_DT, GLOBAL_MAX_DT, NUM_ACTIONS)
#
#         # ג. יצירת ה-State (דיסקרטיזציה)
#         w_grid = round(w_real / gran['weight']) * gran['weight']
#         t_grid = round(t_real / gran['temp']) * gran['temp']
#         h_grid = round(h_real / gran['humid']) * gran['humid']
#
#         # בניית ה-State בהתאם למה שהמודל מכיר
#         if has_par:
#             p_grid = round(p_real / gran['par']) * gran['par']
#             state = (w_grid, t_grid, h_grid, p_grid)  # 4 מימדים
#         else:
#             state = (w_grid, t_grid, h_grid)  # 3 מימדים (תמיכה לאחור)
#
#         # ד. בדיקה מול המדיניות
#         if state in policy:
#             agent_action = policy[state]
#             total_known_states += 1
#             if agent_action == real_action:
#                 correct_predictions += 1
#         else:
#             agent_action = np.nan
#
#         results.append({
#             'Day': row['day_num'],
#             'Real_Action': real_action,
#             'Agent_Action': agent_action,
#             'Diff': abs(real_action - agent_action) if not pd.isna(agent_action) else None
#         })
#
#     res_df = pd.DataFrame(results)
#
#     # ==========================================
#     # 4. ויזואליזציה (אותו דבר כמו קודם)
#     # ==========================================
#     valid_rows = res_df.dropna(subset=['Agent_Action'])
#
#     if len(valid_rows) == 0:
#         print("CRITICAL: The Agent recognized NONE of the states. Check Granularity match!")
#         return
#
#     accuracy = correct_predictions / total_known_states
#     mae = valid_rows['Diff'].mean()
#     recognition_rate = len(valid_rows) / len(res_df)
#
#     stats_text = (
#         f"Plant ID: {chosen_plant_id}\n"
#         f"Total Days: {len(res_df)}\n"
#         f"Recognized States: {recognition_rate:.1%}\n"
#         f"Exact Accuracy: {accuracy:.1%}\n"
#         f"Mean Error (MAE): {mae:.2f}"
#     )
#
#     print(f"\n=== RESULTS ===\n{stats_text}")
#
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
#
#     ax1.plot(res_df['Day'], res_df['Real_Action'], label='Real Plant (Observed)',
#              color='blue', marker='o', alpha=0.6, linewidth=2)
#     ax1.plot(res_df['Day'], res_df['Agent_Action'], label='Agent Policy (Predicted)',
#              color='red', marker='x', linestyle='--', linewidth=2)
#
#     ax1.set_title(f'Generalization Test: Real Plant vs. Agent Strategy', fontsize=16)
#     ax1.set_ylabel(f'Action Level (0-{NUM_ACTIONS - 1})', fontsize=12)
#     ax1.legend(loc='upper left')
#     ax1.grid(True, alpha=0.3)
#
#     props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#     ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, fontsize=11,
#              verticalalignment='top', bbox=props)
#
#     ax2.bar(valid_rows['Day'], valid_rows['Diff'], color='purple', alpha=0.7)
#     ax2.set_title('Prediction Error per Day', fontsize=14)
#     ax2.set_xlabel('Day in Experiment')
#     ax2.set_ylabel('Diff')
#     ax2.grid(axis='y', alpha=0.3)
#
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     analyze_experiment_prediction_final()