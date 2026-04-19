import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import random
import ast
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
    model_file = "q_agent_soil_w5_t2_h10_p100_actions_50_21-3-2026.pkl"
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
def analyze_experiment_prediction_final(model_file):
    # ==========================================
    # 1. הגדרות וטעינה
    # ==========================================
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
    NUM_ACTIONS = agent_data['num_actions']
    target_soil = agent_data.get('soil_type', 'sand')  # ברירת מחדל

    # בדיקה האם זה מודל מבוסס קלאסטרים (GMM/K-Means) או רשת ישנה (Granularities)
    is_cluster_model = 'clustering_method' in agent_data
    print(f"Agent Config -> Soil: {target_soil}, Actions: {NUM_ACTIONS}, Is Cluster Model: {is_cluster_model}")

    # טעינת הדאטה
    print("Loading Data...")
    df = pd.read_parquet(data_file)
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

    chosen_plant_id = df['unique_id'].value_counts().idxmax()
    print(f"\nAnalyzing Prediction for Plant ID: {chosen_plant_id}")

    plant_df = df[df['unique_id'] == chosen_plant_id].sort_values('day_num')

    # ==========================================
    # 3. הריצה: יום אחרי יום
    # ==========================================
    results = []
    correct_predictions = 0
    total_known_states = 0

    # שליפת כל המצבים המוכרים כדי למצוא את הקרוב ביותר (עבור מודל קלאסטרים)
    known_states = list(policy.keys())

    for _, row in plant_df.iterrows():
        # א. נתונים גולמיים
        w_real = row['start_weight']
        t_real = row['avg_temp']
        h_real = row['avg_humidity']
        dt_real = row['dt']
        p_real = row.get('avg_par', 0)

        # ב. המרה לפעולה אמיתית
        real_action = get_real_action_from_dt(dt_real, GLOBAL_MIN_DT, GLOBAL_MAX_DT, NUM_ACTIONS)

        # ג. יצירת ה-State
        if is_cluster_model:
            # במודל החדש ה-State הוא רק 3 ערכים: (טמפרטורה, לחות, אור)
            current_env = np.array([t_real, h_real, p_real])

            # מציאת קלאסטר האקלים הקרוב ביותר מתוך אלו שהסוכן למד
            # משתמשים במרחק אוקלידי למציאת המרכז הקרוב
            closest_state = min(known_states, key=lambda s: np.sum((np.array(s) - current_env) ** 2))
            state = closest_state
        else:
            # תמיכה לאחור במודל הרשת הישן
            gran = agent_data['granularities']
            w_grid = round(w_real / gran['weight']) * gran['weight']
            t_grid = round(t_real / gran['temp']) * gran['temp']
            h_grid = round(h_real / gran['humid']) * gran['humid']

            if 'par' in gran:
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
    mae = valid_rows['Diff'].mean()
    mse = (valid_rows['Diff'] ** 2).mean()
    recognition_rate = len(valid_rows) / len(res_df)

    # עדכון הטקסט בתיבה
    stats_text = (
        f"Plant ID: {chosen_plant_id}\n"
        f"Total Days: {len(res_df)}\n"
        f"Recognized States: {recognition_rate:.1%}\n"
        f"Exact Accuracy: {accuracy:.1%}\n"
        f"MAE (Avg Error): {mae:.2f}\n"
        f"MSE (Loss): {mse:.2f}"
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

    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

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

###########################
def analyze_experiment_prediction_final_evaporation_percentage(model_file):
    data_file = os.path.join("data", "tomato_mdp_final_with_pnw.parquet")

    if not os.path.exists(model_file):
        print(f"Error: Model file '{model_file}' not found.")
        return
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found.")
        return

    print(f"Loading Agent: {model_file}...")
    with open(model_file, 'rb') as f:
        agent_data = pickle.load(f)

    policy = agent_data['optimal_policy']
    NUM_ACTIONS = agent_data['num_actions']
    target_soil = agent_data.get('soil_type', 'sand')
    action_method = agent_data.get('action_method', 'DT_NORMALIZED')

    # המילון עכשיו נקי לחלוטין (Tuple -> Float)
    expected_rewards = agent_data.get('expected_rewards', {})

    is_cluster_model = 'clustering_method' in agent_data
    print(f"Agent Config -> Soil: {target_soil}, Actions: {NUM_ACTIONS}, Method: {action_method}")

    print("Loading Data...")
    df = pd.read_parquet(data_file)
    df = df[df['soil_type'].astype(str).str.strip() == target_soil]

    if action_method == 'DT_NORMALIZED' or action_method == 'DT_GRANULARITY':
        min_dt = df['dt'].min()
        max_dt = df['dt'].max()
        df['stomatal_opening'] = (df['dt'] - min_dt) / (max_dt - min_dt)
        df['real_action_discrete'] = pd.cut(df['stomatal_opening'], bins=NUM_ACTIONS, labels=False)

    elif action_method == 'EVAPORATION_PERCENTAGE':
        df['evap_pct'] = (df['dt'] / df['pnw']) * 100
        df['evap_pct'] = df['evap_pct'].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['evap_pct'])
        df['real_action_discrete'] = pd.cut(df['evap_pct'], bins=NUM_ACTIONS, labels=False)
    else:
        print("Error: Unknown action method.")
        return

    df = df.dropna(subset=['real_action_discrete'])

    plant_counts = df['unique_id'].value_counts()
    valid_plants = plant_counts[plant_counts >= 10].index.tolist()

    if not valid_plants:
        print("No plants with enough data (>10 days) found.")
        return

    chosen_plant_id = df['unique_id'].value_counts().idxmax()
    print(f"\nAnalyzing Prediction for Plant ID: {chosen_plant_id}")

    plant_df = df[df['unique_id'] == chosen_plant_id].sort_values('day_num').copy()
    plant_df['Real_Reward_Daily'] = plant_df['start_weight'].shift(-1) - plant_df['start_weight']
    plant_df['Real_Reward_Daily'] = plant_df['Real_Reward_Daily'].fillna(0)

    results = []
    correct_predictions = 0
    total_known_states = 0
    known_states = list(policy.keys())

    for _, row in plant_df.iterrows():
        w_real = row['start_weight']
        t_real = row['avg_temp']
        h_real = row['avg_humidity']
        p_real = row.get('avg_par', 0)

        real_reward_daily = row['Real_Reward_Daily']
        real_action = int(row['real_action_discrete'])

        if is_cluster_model:
            current_climate = np.array([t_real, h_real, p_real])
            unique_climates = list(set([(s[0], s[1], s[2]) for s in known_states]))
            closest_climate = min(unique_climates, key=lambda c: np.sum((np.array(c) - current_climate) ** 2))
            states_with_this_climate = [s for s in known_states if (s[0], s[1], s[2]) == closest_climate]
            closest_state = min(states_with_this_climate, key=lambda s: abs(s[3] - w_real))
            state = closest_state
        else:
            state = None

        if state in policy:
            agent_action = policy[state]
            total_known_states += 1
            if agent_action == real_action:
                correct_predictions += 1

            # --- שליפה טהורה וישירה מהמילון! ---
            search_key = (tuple(state), int(agent_action))
            agent_reward_daily = expected_rewards.get(search_key, 0)
        else:
            agent_action = np.nan
            agent_reward_daily = 0

        results.append({
            'Day': row['day_num'],
            'Real_Action': real_action,
            'Agent_Action': agent_action,
            'Diff': abs(real_action - agent_action) if not pd.isna(agent_action) else None,
            'Real_Reward': real_reward_daily,
            'Agent_Reward': agent_reward_daily
        })

    res_df = pd.DataFrame(results)
    res_df['Real_Accumulated'] = res_df['Real_Reward'].cumsum()
    res_df['Agent_Accumulated'] = res_df['Agent_Reward'].cumsum()

    valid_rows = res_df.dropna(subset=['Agent_Action'])

    if len(valid_rows) == 0:
        print("CRITICAL: The Agent recognized NONE of the states.")
        return

    accuracy = correct_predictions / total_known_states
    mae = valid_rows['Diff'].mean()
    mse = (valid_rows['Diff'] ** 2).mean()
    recognition_rate = len(valid_rows) / len(res_df)

    stats_text = (
        f"Plant ID: {chosen_plant_id}\n"
        f"Method: {action_method}\n"
        f"Total Days: {len(res_df)}\n"
        f"Recognized States: {recognition_rate:.1%}\n"
        f"Exact Accuracy: {accuracy:.1%}\n"
        f"MAE (Avg Error): {mae:.2f}\n"
        f"Total Plant Gain: {res_df['Real_Accumulated'].iloc[-1]:.1f}g\n"
        f"Total Agent Expected Gain: {res_df['Agent_Accumulated'].iloc[-1]:.1f}g"
    )

    print(f"\n=== RESULTS ===\n{stats_text}")

    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
    ax1.plot(res_df['Day'], res_df['Real_Action'], label='Real Plant (Observed)', color='blue', marker='o', alpha=0.6, linewidth=2)
    ax1.plot(res_df['Day'], res_df['Agent_Action'], label='Agent Policy (Predicted)', color='red', marker='x', linestyle='--', linewidth=2)
    ax1.set_title('Generalization Test: Real Plant vs. Agent Strategy', fontsize=16)
    ax1.set_ylabel(f'Action Level (0-{NUM_ACTIONS - 1})', fontsize=12)
    ax1.set_ylim(-0.5, NUM_ACTIONS - 0.5)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, fontsize=11, verticalalignment='top', bbox=props)

    ax2.bar(valid_rows['Day'], valid_rows['Diff'], color='purple', alpha=0.7)
    ax2.set_title('Prediction Error per Day', fontsize=14)
    ax2.set_ylabel('Diff (Abs Error)')
    ax2.grid(axis='y', alpha=0.3)
    fig1.tight_layout()

    fig2, ax3 = plt.subplots(figsize=(14, 6))
    ax3.plot(res_df['Day'], res_df['Real_Accumulated'], label='Real Plant Growth (Actual)', color='dodgerblue', marker='o', linewidth=3)
    ax3.plot(res_df['Day'], res_df['Agent_Accumulated'], label='Agent Policy Growth (Expected)', color='forestgreen', marker='^', linestyle='-', linewidth=3)
    ax3.set_title(f'Accumulated Reward (Weight Gain) over {len(res_df)} Days', fontsize=16)
    ax3.set_xlabel('Day in Experiment', fontsize=12)
    ax3.set_ylabel('Accumulated Weight Gain (grams)', fontsize=12)
    ax3.legend(loc='upper left', fontsize=12)
    ax3.grid(True, alpha=0.4)
    fig2.tight_layout()

    plt.show()


def analyze_experiment_prediction_final_evaporation_percentage_new(model_file):
    # ==========================================
    # 1. הגדרות וטעינה
    # ==========================================
    data_file = os.path.join("data", "tomato_mdp_final_with_pnw.parquet")
    raw_data_file = os.path.join("data", "tomato_raw_data_v2.parquet")

    if not os.path.exists(model_file):
        print(f"Error: Model file '{model_file}' not found.")
        return
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found.")
        return
    if not os.path.exists(raw_data_file):
        print(f"Error: Raw Data file '{raw_data_file}' not found.")
        return

    print(f"Loading Agent: {model_file}...")
    with open(model_file, 'rb') as f:
        agent_data = pickle.load(f)

    policy = agent_data['optimal_policy']
    NUM_ACTIONS = agent_data['num_actions']
    target_soil = agent_data.get('soil_type', 'sand')
    action_method = agent_data.get('action_method', 'DT_NORMALIZED')
    expected_rewards = agent_data.get('expected_rewards', {})
    is_cluster_model = 'clustering_method' in agent_data

    print(f"Agent Config -> Soil: {target_soil}, Actions: {NUM_ACTIONS}, Method: {action_method}")

    print("Loading Data...")
    df = pd.read_parquet(data_file)
    df = df[df['soil_type'].astype(str).str.strip() == target_soil]

    # ============================================================
    # --- הוספת עמודת exp_ID מתוך הקובץ המקורי (מיזוג חכם ובטוח) ---
    # ============================================================
    print("Ensuring exp_ID is in the data...")

    # נבדוק אם כבר קיים exp_ID או exp_id בקובץ הנוכחי
    if 'exp_ID' not in df.columns and 'exp_id' not in df.columns:
        print("Merging experiment data from processed data...")
        df_raw = pd.read_parquet(raw_data_file)

        # מחפש את השם המדויק בקובץ ה-RAW (מתגבר על בעיות אותיות גדולות/קטנות)
        raw_col_name = None
        if 'exp_ID' in df_raw.columns:
            raw_col_name = 'exp_ID'
        elif 'exp_id' in df_raw.columns:
            raw_col_name = 'exp_id'

        if raw_col_name is None:
            print("CRITICAL ERROR: Neither 'exp_ID' nor 'exp_id' found in the raw data file.")
            return

        df_raw = df_raw[['unique_id', raw_col_name]].drop_duplicates()
        df = df.merge(df_raw, on='unique_id', how='inner')

        # תמיד ננרמל את השם ל-exp_ID
        if raw_col_name == 'exp_id':
            df = df.rename(columns={'exp_id': 'exp_ID'})

    elif 'exp_id' in df.columns:
        # אם זה קיים באותיות קטנות, נשנה לגדולות
        df = df.rename(columns={'exp_id': 'exp_ID'})

    # ============================================================
    # === יצירת הפעולות האמיתיות ===
    # ============================================================
    if action_method == 'DT_NORMALIZED' or action_method == 'DT_GRANULARITY':
        min_dt = df['dt'].min()
        max_dt = df['dt'].max()
        df['stomatal_opening'] = (df['dt'] - min_dt) / (max_dt - min_dt)
        df['real_action_discrete'] = pd.cut(df['stomatal_opening'], bins=NUM_ACTIONS, labels=False)

    elif action_method == 'EVAPORATION_PERCENTAGE':
        df['evap_pct'] = (df['dt'] / df['pnw']) * 100
        df['evap_pct'] = df['evap_pct'].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['evap_pct'])
        df['real_action_discrete'] = pd.cut(df['evap_pct'], bins=NUM_ACTIONS, labels=False)
    else:
        print("Error: Unknown action method.")
        return

    df = df.dropna(subset=['real_action_discrete'])

    # ==========================================
    # 2. בחירת נציג אחד (צמח) מכל ניסוי (exp_ID)
    # ==========================================
    selected_plants = []

    for exp_id_val, group in df.groupby('exp_ID'):
        plant_counts = group['unique_id'].value_counts()
        valid_plants = plant_counts[plant_counts >= 10]

        if not valid_plants.empty:
            best_plant_for_exp = valid_plants.idxmax()
            selected_plants.append(best_plant_for_exp)

    if not selected_plants:
        print("No valid plants found across experiments.")
        return

    print(f"\nSelected {len(selected_plants)} representative plants (one from each valid experiment):")
    for p in selected_plants:
        print(f" - {p}")

    # ==========================================
    # 3. הריצה יום אחרי יום (על כל הצמחים שנבחרו)
    # ==========================================
    all_results = []
    correct_predictions = 0
    total_known_states = 0
    known_states = list(policy.keys())

    for pid in selected_plants:
        plant_df = df[df['unique_id'] == pid].sort_values('day_num').copy()
        plant_df['Real_Reward_Daily'] = plant_df['start_weight'].shift(-1) - plant_df['start_weight']
        plant_df['Real_Reward_Daily'] = plant_df['Real_Reward_Daily'].fillna(0)

        for _, row in plant_df.iterrows():
            w_real = row['start_weight']
            t_real = row['avg_temp']
            h_real = row['avg_humidity']
            p_real = row.get('avg_par', 0)

            real_reward_daily = row['Real_Reward_Daily']
            real_action = int(row['real_action_discrete'])

            if is_cluster_model:
                current_climate = np.array([t_real, h_real, p_real])
                unique_climates = list(set([(s[0], s[1], s[2]) for s in known_states]))
                closest_climate = min(unique_climates, key=lambda c: np.sum((np.array(c) - current_climate) ** 2))
                states_with_this_climate = [s for s in known_states if (s[0], s[1], s[2]) == closest_climate]
                closest_state = min(states_with_this_climate, key=lambda s: abs(s[3] - w_real))
                state = closest_state
            else:
                state = None

            if state in policy:
                agent_action = policy[state]
                total_known_states += 1
                if agent_action == real_action:
                    correct_predictions += 1

                search_key = (tuple(state), int(agent_action))
                agent_reward_daily = expected_rewards.get(search_key, 0)
            else:
                agent_action = np.nan
                agent_reward_daily = 0

            all_results.append({
                'Plant_ID': pid,
                'Day': row['day_num'],
                'Real_Action': real_action,
                'Agent_Action': agent_action,
                'Diff': abs(real_action - agent_action) if not pd.isna(agent_action) else None,
                'Real_Reward': real_reward_daily,
                'Agent_Reward': agent_reward_daily
            })

    # ==========================================
    # סכימה וממוצע לפי ימים
    # ==========================================
    full_res_df = pd.DataFrame(all_results)

    avg_res_df = full_res_df.groupby('Day').agg({
        'Real_Action': 'mean',
        'Agent_Action': 'mean',
        'Diff': 'mean',
        'Real_Reward': 'mean',
        'Agent_Reward': 'mean'
    }).reset_index()

    avg_res_df['Real_Accumulated'] = avg_res_df['Real_Reward'].cumsum()
    avg_res_df['Agent_Accumulated'] = avg_res_df['Agent_Reward'].cumsum()

    valid_rows = avg_res_df.dropna(subset=['Agent_Action'])

    if len(valid_rows) == 0:
        print("CRITICAL: The Agent recognized NONE of the states.")
        return

    accuracy = correct_predictions / total_known_states if total_known_states > 0 else 0
    mae = valid_rows['Diff'].mean()
    mse = (valid_rows['Diff'] ** 2).mean()

    stats_text = (
        f"Averaged across {len(selected_plants)} Experiments\n"
        f"Method: {action_method}\n"
        f"Total Avg Days: {len(avg_res_df)}\n"
        f"Overall Accuracy: {accuracy:.1%}\n"
        f"MAE (Avg Error): {mae:.2f}\n"
        f"Total Plant Avg Gain: {avg_res_df['Real_Accumulated'].iloc[-1]:.1f}g\n"
        f"Total Agent Avg Gain: {avg_res_df['Agent_Accumulated'].iloc[-1]:.1f}g"
    )

    print(f"\n=== AVERAGE RESULTS ===\n{stats_text}")

    # ==========================================
    # 4. ויזואליזציה של הממוצעים הנפרדת
    # ==========================================
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})

    ax1.plot(avg_res_df['Day'], avg_res_df['Real_Action'], label='Average Real Plant', color='blue', marker='o',
             alpha=0.6, linewidth=2)
    ax1.plot(avg_res_df['Day'], avg_res_df['Agent_Action'], label='Average Agent Policy', color='red', marker='x',
             linestyle='--', linewidth=2)
    ax1.set_title(f'Generalization Test: Averaged Over {len(selected_plants)} Plants/Experiments', fontsize=16)
    ax1.set_ylabel(f'Avg Action Level (0-{NUM_ACTIONS - 1})', fontsize=12)
    ax1.set_ylim(-0.5, NUM_ACTIONS - 0.5)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, fontsize=11, verticalalignment='top', bbox=props)

    ax2.bar(valid_rows['Day'], valid_rows['Diff'], color='purple', alpha=0.7)
    ax2.set_title('Average Prediction Error per Day', fontsize=14)
    ax2.set_ylabel('Avg Diff (Error)')
    ax2.grid(axis='y', alpha=0.3)
    fig1.tight_layout()

    fig2, ax3 = plt.subplots(figsize=(14, 6))
    ax3.plot(avg_res_df['Day'], avg_res_df['Real_Accumulated'], label='Avg Real Plant Growth', color='dodgerblue',
             marker='o', linewidth=3)
    ax3.plot(avg_res_df['Day'], avg_res_df['Agent_Accumulated'], label='Avg Agent Policy Growth', color='forestgreen',
             marker='^', linestyle='-', linewidth=3)
    ax3.set_title(f'Average Accumulated Reward over {len(avg_res_df)} Days', fontsize=16)
    ax3.set_xlabel('Day in Experiment', fontsize=12)
    ax3.set_ylabel('Average Accumulated Gain (grams)', fontsize=12)
    ax3.legend(loc='upper left', fontsize=12)
    ax3.grid(True, alpha=0.4)
    fig2.tight_layout()

    plt.show()


def analyze_experiment_prediction_final_evaporation_percentage_all_plants(model_file):
    # ==========================================
    # 1. הגדרות וטעינה
    # ==========================================
    data_file = os.path.join("data", "tomato_mdp_final_with_pnw.parquet")
    raw_data_file = os.path.join("data", "tomato_raw_data_v2.parquet")

    if not os.path.exists(model_file):
        print(f"Error: Model file '{model_file}' not found.")
        return
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found.")
        return
    if not os.path.exists(raw_data_file):
        print(f"Error: Raw Data file '{raw_data_file}' not found.")
        return

    print(f"Loading Agent: {model_file}...")
    with open(model_file, 'rb') as f:
        agent_data = pickle.load(f)

    policy = agent_data['optimal_policy']
    NUM_ACTIONS = agent_data['num_actions']
    target_soil = agent_data.get('soil_type', 'sand')
    action_method = agent_data.get('action_method', 'DT_NORMALIZED')
    expected_rewards = agent_data.get('expected_rewards', {})
    is_cluster_model = 'clustering_method' in agent_data

    # --- הגדרת החציון לפי סוג האדמה ---
    median_day_value = 17 if target_soil == 'sand' else 12

    print(f"Agent Config -> Soil: {target_soil}, Actions: {NUM_ACTIONS}, Method: {action_method}")

    print("Loading Data...")
    df = pd.read_parquet(data_file)
    df = df[df['soil_type'].astype(str).str.strip() == target_soil]

    # ============================================================
    # --- הוספת עמודת exp_ID מתוך הקובץ המקורי (מיזוג חכם ובטוח) ---
    # ============================================================
    print("Ensuring exp_ID is in the data...")

    if 'exp_ID' not in df.columns and 'exp_id' not in df.columns:
        print("Merging experiment data from processed data...")
        df_raw = pd.read_parquet(raw_data_file)

        raw_col_name = None
        if 'exp_ID' in df_raw.columns:
            raw_col_name = 'exp_ID'
        elif 'exp_id' in df_raw.columns:
            raw_col_name = 'exp_id'

        if raw_col_name is None:
            print("CRITICAL ERROR: Neither 'exp_ID' nor 'exp_id' found in the raw data file.")
            return

        df_raw = df_raw[['unique_id', raw_col_name]].drop_duplicates()
        df = df.merge(df_raw, on='unique_id', how='inner')

        if raw_col_name == 'exp_id':
            df = df.rename(columns={'exp_id': 'exp_ID'})

    elif 'exp_id' in df.columns:
        df = df.rename(columns={'exp_id': 'exp_ID'})

    # ============================================================
    # === יצירת הפעולות האמיתיות ===
    # ============================================================
    if action_method == 'DT_NORMALIZED' or action_method == 'DT_GRANULARITY':
        min_dt = df['dt'].min()
        max_dt = df['dt'].max()
        df['stomatal_opening'] = (df['dt'] - min_dt) / (max_dt - min_dt)
        df['real_action_discrete'] = pd.cut(df['stomatal_opening'], bins=NUM_ACTIONS, labels=False)

    elif action_method == 'EVAPORATION_PERCENTAGE':
        df['evap_pct'] = (df['dt'] / df['pnw']) * 100
        df['evap_pct'] = df['evap_pct'].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['evap_pct'])
        df['real_action_discrete'] = pd.cut(df['evap_pct'], bins=NUM_ACTIONS, labels=False)
    else:
        print("Error: Unknown action method.")
        return

    df = df.dropna(subset=['real_action_discrete'])

    # ==========================================
    # 2. בחירת כל הצמחים הרלוונטיים (מכל הניסויים)
    # ==========================================
    plant_counts = df['unique_id'].value_counts()
    selected_plants = plant_counts[plant_counts >= 10].index.tolist()

    if not selected_plants:
        print("No valid plants (with >= 10 days) found.")
        return

    print(f"\nSelected {len(selected_plants)} plants across all experiments.")

    # ==========================================
    # 3. הריצה יום אחרי יום (על כל הצמחים שנבחרו)
    # ==========================================
    all_results = []
    correct_predictions = 0
    total_known_states = 0
    known_states = list(policy.keys())

    for pid in selected_plants:
        plant_df = df[df['unique_id'] == pid].sort_values('day_num').copy()
        plant_df['Real_Reward_Daily'] = plant_df['start_weight'].shift(-1) - plant_df['start_weight']
        plant_df['Real_Reward_Daily'] = plant_df['Real_Reward_Daily'].fillna(0)

        for _, row in plant_df.iterrows():
            w_real = row['start_weight']
            t_real = row['avg_temp']
            h_real = row['avg_humidity']
            p_real = row.get('avg_par', 0)

            real_reward_daily = row['Real_Reward_Daily']
            real_action = int(row['real_action_discrete'])

            if is_cluster_model:
                current_climate = np.array([t_real, h_real, p_real])
                unique_climates = list(set([(s[0], s[1], s[2]) for s in known_states]))
                closest_climate = min(unique_climates, key=lambda c: np.sum((np.array(c) - current_climate) ** 2))
                states_with_this_climate = [s for s in known_states if (s[0], s[1], s[2]) == closest_climate]
                closest_state = min(states_with_this_climate, key=lambda s: abs(s[3] - w_real))
                state = closest_state
            else:
                state = None

            if state in policy:
                agent_action = policy[state]
                total_known_states += 1
                if agent_action == real_action:
                    correct_predictions += 1

                search_key = (tuple(state), int(agent_action))
                agent_reward_daily = expected_rewards.get(search_key, 0)
            else:
                agent_action = np.nan
                agent_reward_daily = 0

            all_results.append({
                'Plant_ID': pid,
                'Day': row['day_num'],
                'Real_Action': real_action,
                'Agent_Action': agent_action,
                'Diff': abs(real_action - agent_action) if not pd.isna(agent_action) else None,
                'Real_Reward': real_reward_daily,
                'Agent_Reward': agent_reward_daily
            })

    # ==========================================
    # סכימה וממוצע לפי ימים
    # ==========================================
    full_res_df = pd.DataFrame(all_results)

    avg_res_df = full_res_df.groupby('Day').agg({
        'Real_Action': 'mean',
        'Agent_Action': 'mean',
        'Diff': 'mean',
        'Real_Reward': 'mean',
        'Agent_Reward': 'mean'
    }).reset_index()

    avg_res_df['Real_Accumulated'] = avg_res_df['Real_Reward'].cumsum()
    avg_res_df['Agent_Accumulated'] = avg_res_df['Agent_Reward'].cumsum()

    valid_rows = avg_res_df.dropna(subset=['Agent_Action'])

    if len(valid_rows) == 0:
        print("CRITICAL: The Agent recognized NONE of the states.")
        return

    accuracy = correct_predictions / total_known_states if total_known_states > 0 else 0
    mae = valid_rows['Diff'].mean()
    mse = (valid_rows['Diff'] ** 2).mean()

    stats_text = (
        f"Averaged across {len(selected_plants)} Plants\n"
        f"Method: {action_method}\n"
        f"Total Avg Days: {len(avg_res_df)}\n"
        f"Overall Accuracy: {accuracy:.1%}\n"
        f"MAE (Avg Error): {mae:.2f}\n"
        f"Total Plant Avg Gain: {avg_res_df['Real_Accumulated'].iloc[-1]:.1f}g\n"
        f"Total Agent Avg Gain: {avg_res_df['Agent_Accumulated'].iloc[-1]:.1f}g"
    )

    print(f"\n=== AVERAGE RESULTS ===\n{stats_text}")

    # ==========================================
    # 4. ויזואליזציה של הממוצעים הנפרדת + קו חציון
    # ==========================================
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})

    # --- גרף 1: פעולות ---
    ax1.plot(avg_res_df['Day'], avg_res_df['Real_Action'], label='Average Real Plant', color='blue', marker='o',
             alpha=0.6, linewidth=2)
    ax1.plot(avg_res_df['Day'], avg_res_df['Agent_Action'], label='Average Agent Policy', color='red', marker='x',
             linestyle='--', linewidth=2)

    # הוספת קו חציון
    ax1.axvline(x=median_day_value, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Median Day')
    ax1.text(median_day_value + 0.5, NUM_ACTIONS * 0.9, f'Median: {median_day_value}', color='black', fontsize=10,
             fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax1.set_title(f'Generalization Test: Averaged Over {len(selected_plants)} Plants', fontsize=16)
    ax1.set_ylabel(f'Avg Action Level (0-{NUM_ACTIONS - 1})', fontsize=12)
    ax1.set_ylim(-0.5, NUM_ACTIONS - 0.5)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, fontsize=11, verticalalignment='top', bbox=props)

    # --- גרף 2: שגיאה ---
    ax2.bar(valid_rows['Day'], valid_rows['Diff'], color='purple', alpha=0.7)

    # הוספת קו חציון
    ax2.axvline(x=median_day_value, color='black', linestyle='--', linewidth=2, alpha=0.7)

    ax2.set_title('Average Prediction Error per Day', fontsize=14)
    ax2.set_ylabel('Avg Diff (Error)')
    ax2.grid(axis='y', alpha=0.3)
    fig1.tight_layout()

    # --- גרף 3: ריוורד מצטבר ---
    fig2, ax3 = plt.subplots(figsize=(14, 6))
    ax3.plot(avg_res_df['Day'], avg_res_df['Real_Accumulated'], label='Avg Real Plant Growth', color='dodgerblue',
             marker='o', linewidth=3)
    ax3.plot(avg_res_df['Day'], avg_res_df['Agent_Accumulated'], label='Avg Agent Policy Growth', color='forestgreen',
             marker='^', linestyle='-', linewidth=3)

    # הוספת קו חציון
    ax3.axvline(x=median_day_value, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Median Day')
    ax3.text(median_day_value + 0.5, ax3.get_ylim()[1] * 0.9, f'Median: {median_day_value}', color='black', fontsize=10,
             fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax3.set_title(f'Average Accumulated Reward over {len(avg_res_df)} Days', fontsize=16)
    ax3.set_xlabel('Day in Experiment', fontsize=12)
    ax3.set_ylabel('Average Accumulated Gain (grams)', fontsize=12)
    ax3.legend(loc='upper left', fontsize=12)
    ax3.grid(True, alpha=0.4)
    fig2.tight_layout()

    plt.show()




def check_state_action_coverage(model_file):
    if not os.path.exists(model_file):
        print(f"Error: Model file '{model_file}' not found.")
        return

    print(f"Loading Agent: {model_file}...")
    with open(model_file, 'rb') as f:
        agent_data = pickle.load(f)

    # שליפת הנתונים מתוך המודל
    num_states = agent_data.get('num_states', 700)
    num_actions = agent_data.get('num_actions', 50)
    expected_rewards = agent_data.get('expected_rewards', {})
    q_table = agent_data.get('q_table', {})

    # חישוב הצירופים (State X Action)
    total_theoretical_combinations = num_states * num_actions
    valid_combinations = len(expected_rewards)
    coverage_percentage = (valid_combinations / total_theoretical_combinations) * 100

    # חישוב המצבים הפעילים (כמה מצבים מתוך ה-700 ראו לפחות פעולה אחת)
    active_states = len([s for s, actions in q_table.items() if actions])
    states_coverage = (active_states / num_states) * 100 if num_states > 0 else 0

    print("\n" + "="*50)
    print("=== State-Action Coverage Analysis ===")
    print("="*50)
    print(f"Total States (Clusters X Weight): {num_states}")
    print(f"Total Actions (Stomatal/Evap): {num_actions}")
    print("-" * 50)
    print(f"Theoretical Max Combinations: {total_theoretical_combinations:,}")
    print(f"Actual VALID Combinations (with Reward): {valid_combinations:,}")
    print(f"Matrix Coverage (Density): {coverage_percentage:.2f}%")
    print("-" * 50)
    print(f"States with AT LEAST ONE valid action: {active_states} out of {num_states} ({states_coverage:.1f}%)")
    print("="*50 + "\n")
#q_agent_soil_kmeans_400_actions_50.pkl
#q_agent_sand_kmeans_1400_actions_50.pkl

#q_agent_soil_gmm_121_act_50_DT_GRANULARITY.pkl
#q_agent_sand_gmm_500_act_50_DT_GRANULARITY.pkl
#q_agent_soil_gmm_121_act_50_EVAPORATION_PERCENTAGE.pkl
#q_agent_sand_gmm_500_act_50_EVAPORATION_PERCENTAGE.pkl
#q_agent_sand_gmm_500_actions_50.pkl this is with 3 elements in the state
#q_agent_soil_gmm_121_actions_50.pkl this is with 3 elements in the state

#q_agent_sand_gmm_500_act_50_DT_GRANULARITY_new_state.pkl
#q_agent_sand_gmm_500_act_50_EVAPORATION_PERCENTAGE_new_state.pkl
#q_agent_sand_gmm_500_act_100_EVAPORATION_PERCENTAGE_new_state.pkl

#q_agent_soil_gmm_121_act_50_DT_GRANULARITY_new_state.pkl
#q_agent_soil_gmm_121_act_50_EVAPORATION_PERCENTAGE_new_state.pkl
#q_agent_soil_gmm_121_act_100_EVAPORATION_PERCENTAGE_new_state.pkl
if __name__ == "__main__":
    target_model = "q_agent_soil_gmm_121_act_100_EVAPORATION_PERCENTAGE_new_state.pkl"
    # check_state_action_coverage(target_model)
    analyze_experiment_prediction_final_evaporation_percentage_all_plants(target_model)
