import pandas as pd
import numpy as np
import os
from MDP import PlantMDP
import pickle

def train_plant_agent():
    # ==============================================================================
    # 1. הכנת הנתונים (Data Preprocessing)
    # ==============================================================================
    input_file = os.path.join("data", "tomato_daily_summary_per_plant_with_dt.parquet")
    print(f"Loading data from {input_file}...")

    df = pd.read_parquet(input_file)

    # --- סינון לפי סוג קרקע ---
    target_soil = 'sand' #todo change by the soil we want
    print(f"Filtering data for soil type: '{target_soil}'...")
    if 'soil_type' in df.columns:
        df = df[df['soil_type'].astype(str).str.strip() == target_soil]

    if df.empty:
        print("CRITICAL: No data found.")
        return

    print(f"Rows remaining after filter: {len(df)}")
    df = df.dropna(subset=['dt', 'start_weight', 'end_weight'])

    # --- א. יצירת ה-Reward ---
    df['daily_growth'] = df['end_weight'] - df['start_weight']

    # --- ב. יצירת ה-Action ---
    min_dt = df['dt'].min()
    max_dt = df['dt'].max()
    df['stomatal_opening'] = (df['dt'] - min_dt) / (max_dt - min_dt)

    NUM_ACTIONS = 10
    df['action_discrete'] = pd.cut(df['stomatal_opening'], bins=NUM_ACTIONS, labels=False)

    # ============================================================
    # === תיקון קריטי: עיגול המשקלים לרשת (Grid) של 50 גרם ===
    # ============================================================
    GRANULARITY = 50

    # הפעולה: מחלקים ב-50, מעגלים למספר השלם הקרוב, ומכפילים חזרה ב-50
    # דוגמה: 2349.15 -> 46.98 -> 47.0 -> 2350.0
    df['start_weight'] = (df['start_weight'] / GRANULARITY).round() * GRANULARITY
    df['end_weight'] = (df['end_weight'] / GRANULARITY).round() * GRANULARITY

    # ============================================================

    # שמירה לקובץ זמני
    temp_file = "temp_data_for_training.parquet"
    df.to_parquet(temp_file)

    print(f"Data ready for {target_soil}. Weights rounded to nearest {GRANULARITY}g.")

    # ==============================================================================
    # 2. בניית המודל (MDP Construction)
    # ==============================================================================
    print("Building MDP from data...")

    state_config = {
        'start_weight': {
            'bounds': (df['start_weight'].min(), df['start_weight'].max()),
            'granularity': GRANULARITY
        }
    }

    mdp_model = PlantMDP(
        data_path=temp_file,
        state_map=state_config,
        action_col='action_discrete'
    )
    mdp_model.process_data()

    print(f"MDP Built! Total states explored: {len(mdp_model.transitions)}")

    # ==============================================================================
    # 3. הגדרת הסביבה
    # ==============================================================================
    def get_env_step(state, action):
        # טיפול במצבים לא מוכרים (עכשיו זה יקרה פחות בזכות העיגול)
        if state not in mdp_model.transitions or action not in mdp_model.transitions[state]:
            return state, -10

        next_states_dict = mdp_model.transitions[state][action]
        candidates = list(next_states_dict.keys())
        counts = list(next_states_dict.values())
        probs = np.array(counts) / sum(counts)

        next_state_idx = np.random.choice(len(candidates), p=probs)
        next_state = candidates[next_state_idx]

        reward = next_state[0] - state[0]
        return next_state, reward

    # ==============================================================================
    # 4. הרצת Q-Learning
    # ==============================================================================
    print(f"\nStarting Q-Learning Training for {target_soil}...")

    ACTIONS = list(range(NUM_ACTIONS))

    # אתחול הטבלה - עכשיו המפתחות יתאימו למה שמגיע מה-MDP
    Q_table = {s: {a: 0.0 for a in ACTIONS} for s in mdp_model.states}

    ALPHA = 0.1
    GAMMA = 0.95
    EPSILON = 1.0
    EPSILON_MIN = 0.01
    DECAY = 0.9995
    EPISODES = 10000

    for episode in range(EPISODES):
        # בחירת מצב התחלה אקראי מתוך המצבים הקיימים במודל
        # סינון למשקלים נמוכים כדי לדמות שתיל צעיר
        possible_starts = [s for s in mdp_model.states if s[0] < 250]

        # אם אין נתונים למשקלים נמוכים, קח כל מצב התחלתי שקיים בנתונים
        if not possible_starts:
            possible_starts = list(mdp_model.transitions.keys())

        current_state = possible_starts[np.random.choice(len(possible_starts))]

        for _ in range(60):
            if np.random.random() < EPSILON:
                action = np.random.choice(ACTIONS)
            else:
                action = max(Q_table[current_state], key=Q_table[current_state].get)

            next_state, reward = get_env_step(current_state, action)

            # בדיקה שמצב העתיד קיים בטבלה (למקרה של חריגה קטנה)
            max_future_q = max(Q_table[next_state].values()) if next_state in Q_table else 0

            current_q = Q_table[current_state][action]
            new_q = current_q + ALPHA * (reward + GAMMA * max_future_q - current_q)
            Q_table[current_state][action] = new_q

            current_state = next_state

        if EPSILON > EPSILON_MIN:
            EPSILON *= DECAY

        if episode % 1000 == 0:
            print(f"Episode {episode}/{EPISODES} | Epsilon: {EPSILON:.4f}")

    print("Training Finished.")

    # ==============================================================================
    # 5. הצגת התוצאות
    # ==============================================================================
    print(f"\n=== OPTIMAL POLICY ({target_soil.upper()}) ===")
    print("Action 0 = Closed Stomata (Min Transpiration)")
    print("Action 9 = Open Stomata (Max Transpiration)")
    print("-" * 60)
    print(f"{'Plant Weight (g)':<20} | {'Best Action (0-9)':<20} | {'Expected Value':<15}")
    print("-" * 60)

    sorted_states = sorted(mdp_model.states, key=lambda x: x[0])

    for state in sorted_states:
        # בדיקה שהמצב קיים בטבלה
        if state in Q_table:
            best_action = max(Q_table[state], key=Q_table[state].get)
            max_val = Q_table[state][best_action]

            if max_val != 0:
                print(f"{state[0]:<20.1f} | {best_action:<20} | {max_val:<15.2f}")

    if os.path.exists(temp_file):
        os.remove(temp_file)

    # ==============================================================================
    # 6. שמירת המודל (Saving the Agent)
    # ==============================================================================
    model_filename = f"q_learning_agent_{target_soil}.pkl"
    print(f"\nSaving model to {model_filename}...")

    model_data = {
        "q_table": Q_table,
        "soil_type": target_soil,
        "granularity": 50,  # חשוב לדעת איך עיגלנו
        "optimal_policy": {s: max(Q_table[s], key=Q_table[s].get) for s in mdp_model.states}
    }

    with open(model_filename, "wb") as f:
        pickle.dump(model_data, f)

    print("Model saved successfully!")

if __name__ == "__main__":
    train_plant_agent()
