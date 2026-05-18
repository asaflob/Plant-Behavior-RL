import pandas as pd
import numpy as np
import os
from MDP import PlantMDP
import pickle


def train_plant_agent():
    # ==============================================================================
    # 0. הגדרות (Configuration)
    # ==============================================================================
    # הגדרת הרזולוציה לכל מימד בנפרד
    GRANULARITY_WEIGHT = 5  # קפיצות של 5 גרם (שים לב: בקוד המקורי כתבת 50 בהערה אבל 5 בקוד)
    GRANULARITY_TEMP = 2  # קפיצות של 2 מעלות
    GRANULARITY_HUMID = 10  # קפיצות של 10 אחוז לחות
    GRANULARITY_PAR = 100  # <--- חדש: קפיצות של 100 יחידות אור (טווח 0-2000 בערך)

    NUM_ACTIONS = 10  # מספר דרגות פתיחת פיוניות

    # שם הקובץ הסופי שיצרנו ב-Pipeline
    input_file = os.path.join("data", "tomato_mdp_final_filtered.parquet")

    # ==============================================================================
    # 1. הכנת הנתונים (Data Preprocessing)
    # ==============================================================================
    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_parquet(input_file)
    except FileNotFoundError:
        print("Error: Input file not found. Run the data_execution pipeline first.")
        return

    # --- סינון לפי סוג קרקע ---
    target_soil = 'sand'  # ('soil' / 'sand')
    print(f"Filtering data for soil type: '{target_soil}'...")

    if 'soil_type' in df.columns:
        df = df[df['soil_type'].astype(str).str.strip() == target_soil]

    if df.empty:
        print("CRITICAL: No data found.")
        return

    print(f"Rows remaining after filter: {len(df)}")

    # ניקוי NaN מכל העמודות הרלוונטיות - כולל avg_par
    df = df.dropna(subset=['dt', 'start_weight', 'end_weight', 'avg_temp', 'avg_humidity', 'avg_par'])

    # --- א. יצירת ה-Action ---
    min_dt = df['dt'].min()
    max_dt = df['dt'].max()

    # הגנה מפני חילוק באפס (אם יש רק ערך אחד)
    # if max_dt == min_dt:
    #     df['stomatal_opening'] = 0
    # else:
    #     df['stomatal_opening'] = (df['dt'] - min_dt) / (max_dt - min_dt)
    #
    # df['action_discrete'] = pd.cut(df['stomatal_opening'], bins=NUM_ACTIONS, labels=False)
    df['stomatal_opening'] = (df['dt'] - min_dt) / (max_dt - min_dt)
    df['action_discrete'] = pd.cut(df['stomatal_opening'], bins=NUM_ACTIONS, labels=False)

    # ============================================================
    # === תהליך הדיסקרטיזציה (Discretization / Rounding) ===
    # ============================================================
    print("Rounding data to grid...")

    # 1. משקל
    df['start_weight'] = (df['start_weight'] / GRANULARITY_WEIGHT).round() * GRANULARITY_WEIGHT
    df['end_weight'] = (df['end_weight'] / GRANULARITY_WEIGHT).round() * GRANULARITY_WEIGHT

    # 2. טמפרטורה
    df['avg_temp'] = (df['avg_temp'] / GRANULARITY_TEMP).round() * GRANULARITY_TEMP

    # 3. לחות
    df['avg_humidity'] = (df['avg_humidity'] / GRANULARITY_HUMID).round() * GRANULARITY_HUMID

    # 4. קרינה (PAR)
    df['avg_par'] = (df['avg_par'] / GRANULARITY_PAR).round() * GRANULARITY_PAR

    # שמירה לקובץ זמני
    temp_file = "temp_data_for_training.parquet"
    df.to_parquet(temp_file)

    # ==============================================================================
    # 2. בניית המודל (MDP Construction)
    # ==============================================================================
    print("Building MDP from data...")

    # כאן אנחנו מגדירים את ה-State המלא! (משקל, טמפ, לחות, אור)
    # חשוב: start_weight חייב להישאר ראשון כדי שהלוגיקה בהמשך תעבוד
    state_config = {
        'start_weight': {
            'bounds': (df['start_weight'].min(), df['start_weight'].max()),
            'granularity': GRANULARITY_WEIGHT
        },
        'avg_temp': {
            'bounds': (df['avg_temp'].min(), df['avg_temp'].max()),
            'granularity': GRANULARITY_TEMP
        },
        'avg_humidity': {
            'bounds': (df['avg_humidity'].min(), df['avg_humidity'].max()),
            'granularity': GRANULARITY_HUMID
        },
        'avg_par': {
            'bounds': (df['avg_par'].min(), df['avg_par'].max()),
            'granularity': GRANULARITY_PAR
        }
    }

    # אתחול המחלקה
    mdp_model = PlantMDP(
        data_path=temp_file,
        state_map=state_config,
        action_col='action_discrete',
        weight_col='start_weight'
    )
    mdp_model.process_data()
    mdp_model.print_occupancy_stats()

    # ==============================================================================
    # 3. הגדרת הסביבה
    # ==============================================================================
    def get_env_step(state, action):
        if state not in mdp_model.transitions or action not in mdp_model.transitions[state]:
            return state, -10

        next_states_dict = mdp_model.transitions[state][action]
        candidates = list(next_states_dict.keys())
        counts = list(next_states_dict.values())
        probs = np.array(counts) / sum(counts)

        next_state_idx = np.random.choice(len(candidates), p=probs)
        next_state = candidates[next_state_idx]

        # חישוב ה-Reward:
        # State[0] הוא המשקל כי 'start_weight' הוגדר ראשון ב-state_config
        current_weight = state[0]
        next_weight = next_state[0]

        reward = next_weight - current_weight
        return next_state, reward

    # ==============================================================================
    # 4. הרצת Q-Learning
    # ==============================================================================
    print(f"\nStarting Q-Learning Training for {target_soil}...")

    ACTIONS = list(range(NUM_ACTIONS))
    Q_table = {s: {a: 0.0 for a in ACTIONS} for s in mdp_model.states}

    ALPHA = 0.1
    GAMMA = 0.95
    EPSILON = 1.0
    EPSILON_MIN = 0.01
    DECAY = 0.9995
    EPISODES = 25000  # העליתי קצת את מספר האפיזודות כי מרחב המצבים גדל (4 מימדים)


    for episode in range(EPISODES):
        # בחירת מצב התחלה: שתיל קטן (משקל < 250) שקיים בדאטה
        possible_starts = [s for s in mdp_model.states if s[0] < 250]
        possible_starts = [s for s in possible_starts if s in mdp_model.transitions]

        if not possible_starts:
            # Fallback אם אין שתילים קטנים
            possible_starts = list(mdp_model.transitions.keys())

        current_state = possible_starts[np.random.choice(len(possible_starts))]

        for _ in range(60):  # סימולציה של 60 יום
            if np.random.random() < EPSILON:
                action = np.random.choice(ACTIONS)
            else:
                action = max(Q_table[current_state], key=Q_table[current_state].get)

            next_state, reward = get_env_step(current_state, action)

            # עדכון Q
            max_future_q = max(Q_table[next_state].values()) if next_state in Q_table else 0
            current_q = Q_table[current_state][action]
            new_q = current_q + ALPHA * (reward + GAMMA * max_future_q - current_q)
            Q_table[current_state][action] = new_q

            current_state = next_state

        if EPSILON > EPSILON_MIN:
            EPSILON *= DECAY

        if episode % 5000 == 0:
            print(f"Episode {episode}/{EPISODES} | Epsilon: {EPSILON:.4f}")

    print("Training Finished.")

    if os.path.exists(temp_file):
        os.remove(temp_file)

    # ==============================================================================
    # 6. שמירת המודל (כולל המטא-דאטה החדש)
    # ==============================================================================
    # הוספתי _p לציון PAR בשם הקובץ
    model_filename = f"q_agent_{target_soil}_w{GRANULARITY_WEIGHT}_t{GRANULARITY_TEMP}_h{GRANULARITY_HUMID}_p{GRANULARITY_PAR}_actions_{NUM_ACTIONS}.pkl"
    print(f"\nSaving model to {model_filename}...")

    model_data = {
        "q_table": Q_table,
        "soil_type": target_soil,
        "granularities": {
            "weight": GRANULARITY_WEIGHT,
            "temp": GRANULARITY_TEMP,
            "humid": GRANULARITY_HUMID,
            "par": GRANULARITY_PAR
        },
        "optimal_policy": {s: max(Q_table[s], key=Q_table[s].get) for s in mdp_model.states}
    }

    with open(model_filename, "wb") as f:
        pickle.dump(model_data, f)

    print("Model saved successfully!")


if __name__ == "__main__":
    train_plant_agent()