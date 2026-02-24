import pandas as pd
import numpy as np
import os
from MDP import PlantMDP
import pickle
import matplotlib.pyplot as plt  # <--- הוספנו את ספריית הציור


def train_plant_agent():
    # ==============================================================================
    # 0. הגדרות (Configuration)
    # ==============================================================================
    # הגדרת הרזולוציה לכל מימד בנפרד
    GRANULARITY_WEIGHT = 5
    GRANULARITY_TEMP = 2
    GRANULARITY_HUMID = 10
    GRANULARITY_PAR = 100

    NUM_ACTIONS = 10

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
    target_soil = 'soil' #todo change to soil/sand
    print(f"Filtering data for soil type: '{target_soil}'...")

    if 'soil_type' in df.columns:
        df = df[df['soil_type'].astype(str).str.strip() == target_soil]

    if df.empty:
        print("CRITICAL: No data found.")
        return

    print(f"Rows remaining after filter: {len(df)}")

    df = df.dropna(subset=['dt', 'start_weight', 'end_weight', 'avg_temp', 'avg_humidity', 'avg_par'])

    # --- א. יצירת ה-Action ---
    min_dt = df['dt'].min()
    max_dt = df['dt'].max()

    df['stomatal_opening'] = (df['dt'] - min_dt) / (max_dt - min_dt)
    df['action_discrete'] = pd.cut(df['stomatal_opening'], bins=NUM_ACTIONS, labels=False)

    # ============================================================
    # === תהליך הדיסקרטיזציה (Discretization / Rounding) ===
    # ============================================================
    print("Rounding data to grid...")

    df['start_weight'] = (df['start_weight'] / GRANULARITY_WEIGHT).round() * GRANULARITY_WEIGHT
    df['end_weight'] = (df['end_weight'] / GRANULARITY_WEIGHT).round() * GRANULARITY_WEIGHT
    df['avg_temp'] = (df['avg_temp'] / GRANULARITY_TEMP).round() * GRANULARITY_TEMP
    df['avg_humidity'] = (df['avg_humidity'] / GRANULARITY_HUMID).round() * GRANULARITY_HUMID
    df['avg_par'] = (df['avg_par'] / GRANULARITY_PAR).round() * GRANULARITY_PAR

    temp_file = "temp_data_for_training.parquet"
    df.to_parquet(temp_file)

    # ==============================================================================
    # 2. בניית המודל (MDP Construction)
    # ==============================================================================
    print("Building MDP from data...")

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

    MAX_EPISODES = 100000
    TOLERANCE = 1e-4
    CONSECUTIVE_EPISODES = 10
    converged_count = 0

    # <--- חדש: רשימה לשמירת היסטוריית ההתכנסות --->
    convergence_history = []

    for episode in range(MAX_EPISODES):
        possible_starts = [s for s in mdp_model.states if s[0] < 250]
        possible_starts = [s for s in possible_starts if s in mdp_model.transitions]

        if not possible_starts:
            possible_starts = list(mdp_model.transitions.keys())

        current_state = possible_starts[np.random.choice(len(possible_starts))]
        max_change_this_episode = 0.0

        for _ in range(60):
            if np.random.random() < EPSILON:
                action = np.random.choice(ACTIONS)
            else:
                action = max(Q_table[current_state], key=Q_table[current_state].get)

            next_state, reward = get_env_step(current_state, action)

            max_future_q = max(Q_table[next_state].values()) if next_state in Q_table else 0
            current_q = Q_table[current_state][action]
            new_q = current_q + ALPHA * (reward + GAMMA * max_future_q - current_q)

            change = abs(new_q - current_q)
            if change > max_change_this_episode:
                max_change_this_episode = change

            Q_table[current_state][action] = new_q
            current_state = next_state

        # שמירת השינוי המקסימלי של האפיזודה הזו לטובת הגרף
        convergence_history.append(max_change_this_episode)

        if EPSILON > EPSILON_MIN:
            EPSILON *= DECAY

        if max_change_this_episode < TOLERANCE:
            converged_count += 1
        else:
            converged_count = 0

        if converged_count >= CONSECUTIVE_EPISODES:
            print(f"Algorithm converged successfully at episode {episode}!")
            break

        if episode % 5000 == 0:
            print(
                f"Episode {episode}/{MAX_EPISODES} | Epsilon: {EPSILON:.4f}| Max Change: {max_change_this_episode:.6f}")

    print("Training Finished.")

    if os.path.exists(temp_file):
        os.remove(temp_file)

    # ==============================================================================
    # 5. יצירת גרף התכנסות (Convergence Plot)
    # ==============================================================================
    print("Generating convergence plot...")
    plt.figure(figsize=(10, 6))

    # ציור ההיסטוריה בסקאלה רגילה (ליניארית)
    plt.plot(convergence_history, color='blue', alpha=0.5, label='Max Q-value Change (Delta)')

    # קו אדום מקווקו המייצג את סף ההתכנסות שלנו
    plt.axhline(y=TOLERANCE, color='red', linestyle='--', label=f'Convergence Threshold ({TOLERANCE})')

    plt.title(f'Q-Learning Convergence for {target_soil} Soil', fontsize=14)
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Max Delta (Change) in Q-values', fontsize=12)

    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()

    # שמירת הגרף כתמונה
    plot_filename = f"convergence_plot_{target_soil}.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot saved as {plot_filename}")
    # plt.show() # הסר את סימן ההערה אם אתה רוצה שהגרף יקפוץ לך על המסך בסיום ההרצה

    # ==============================================================================
    # 6. שמירת המודל
    # ==============================================================================
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