import pandas as pd
import numpy as np
import os
from MDP import PlantMDP
import pickle
import matplotlib.pyplot as plt
from q_learning_algo import q_learning
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


def train_plant_agent():
    # ==============================================================================
    # 0. הגדרות (Configuration)
    # ==============================================================================
    # הגדרת הרזולוציה לכל מימד בנפרד
    GRANULARITY_WEIGHT = 5
    GRANULARITY_TEMP = 2
    GRANULARITY_HUMID = 10
    GRANULARITY_PAR = 100

    NUM_ACTIONS = 50

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

    # ============================================================
    # === Normalization + K-Means ===
    # ============================================================

    print("Normalizing data and applying K-Means...")

    # א. הגדרת העמודות שמרכיבות את המצב
    state_cols = ['start_weight', 'avg_temp', 'avg_humidity', 'avg_par']

    # ב. נרמול
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df[state_cols]), columns=state_cols)

    ###########################################
    print("Generating Elbow Method plot to justify the number of states...")
    k_values_to_test = [100, 300, 500, 1000, 1500, 2000, 2500, 3000]
    inertias = []

    # בודקים כמה "טעות" יש בכל בחירה של מספר מצבים
    for k in k_values_to_test:
        temp_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        temp_kmeans.fit(df_normalized)
        inertias.append(temp_kmeans.inertia_)#todo understand the meaning here

    # מציירים את הגרף ושומרים אותו
    plt.figure(figsize=(10, 6))
    plt.plot(k_values_to_test, inertias, marker='o', linestyle='-', color='purple', linewidth=2)
    plt.title(f'Elbow Method For Optimal States (Soil: {target_soil})', fontsize=14)
    plt.xlabel('Number of States (K)', fontsize=12)
    plt.ylabel('Inertia (Error / Distances)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    elbow_plot_filename = f"elbow_method_{target_soil}.png"
    plt.savefig(elbow_plot_filename, dpi=300)
    print(f"Saved Elbow plot as '{elbow_plot_filename}'. Look at this image to find the optimal NUM_STATES!")
    plt.close()
    ###########################################

    # ג. K-Means
    NUM_STATES = 1000
    kmeans = KMeans(n_clusters=NUM_STATES, random_state=42, n_init=10)

    # מקבלים לאיזה אשכול שייכת כל שורה
    cluster_labels = kmeans.fit_predict(df_normalized)
    centroids_real = scaler.inverse_transform(kmeans.cluster_centers_)

    df[state_cols] = centroids_real[cluster_labels]

    # ============================================================
    # הפתרון שלך: "Snap to Grid" של מרכזי הכובד
    # ============================================================
    print("Snapping K-Means centroids back to the original grid...")
    df['start_weight'] = (df['start_weight'] / GRANULARITY_WEIGHT).round() * GRANULARITY_WEIGHT
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
    # Q-Learning
    # ==============================================================================
    print(f"\nStarting Q-Learning Execution for {target_soil}...")

    THRESHOLD = 1e-4
    # קריאה לפונקציה המבודדת מתוך הקובץ q_learning_algo.py
    Q_table, convergence_history = q_learning(
        mdp_model=mdp_model,
        env_step_func=get_env_step,
        num_actions=NUM_ACTIONS,
        threshold = THRESHOLD
    )

    if os.path.exists(temp_file):
        os.remove(temp_file)

    # ==============================================================================
    # 5. יצירת גרף התכנסות (Convergence Plot)
    # ==============================================================================
    print("Generating convergence plot...")

    # plt.figure(figsize=(12, 6))
    #
    # window_size = 100
    # smoothed_history = pd.Series(convergence_history).rolling(window=window_size, min_periods=1).mean()
    #
    # # 1. מציירים את הנתונים הגולמיים אבל חלש חלש ברקע (כדי שיראו שיש רעש)
    # plt.plot(convergence_history, color='cornflowerblue', alpha=0.2, label='Raw Delta (Noisy)')
    #
    # # 2. מציירים את הממוצע הנע - קו כהה, עבה וברור!
    # plt.plot(smoothed_history, color='navy', linewidth=2, label=f'Moving Average ({window_size} episodes)')
    #
    # # קו אדום מקווקו המייצג את סף ההתכנסות שלנו
    # plt.axhline(y=THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Convergence Threshold ({THRESHOLD})')
    #
    # plt.title(f'Q-Learning Convergence for {target_soil} Soil', fontsize=16)
    # plt.xlabel('Episodes', fontsize=14)
    # plt.ylabel('Max Delta in Q-values', fontsize=14)
    #
    # # אפשר לחזור לסקאלה רגילה אם הממוצע הנע מחליק את זה מספיק,
    # # או להשאיר את ה-symlog אם זה עדיין קופצני. ננסה symlog עדין:
    # plt.yscale('symlog', linthresh=THRESHOLD)
    #
    # plt.grid(True, which="both", ls="--", alpha=0.5)
    # plt.legend()
    #
    # # שמירת הגרף כתמונה
    # plot_filename = f"convergence_plot_{target_soil}_smoothed.png"
    # plt.savefig(plot_filename, dpi=300)
    # print(f"Plot saved as {plot_filename}")

    plt.figure(figsize=(10, 6))

    # ציור ההיסטוריה בסקאלה רגילה (ליניארית)
    plt.plot(convergence_history, color='blue', alpha=0.5, label='Max Q-value Change (Delta)')

    # קו אדום מקווקו המייצג את סף ההתכנסות שלנו
    plt.axhline(y=THRESHOLD, color='red', linestyle='--', label=f'Convergence Threshold ({THRESHOLD})')

    plt.title(f'Q-Learning Convergence for {target_soil} Soil', fontsize=14)
    plt.xlabel('iteration', fontsize=12)
    plt.ylabel('Max Delta (Change) in Q-values', fontsize=12)

    # plt.yscale('symlog', linthresh=THRESHOLD)

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
        "num_actions": NUM_ACTIONS,
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
    # input_file = os.path.join("data", "tomato_mdp_final_filtered.parquet")
    #
    # # קריאת הנתונים
    # df = pd.read_parquet(input_file)
    #
    # # 1. שליפת המשקל המקסימלי
    # max_start = df['start_weight'].max()
    # max_end = df['end_weight'].max()
    # max_weight = max(max_start, max_end)
    #
    # # 2. שליפת מקסימום טמפרטורה
    # max_temp = df['avg_temp'].max()
    #
    # # 3. שליפת מקסימום לחות
    # max_humidity = df['avg_humidity'].max()
    #
    # # 4. שליפת מקסימום אור (PAR)
    # max_par = df['avg_par'].max()
    #
    # # הדפסות מסודרות
    # print("=== Maximum Values in Dataset ===")
    # print(f"Maximum Weight: {max_weight:.2f} grams")
    # print(f"Maximum Temperature: {max_temp:.2f} °C")
    # print(f"Maximum Humidity: {max_humidity:.2f} %")
    # print(f"Maximum PAR (Light): {max_par:.2f} PPFD")
    # print("=================================")