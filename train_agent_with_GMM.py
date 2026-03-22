import pandas as pd
import numpy as np
import os
from MDP_cluster import PlantMDPCluster
import pickle
import matplotlib.pyplot as plt
from q_learning_algo import q_learning
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture  # <--- ייבוא ה-GMM


def train_plant_agent():
    # ==============================================================================
    # 0. הגדרות (Configuration)
    # ==============================================================================

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
    target_soil = 'sand'  # todo change to soil/sand
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
    # === Normalization + GMM ===
    # ============================================================

    print("Normalizing data and applying GMM...")

    # א. הגדרת העמודות שמרכיבות את המצב
    state_cols = ['start_weight', 'avg_temp', 'avg_humidity', 'avg_par']

    # ב. נרמול
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df[state_cols]), columns=state_cols)

    ###########################################
    print("Generating BIC Score plot to justify the number of states...")
    k_values_to_test = [10, 20, 30, 40, 50, 70, 100, 150, 200, 300, 400]
    bic_scores = []

    # בודקים כמה "טעות" יש בכל בחירה של מספר מצבים לפי BIC
    for k in k_values_to_test:
        temp_gmm = GaussianMixture(n_components=k, random_state=42, n_init=3)
        temp_gmm.fit(df_normalized)
        bic_scores.append(temp_gmm.bic(df_normalized))

    # מציירים את הגרף ושומרים אותו
    plt.figure(figsize=(10, 6))
    plt.plot(k_values_to_test, bic_scores, marker='o', linestyle='-', color='teal', linewidth=2)
    plt.title(f'GMM BIC Score For Optimal States (Soil: {target_soil})', fontsize=14)
    plt.xlabel('Number of States (Components)', fontsize=12)
    plt.ylabel('BIC Score (Lower is Better)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    gmm_plot_filename = f"gmm_bic_method_{target_soil}.png"
    plt.savefig(gmm_plot_filename, dpi=300)
    print(f"Saved BIC plot as '{gmm_plot_filename}'. Look for the MINIMUM point!")
    plt.close()

    ###########################################

    # ג. GMM
    NUM_STATES = 200  # תשנה את זה לפי הנקודה הכי נמוכה שתראה בגרף ה-BIC החדש!
    print(f"Running final GMM with {NUM_STATES} components...")
    gmm = GaussianMixture(n_components=NUM_STATES, random_state=42, n_init=5)

    # מקבלים לאיזה אשכול שייכת כל שורה
    cluster_labels = gmm.fit_predict(df_normalized)

    # ב-GMM מרכזי הכובד נקראים means_
    centroids_real = scaler.inverse_transform(gmm.means_)

    df[state_cols] = centroids_real[cluster_labels]

    temp_file = "temp_data_for_training.parquet"
    df.to_parquet(temp_file)

    # ==============================================================================
    # 2. בניית המודל (MDP Construction)
    # ==============================================================================
    print("Building MDP directly from GMM states...")

    mdp_model = PlantMDPCluster(  # <--- שימוש במחלקה החדשה
        data_path=temp_file,
        state_cols=state_cols,  # <--- רשימת העמודות הטהורה
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
    Q_table, convergence_history = q_learning(
        mdp_model=mdp_model,
        env_step_func=get_env_step,
        num_actions=NUM_ACTIONS,
        threshold=THRESHOLD
    )

    if os.path.exists(temp_file):
        os.remove(temp_file)

    # ==============================================================================
    # 5. יצירת גרף התכנסות (Convergence Plot)
    # ==============================================================================
    print("Generating convergence plot...")

    plt.figure(figsize=(10, 6))
    plt.plot(convergence_history, color='blue', alpha=0.5, label='Max Q-value Change (Delta)')
    plt.axhline(y=THRESHOLD, color='red', linestyle='--', label=f'Convergence Threshold ({THRESHOLD})')

    plt.title(f'Q-Learning Convergence for {target_soil} Soil (GMM)', fontsize=14)
    plt.xlabel('iteration', fontsize=12)
    plt.ylabel('Max Delta (Change) in Q-values', fontsize=12)

    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()

    plot_filename = f"convergence_plot_{target_soil}_gmm.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot saved as {plot_filename}")

    # ==============================================================================
    # 6. שמירת המודל
    # ==============================================================================
    model_filename = f"q_agent_{target_soil}_gmm_{NUM_STATES}_actions_{NUM_ACTIONS}.pkl"
    print(f"\nSaving model to {model_filename}...")

    model_data = {
        "q_table": Q_table,
        "soil_type": target_soil,
        "num_actions": NUM_ACTIONS,
        "clustering_method": "GMM",
        "num_states": NUM_STATES,
        "optimal_policy": {s: max(Q_table[s], key=Q_table[s].get) for s in mdp_model.states if Q_table[s]}
    }

    with open(model_filename, "wb") as f:
        pickle.dump(model_data, f)

    print("Model saved successfully!")


if __name__ == "__main__":
    train_plant_agent()