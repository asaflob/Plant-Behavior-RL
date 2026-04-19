import pandas as pd
import numpy as np
import os
from MDP_cluster import PlantMDPCluster
import pickle
import matplotlib.pyplot as plt
from q_learning_algo import q_learning
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture  # <--- ייבוא ה-GMM

#todo fix this visualize
def visualize_clustering_process(df, state_cols, num_states, soil_type):
    """
    מייצר גרף 'לפני ואחרי' של תהליך ה-GMM על טמפרטורה ולחות.
    """
    print("Generating Clustering Visualization (Before vs. After)...")

    # 1. נרמול (כמו באימון האמיתי)
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df[state_cols]), columns=state_cols)

    # 2. הרצת GMM
    gmm = GaussianMixture(n_components=num_states, random_state=42, n_init=5)
    cluster_labels = gmm.fit_predict(df_normalized)

    # חילוץ מרכזי הכובד והמרתם חזרה ליחידות המקוריות (מעלות ואחוזים)
    centroids_real = scaler.inverse_transform(gmm.means_)

    # 3. ציור הגרפים
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # גרף 1: לפני (נתונים גולמיים)
    ax1.scatter(df['avg_temp'], df['avg_humidity'], c='gray', alpha=0.5, edgecolors='w', s=40)
    ax1.set_title(f'Before GMM: Raw Climate Data ({soil_type})', fontsize=16)
    ax1.set_xlabel('Average Temperature (°C)', fontsize=12)
    ax1.set_ylabel('Average Humidity (%)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # גרף 2: אחרי (צבוע לפי קלאסטרים + סימון מרכזים)
    # משתמשים במפת צבעים (cmap) כדי לתת צבע שונה לכל קלאסטר
    scatter = ax2.scatter(df['avg_temp'], df['avg_humidity'], c=cluster_labels, cmap='tab20', alpha=0.6, s=40)

    # מוסיפים את מרכזי הכובד (Centroids) - ניקח רק את העמודות של טמפ' ולחות
    # נניח ש-state_cols מסודר ככה: ['avg_temp', 'avg_humidity', 'avg_par']
    temp_idx = state_cols.index('avg_temp')
    humid_idx = state_cols.index('avg_humidity')

    ax2.scatter(centroids_real[:, temp_idx], centroids_real[:, humid_idx],
                c='red', marker='*', s=150, edgecolor='black', linewidth=1, label='Cluster Centroids (States)')

    ax2.set_title(f'After GMM: {num_states} Climate States Assigned', fontsize=16)
    ax2.set_xlabel('Average Temperature (°C)', fontsize=12)
    ax2.set_ylabel('Average Humidity (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_filename = f"gmm_clustering_visual_{soil_type}_{num_states}.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"Saved clustering visualization as '{plot_filename}'")
    plt.show()
    plt.close()

def train_plant_agent():
    # ==============================================================================
    # 0. הגדרות (Configuration)
    # ==============================================================================

    NUM_ACTIONS = 100

    # שם הקובץ הסופי שיצרנו ב-Pipeline
    input_file = os.path.join("data", "tomato_mdp_final_with_pnw.parquet")

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
    ACTION_CALC_METHOD = 'EVAPORATION_PERCENTAGE' #EVAPORATION_PERCENTAGE, DT_NORMALIZED, DT_GRANULARITY
    target_soil = 'sand'  # todo change to soil/sand
    print(f"Filtering data for soil type: '{target_soil}'...")

    if 'soil_type' in df.columns:
        df = df[df['soil_type'].astype(str).str.strip() == target_soil]

    if df.empty:
        print("CRITICAL: No data found.")
        return

    # ============================================================
    # === Action Calculation Logic ===
    # ============================================================
    df = df.dropna(subset=['dt', 'start_weight', 'end_weight', 'avg_temp', 'avg_humidity', 'avg_par'])

    print(f"Calculating Actions using method: {ACTION_CALC_METHOD}")

    if ACTION_CALC_METHOD == 'DT_NORMALIZED':
        min_dt = df['dt'].min()
        max_dt = df['dt'].max()

        df['stomatal_opening'] = (df['dt'] - min_dt) / (max_dt - min_dt)
        df['action_discrete'] = pd.cut(df['stomatal_opening'], bins=NUM_ACTIONS, labels=False)

    elif ACTION_CALC_METHOD == 'EVAPORATION_PERCENTAGE':
        # Drop rows where pnw is non-positive — otherwise 100*dt/pnw explodes
        # (we saw ~287 such rows with pnw<=0 producing E% in the ±40,000% range).
        PNW_MIN = 1.0  # grams
        bad_pnw = (df['pnw'].isna()) | (df['pnw'] <= PNW_MIN)
        if bad_pnw.any():
            print(f"  Dropping {int(bad_pnw.sum())} rows with pnw <= {PNW_MIN}")
            df = df[~bad_pnw].copy()

        df['evap_pct'] = (df['dt'] / df['pnw']) * 100

        df['evap_pct'] = df['evap_pct'].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['evap_pct'])

        # evap_pct stays long-tailed even after the pnw>MIN filter, so equal-width
        # pd.cut wastes ~half the buckets (e.g. only 22/50 filled on soil). qcut
        # assigns equal counts per bin — all NUM_ACTIONS actions get real data.
        # duplicates='drop' handles ties at identical bin edges.
        df['action_discrete'] = pd.qcut(
            df['evap_pct'], q=NUM_ACTIONS, labels=False, duplicates='drop'
        )

    elif ACTION_CALC_METHOD == 'DT_GRANULARITY':
        df['action_discrete'] = pd.cut(df['dt'], bins=NUM_ACTIONS, labels=False)

    # ============================================================
    # === Normalization + GMM ===
    # ============================================================

    print("Normalizing data and applying GMM...")

    state_cols = ['avg_temp', 'avg_humidity', 'avg_par']#['start_weight', 'avg_temp', 'avg_humidity', 'avg_par']

    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df[state_cols]), columns=state_cols)

    ###########################################
    # print("Generating BIC Score plot to justify the number of states...")
    # k_values_to_test = [200, 300, 400, 500, 510,520,682, 720]
    # bic_scores = []
    #
    # # בודקים כמה "טעות" יש בכל בחירה של מספר מצבים לפי BIC
    # for k in k_values_to_test:
    #     temp_gmm = GaussianMixture(n_components=k, random_state=42, n_init=3)
    #     temp_gmm.fit(df_normalized)
    #     bic_scores.append(temp_gmm.bic(df_normalized))
    #
    # # מציירים את הגרף ושומרים אותו
    # plt.figure(figsize=(10, 6))
    # plt.plot(k_values_to_test, bic_scores, marker='o', linestyle='-', color='teal', linewidth=2)
    # plt.title(f'GMM BIC Score For Optimal States (Soil: {target_soil})', fontsize=14)
    # plt.xlabel('Number of States (Components)', fontsize=12)
    # plt.ylabel('BIC Score (Lower is Better)', fontsize=12)
    # plt.grid(True, linestyle='--', alpha=0.7)
    #
    # gmm_plot_filename = f"gmm_bic_method_{target_soil}.png"
    # plt.savefig(gmm_plot_filename, dpi=300)
    # print(f"Saved BIC plot as '{gmm_plot_filename}'. Look for the MINIMUM point!")
    # plt.close()

    ###########################################

    # ג. GMM
    NUM_STATES = 500 #sand 500 #soil 121 # todo check by the graph
    print(f"Running final GMM with {NUM_STATES} components...")
    # visualize_clustering_process(df, state_cols, NUM_STATES, target_soil) #todo fix this visualize

    gmm = GaussianMixture(n_components=NUM_STATES, random_state=42, n_init=5)

    # מקבלים לאיזה אשכול שייכת כל שורה
    cluster_labels = gmm.fit_predict(df_normalized)

    # ב-GMM מרכזי הכובד נקראים means_
    centroids_real = scaler.inverse_transform(gmm.means_)

    df[state_cols] = centroids_real[cluster_labels]

    WEIGHT_GRAN_FOR_STATE = 1
    df['weight_state'] = (df['start_weight'] / WEIGHT_GRAN_FOR_STATE).round() * WEIGHT_GRAN_FOR_STATE

    temp_file = "temp_data_for_training.parquet"
    df.to_parquet(temp_file)

    # ==============================================================================
    # 2. בניית המודל (MDP Construction)
    # ==============================================================================
    print("Building MDP directly from GMM states X Weight...")

    mdp_state_cols = state_cols + ['weight_state']

    mdp_model = PlantMDPCluster(  # <--- שימוש במחלקה החדשה
        data_path=temp_file,
        state_cols=mdp_state_cols,  # <--- רשימת העמודות הטהורה
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

        reward = mdp_model.expected_rewards.get(str((state, action)), 0)

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
    model_filename = f"q_agent_{target_soil}_gmm_{NUM_STATES}_act_{NUM_ACTIONS}_{ACTION_CALC_METHOD}_new_state.pkl"
    print(f"\nSaving model to {model_filename}...")

    model_data = {
        "q_table": Q_table,
        "soil_type": target_soil,
        "num_actions": NUM_ACTIONS,
        "action_method": ACTION_CALC_METHOD,
        "clustering_method": "GMM",
        "num_states": NUM_STATES,
        "optimal_policy": {s: max(Q_table[s], key=Q_table[s].get) for s in mdp_model.states if Q_table[s]},
        "expected_rewards": mdp_model.expected_rewards
    }

    with open(model_filename, "wb") as f:
        pickle.dump(model_data, f)

    print("Model saved successfully!")


if __name__ == "__main__":
    train_plant_agent()