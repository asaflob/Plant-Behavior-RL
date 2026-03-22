import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


def generate_normalization_plot():
    # 1. טעינת הנתונים
    input_file = os.path.join("data", "tomato_mdp_final_filtered.parquet")
    print(f"Loading data from {input_file}...")
    df = pd.read_parquet(input_file)

    # סינון לאדמת חול
    df = df[df['soil_type'].astype(str).str.strip() == 'sand']
    df = df.dropna(subset=['dt', 'start_weight', 'end_weight', 'avg_temp', 'avg_humidity', 'avg_par'])

    state_cols = ['start_weight', 'avg_temp', 'avg_humidity', 'avg_par']
    DEMO_STATES = 6  # בחרתי 6 כדי שהצבעים יהיו ברורים מאוד בעין

    # 2. חלוקה ללא נרמול (The Flawed Way)
    kmeans_unnormalized = KMeans(n_clusters=DEMO_STATES, random_state=42, n_init=10)
    labels_unnormalized = kmeans_unnormalized.fit_predict(df[state_cols])

    # 3. חלוקה עם נרמול (The Correct Way)
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df[state_cols]), columns=state_cols)
    kmeans_normalized = KMeans(n_clusters=DEMO_STATES, random_state=42, n_init=10)
    labels_normalized = kmeans_normalized.fit_predict(df_normalized)

    # =========================================================
    # יצירת הגרף (Visual Plotting)
    # =========================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- גרף 1: ללא נרמול ---
    scatter1 = ax1.scatter(df['avg_temp'], df['start_weight'], c=labels_unnormalized, cmap='tab10', alpha=0.6, s=15)
    ax1.set_title("A: K-Means WITHOUT Normalization\n(Clustering relies heavily on Weight)", fontsize=14,
                  fontweight='bold', color='darkred')
    ax1.set_xlabel("Average Temperature (°C)", fontsize=12)
    ax1.set_ylabel("Start Weight (grams)", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- גרף 2: עם נרמול ---
    scatter2 = ax2.scatter(df['avg_temp'], df['start_weight'], c=labels_normalized, cmap='tab10', alpha=0.6, s=15)
    ax2.set_title("B: K-Means WITH Normalization\n(Clustering balances Weight & Temperature)", fontsize=14,
                  fontweight='bold', color='darkgreen')
    ax2.set_xlabel("Average Temperature (°C)", fontsize=12)
    ax2.set_ylabel("Start Weight (grams)", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # עיצוב סופי ושמירה
    plt.suptitle("Why Normalization is Critical Before Clustering States", fontsize=18, y=1.02)
    plt.tight_layout()

    output_filename = "normalization_proof_for_presentation.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Open '{output_filename}' and put it straight into your presentation.")


if __name__ == "__main__":
    generate_normalization_plot()