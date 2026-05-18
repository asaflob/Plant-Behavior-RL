# Explaining Plant Behavior via Reinforcement Learning

Welcome to the **Plant Behavior RL** repository! This project models a plant as an intelligent Reinforcement Learning (RL) agent. By analyzing seven years of real-world greenhouse data, we aim to understand and explain the critical trade-off plants make: opening their stomata to absorb CO₂ (maximizing energy/reward) versus preventing water loss via transpiration (minimizing cost).

We utilize **Markov Decision Processes (MDP)**, **Q-Learning**, and **Gaussian Mixture Models (GMM)** to cluster complex environmental states and learn the plant's optimal behavior policy.

---

## Project Structure

### Core training pipeline
* **`train_agent.py`** — **main entry point**. Single CLI for training under either clustering method (GMM or K-Means) and any of the supported action discretisations.
* **`q_learning_algo.py`** — the tabular Q-learning loop shared by every trainer.
* **`MDP_cluster.py`** — `PlantMDPCluster`: builds transitions and expected rewards over cluster-based states.
* **`MDP.py`** — `PlantMDP`: legacy grid-based MDP (kept for `PlantGrowthTrainer`).
* **`PlantGrowthTrainer.py`** — legacy config-driven trainer, driven by `config.json`.

### Reusable building blocks
* **`clustering.py`** — `Clusterer` protocol plus `GMMClusterer` / `KMeansClusterer`. Add a new method here without touching the trainer.
* **`actions.py`** — action-discretisation strategies (`DT_NORMALIZED`, `EVAPORATION_PERCENTAGE`, `DT_GRANULARITY`).
* **`agent_io.py`** — `SavedAgent` schema, `build_saved_agent`, and `load_agent` (with migration for older pickles).

### Backwards-compat shims
* **`train_agent_with_GMM.py`**, **`train_agent_knn.py`** — thin wrappers that warn and forward to `train_agent.py`. Prefer the unified script.

### Evaluation & visualisation
* **`models_plant_vs_model.py`** — compares the trained policy against the average real plant per experiment. Reports exact and relaxed accuracy plus MAE.
* **`prove_the_norm_in_k-means.py`** — illustrative plot showing why feature normalisation is required before clustering.
* **`visualization_for_model_only_weight.py`** — overlay of soil-vs-sand average policy across the relative growth stage.

### Explainability (XAI)
* **`XAI/shap_XAI.py`** — fits a Random Forest surrogate on the policy, then explains feature attributions via SHAP.
* **`XAI/summarize_using_transitions.py`** — surfaces the top-K most "critical" states (largest spread between best and worst action Q-values).

### Tests & archive
* **`tests/smoke_test.py`** — synthetic-data end-to-end smoke test for the training pipeline. Run before merging changes that touch trainers, MDP, or the SavedAgent schema.
* **`archive/`** — historical dev scripts and the original 1400-line `models_plant_vs_model.py`.

---

## How to Run the Project

### 1. Install Requirements
```bash
pip install -r requirements.txt
```
(Dependencies include `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `shap`.)

### 2. Process the Data
Run the data preprocessing pipeline to generate the `.parquet` files in `data/`.
```bash
python data/data_execution.py
```

### 3. Train the Agent
The unified trainer covers both clustering methods:

```bash
# Default: GMM, sand, 500 states, 50 actions, EVAPORATION_PERCENTAGE
python train_agent.py

# K-Means alternative
python train_agent.py --clustering kmeans --soil soil --num-states 121

# Tune number of actions or action method
python train_agent.py --num-actions 50 --action-method DT_GRANULARITY
```

The script prints occupancy stats, runs Q-Learning until convergence, saves a convergence plot (`convergence_plot_*.png`), and writes a `SavedAgent` pickle (`q_agent_*.pkl`).

### 4. Evaluate the Model
```bash
python models_plant_vs_model.py path/to/q_agent_*.pkl
python models_plant_vs_model.py path/to/q_agent_*.pkl --coverage-only
```

### 5. Explain the Policy
```bash
python XAI/summarize_using_transitions.py   # top-K critical states
python XAI/shap_XAI.py                       # surrogate + SHAP plots
```

---

## Methodology Highlights
* **Action Granularity:** Several stomatal action definitions are supported, including Discrete Transpiration (DT) and Evaporation Percentage relative to Plant Net Weight (PNW).
* **State Dimensionality Reduction:** Raw combinations of Temperature, Humidity, and Light create an enormous state space. **GMM** groups them into a tractable number of environmental "states" (e.g. 500 for Sand, ~120 for Soil) so Q-learning converges efficiently.
* **Future Work:** Moving from daily averages to 15-minute intervals via Monte Carlo, and deepening the XAI integration to interpret the policy at scale.

---

## Extending the Project

* **New clustering method.** Implement the `Clusterer` protocol in `clustering.py`, register it in `build_clusterer`. The trainer picks it up automatically.
* **New action discretisation.** Add a function in `actions.py` and register it in `DISCRETIZERS`. CLI choices update on the next change to `train_agent.parse_args`.
* **New saved-model field.** Add it to the `SavedAgent` dataclass in `agent_io.py` and bump `SCHEMA_VERSION`. `load_agent` handles migrating old pickles forward.

---
*Developed by Team 118: Asaf Vitenshtein & Dor Snapiri at the Hebrew University of Jerusalem.*
