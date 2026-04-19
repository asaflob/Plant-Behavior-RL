# Explaining Plant Behavior via Reinforcement Learning

Welcome to the **Plant Behavior RL** repository! This project models a plant as an intelligent Reinforcement Learning (RL) agent. By analyzing seven years of real-world greenhouse data, we aim to understand and explain the critical trade-off plants make: opening their stomata to absorb CO₂ (maximizing energy/reward) versus preventing water loss via transpiration (minimizing cost).

We utilize **Markov Decision Processes (MDP)**, **Q-Learning**, and **Gaussian Mixture Models (GMM)** to cluster complex environmental states and learn the plant's optimal behavior policy.

---

##Project Structure

Here is an overview of the files and modules in this repository:

### Core Pipeline & Training
* **`data/data_execution.py`**: The initial data preprocessing pipeline. Cleans raw sensor data and prepares it for the MDP.
* **`train_agent_with_GMM.py`**: **[MAIN EXECUTABLE]** The primary script for training the agent. It loads processed data, applies GMM clustering to reduce the state space, builds the MDP, runs the Q-learning algorithm, and saves the trained model (`.pkl`) and convergence plots.
* **`train_agent_knn.py`**: An alternative training script utilizing K-Means clustering instead of GMM.
* **`q_learning_algo.py`**: Contains the core Q-Learning algorithm loop, separated for modularity and clarity.

### Environment & MDP Models
* **`MDP_cluster.py`**: The updated MDP class where environmental states are defined dynamically by clusters (GMM/K-Means) rather than manual bins.
* **`MDP..py`**: The legacy MDP class used before transitioning to clustering methods.
* **`PlantGrowthTrainer.py`**: A validation script used to test and verify the integrity of the MDP class before full model integration.

### Evaluation & Visualization
* **`models_plant_vs_model.py`**: The evaluation module. Compares the RL agent's predicted actions against actual real-world plant behavior to measure exact accuracy and Mean Absolute Error (MAE).
* **`prove_the_norm_in_k-means.py`**: Generates the "Elbow Method" plot for K-Means to justify the optimal number of clusters.
* **`visualization_for_model_only_weight.py`**: Generates visualizations tracking the progression of plant weight over time.

### Testing & Development 
* **`test_the_class.py`, `test_the_class_check.py`, etc.**: Various historical development scripts used during the initial phases of the project to ensure basic code execution before migrating to the clustering architecture.

---

## How to Run the Project

### 1. Install Requirements
Ensure you have Python installed, then install the necessary dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
*(Dependencies include: `pandas`, `numpy`, `scikit-learn`, `matplotlib`)*

### 2. Process the Data
Before training, you must process the raw greenhouse data. Run the data execution script:
```bash
python data/data_execution.py
```
*This will generate the required `.parquet` files (e.g., `tomato_mdp_final_with_pnw.parquet`) in the `data/` directory.*

### 3. Train the Agent
To train the main model using GMM clustering (recommended):
```bash
python train_agent_with_GMM.py
```
**What this script does:**
1. Filters data by soil type (Sand/Soil).
2. Calculates discrete action spaces (e.g., Evaporation Percentage).
3. Normalizes environmental states and clusters them using GMM.
4. Builds the transition matrix and MDP.
5. Runs Q-Learning until convergence.
6. Outputs a convergence plot (`.png`) and saves the trained policy (`.pkl`).

### 4. Evaluate the Model
To see how well the model performs compared to a real plant, run:
```bash
python models_plant_vs_model.py
```

---

## Methodology Highlights
* **Action Granularity:** We experiment with different stomatal action definitions, including Discrete Transpiration (DT) and Evaporation Percentage relative to Plant Net Weight (PNW).
* **State Dimensionality Reduction:** Raw combinations of Temperature, Humidity, and Light create an impossibly large state space. We use **GMM** to group these into realistic, distinct environmental "states" (e.g., 200 states for Sand), allowing the Q-learning algorithm to converge efficiently.
* **Future Work:** Transitioning from daily averages to 15-minute intervals using Monte Carlo algorithms, and integrating Explainable AI (XAI) to interpret the policy.

---
*Developed by Team 118: Asaf Vitenshtein & Dor Snapiri at the Hebrew University of Jerusalem.*
