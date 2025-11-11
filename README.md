# Plant-Behavior-RL
# Explaining Plant Behavior

**Group 188 - HUJI Project**
* [cite_start]Asaf Vitenshtein (asaf.vitenshtein@mail.huji.ac.il) [cite: 3]
* [cite_start]Dor Snapiri (dor.snapiri@mail.huji.ac.il) [cite: 4]
* [cite_start]Mentor: Gal Fiebelman (gal.fiebelman@mail.huji.ac.il) [cite: 7]

[cite_start]In association with HUJI School of Computer Science & Engineering and HUJI Faculty of Agriculture, Food and Environment[cite: 9, 10].

## 1. Project Goal

[cite_start]Plants constantly manage the trade-off between $CO_2$ uptake and water loss via their stomata[cite: 12]. [cite_start]Due to numerous environmental inputs, it is difficult to predict or explain why a plant adjusts its stomatal openings in real-time[cite: 13].

[cite_start]Our project's goal is to model this process as a Reinforcement Learning agent[cite: 39]. [cite_start]This will allow us to move beyond simple prediction and use Explainable AI (XAI) to interpret the plant's decision-making strategy[cite: 42]. [cite_start]This is vital for researchers and farmers seeking to optimize crop efficiency and water usage[cite: 14].

## 2. Project Setup & Technologies

This project is developed in Python. The environment requires the following libraries and tools:

### Core Libraries
* [cite_start]**Data Handling:** Pandas and NumPy [cite: 60]
* [cite_start]**RL Algorithms:** PyTorch [cite: 61] [cite_start]and Stable Baselines3 [cite: 62]
* [cite_start]**Explainability (XAI):** SHAP and Captum [cite: 63]
* [cite_start]**Visualization:** Matplotlib and Seaborn [cite: 64]

### Tools & Hardware
* [cite_start]**IDE:** Jupyter Notebook [cite: 65]
* [cite_start]**Version Control:** Git [cite: 65]
* [cite_start]**Hardware:** GPU-enabled systems for efficient training [cite: 65]

### Setup Instructions
To set up a local environment, clone this repository and install the required packages. A `requirements.txt` file will be added as the project progresses.

```bash
# Recommended to use a virtual environment
pip install pandas numpy torch stable-baselines3 shap captum matplotlib seaborn jupyter
