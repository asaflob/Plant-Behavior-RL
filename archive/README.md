# Archive

Historical development scripts kept for reference. These pre-date the migration to
clustering-based MDPs (`MDP_cluster.py`) and the GMM training pipeline. They are not
expected to run against the current data schema.

- `test_the_class.py`, `test_the_class_checkk.py`, `test_the_class_checkk(2).py` —
  early validation scripts used while bringing up the original `PlantMDP` class.
- `models_plant_vs_model_legacy.py` — the 1400-line evaluation file containing
  multiple historical iterations of the real-vs-agent comparison
  (`compare_real_vs_agent`, `analyze_experiment_prediction`,
  `analyze_experiment_prediction_final`, etc.). The active evaluation path now
  lives in `models_plant_vs_model.py` at the repo root, built around the
  shared `load_agent` and `actions.discretize` helpers.
