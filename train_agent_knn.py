"""Backwards-compat shim: forwards to the unified ``train_agent`` pipeline.

The actual training logic now lives in ``train_agent.py``. Prefer running
``python train_agent.py --clustering kmeans ...`` directly.

Note: the previous version of this script had a silent bug — it looked up
expected rewards by ``str((state, action))`` against a tuple-keyed dict, so
every Bellman update saw a reward of 0. The new pipeline fixes that.
"""
from __future__ import annotations

import warnings

from train_agent import TrainConfig, train


def train_plant_agent() -> None:
    warnings.warn(
        "train_agent_knn.py is deprecated; use 'python train_agent.py "
        "--clustering kmeans' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    train(TrainConfig(
        input_file=__import__("pathlib").Path("data") / "tomato_mdp_final_filtered.parquet",
        soil_type="soil",
        num_actions=50,
        num_states=121,
        action_method="DT_NORMALIZED",
        clustering_method="kmeans",
    ))


if __name__ == "__main__":
    train_plant_agent()
