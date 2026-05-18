"""Backwards-compat shim: forwards to the unified ``train_agent`` pipeline.

The actual training logic now lives in ``train_agent.py``. Prefer running
``python train_agent.py --clustering gmm ...`` directly.
"""
from __future__ import annotations

import warnings

from train_agent import TrainConfig, train


def train_plant_agent() -> None:
    warnings.warn(
        "train_agent_with_GMM.py is deprecated; use 'python train_agent.py "
        "--clustering gmm' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    train(TrainConfig(
        soil_type="sand",
        num_actions=50,
        num_states=500,
        action_method="EVAPORATION_PERCENTAGE",
        clustering_method="gmm",
    ))


if __name__ == "__main__":
    train_plant_agent()
