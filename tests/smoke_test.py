"""End-to-end smoke test for the training pipeline.

Synthesises a tiny dataset shaped like the real greenhouse data, runs the full
pipeline (action discretisation -> clustering -> MDP -> Q-learning -> save ->
load), and asserts the resulting SavedAgent is usable.

Run from the repo root:

    python tests/smoke_test.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running this file directly from tests/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from actions import discretize  # noqa: E402
from agent_io import build_saved_agent, load_agent  # noqa: E402
from clustering import build_clusterer  # noqa: E402
from MDP_cluster import PlantMDPCluster  # noqa: E402
from q_learning_algo import UNKNOWN_ACTION_PENALTY, q_learning  # noqa: E402
from train_agent import (  # noqa: E402
    DEFAULT_STATE_COLS, TrainConfig, cluster_states, make_env_step, train,
)


def _synthetic_dataframe(num_plants: int = 12, days_per_plant: int = 10) -> pd.DataFrame:
    """Build a small, well-shaped dataset that exercises every column."""
    rng = np.random.default_rng(42)
    rows = []
    for plant_id in range(num_plants):
        base_weight = rng.uniform(30, 80)
        for day in range(days_per_plant):
            weight = base_weight + day * rng.uniform(2, 5)
            rows.append({
                "unique_id": plant_id,
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                "day_num": day,
                "soil_type": "sand",
                "avg_temp": rng.uniform(18, 30),
                "avg_humidity": rng.uniform(40, 80),
                "avg_par": rng.uniform(100, 400),
                "start_weight": weight,
                "end_weight": weight + rng.uniform(2, 6),
                "dt": rng.uniform(0.5, 8.0),
                "pnw": weight,
            })
    return pd.DataFrame(rows)


def test_pipeline_with_synthetic_data() -> None:
    df = _synthetic_dataframe()
    df = discretize("EVAPORATION_PERCENTAGE", df, num_actions=5)
    clusterer = build_clusterer("gmm", n_states=6)
    df = cluster_states(df, DEFAULT_STATE_COLS, clusterer)

    mdp = PlantMDPCluster(
        data=df,
        state_cols=list(DEFAULT_STATE_COLS) + ["weight_state"],
        action_col="action_discrete",
        weight_col="start_weight",
    )
    mdp.process_data()
    assert mdp.transitions, "Expected transitions populated from synthetic data"
    assert mdp.expected_rewards, "Reward map should not be empty after process_data"

    env_step = make_env_step(mdp)
    sample_state = next(iter(mdp.transitions))
    next_state, reward = env_step(sample_state, next(iter(mdp.transitions[sample_state])))
    assert next_state in mdp.transitions or reward == UNKNOWN_ACTION_PENALTY

    q_table, history = q_learning(
        mdp_model=mdp, env_step_func=env_step, num_actions=5,
        max_iterations=30, threshold=1e-2, consecutive_iterations=2,
    )
    assert q_table, "Q-table should not be empty"
    assert history, "Convergence history should not be empty"

    agent = build_saved_agent(
        q_table=q_table,
        soil_type="sand",
        num_actions=5,
        num_states=6,
        state_cols=list(DEFAULT_STATE_COLS) + ["weight_state"],
        action_method="EVAPORATION_PERCENTAGE",
        clustering_method="GMM",
        expected_rewards=mdp.expected_rewards,
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "agent.pkl"
        agent.save(path)
        roundtripped = load_agent(path)

    assert roundtripped.num_actions == 5
    assert roundtripped.clustering_method == "GMM"
    assert roundtripped.action_method == "EVAPORATION_PERCENTAGE"
    assert roundtripped.q_table.keys() == q_table.keys()


def test_train_writes_model_and_plot() -> None:
    """Exercise the full ``train`` entry point against synthetic data."""
    df = _synthetic_dataframe(num_plants=8, days_per_plant=8)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "synthetic.parquet"
        df.to_parquet(input_path)

        cfg = TrainConfig(
            input_file=input_path,
            soil_type="sand",
            num_actions=4,
            num_states=4,
            action_method="EVAPORATION_PERCENTAGE",
            clustering_method="gmm",
            output_dir=tmp_path,
        )
        model_path = train(cfg)
        assert model_path.exists(), f"Trainer should produce {model_path}"
        assert any(tmp_path.glob("convergence_plot_*.png")), "Expected a convergence plot"
        agent = load_agent(model_path)
        assert agent.num_actions == 4
        assert agent.optimal_policy, "Optimal policy should not be empty"


def main() -> None:
    print("Running pipeline smoke test...")
    test_pipeline_with_synthetic_data()
    print("  test_pipeline_with_synthetic_data PASSED")
    test_train_writes_model_and_plot()
    print("  test_train_writes_model_and_plot PASSED")
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
