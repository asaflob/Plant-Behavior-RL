"""SavedAgent schema and save/load helpers.

All trainers persist a SavedAgent. Evaluators and XAI scripts load through
``load_agent``, which migrates older pickles (v1) into the current schema.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 2


@dataclass
class SavedAgent:
    q_table: dict
    optimal_policy: dict
    soil_type: str
    num_actions: int
    num_states: int
    state_cols: list[str]
    action_method: str
    clustering_method: str
    expected_rewards: dict
    extra: dict[str, Any] = field(default_factory=dict)
    version: int = SCHEMA_VERSION

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(asdict(self), f)


def _policy_from_q_table(q_table: dict) -> dict:
    return {s: max(actions, key=actions.get) for s, actions in q_table.items() if actions}


def load_agent(path: str | Path) -> SavedAgent:
    """Load a SavedAgent from disk, migrating from older formats if needed."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    version = data.get("version", 1)
    if version == SCHEMA_VERSION:
        return SavedAgent(**data)

    # ----- v1 migration -----
    # Older GMM trainer pickles had: q_table, soil_type, num_actions,
    # action_method, clustering_method, num_states, optimal_policy,
    # expected_rewards (no version field, no state_cols).
    # Older KNN pickles lacked expected_rewards and action_method.
    # Older PlantGrowthTrainer pickles had: q_table, policy, soil_type,
    # state_config, transitions (no num_actions etc.).
    q_table = data.get("q_table", {})
    policy = data.get("optimal_policy") or data.get("policy") or _policy_from_q_table(q_table)

    known_methods = ("DT_NORMALIZED", "EVAPORATION_PERCENTAGE", "DT_GRANULARITY")
    action_method = data.get("action_method") or "DT_NORMALIZED"
    if action_method not in known_methods:
        action_method = "DT_NORMALIZED"

    return SavedAgent(
        q_table=q_table,
        optimal_policy=policy,
        soil_type=data.get("soil_type", "unknown"),
        num_actions=data.get("num_actions") or (max(policy.values()) + 1 if policy else 0),
        num_states=data.get("num_states", len(q_table)),
        state_cols=data.get("state_cols", []),
        action_method=action_method,
        clustering_method=data.get("clustering_method", "unknown"),
        expected_rewards=data.get("expected_rewards", {}),
        extra={
            k: v for k, v in data.items()
            if k not in {
                "q_table", "optimal_policy", "policy", "soil_type", "num_actions",
                "num_states", "state_cols", "action_method", "clustering_method",
                "expected_rewards", "version",
            }
        },
        version=SCHEMA_VERSION,
    )


def build_saved_agent(
    q_table: dict,
    soil_type: str,
    num_actions: int,
    num_states: int,
    state_cols: list[str],
    action_method: str,
    clustering_method: str,
    expected_rewards: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> SavedAgent:
    """Convenience constructor used by trainers."""
    return SavedAgent(
        q_table=q_table,
        optimal_policy=_policy_from_q_table(q_table),
        soil_type=soil_type,
        num_actions=num_actions,
        num_states=num_states,
        state_cols=list(state_cols),
        action_method=action_method,
        clustering_method=clustering_method,
        expected_rewards=expected_rewards or {},
        extra=extra or {},
    )
