"""Discretization strategies for the plant's stomatal-opening action.

Each strategy turns a continuous physiological quantity (typically daily
transpiration ``dt``) into an integer action in [0, num_actions).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

ActionDiscretizer = Callable[[pd.DataFrame, int], pd.DataFrame]


def dt_normalized(df: pd.DataFrame, num_actions: int) -> pd.DataFrame:
    """Min-max-normalise dt across the dataset, then bin."""
    min_dt, max_dt = df["dt"].min(), df["dt"].max()
    out = df.copy()
    out["stomatal_opening"] = (out["dt"] - min_dt) / (max_dt - min_dt)
    out["action_discrete"] = pd.cut(out["stomatal_opening"], bins=num_actions, labels=False)
    return out


def evaporation_percentage(df: pd.DataFrame, num_actions: int) -> pd.DataFrame:
    """Daily transpiration as a percentage of plant net weight, then bin."""
    out = df.copy()
    out["evap_pct"] = (out["dt"] / out["pnw"]) * 100
    out["evap_pct"] = out["evap_pct"].replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["evap_pct"])
    out["action_discrete"] = pd.cut(out["evap_pct"], bins=num_actions, labels=False)
    return out


def dt_granularity(df: pd.DataFrame, num_actions: int) -> pd.DataFrame:
    """Bin raw dt directly into ``num_actions`` equal-width bins."""
    out = df.copy()
    out["action_discrete"] = pd.cut(out["dt"], bins=num_actions, labels=False)
    return out


DISCRETIZERS: dict[str, ActionDiscretizer] = {
    "DT_NORMALIZED": dt_normalized,
    "EVAPORATION_PERCENTAGE": evaporation_percentage,
    "DT_GRANULARITY": dt_granularity,
}


def discretize(method: str, df: pd.DataFrame, num_actions: int) -> pd.DataFrame:
    try:
        return DISCRETIZERS[method](df, num_actions)
    except KeyError as exc:
        raise ValueError(
            f"Unknown action discretization method: {method!r}. "
            f"Options: {sorted(DISCRETIZERS)}"
        ) from exc
