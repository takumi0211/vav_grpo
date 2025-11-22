"""Shared helpers for sampling and cleaning critic_value datasets."""

from __future__ import annotations

from typing import Iterable, Iterator

import numpy as np
import pandas as pd

DECISION_COLUMN = "decision_step"
ACTION_PREFIX = "action_"


def sample_evenly(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return up to *n* roughly-evenly spaced rows from *df* (preserves order)."""

    if df.empty:
        return df.copy()
    n = max(1, int(n))
    if len(df) <= n:
        return df.copy()
    idx = np.linspace(0, len(df) - 1, n, dtype=int)
    idx = np.unique(idx)
    return df.iloc[idx].copy()


def iter_episode_groups(df: pd.DataFrame) -> Iterator[pd.DataFrame]:
    """Yield per-episode slices if an episode_id column is present."""

    if "episode_id" not in df.columns:
        yield df
        return
    for _, group in df.groupby("episode_id"):
        yield group


def filter_decision_steps(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only decision_step == 1 rows when the column exists."""

    if DECISION_COLUMN not in df.columns:
        return df.copy()
    mask = df[DECISION_COLUMN] == 1
    return df[mask].copy()


def drop_action_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy without action_* columns (useful for state-only datasets)."""

    cols = [c for c in df.columns if not c.startswith(ACTION_PREFIX)]
    return df[cols].copy()


def ensure_sample_ids(df: pd.DataFrame, column: str = "sample_id") -> pd.DataFrame:
    """Ensure *column* exists and is a simple 0..N-1 range."""

    df = df.copy()
    if column in df.columns:
        df = df.drop(columns=[column])
    df.insert(0, column, np.arange(len(df), dtype=np.int32))
    return df


def clamp_action_columns(df: pd.DataFrame, lo: float = -0.999, hi: float = 0.999) -> pd.DataFrame:
    """Clamp any action_* columns to [lo, hi] to avoid tanh saturation."""

    df = df.copy()
    action_cols = [c for c in df.columns if c.startswith(ACTION_PREFIX)]
    if not action_cols:
        return df
    df[action_cols] = df[action_cols].clip(lower=lo, upper=hi)
    return df

