"""Fallback synthetic-series generator for production images that ship
without the tests/ folder.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def synthetic_series(
    days: int = 30,
    step: str = "5min",
    start: pd.Timestamp | None = None,
    *,
    seed: int = 0,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    start = start or (pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(days=days))
    n = int(days * 24 * 60 / pd.Timedelta(step).total_seconds() * 60)
    idx = pd.date_range(start=start, periods=n, freq=step, tz="UTC")
    t = np.arange(len(idx), dtype=float)
    step_per_day = pd.Timedelta("1D") / pd.Timedelta(step)
    daily = 8.0 * np.sin(2 * math.pi * t / step_per_day)
    weekly = 4.0 * np.sin(2 * math.pi * t / (step_per_day * 7))
    trend = (t / step_per_day) * 0.5
    noise = rng.normal(0, 1.5, size=len(idx))
    return pd.Series(40.0 + trend + daily + weekly + noise, index=idx, name="value")
