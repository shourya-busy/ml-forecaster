"""Synthetic time-series generator with trend, daily/weekly seasonality + noise."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def synthetic_series(
    days: int = 30,
    step: str = "5min",
    start: pd.Timestamp | None = None,
    *,
    trend_per_day: float = 0.5,
    daily_amp: float = 8.0,
    weekly_amp: float = 4.0,
    noise_sigma: float = 1.5,
    base: float = 40.0,
    seed: int = 0,
) -> pd.Series:
    """Return a pd.Series indexed by tz-aware UTC timestamps."""
    rng = np.random.default_rng(seed)
    start = start or (pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(days=days))
    idx = pd.date_range(start=start, periods=int(days * 24 * 60 / pd.Timedelta(step).total_seconds() * 60), freq=step, tz="UTC")
    t = np.arange(len(idx), dtype=float)
    step_per_day = pd.Timedelta("1D") / pd.Timedelta(step)
    daily = daily_amp * np.sin(2 * math.pi * t / step_per_day)
    weekly = weekly_amp * np.sin(2 * math.pi * t / (step_per_day * 7))
    trend = (t / step_per_day) * trend_per_day
    noise = rng.normal(0, noise_sigma, size=len(idx))
    values = base + trend + daily + weekly + noise
    return pd.Series(values, index=idx, name="value")
