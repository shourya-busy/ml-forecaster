"""Shared lag / calendar feature builder for tabular ML models.

Models like XGBoost and LightGBM consume tabular features; this module
centralises lag and calendar feature engineering so individual algorithm
files stay tiny.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def build_lag_frame(series: pd.Series, lags: int = 48) -> tuple[pd.DataFrame, pd.Series]:
    """Build X, y matrices with `lags` autoregressive lag features + calendar."""
    df = pd.DataFrame({"y": series.astype(float)})
    for k in range(1, lags + 1):
        df[f"lag_{k}"] = df["y"].shift(k)
    idx = df.index
    df["hour"] = idx.hour
    df["dow"] = idx.dayofweek
    df["minute"] = idx.minute
    df = df.dropna()
    return df.drop(columns=["y"]), df["y"]


def future_index(last_ts: pd.Timestamp, step: pd.Timedelta, steps: int) -> pd.DatetimeIndex:
    return pd.date_range(start=last_ts + step, periods=steps, freq=step)


def recursive_forecast(
    history: pd.Series,
    steps: int,
    lags: int,
    predict_one: Callable[[np.ndarray], float],
    step: pd.Timedelta,
) -> np.ndarray:
    """Recursive one-step-ahead forecasting.

    `predict_one` takes the feature vector [lag_1..lag_n, hour, dow, minute]
    and returns the next predicted value.
    """
    history = history.astype(float).copy()
    preds: list[float] = []
    last_ts = history.index[-1]
    for h in range(1, steps + 1):
        next_ts = last_ts + step * h
        lag_vals = history.iloc[-lags:].to_numpy()[::-1]  # lag_1 first
        if lag_vals.size < lags:
            lag_vals = np.pad(lag_vals, (0, lags - lag_vals.size), constant_values=history.mean())
        feats = np.concatenate([lag_vals, [next_ts.hour, next_ts.dayofweek, next_ts.minute]])
        y_hat = float(predict_one(feats))
        preds.append(y_hat)
        history.loc[next_ts] = y_hat
    return np.asarray(preds, dtype=float)
