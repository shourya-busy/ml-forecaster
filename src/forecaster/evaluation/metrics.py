"""The 5 ranking metrics: MAE, RMSE, MAPE, sMAPE, R².

All metrics accept numpy arrays of equal length. They are robust to NaNs
(rows with any NaN are dropped) and to zero-division (MAPE/sMAPE skip
points where the denominator is ~0).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

_EPS = 1e-9


def _clean(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return y_true[mask], y_pred[mask]


def mae(y_true, y_pred) -> float:
    t, p = _clean(y_true, y_pred)
    if t.size == 0:
        return float("nan")
    return float(np.mean(np.abs(t - p)))


def rmse(y_true, y_pred) -> float:
    t, p = _clean(y_true, y_pred)
    if t.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((t - p) ** 2)))


def mape(y_true, y_pred) -> float:
    t, p = _clean(y_true, y_pred)
    if t.size == 0:
        return float("nan")
    mask = np.abs(t) > _EPS
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((t[mask] - p[mask]) / t[mask]))) * 100.0


def smape(y_true, y_pred) -> float:
    t, p = _clean(y_true, y_pred)
    if t.size == 0:
        return float("nan")
    denom = (np.abs(t) + np.abs(p)) / 2 + _EPS
    return float(np.mean(np.abs(t - p) / denom)) * 100.0


def r2(y_true, y_pred) -> float:
    t, p = _clean(y_true, y_pred)
    if t.size < 2:
        return float("nan")
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    if ss_tot < _EPS:
        return float("nan")
    return 1.0 - ss_res / ss_tot


METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "mae": mae,
    "rmse": rmse,
    "mape": mape,
    "smape": smape,
    "r2": r2,
}


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {name: fn(y_true, y_pred) for name, fn in METRICS.items()}
