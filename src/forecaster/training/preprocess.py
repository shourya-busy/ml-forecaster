"""Training-time series preprocessing.

Currently a single step — outlier removal via Isolation Forest — but the
shape is designed so additional cleaners can be layered in later (e.g.
imputation, detrending, robust scaling) without touching the pipeline.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config.schema import AnomalyFilter

log = logging.getLogger(__name__)


def _build_windows(values: np.ndarray, window: int) -> np.ndarray:
    """Slide a `window`-sized vector over the series; each point becomes
    the lagging window ending at it (the head is left-padded with the
    first value so every original point has a feature vector)."""
    if window <= 1:
        return values.reshape(-1, 1)
    padded = np.concatenate([np.full(window - 1, values[0]), values])
    return np.stack(
        [padded[i: i + window] for i in range(len(values))]
    )


def remove_anomalies(series: pd.Series, cfg: AnomalyFilter) -> tuple[pd.Series, int]:
    """Drop outliers detected by IsolationForest on lag windows.

    Returns (cleaned_series, n_dropped). Failures (e.g. sklearn not
    available, series too short) return the series unchanged with 0.
    """
    if not cfg.enabled or len(series) < max(20, 2 * cfg.window):
        return series, 0
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        log.warning("anomaly_filter enabled but sklearn missing; skipping")
        return series, 0

    values = series.astype(float).to_numpy()
    if not np.isfinite(values).all():
        return series, 0
    X = _build_windows(values, cfg.window)
    try:
        model = IsolationForest(
            contamination=float(cfg.contamination),
            random_state=0, n_jobs=1,
        )
        labels = model.fit_predict(X)   # +1 inlier, -1 outlier
    except Exception as exc:  # noqa: BLE001
        log.warning("isolation_forest fit failed: %s — skipping", exc)
        return series, 0

    keep = labels == 1
    n_dropped = int((~keep).sum())
    if n_dropped == 0:
        return series, 0
    cleaned = series.iloc[keep]
    log.info("anomaly_filter: dropped %d / %d points (%.2f%%)",
             n_dropped, len(series), 100.0 * n_dropped / len(series))
    return cleaned, n_dropped
