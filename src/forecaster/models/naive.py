"""Naive last-value forecaster."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseForecaster
from .registry import register


@register("naive")
class NaiveForecaster(BaseForecaster):
    """Predict the last observed value for all future steps."""

    def fit(self, series: pd.Series) -> None:
        if series.empty:
            raise ValueError("naive: cannot fit on empty series")
        self._last = float(series.iloc[-1])
        # Residuals = lag-1 differences for confidence bands.
        diffs = series.diff().dropna().to_numpy()
        self._residuals = diffs if diffs.size else np.array([0.0])
        self._fitted = True

    def predict(self, steps: int) -> np.ndarray:
        return np.full(steps, self._last, dtype=float)
