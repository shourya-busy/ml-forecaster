"""Drift forecaster: random walk with a constant slope estimated from the
endpoints of the training series (y_n - y_1)/(n-1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseForecaster
from .registry import register


@register("drift")
class DriftForecaster(BaseForecaster):
    def fit(self, series: pd.Series) -> None:
        if series.empty:
            raise ValueError("drift: cannot fit on empty series")
        values = series.astype(float).to_numpy()
        self._last = float(values[-1])
        self._slope = (values[-1] - values[0]) / max(1, len(values) - 1)
        diffs = series.diff().dropna().to_numpy()
        self._residuals = diffs if diffs.size else np.array([0.0])
        self._fitted = True

    def predict(self, steps: int) -> np.ndarray:
        return np.array(
            [self._last + (i + 1) * self._slope for i in range(steps)],
            dtype=float,
        )
