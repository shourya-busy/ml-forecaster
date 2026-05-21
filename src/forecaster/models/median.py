"""Median forecaster — predicts the historical median (robust to outliers)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseForecaster
from .registry import register


@register("median")
class MedianForecaster(BaseForecaster):
    def fit(self, series: pd.Series) -> None:
        if series.empty:
            raise ValueError("median: cannot fit on empty series")
        values = series.astype(float).to_numpy()
        self._median = float(np.median(values))
        self._residuals = values - self._median
        self._fitted = True

    def predict(self, steps: int) -> np.ndarray:
        return np.full(steps, self._median, dtype=float)
