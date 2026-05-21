"""Mean forecaster — predicts the historical arithmetic mean."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseForecaster
from .registry import register


@register("mean")
class MeanForecaster(BaseForecaster):
    def fit(self, series: pd.Series) -> None:
        if series.empty:
            raise ValueError("mean: cannot fit on empty series")
        values = series.astype(float).to_numpy()
        self._mean = float(values.mean())
        self._residuals = values - self._mean
        self._fitted = True

    def predict(self, steps: int) -> np.ndarray:
        return np.full(steps, self._mean, dtype=float)
