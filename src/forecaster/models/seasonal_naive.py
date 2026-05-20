"""Seasonal naive: y_{t+h} = y_{t+h-S} where S is the season length."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseForecaster
from .registry import register


@register("seasonal_naive")
class SeasonalNaiveForecaster(BaseForecaster):
    def __init__(self, season_length: int = 288, **hp):
        super().__init__(season_length=season_length, **hp)
        self.season_length = int(season_length)

    def fit(self, series: pd.Series) -> None:
        if len(series) < self.season_length + 1:
            # Fall back to last value if not enough history
            self.season_length = max(1, len(series) - 1)
        self._tail = series.iloc[-self.season_length:].to_numpy(dtype=float)
        # Seasonal residuals (y_t - y_{t-S})
        if len(series) > self.season_length:
            diffs = (series.values[self.season_length:] - series.values[:-self.season_length]).astype(float)
            self._residuals = diffs
        else:
            self._residuals = np.array([0.0])
        self._fitted = True

    def predict(self, steps: int) -> np.ndarray:
        n = self.season_length
        return np.array([self._tail[i % n] for i in range(steps)], dtype=float)

    def lookback_required(self) -> int:
        return self.season_length + 1
