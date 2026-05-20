"""Holt-Winters triple exponential smoothing."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseForecaster
from .registry import register


@register("holt_winters")
class HoltWintersForecaster(BaseForecaster):
    def __init__(self, seasonal_periods: int = 288, trend: str = "add", seasonal: str = "add", **hp: Any):
        super().__init__(seasonal_periods=seasonal_periods, trend=trend, seasonal=seasonal, **hp)
        self.seasonal_periods = int(seasonal_periods)
        self.trend = trend
        self.seasonal = seasonal

    def fit(self, series: pd.Series) -> None:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        warnings.filterwarnings("ignore")

        # Holt-Winters seasonal requires at least 2 full seasons.
        sp = self.seasonal_periods
        if len(series) < 2 * sp:
            sp = max(1, len(series) // 2)
        try:
            model = ExponentialSmoothing(
                series.values,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=sp,
                initialization_method="estimated",
            ).fit(optimized=True)
        except Exception:
            # Fall back to non-seasonal smoothing on short series.
            model = ExponentialSmoothing(
                series.values,
                trend=self.trend,
                seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True)
        self._model = model
        self._residuals = np.asarray(model.resid)
        self._fitted = True

    def predict(self, steps: int) -> np.ndarray:
        return np.asarray(self._model.forecast(steps=steps), dtype=float)

    def lookback_required(self) -> int:
        return 2 * self.seasonal_periods
