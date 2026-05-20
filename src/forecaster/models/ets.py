"""ETS / Exponential smoothing (statsmodels)."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseForecaster
from .registry import register


@register("ets")
class ETSForecaster(BaseForecaster):
    def __init__(self, trend: str | None = "add", seasonal: str | None = None, seasonal_periods: int | None = None, **hp: Any):
        super().__init__(trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods, **hp)
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods

    def fit(self, series: pd.Series) -> None:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        warnings.filterwarnings("ignore")

        model = ExponentialSmoothing(
            series.values,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            initialization_method="estimated",
        ).fit(optimized=True)
        self._model = model
        self._residuals = np.asarray(model.resid)
        self._fitted = True

    def predict(self, steps: int) -> np.ndarray:
        return np.asarray(self._model.forecast(steps=steps), dtype=float)

    def lookback_required(self) -> int:
        return max(10, 2 * (self.seasonal_periods or 1))
