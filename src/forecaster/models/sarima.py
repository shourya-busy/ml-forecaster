"""SARIMA (seasonal ARIMA) via statsmodels SARIMAX."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseForecaster
from .registry import register


@register("sarima")
class SARIMAForecaster(BaseForecaster):
    """Seasonal ARIMA. Slower than `arima` but captures seasonality directly."""

    def __init__(
        self,
        order: tuple = (1, 1, 1),
        seasonal_order: tuple = (1, 0, 1, 288),
        **hp: Any,
    ) -> None:
        super().__init__(order=order, seasonal_order=seasonal_order, **hp)
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)

    def fit(self, series: pd.Series) -> None:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        warnings.filterwarnings("ignore")
        # Auto-shrink seasonal period if the series is too short
        p, d, q, s = self.seasonal_order
        if s and len(series) < 2 * s:
            s = max(1, len(series) // 2)
        seasonal_order = (p, d, q, s) if s > 0 else (0, 0, 0, 0)
        model = SARIMAX(
            series.astype(float).values,
            order=self.order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        self._model = model
        self._residuals = np.asarray(model.resid, dtype=float)
        self._fitted = True

    def predict(self, steps: int) -> np.ndarray:
        return np.asarray(self._model.forecast(steps), dtype=float)

    def predict_interval(self, steps: int, alpha: float = 0.05):  # type: ignore[override]
        fc = self._model.get_forecast(steps=steps)
        ci = fc.conf_int(alpha=alpha)
        if hasattr(ci, "values"):
            ci = ci.values
        return np.asarray(ci[:, 0], dtype=float), np.asarray(ci[:, 1], dtype=float)

    def lookback_required(self) -> int:
        s = self.seasonal_order[3]
        return max(50, 2 * s)
