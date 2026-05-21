"""Theta method (Assimakopoulos & Nikolopoulos, 2000) — strong M3 baseline."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseForecaster
from .registry import register


@register("theta")
class ThetaForecaster(BaseForecaster):
    """Classical Theta. Period is inferred from the series frequency."""

    def __init__(self, period: int | None = None, **hp: Any) -> None:
        super().__init__(period=period, **hp)
        self.period = period

    def fit(self, series: pd.Series) -> None:
        from statsmodels.tsa.forecasting.theta import ThetaModel

        warnings.filterwarnings("ignore")
        period = self.period
        # Theta needs a season; for our 5-min step that's 288 (daily).
        # Fall back to non-seasonal if the series is too short.
        if period and len(series) < 2 * period:
            period = None
        # Pass raw values to bypass statsmodels' freq-inference on pandas
        # DatetimeIndex (it raises on minute-resolution aliases like
        # "15min"). When period is unknown we disable deseasonalization
        # rather than guessing wrong.
        values = series.astype(float).values
        if period is None:
            model = ThetaModel(values, period=1, deseasonalize=False).fit()
        else:
            model = ThetaModel(values, period=period).fit()
        self._model = model
        try:
            self._residuals = np.asarray(values - model.fittedvalues, dtype=float)
        except Exception:  # noqa: BLE001
            self._residuals = np.zeros(len(series))
        self._fitted = True

    def predict(self, steps: int) -> np.ndarray:
        return np.asarray(self._model.forecast(steps), dtype=float)

    def lookback_required(self) -> int:
        return max(20, 2 * (self.period or 1))
