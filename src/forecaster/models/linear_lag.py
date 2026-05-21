"""Ridge regression on autoregressive lag features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..features.lag_features import build_lag_frame, recursive_forecast
from .base import BaseForecaster
from .registry import register


@register("linear_lag")
class LinearLagForecaster(BaseForecaster):
    """L2-regularised linear regression on lag + calendar features.

    Fast, interpretable, hard to over-fit. A solid baseline for the
    "is non-linearity actually helping?" question.
    """

    def __init__(self, lags: int = 48, alpha: float = 1.0, **hp: Any) -> None:
        super().__init__(lags=lags, alpha=alpha, **hp)
        self.lags = int(lags)
        self.alpha = float(alpha)

    def fit(self, series: pd.Series) -> None:
        from sklearn.linear_model import Ridge

        X, y = build_lag_frame(series, self.lags)
        if X.empty:
            raise ValueError(f"linear_lag: need >{self.lags} points, got {len(series)}")
        model = Ridge(alpha=self.alpha)
        model.fit(X.values, y.values)
        self._model = model
        self._history = series.astype(float).copy()
        self._step = self._history.index[1] - self._history.index[0]
        self._residuals = (y.values - model.predict(X.values)).astype(float)
        self._fitted = True

    def predict(self, steps: int) -> np.ndarray:
        return recursive_forecast(
            self._history, steps, self.lags,
            lambda f: float(self._model.predict(f.reshape(1, -1))[0]),
            self._step,
        )

    def lookback_required(self) -> int:
        return self.lags * 2
