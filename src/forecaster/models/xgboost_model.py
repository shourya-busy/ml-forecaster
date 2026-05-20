"""XGBoost on autoregressive lag features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..features.lag_features import build_lag_frame, recursive_forecast
from .base import BaseForecaster
from .registry import register


@register("xgboost")
class XGBoostForecaster(BaseForecaster):
    def __init__(self, n_estimators: int = 200, max_depth: int = 6, lags: int = 48, learning_rate: float = 0.05, **hp: Any):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth, lags=lags, learning_rate=learning_rate, **hp)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.lags = int(lags)
        self.learning_rate = float(learning_rate)

    def fit(self, series: pd.Series) -> None:
        from xgboost import XGBRegressor

        X, y = build_lag_frame(series, self.lags)
        if X.empty:
            raise ValueError(f"xgboost: not enough data ({len(series)}) for lags={self.lags}")
        model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            tree_method="hist",
            n_jobs=1,
            verbosity=0,
        )
        model.fit(X.values, y.values)
        self._model = model
        self._history = series.astype(float).copy()
        self._step = self._history.index[1] - self._history.index[0]
        self._residuals = (y.values - model.predict(X.values)).astype(float)
        self._fitted = True

    def predict(self, steps: int) -> np.ndarray:
        return recursive_forecast(
            self._history,
            steps,
            self.lags,
            lambda f: float(self._model.predict(f.reshape(1, -1))[0]),
            self._step,
        )

    def lookback_required(self) -> int:
        return self.lags * 2
