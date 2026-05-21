"""K-Nearest-Neighbors regression on autoregressive lag features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..features.lag_features import build_lag_frame, recursive_forecast
from .base import BaseForecaster
from .registry import register


@register("knn")
class KNNForecaster(BaseForecaster):
    """KNN finds historical windows that look like the current one and
    averages their next-step values. Non-parametric; surprisingly strong
    when the series has repeating local motifs."""

    def __init__(self, n_neighbors: int = 5, lags: int = 48, **hp: Any) -> None:
        super().__init__(n_neighbors=n_neighbors, lags=lags, **hp)
        self.n_neighbors = int(n_neighbors)
        self.lags = int(lags)

    def fit(self, series: pd.Series) -> None:
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.preprocessing import StandardScaler

        X, y = build_lag_frame(series, self.lags)
        if X.empty:
            raise ValueError(f"knn: need >{self.lags} points")
        scaler = StandardScaler().fit(X.values)
        Xs = scaler.transform(X.values)
        model = KNeighborsRegressor(
            n_neighbors=min(self.n_neighbors, len(Xs)),
            weights="distance",
        )
        model.fit(Xs, y.values)
        self._scaler = scaler
        self._model = model
        self._history = series.astype(float).copy()
        self._step = self._history.index[1] - self._history.index[0]
        self._residuals = (y.values - model.predict(Xs)).astype(float)
        self._fitted = True

    def predict(self, steps: int) -> np.ndarray:
        def _one(f: np.ndarray) -> float:
            fs = self._scaler.transform(f.reshape(1, -1))
            return float(self._model.predict(fs)[0])
        return recursive_forecast(
            self._history, steps, self.lags, _one, self._step,
        )

    def lookback_required(self) -> int:
        return max(self.lags * 2, self.n_neighbors * 2)
