"""Shared adapter for Nixtla statsforecast models.

statsforecast's single-series API (`Model().fit(y)` / `Model().predict(h)`)
is similar to our Forecaster protocol but takes raw numpy arrays. This
module is the bridge so each statsforecast wrapper is a 5-line file.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import BaseForecaster


class NixtlaForecasterBase(BaseForecaster):
    """Each concrete subclass overrides `_make_model()` returning a
    statsforecast model instance configured for the requested season.

    All these models are numba-compiled and fit on a 1-D numpy array.
    """

    season_length: int = 1

    def _make_model(self):
        raise NotImplementedError

    def fit(self, series: pd.Series) -> None:
        if series.empty:
            raise ValueError(f"{type(self).__name__}: empty series")
        y = series.astype(float).to_numpy()
        # statsforecast expects season_length <= len(y). Shrink if too short.
        if self.season_length and len(y) < 2 * self.season_length:
            self._effective_season = max(1, len(y) // 2)
        else:
            self._effective_season = self.season_length
        model = self._make_model()
        # The fit/predict API differs slightly between statsforecast versions;
        # both forms accept a numpy array.
        try:
            model = model.fit(y)
        except TypeError:
            model.fit(y)  # in-place fit on some versions
        self._model = model
        # Residuals for the residual-based BaseForecaster interval. Some
        # models expose fitted values via .model_['residuals'] or similar;
        # falling back to a single-step in-sample prediction is robust.
        try:
            fitted = np.asarray(getattr(model, "model_", {}).get("fitted", []))
            if fitted.size == y.size:
                self._residuals = (y - fitted).astype(float)
            else:
                self._residuals = np.diff(y, prepend=y[0]).astype(float)
        except Exception:  # noqa: BLE001
            self._residuals = np.diff(y, prepend=y[0]).astype(float)
        self._fitted = True

    def predict(self, steps: int) -> np.ndarray:
        out = self._model.predict(h=steps)
        # statsforecast returns a dict like {'mean': ndarray, 'lo-95': ndarray, ...}
        if isinstance(out, dict):
            return np.asarray(out.get("mean", list(out.values())[0]), dtype=float)
        # Some models return a numpy array directly
        return np.asarray(out, dtype=float)

    def predict_interval(self, steps: int, alpha: float = 0.05):  # type: ignore[override]
        # statsforecast's predict() supports level= for CIs in many models.
        level = int(round((1 - alpha) * 100))
        try:
            out = self._model.predict(h=steps, level=[level])
        except Exception:  # noqa: BLE001
            return super().predict_interval(steps, alpha=alpha)
        if not isinstance(out, dict):
            return super().predict_interval(steps, alpha=alpha)
        lo_key = next((k for k in out if k.startswith(f"lo-{level}")), None)
        hi_key = next((k for k in out if k.startswith(f"hi-{level}")), None)
        if lo_key and hi_key:
            return (np.asarray(out[lo_key], dtype=float),
                    np.asarray(out[hi_key], dtype=float))
        return super().predict_interval(steps, alpha=alpha)

    def lookback_required(self) -> int:
        return max(20, 2 * (self.season_length or 1))


def _common_init(self, season_length: int, **hp: Any) -> None:  # helper
    BaseForecaster.__init__(self, season_length=season_length, **hp)
    self.season_length = int(season_length)
