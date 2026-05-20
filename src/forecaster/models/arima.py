"""ARIMA via statsmodels.

Uses a small grid search over (p,d,q) when pmdarima isn't available;
otherwise auto_arima.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseForecaster
from .registry import register

log = logging.getLogger(__name__)


@register("arima")
class ARIMAForecaster(BaseForecaster):
    def __init__(self, max_p: int = 3, max_q: int = 3, seasonal: bool = False, **hp: Any):
        super().__init__(max_p=max_p, max_q=max_q, seasonal=seasonal, **hp)
        self.max_p = int(max_p)
        self.max_q = int(max_q)
        self.seasonal = bool(seasonal)

    def fit(self, series: pd.Series) -> None:
        from statsmodels.tsa.arima.model import ARIMA  # local import: heavy
        warnings.filterwarnings("ignore")

        try:
            import pmdarima as pm  # type: ignore[import-not-found]

            model = pm.auto_arima(
                series.values,
                seasonal=self.seasonal,
                stepwise=True,
                suppress_warnings=True,
                max_p=self.max_p,
                max_q=self.max_q,
                error_action="ignore",
            )
            self._model = model
            self._kind = "pmdarima"
            self._residuals = np.asarray(model.resid())
        except Exception as exc:  # noqa: BLE001 - fall back to small grid
            log.debug("pmdarima unavailable, using grid search: %s", exc)
            best = None
            best_aic = np.inf
            for p in range(self.max_p + 1):
                for q in range(self.max_q + 1):
                    if p == 0 and q == 0:
                        continue
                    try:
                        m = ARIMA(series.values, order=(p, 1, q)).fit()
                        if m.aic < best_aic:
                            best_aic = m.aic
                            best = m
                    except Exception:  # noqa: BLE001
                        continue
            if best is None:
                raise RuntimeError("ARIMA: no candidate fit") from None
            self._model = best
            self._kind = "statsmodels"
            self._residuals = np.asarray(best.resid)
        self._fitted = True

    def predict(self, steps: int) -> np.ndarray:
        if self._kind == "pmdarima":
            return np.asarray(self._model.predict(n_periods=steps), dtype=float)
        return np.asarray(self._model.forecast(steps=steps), dtype=float)

    def predict_interval(
        self, steps: int, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._kind == "pmdarima":
            point, ci = self._model.predict(n_periods=steps, return_conf_int=True, alpha=alpha)
            return np.asarray(ci[:, 0]), np.asarray(ci[:, 1])
        fc = self._model.get_forecast(steps=steps)
        ci = fc.conf_int(alpha=alpha)
        # ci can be a DataFrame or ndarray depending on the version.
        if hasattr(ci, "values"):
            ci = ci.values
        return np.asarray(ci[:, 0]), np.asarray(ci[:, 1])

    def lookback_required(self) -> int:
        return max(50, (self.max_p + self.max_q) * 5)
