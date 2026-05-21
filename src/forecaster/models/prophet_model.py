"""Prophet wrapper."""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseForecaster
from .registry import register

log = logging.getLogger(__name__)


@register("prophet")
class ProphetForecaster(BaseForecaster):
    def __init__(
        self,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
        yearly_seasonality: bool = False,
        **hp: Any,
    ):
        super().__init__(
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            yearly_seasonality=yearly_seasonality,
            **hp,
        )
        self.weekly = weekly_seasonality
        self.daily = daily_seasonality
        self.yearly = yearly_seasonality

    def fit(self, series: pd.Series) -> None:
        from prophet import Prophet  # heavy import
        warnings.filterwarnings("ignore")
        logging.getLogger("prophet").setLevel(logging.ERROR)
        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

        df = pd.DataFrame({"ds": series.index.tz_localize(None), "y": series.values})
        m = Prophet(
            weekly_seasonality=self.weekly,
            daily_seasonality=self.daily,
            yearly_seasonality=self.yearly,
            interval_width=0.95,
        )
        m.fit(df)
        self._model = m
        self._last_ts = series.index[-1]
        # pandas 4.x rejects bare alias strings like "min" / "h" / "D" in
        # pd.Timedelta(), so always compute step from observed spacing.
        self._step = pd.Timedelta(series.index[1] - series.index[0])
        # In-sample residuals for free residual-based bounds (we still
        # prefer Prophet's native intervals in predict_interval).
        in_pred = m.predict(df[["ds"]])["yhat"].values
        self._residuals = (df["y"].values - in_pred).astype(float)
        self._fitted = True

    def _future_frame(self, steps: int) -> pd.DataFrame:
        future = pd.date_range(
            start=self._last_ts + pd.Timedelta(self._step),
            periods=steps,
            freq=self._step,
        ).tz_localize(None)
        return pd.DataFrame({"ds": future})

    def predict(self, steps: int) -> np.ndarray:
        out = self._model.predict(self._future_frame(steps))
        return np.asarray(out["yhat"].values, dtype=float)

    def predict_interval(self, steps: int, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        # Prophet's interval_width is set at fit time; alpha is informational here.
        out = self._model.predict(self._future_frame(steps))
        return (
            np.asarray(out["yhat_lower"].values, dtype=float),
            np.asarray(out["yhat_upper"].values, dtype=float),
        )

    def lookback_required(self) -> int:
        return 30
