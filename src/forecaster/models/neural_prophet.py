"""NeuralProphet — torch-based successor to Prophet with AR-Net.

Differs from Prophet by adding a small neural autoregressive component on
top of the decomposable trend + seasonality. Generally beats Prophet on
series with strong short-term autocorrelation.
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


@register("neural_prophet")
class NeuralProphetForecaster(BaseForecaster):
    def __init__(
        self,
        n_lags: int = 24,
        epochs: int = 30,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
        yearly_seasonality: bool = False,
        **hp: Any,
    ):
        super().__init__(
            n_lags=n_lags, epochs=epochs,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            yearly_seasonality=yearly_seasonality,
            **hp,
        )
        self.n_lags = int(n_lags)
        self.epochs = int(epochs)
        self.weekly = weekly_seasonality
        self.daily = daily_seasonality
        self.yearly = yearly_seasonality

    def fit(self, series: pd.Series) -> None:
        import torch
        from neuralprophet import NeuralProphet, set_log_level

        warnings.filterwarnings("ignore")
        set_log_level("ERROR")
        logging.getLogger("NP").setLevel(logging.ERROR)

        idx = series.index.tz_localize(None) if series.index.tz is not None else series.index
        df = pd.DataFrame({"ds": idx, "y": series.astype(float).values})

        m = NeuralProphet(
            n_lags=self.n_lags,
            n_forecasts=1,           # single-step model; predict() does the recursion
            epochs=self.epochs,
            weekly_seasonality=self.weekly,
            daily_seasonality=self.daily,
            yearly_seasonality=self.yearly,
            quantiles=[0.025, 0.975],
        )
        # PyTorch 2.6+ defaults torch.load(weights_only=True). PL's LR-finder
        # writes a checkpoint mid-fit then loads it back, and that checkpoint
        # contains NeuralProphet config dataclasses (ConfigSeasonality etc.)
        # which aren't in torch's safe-globals allowlist. We're loading a
        # checkpoint we wrote two seconds ago in-process, so weights_only=False
        # is safe. Patch torch.load for the duration of m.fit() only.
        _orig_torch_load = torch.load
        def _trusted_load(*a, **kw):
            kw.setdefault("weights_only", False)
            return _orig_torch_load(*a, **kw)
        torch.load = _trusted_load
        try:
            m.fit(df, freq="auto", progress=None)
        finally:
            torch.load = _orig_torch_load
        self._model = m
        self._train_df = df
        self._last_ts = series.index[-1]
        # Always derive step from observed spacing — pd.infer_freq() can
        # return bare aliases like "min" which pd.Timedelta() rejects on
        # pandas 4.x.
        self._step = pd.Timedelta(series.index[1] - series.index[0])
        # In-sample residuals for the residual-based interval fallback
        try:
            preds = m.predict(df)
            self._residuals = np.asarray(df["y"].values - preds["yhat1"].values, dtype=float)
        except Exception:  # noqa: BLE001
            self._residuals = np.zeros(len(series))
        self._fitted = True

    def _future_frame(self, steps: int) -> pd.DataFrame:
        # NeuralProphet expects a dataframe with `ds` only for the new
        # future window when n_lags > 0; the helper handles AR carry-over.
        future = self._model.make_future_dataframe(
            df=self._train_df, periods=steps, n_historic_predictions=False,
        )
        return future

    def predict(self, steps: int) -> np.ndarray:
        out = self._model.predict(self._future_frame(steps))
        # Pick the yhat for the latest origin; drop NaN rows that
        # NeuralProphet emits for the AR warm-up.
        yhat = out["yhat1"].dropna().to_numpy()[:steps]
        # Pad if we got fewer than steps (rare on short series)
        if len(yhat) < steps:
            yhat = np.concatenate([yhat, np.full(steps - len(yhat), yhat[-1] if len(yhat) else 0.0)])
        return np.asarray(yhat, dtype=float)

    def predict_interval(self, steps: int, alpha: float = 0.05):  # type: ignore[override]
        try:
            out = self._model.predict(self._future_frame(steps))
            lo_col = next((c for c in out.columns if c.startswith("yhat1 ") and "2.5%" in c), None)
            hi_col = next((c for c in out.columns if c.startswith("yhat1 ") and "97.5%" in c), None)
            if lo_col and hi_col:
                lo = out[lo_col].dropna().to_numpy()[:steps]
                hi = out[hi_col].dropna().to_numpy()[:steps]
                if len(lo) >= steps and len(hi) >= steps:
                    return lo.astype(float), hi.astype(float)
        except Exception as exc:  # noqa: BLE001
            log.debug("neuralprophet interval fallback: %s", exc)
        return super().predict_interval(steps, alpha=alpha)

    def lookback_required(self) -> int:
        return max(60, self.n_lags * 4)
