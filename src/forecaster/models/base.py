"""Forecaster protocol.

Every algorithm in src/forecaster/models/ implements the same surface:

    fit(series)
    predict(steps) -> point forecasts (np.ndarray of length steps)
    predict_interval(steps, alpha) -> (lower, upper) of length steps
    lookback_required() -> min number of points needed to train
    save(path) / load(path) -> persist artifact

The pipeline only knows about this protocol, so adding an algorithm is a
one-file affair plus a @register("name") decorator.
"""

from __future__ import annotations

import abc
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ForecastResult:
    """Output of a single (algo, instance, metric, horizon) run."""

    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    timestamps: pd.DatetimeIndex

    def as_records(self) -> list[dict[str, Any]]:
        return [
            {
                "timestamp": ts.isoformat(),
                "point": float(self.point[i]),
                "lower": float(self.lower[i]),
                "upper": float(self.upper[i]),
            }
            for i, ts in enumerate(self.timestamps)
        ]


class Forecaster(abc.ABC):
    """Protocol every algorithm must follow.

    Implementations must be picklable so they can round-trip through the
    artifact store via save/load.
    """

    name: ClassVar[str] = "base"

    def __init__(self, **hyperparams: Any) -> None:
        self.hyperparams = hyperparams
        self._fitted: bool = False

    @abc.abstractmethod
    def fit(self, series: pd.Series) -> None:
        """Train the model on a 1-D pd.Series indexed by a DatetimeIndex."""

    @abc.abstractmethod
    def predict(self, steps: int) -> np.ndarray:
        """Return `steps` point forecasts."""

    @abc.abstractmethod
    def predict_interval(
        self, steps: int, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) prediction bands at confidence 1-alpha."""

    def lookback_required(self) -> int:
        """Minimum number of points needed; default 1."""
        return 1

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "Forecaster":
        with Path(path).open("rb") as f:
            return pickle.load(f)  # noqa: S301 - trusted artifacts on local volume


class BaseForecaster(Forecaster):
    """Convenience base with simple residual-based prediction intervals.

    Subclasses set self._residuals during fit() and get free
    `predict_interval` based on a normal-approximation of residual std.
    """

    def __init__(self, **hyperparams: Any) -> None:
        super().__init__(**hyperparams)
        self._residuals: np.ndarray | None = None

    def predict_interval(
        self, steps: int, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        from scipy.stats import norm  # local import: only models need it

        point = self.predict(steps)
        if self._residuals is None or len(self._residuals) < 2:
            spread = np.full(steps, np.nan)
        else:
            sigma = float(np.nanstd(self._residuals, ddof=1))
            z = float(norm.ppf(1 - alpha / 2))
            # Naive widening: variance grows linearly with step (common in
            # simple baselines without a full state-space model).
            spread = z * sigma * np.sqrt(np.arange(1, steps + 1))
        return point - spread, point + spread
