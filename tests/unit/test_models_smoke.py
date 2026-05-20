"""Smoke test: every registered model can fit + predict on synthetic data.

We use a small series and skip the slow deep-learning models by default
unless RUN_SLOW_MODELS=1 is set. Models whose optional dependency is not
installed in the current environment are skipped.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from forecaster.models import REGISTRY, build
from tests.fixtures.synthetic_series import synthetic_series

SLOW = {"lstm", "nbeats", "prophet"}

# Map algo name -> optional dependency module that must be importable.
OPTIONAL_DEPS = {
    "prophet": "prophet",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "lstm": "torch",
    "nbeats": "torch",
    "arima": "statsmodels",
    "ets": "statsmodels",
    "holt_winters": "statsmodels",
}


def _dep_missing(algo: str) -> bool:
    dep = OPTIONAL_DEPS.get(algo)
    return dep is not None and importlib.util.find_spec(dep) is None


@pytest.mark.parametrize("algo", sorted(REGISTRY.keys()))
def test_model_fit_predict_smoke(algo: str):
    if algo in SLOW and not os.environ.get("RUN_SLOW_MODELS"):
        pytest.skip("slow model — set RUN_SLOW_MODELS=1 to run")
    if _dep_missing(algo):
        pytest.skip(f"optional dep missing for {algo}")
    s = synthetic_series(days=3, step="15min")
    m = build(algo)
    m.fit(s)
    pred = m.predict(24)
    assert pred.shape == (24,)
    assert np.isfinite(pred).all()


@pytest.mark.parametrize("algo", ["naive", "seasonal_naive", "arima", "ets", "holt_winters"])
def test_predict_interval_returns_bounds(algo: str):
    s = synthetic_series(days=3, step="15min")
    m = build(algo)
    m.fit(s)
    lo, hi = m.predict_interval(12, alpha=0.05)
    assert lo.shape == (12,) and hi.shape == (12,)
