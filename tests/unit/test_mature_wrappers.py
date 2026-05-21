"""Spot-checks for the Nixtla statsforecast + NeuralProphet wrappers.

Each test is dep-gated: if the third-party library isn't installed in
the current environment, the test is skipped (the worker Docker image
always has them).
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from tests.fixtures.synthetic_series import synthetic_series


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def test_new_wrappers_registered():
    from forecaster.models import REGISTRY
    for name in ["auto_arima", "auto_ets", "auto_theta", "mstl", "neural_prophet"]:
        assert name in REGISTRY


def test_new_wrappers_have_info():
    from forecaster.models.registry import algo_info
    for name in ["auto_arima", "auto_ets", "auto_theta", "mstl", "neural_prophet"]:
        info = algo_info(name)
        assert info["description"]
        assert info["family"] in {"statistical", "deep-learning"}


def test_new_wrappers_disabled_by_default():
    """The 5 new wrappers must NOT be in default.yaml's `algorithms.enabled`."""
    import os
    from pathlib import Path
    from forecaster.config.loader import load_settings

    config_dir = Path(__file__).resolve().parents[2] / "config"
    os.environ.pop("FORECASTER__ALGORITHMS__ENABLED", None)
    settings = load_settings(config_dir, with_db_overrides=False)
    new_ones = {"auto_arima", "auto_ets", "auto_theta", "mstl", "neural_prophet"}
    assert set(settings.algorithms.enabled) & new_ones == set()


@pytest.mark.skipif(not _have("statsforecast"),
                    reason="statsforecast not installed in test env")
@pytest.mark.parametrize("algo", ["auto_arima", "auto_ets", "auto_theta", "mstl"])
def test_statsforecast_wrapper_smokes(algo: str):
    from forecaster.models import build
    # Smaller series and shorter season for speed
    s = synthetic_series(days=4, step="1h")
    m = build(algo, season_length=24)
    m.fit(s)
    out = m.predict(12)
    assert out.shape == (12,)
    assert np.isfinite(out).all()


@pytest.mark.skipif(not _have("neuralprophet"),
                    reason="neuralprophet not installed in test env")
def test_neural_prophet_smoke():
    from forecaster.models import build
    s = synthetic_series(days=5, step="1h")
    m = build("neural_prophet", n_lags=8, epochs=5)
    m.fit(s)
    out = m.predict(12)
    assert out.shape == (12,)
    assert np.isfinite(out).all()
