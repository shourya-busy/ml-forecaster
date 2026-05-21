"""Tests for the expanded algorithm library + grouped Manage → Training UI."""

from __future__ import annotations

from pathlib import Path

import pytest


def _setup(tmp_path: Path, monkeypatch):
    pytest.importorskip("fastapi")
    db = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("FORECASTER_OVERRIDE_TTL", "0")
    monkeypatch.setenv("FORECASTER__ARTIFACT_STORE__VOLUME_PATH", str(tmp_path / "art"))
    from forecaster.config import loader as cfg_loader
    cfg_loader._settings = None
    cfg_loader._settings_loaded_at = 0.0
    from forecaster.registry.repo import RegistryRepo
    repo = RegistryRepo(f"sqlite:///{db}")
    repo.create_schema()
    return repo


def _client():
    from forecaster.api import deps
    deps._repo.cache_clear()
    from fastapi.testclient import TestClient
    from forecaster.api.main import create_app
    return TestClient(create_app(), follow_redirects=False)


def test_all_new_algos_registered():
    from forecaster.models import REGISTRY
    new_ones = {"drift", "mean", "median", "theta", "sarima",
                "linear_lag", "random_forest", "knn", "gru"}
    assert new_ones.issubset(set(REGISTRY))
    # And the original 10 still there
    originals = {"naive", "seasonal_naive", "arima", "ets", "holt_winters",
                 "prophet", "xgboost", "lightgbm", "lstm", "nbeats"}
    assert originals.issubset(set(REGISTRY))


def test_all_algos_have_info():
    from forecaster.models import REGISTRY
    from forecaster.models.registry import algo_info
    for name in REGISTRY:
        info = algo_info(name)
        assert info["family"] in {"baseline", "statistical", "ml", "deep-learning"}
        assert info["description"]
        assert info["when_to_use"]


def test_new_algos_disabled_by_default(tmp_path, monkeypatch):
    """Out of the box, the new algorithms must NOT appear in `enabled`."""
    _setup(tmp_path, monkeypatch)
    from forecaster.config import loader as cfg_loader
    cfg_loader._settings = None
    settings = cfg_loader.load_settings()
    new_ones = {"drift", "mean", "median", "theta", "sarima",
                "linear_lag", "random_forest", "knn", "gru"}
    enabled = set(settings.algorithms.enabled)
    assert enabled & new_ones == set(), \
        f"new algos should be disabled by default, but found: {enabled & new_ones}"


def test_manage_training_groups_by_family(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    client = _client()
    body = client.get("/ui/manage/training").text
    # Section headers present
    assert "Baseline" in body
    assert "Statistical" in body
    assert "Machine learning" in body
    assert "Deep learning" in body
    # New algorithms present somewhere in the page
    for algo in ["drift", "mean", "median", "theta", "linear_lag",
                 "random_forest", "knn", "gru", "sarima"]:
        assert algo in body, f"missing card for {algo}"
    # Library counter is rendered
    assert "enabled)" in body


def test_manage_training_enable_new_algo(tmp_path, monkeypatch):
    """A user can enable a previously-disabled algo from the UI."""
    repo = _setup(tmp_path, monkeypatch)
    client = _client()
    # Enable a mix across categories: an existing one + a new baseline +
    # a new ML model. This is the cross-category selection the user asked for.
    r = client.post("/ui/manage/training/save", data={
        "lookback_days": "", "backtest_folds": "",
        "workers": "", "algos_per_job": "", "confidence_alpha": "",
        "weight_mae": "", "weight_rmse": "",
        "weight_mape": "", "weight_smape": "", "weight_r2": "",
        "enabled_algos": ["naive", "drift", "linear_lag"],
    })
    assert r.status_code in (302, 303, 307)
    ov = repo.get_all_settings_overrides()
    assert ov["algorithms.enabled"] == ["naive", "drift", "linear_lag"]


def test_smoke_new_simple_algos():
    """The non-heavy new algos must fit + predict on synthetic data."""
    from tests.fixtures.synthetic_series import synthetic_series
    from forecaster.models import build
    import numpy as np
    s = synthetic_series(days=3, step="15min")
    for algo in ["drift", "mean", "median"]:
        m = build(algo)
        m.fit(s)
        pred = m.predict(12)
        assert pred.shape == (12,)
        assert np.isfinite(pred).all()


def test_smoke_new_statsmodels_algos():
    pytest.importorskip("statsmodels")
    from tests.fixtures.synthetic_series import synthetic_series
    from forecaster.models import build
    import numpy as np
    s = synthetic_series(days=5, step="1h")
    for algo in ["theta"]:
        m = build(algo)
        m.fit(s)
        pred = m.predict(24)
        assert pred.shape == (24,)
        assert np.isfinite(pred).all()


def test_smoke_new_sklearn_algos():
    pytest.importorskip("sklearn")
    from tests.fixtures.synthetic_series import synthetic_series
    from forecaster.models import build
    import numpy as np
    s = synthetic_series(days=4, step="15min")
    for algo in ["linear_lag", "random_forest", "knn"]:
        m = build(algo, lags=12)
        m.fit(s)
        pred = m.predict(8)
        assert pred.shape == (8,)
        assert np.isfinite(pred).all()
