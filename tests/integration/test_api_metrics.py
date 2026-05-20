"""Hit /metrics on a populated SQLite DB; confirm Prom-format output."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_metrics_endpoint_emits_after_run(tmp_path: Path, monkeypatch):
    pytest.importorskip("statsmodels")
    pytest.importorskip("fastapi")

    db = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("FORECASTER__ARTIFACT_STORE__VOLUME_PATH", str(tmp_path / "artifacts"))
    monkeypatch.setenv("FORECASTER_USE_CUDA", "0")

    from forecaster.config import loader as cfg_loader
    from forecaster.registry.repo import RegistryRepo
    from forecaster.training import pipeline as pl

    cfg_loader._settings = None
    settings = cfg_loader.load_settings()
    settings.algorithms.enabled = ["naive", "seasonal_naive", "ets"]
    settings.algorithms.per_metric = {}  # override so test runs only the fast algos
    settings.training.backtest_folds = 2
    settings.training.parallelism.algos_per_job = 1
    settings.database_url = f"sqlite:///{db}"
    settings.artifact_store.volume_path = str(tmp_path / "artifacts")
    cfg_loader._settings = settings

    repo = RegistryRepo(settings.database_url)
    repo.create_schema()

    from tests.fixtures.synthetic_series import synthetic_series
    series = synthetic_series(days=3, step="5min")

    monkeypatch.setattr(pl, "_fetch_series", lambda **_kw: series)
    pl.run_pipeline(instance="fake-1", metric="cpu", horizon="medium")

    # Reset the cached repo so the new SQLite path is used.
    from forecaster.api import deps
    deps._repo.cache_clear()

    from fastapi.testclient import TestClient

    from forecaster.api.main import create_app

    client = TestClient(create_app())
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "forecast_best_value" in body
    assert "forecast_model_score" in body
    assert "instance=\"fake-1\"" in body
    assert "metric=\"cpu\"" in body

    runs = client.get("/runs").json()
    assert len(runs) == 1
    rankings = client.get("/rankings").json()
    assert rankings[0]["winning_algo"] in {"naive", "seasonal_naive", "ets"}
