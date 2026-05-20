"""End-to-end pipeline test with SQLite + monkey-patched data source.

Exercises: fetch → backtest → fit → rank → persist → API rendering.
Skips production-style Prometheus/Postgres; both are stubbed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def tmp_env(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("FORECASTER__ARTIFACT_STORE__VOLUME_PATH", str(tmp_path / "artifacts"))
    monkeypatch.setenv("FORECASTER_USE_CUDA", "0")
    # Force fast model subset
    monkeypatch.setattr(
        "forecaster.config.loader._settings", None, raising=False,
    )
    yield tmp_path


def test_pipeline_end_to_end(tmp_env, monkeypatch):
    """Run pipeline against a stubbed data source."""
    pytest.importorskip("statsmodels")

    import pandas as pd

    from forecaster.config import loader as cfg_loader
    from forecaster.data.base import TimeSeries
    from forecaster.registry.repo import RegistryRepo

    # Build a fast settings object with 3 lightweight algos only
    cfg_loader._settings = None
    settings = cfg_loader.load_settings()
    settings.algorithms.enabled = ["naive", "seasonal_naive", "ets"]
    settings.algorithms.per_metric = {}  # override per-metric defaults that include heavy algos
    settings.training.backtest_folds = 2
    settings.horizons["medium"].step = "5min"
    settings.horizons["medium"].horizon = "1h"
    settings.training.parallelism.algos_per_job = 1
    settings.database_url = os.environ["DATABASE_URL"]
    settings.artifact_store.volume_path = str(tmp_env / "artifacts")
    cfg_loader._settings = settings

    # Initialise schema
    repo = RegistryRepo(settings.database_url)
    repo.create_schema()

    # Build synthetic data
    from tests.fixtures.synthetic_series import synthetic_series
    series = synthetic_series(days=3, step="5min")

    # Stub TSDataSource so pipeline._fetch_series gets a series back
    from forecaster.training import pipeline as pl

    def fake_fetch_series(**kwargs):
        return series
    monkeypatch.setattr(pl, "_fetch_series", fake_fetch_series)

    run_id = pl.run_pipeline(instance="fake-1", metric="cpu", horizon="medium")
    assert run_id > 0
    run = repo.get_run(run_id)
    assert run is not None
    assert run.status == "completed", run.error
    rankings = repo.latest_rankings(instance="fake-1", metric="cpu", horizon="medium")
    assert len(rankings) == 1
    forecasts = repo.latest_forecasts(instance="fake-1", metric="cpu", horizon="medium")
    assert len(forecasts) > 0
