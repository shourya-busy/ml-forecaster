"""Tests for the training-data footprint feature."""

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


def test_pipeline_records_data_stats(tmp_path, monkeypatch):
    """run_pipeline must write data_stats with sane fields."""
    pytest.importorskip("statsmodels")
    repo = _setup(tmp_path, monkeypatch)
    from forecaster.config import loader as cfg_loader
    cfg_loader._settings = None
    settings = cfg_loader.load_settings()
    settings.algorithms.enabled = ["naive", "ets"]
    settings.algorithms.per_metric = {}
    settings.training.parallelism.algos_per_job = 1
    settings.database_url = repo.engine.url.render_as_string(hide_password=False)
    settings.artifact_store.volume_path = str(tmp_path / "art")
    cfg_loader._settings = settings

    # Stub Prometheus fetch
    from tests.fixtures.synthetic_series import synthetic_series
    series = synthetic_series(days=3, step="5min")
    from forecaster.training import pipeline as pl
    monkeypatch.setattr(pl, "_fetch_series", lambda **kw: series)

    run_id = pl.run_pipeline(instance="fake-1", metric="cpu", horizon="medium")
    run = repo.get_run(run_id)
    assert run.status == "completed"
    ds = run.data_stats
    assert ds is not None
    # Key shape
    for key in ["step", "lookback_days", "fetched_points", "used_points",
                "dropped_by_filter", "first_ts", "last_ts", "span_seconds",
                "anomaly_filter_applied"]:
        assert key in ds, f"missing key: {key}"
    # Sanity: used_points equals fetched_points when no anomaly filter
    assert ds["used_points"] == ds["fetched_points"]
    assert ds["dropped_by_filter"] == 0
    assert ds["anomaly_filter_applied"] is False
    # 3 days @ 5-min step = 864 points (synthetic_series goes a hair under)
    assert 800 <= ds["fetched_points"] <= 900
    # Span should be roughly 3 days
    assert 2.5 * 86400 <= ds["span_seconds"] <= 3.5 * 86400


def test_pipeline_records_anomaly_drops(tmp_path, monkeypatch):
    """Outlier filter drops show up in data_stats."""
    pytest.importorskip("sklearn")
    pytest.importorskip("statsmodels")
    repo = _setup(tmp_path, monkeypatch)
    from forecaster.config import loader as cfg_loader
    cfg_loader._settings = None
    settings = cfg_loader.load_settings()
    settings.algorithms.enabled = ["naive", "ets"]
    settings.algorithms.per_metric = {}
    settings.training.parallelism.algos_per_job = 1
    settings.training.anomaly_filter.enabled = True
    settings.training.anomaly_filter.contamination = 0.10
    settings.training.anomaly_filter.window = 1
    settings.database_url = repo.engine.url.render_as_string(hide_password=False)
    settings.artifact_store.volume_path = str(tmp_path / "art")
    cfg_loader._settings = settings

    # Inject big spikes so the filter has something to remove
    from tests.fixtures.synthetic_series import synthetic_series
    series = synthetic_series(days=3, step="5min")
    series.iloc[::40] += 50.0   # spike every 40th point

    from forecaster.training import pipeline as pl
    monkeypatch.setattr(pl, "_fetch_series", lambda **kw: series)

    run_id = pl.run_pipeline(instance="fake-1", metric="cpu", horizon="medium")
    run = repo.get_run(run_id)
    ds = run.data_stats
    assert ds is not None
    assert ds["anomaly_filter_applied"] is True
    assert ds["dropped_by_filter"] > 0
    assert ds["used_points"] < ds["fetched_points"]


def test_run_detail_shows_footprint_card(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    run_id = repo.create_run(
        instance="fake-1", metric="cpu", horizon="medium", config_snapshot={},
    )
    repo.record_data_stats(run_id, {
        "step": "5min", "lookback_days": 30,
        "fetched_points": 8640, "used_points": 8632,
        "dropped_by_filter": 8,
        "first_ts": "2026-04-21T10:00:00+00:00",
        "last_ts": "2026-05-21T09:55:00+00:00",
        "span_seconds": 30 * 86400,
        "anomaly_filter_applied": True,
    })
    repo.mark_completed(run_id, duration_seconds=10.0)

    client = _client()
    body = client.get(f"/ui/runs/{run_id}").text
    assert "Training data footprint" in body
    # Numbers we wrote should appear (commas in 8,632)
    assert "8,632" in body
    assert "30d" in body
    assert "8 point" in body or "8 points" in body or "8 dropped" in body


def test_target_detail_shows_latest_footprint(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    run_id = repo.create_run(
        instance="fake-1", metric="cpu", horizon="medium", config_snapshot={},
    )
    repo.mark_completed(run_id, duration_seconds=1.0)
    repo.record_data_stats(run_id, {
        "step": "5min", "lookback_days": 14,
        "fetched_points": 4032, "used_points": 4032,
        "dropped_by_filter": 0,
        "first_ts": "2026-05-07T10:00:00+00:00",
        "last_ts": "2026-05-21T10:00:00+00:00",
        "span_seconds": 14 * 86400,
        "anomaly_filter_applied": False,
    })
    repo.add_ranking(
        run_id=run_id, instance="fake-1", metric="cpu", horizon="medium",
        winning_algo="naive",
        ranked=[{"rank": 1, "algo": "naive", "composite": 0.9,
                 "raw_scores": {}, "normalised_scores": {}}],
    )
    client = _client()
    body = client.get("/ui/targets/fake-1/cpu/medium").text
    assert "Latest training set" in body
    assert "4,032 pts" in body
    assert "14d lookback" in body


def test_schedule_page_shows_training_point_estimate(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    client = _client()
    body = client.get("/ui/schedule").text
    # The new "pts/run/target trained on" footer
    assert "pts/run/target trained on" in body


def test_record_data_stats_overwrites(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    run_id = repo.create_run(
        instance="x", metric="cpu", horizon="medium", config_snapshot={},
    )
    repo.record_data_stats(run_id, {"used_points": 100})
    repo.record_data_stats(run_id, {"used_points": 200})
    assert repo.get_run(run_id).data_stats == {"used_points": 200}
