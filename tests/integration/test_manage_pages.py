"""Tests for the new Schedule / Manage / Compare / Trends pages."""

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


def test_schedule_page_renders(tmp_path: Path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    client = _client()
    r = client.get("/ui/schedule")
    assert r.status_code == 200
    body = r.text
    assert "Per-horizon retrain cadences" in body
    # Every configured horizon should appear
    assert "short" in body and "medium" in body and "long" in body
    # Cron quick reference should render
    assert "*/15 * * * *" in body


def test_schedule_save_horizon(tmp_path: Path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    client = _client()
    r = client.post("/ui/schedule/horizon",
                    data={"horizon": "medium", "retrain": "*/30 * * * *"})
    assert r.status_code in (302, 303, 307)
    assert "training.lookback_days" not in repo.get_all_settings_overrides()
    assert repo.get_all_settings_overrides().get("horizons.medium.retrain") == "*/30 * * * *"


def test_schedule_invalid_cron_rejected(tmp_path: Path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    client = _client()
    r = client.post("/ui/schedule/horizon",
                    data={"horizon": "medium", "retrain": "not a cron"})
    assert r.status_code == 400


def test_manage_targets_save_and_disable(tmp_path: Path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    client = _client()
    r = client.post("/ui/manage/targets/save", data={
        "instance": "host-1", "metric": "cpu", "horizon": "medium",
        # 'enabled' field omitted = unchecked
        "schedule_cron": "", "note": "drift suspect",
    })
    assert r.status_code in (302, 303, 307)
    ov = repo.get_target_overrides_map().get(("host-1", "cpu", "medium"))
    assert ov is not None
    assert ov["enabled"] is False
    assert ov["note"] == "drift suspect"


def test_manage_targets_per_target_cron(tmp_path: Path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    client = _client()
    r = client.post("/ui/manage/targets/save", data={
        "instance": "host-1", "metric": "cpu", "horizon": "medium",
        "enabled": "on", "schedule_cron": "*/10 * * * *", "note": "",
    })
    assert r.status_code in (302, 303, 307)
    ov = repo.get_target_overrides_map().get(("host-1", "cpu", "medium"))
    assert ov["schedule_cron"] == "*/10 * * * *"
    assert ov["enabled"] is True


def test_manage_metrics_save_and_delete(tmp_path: Path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    client = _client()
    # Add a new metric
    r = client.post("/ui/manage/metrics/save",
                    data={"name": "load1", "query": "avg by (instance) (node_load1)"})
    assert r.status_code in (302, 303, 307)
    assert repo.get_all_settings_overrides().get("metrics_to_forecast.queries.load1") == \
           "avg by (instance) (node_load1)"
    # Delete
    r = client.post("/ui/manage/metrics/delete", data={"name": "load1"})
    assert r.status_code in (302, 303, 307)
    assert "metrics_to_forecast.queries.load1" not in repo.get_all_settings_overrides()


def test_manage_metrics_dot_in_name_rejected(tmp_path: Path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    client = _client()
    r = client.post("/ui/manage/metrics/save",
                    data={"name": "bad.name", "query": "up"})
    assert r.status_code == 400


def test_manage_training_save_weights_and_limits(tmp_path: Path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    client = _client()
    r = client.post("/ui/manage/training/save", data={
        "lookback_days": "14", "backtest_folds": "3",
        "workers": "", "algos_per_job": "", "confidence_alpha": "",
        "weight_mae": "0.4", "weight_rmse": "0.3",
        "weight_mape": "0.1", "weight_smape": "0.1", "weight_r2": "0.1",
        "enabled_algos": ["naive", "ets"],
    })
    assert r.status_code in (302, 303, 307)
    ov = repo.get_all_settings_overrides()
    assert ov.get("training.lookback_days") == 14
    assert ov.get("training.backtest_folds") == 3
    assert abs(ov.get("ranking.weights.mae") - 0.4) < 1e-9
    assert ov.get("algorithms.enabled") == ["naive", "ets"]


def test_compare_page_empty(tmp_path: Path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    client = _client()
    r = client.get("/ui/compare")
    assert r.status_code == 200
    assert "Pick A and B" in r.text


def test_trends_page_renders(tmp_path: Path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    client = _client()
    r = client.get("/ui/trends")
    assert r.status_code == 200
    body = r.text
    assert "Mean forecast curve" in body or "Trends" in body
    # Metric selector populated from settings
    assert "cpu" in body


def test_nav_shows_new_tabs(tmp_path: Path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    client = _client()
    body = client.get("/ui/").text
    for tab in ["Schedule", "Manage", "Compare", "Trends"]:
        assert tab in body
