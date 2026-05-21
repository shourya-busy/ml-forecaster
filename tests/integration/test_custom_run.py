"""Tests for the Custom Run panel — save / load / run / delete + pipeline overrides."""

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


# ---- repo CRUD ---------------------------------------------------------

def test_custom_config_save_and_load(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    cfg = repo.upsert_custom_config(
        name="cpu_lstm_only", instance="host-1", metric="cpu", horizon="medium",
        algorithms=["lstm", "nbeats"],
        anomaly_filter={"enabled": True, "contamination": 0.05, "window": 1},
        note="benchmark deep models",
    )
    assert cfg.id is not None
    fetched = repo.get_custom_config(cfg.id)
    assert fetched.algorithms == ["lstm", "nbeats"]
    assert fetched.anomaly_filter["enabled"] is True
    by_name = repo.get_custom_config_by_name("cpu_lstm_only")
    assert by_name.id == cfg.id


def test_custom_config_upsert_replaces_by_name(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    repo.upsert_custom_config(
        name="probe", instance="a", metric="cpu", horizon="short", algorithms=["naive"],
    )
    repo.upsert_custom_config(
        name="probe", instance="b", metric="mem", horizon="long", algorithms=["ets"],
    )
    rows = repo.list_custom_configs()
    assert len(rows) == 1
    assert rows[0].instance == "b"
    assert rows[0].algorithms == ["ets"]


def test_custom_config_touch_increments_run_count(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    cfg = repo.upsert_custom_config(
        name="x", instance="a", metric="cpu", horizon="medium",
    )
    assert repo.get_custom_config(cfg.id).run_count == 0
    repo.touch_custom_config(cfg.id)
    repo.touch_custom_config(cfg.id)
    assert repo.get_custom_config(cfg.id).run_count == 2
    assert repo.get_custom_config(cfg.id).last_used_at is not None


def test_custom_config_delete(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    cfg = repo.upsert_custom_config(name="x", instance="a", metric="cpu", horizon="short")
    assert repo.delete_custom_config(cfg.id) is True
    assert repo.get_custom_config(cfg.id) is None
    assert repo.delete_custom_config(cfg.id) is False  # idempotent


# ---- UI ---------------------------------------------------------------

def test_custom_run_page_renders(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    client = _client()
    r = client.get("/ui/custom-run")
    assert r.status_code == 200
    body = r.text
    assert "Custom Run" in body
    # All 4 sections present
    assert "Target" in body and "Algorithms" in body
    assert "Outlier filtering" in body and "Save (optional)" in body
    assert "Saved configurations" in body
    assert "Active runs" in body


def test_custom_run_save_creates_row(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    client = _client()
    r = client.post("/ui/custom-run/save", data={
        "name": "cpu_lstm_only",
        "instance": "host-1",
        "metric": "cpu",
        "horizon": "medium",
        "algorithms": ["lstm", "nbeats"],
        "anomaly_enabled": "on",
        "anomaly_contamination": "0.05",
        "anomaly_window": "1",
        "note": "test",
    })
    assert r.status_code in (302, 303, 307)
    cfg = repo.get_custom_config_by_name("cpu_lstm_only")
    assert cfg is not None
    assert cfg.algorithms == ["lstm", "nbeats"]
    assert cfg.anomaly_filter["enabled"] is True
    assert abs(cfg.anomaly_filter["contamination"] - 0.05) < 1e-9


def test_custom_run_run_dispatches_with_overrides(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    enqueued: list[dict] = []

    # Stub apply_async — capture args/kwargs
    class _Stub:
        def apply_async(self, *args, **kwargs):
            enqueued.append({"args": args, "kwargs": kwargs})
            class _Res: id = "fake-task-id"
            return _Res()
    monkeypatch.setattr("forecaster.training.tasks.train_task", _Stub())

    client = _client()
    r = client.post("/ui/custom-run/run", data={
        "instance": "host-1",
        "metric": "cpu",
        "horizon": "short",
        "algorithms": ["naive", "ets"],
        "anomaly_enabled": "on",
        "anomaly_contamination": "0.10",
        "anomaly_window": "5",
    })
    assert r.status_code in (302, 303, 307)
    assert len(enqueued) == 1
    call = enqueued[0]
    # The train_task is invoked with (instance, metric, horizon) and overrides kw
    assert call["kwargs"]["args"] == ["host-1", "cpu", "short"]
    ov = call["kwargs"]["kwargs"]["overrides"]
    assert ov["algorithms"] == ["naive", "ets"]
    assert ov["anomaly_filter"]["enabled"] is True
    assert ov["anomaly_filter"]["contamination"] == 0.10
    assert ov["anomaly_filter"]["window"] == 5


def test_custom_run_run_saved_increments_touch(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    cfg = repo.upsert_custom_config(
        name="probe", instance="host-1", metric="cpu", horizon="medium",
        algorithms=["ets"],
        anomaly_filter={"enabled": False, "contamination": None, "window": None},
    )
    class _Stub:
        def apply_async(self, *args, **kwargs):
            class _Res: id = "t"
            return _Res()
    monkeypatch.setattr("forecaster.training.tasks.train_task", _Stub())

    client = _client()
    r = client.post(f"/ui/custom-run/run-saved/{cfg.id}")
    assert r.status_code in (302, 303, 307)
    assert repo.get_custom_config(cfg.id).run_count == 1
    assert repo.get_custom_config(cfg.id).last_used_at is not None


def test_custom_run_delete_route(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    cfg = repo.upsert_custom_config(name="x", instance="a", metric="cpu", horizon="short")
    client = _client()
    r = client.post(f"/ui/custom-run/delete/{cfg.id}")
    assert r.status_code in (302, 303, 307)
    assert repo.get_custom_config(cfg.id) is None


def test_custom_run_validation_rejects_missing_target(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    client = _client()
    r = client.post("/ui/custom-run/run", data={"instance": "", "metric": "", "horizon": ""})
    assert r.status_code == 400


def test_nav_includes_custom_run(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    client = _client()
    body = client.get("/ui/").text
    assert "Custom Run" in body


# ---- Pipeline override semantics -------------------------------------

def test_pipeline_overrides_used_in_config_snapshot(tmp_path, monkeypatch):
    """run_pipeline must record overrides into the config_snapshot."""
    repo = _setup(tmp_path, monkeypatch)
    from forecaster.config import loader as cfg_loader
    cfg_loader._settings = None
    cfg_loader._settings_loaded_at = 0.0
    settings = cfg_loader.load_settings()
    settings.algorithms.enabled = ["naive", "ets"]
    settings.algorithms.per_metric = {}
    settings.database_url = repo.engine.url.render_as_string(hide_password=False)
    settings.artifact_store.volume_path = str(tmp_path / "art")
    cfg_loader._settings = settings

    # Stub the data fetch so we don't talk to Prometheus
    from tests.fixtures.synthetic_series import synthetic_series
    series = synthetic_series(days=3, step="5min")
    from forecaster.training import pipeline as pl
    monkeypatch.setattr(pl, "_fetch_series", lambda **kw: series)

    run_id = pl.run_pipeline(
        instance="fake-1", metric="cpu", horizon="medium",
        overrides={"algorithms": ["naive"], "anomaly_filter": {"enabled": False}},
    )
    run = repo.get_run(run_id)
    assert run.status == "completed"
    ov = run.config_snapshot["overrides"]
    assert ov["algorithms"] == ["naive"]
    assert ov["anomaly_filter"]["enabled"] is False
    # The override list must have trained only naive
    with repo.session() as s:
        from forecaster.registry.models import RunMetric
        from sqlalchemy import select as _select
        algos = {m.algo for m in s.scalars(
            _select(RunMetric).where(RunMetric.run_id == run_id).where(RunMetric.fold == -1)
        )}
    assert algos == {"naive"}
