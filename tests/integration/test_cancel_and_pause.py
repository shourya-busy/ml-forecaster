"""Tests for cancel-run / cancel-active / pause-resume / target lookback."""

from __future__ import annotations

from datetime import datetime, timezone
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


# ---- cancellation ------------------------------------------------------

def test_cancel_run_marks_status(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    # Stub out Celery so we don't try to talk to redis in tests
    monkeypatch.setattr("forecaster.training.tasks.revoke_task", lambda _: True)
    run_id = repo.create_run(
        instance="fake-1", metric="cpu", horizon="medium", config_snapshot={},
    )
    repo.record_celery_task_id(run_id, "fake-task-123")

    client = _client()
    r = client.post(f"/ui/runs/{run_id}/cancel")
    assert r.status_code in (302, 303, 307)
    assert repo.get_run(run_id).status == "cancelled"
    assert repo.get_run(run_id).error == "cancelled via UI"


def test_cancel_already_completed_run_is_noop(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr("forecaster.training.tasks.revoke_task", lambda _: True)
    run_id = repo.create_run(
        instance="fake-1", metric="cpu", horizon="medium", config_snapshot={},
    )
    repo.mark_completed(run_id, duration_seconds=1.0)
    assert repo.get_run(run_id).status == "completed"

    client = _client()
    r = client.post(f"/ui/runs/{run_id}/cancel")
    assert r.status_code in (302, 303, 307)
    # Status stays completed
    assert repo.get_run(run_id).status == "completed"


def test_cancel_active_runs_bulk(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    revoked: list[str | None] = []
    monkeypatch.setattr(
        "forecaster.training.tasks.revoke_task",
        lambda tid: (revoked.append(tid), True)[1],
    )
    # Three active runs + one completed
    ids = []
    for i in range(3):
        rid = repo.create_run(
            instance=f"fake-{i}", metric="cpu", horizon="medium", config_snapshot={},
        )
        repo.record_celery_task_id(rid, f"task-{i}")
        ids.append(rid)
    done = repo.create_run(
        instance="fake-done", metric="cpu", horizon="medium", config_snapshot={},
    )
    repo.mark_completed(done, duration_seconds=1.0)

    client = _client()
    r = client.post("/ui/runs/cancel-active")
    assert r.status_code in (302, 303, 307)
    for rid in ids:
        assert repo.get_run(rid).status == "cancelled"
    assert repo.get_run(done).status == "completed"   # untouched
    assert sorted(revoked) == ["task-0", "task-1", "task-2"]


# ---- pause / resume ----------------------------------------------------

def test_pause_and_resume_flag(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    client = _client()
    assert repo.is_training_paused() is False

    r = client.post("/ui/training/pause")
    assert r.status_code in (302, 303, 307)
    assert repo.is_training_paused() is True

    r = client.post("/ui/training/resume")
    assert r.status_code in (302, 303, 307)
    assert repo.is_training_paused() is False


def test_pause_suppresses_fan_out(tmp_path, monkeypatch):
    """When training.paused=True, scheduler.fan_out returns 0 without
    enqueueing any tasks."""
    repo = _setup(tmp_path, monkeypatch)
    repo.set_training_paused(True)

    from forecaster.config import loader as cfg_loader
    cfg_loader._settings = None
    cfg_loader._settings_loaded_at = 0.0
    s = cfg_loader.load_settings()
    assert s.training.paused is True

    # Stub apply_async so any accidental enqueue surfaces loudly
    monkeypatch.setattr(
        "forecaster.training.tasks.train_task.apply_async",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("should not enqueue when paused")),
    )
    from forecaster.scheduling.jobs import fan_out
    assert fan_out("medium") == 0


def test_overview_shows_paused_banner(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    repo.set_training_paused(True)
    client = _client()
    body = client.get("/ui/").text
    # Notice + sidebar status both reflect the paused state
    assert "Training paused" in body
    # Resume button visible when paused
    assert "Start all training" in body

    # Resume → indicator flips, Stop button appears
    repo.set_training_paused(False)
    body2 = client.get("/ui/").text
    assert "active" in body2.lower()
    assert "Stop all training" in body2


# ---- target detail lookback -------------------------------------------

def test_target_detail_lookback_param(tmp_path, monkeypatch):
    """Custom lookback_hours flows into the rendered subtitle + form value."""
    repo = _setup(tmp_path, monkeypatch)
    # Seed a run so the page has data
    run_id = repo.create_run(
        instance="fake-1", metric="cpu", horizon="medium", config_snapshot={},
    )
    repo.mark_completed(run_id, duration_seconds=1.0)
    repo.add_forecasts(
        run_id=run_id, instance="fake-1", metric="cpu", horizon="medium",
        algo="naive", is_best=True,
        timestamps=[datetime(2026, 5, 21, 12, i, tzinfo=timezone.utc) for i in [0, 5, 10]],
        point=[1.0, 1.1, 1.2],
        lower=[0.9, 1.0, 1.1], upper=[1.1, 1.2, 1.3],
    )
    repo.add_ranking(
        run_id=run_id, instance="fake-1", metric="cpu", horizon="medium",
        winning_algo="naive",
        ranked=[{"rank": 1, "algo": "naive", "composite": 0.9,
                 "raw_scores": {}, "normalised_scores": {}}],
    )

    client = _client()
    r = client.get("/ui/targets/fake-1/cpu/medium?lookback_hours=72")
    assert r.status_code == 200
    body = r.text
    # The lookback control is rendered with our value
    assert 'name="lookback_hours"' in body
    assert 'value="72"' in body
    # Subtitle echoes the value
    assert "72h of past actuals" in body


def test_target_detail_default_lookback_per_horizon(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    run_id = repo.create_run(
        instance="fake-1", metric="cpu", horizon="long", config_snapshot={},
    )
    repo.mark_completed(run_id, duration_seconds=1.0)
    repo.add_forecasts(
        run_id=run_id, instance="fake-1", metric="cpu", horizon="long",
        algo="naive", is_best=True,
        timestamps=[datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc)],
        point=[1.0], lower=[0.9], upper=[1.1],
    )
    repo.add_ranking(
        run_id=run_id, instance="fake-1", metric="cpu", horizon="long",
        winning_algo="naive",
        ranked=[{"rank": 1, "algo": "naive", "composite": 0.9,
                 "raw_scores": {}, "normalised_scores": {}}],
    )
    client = _client()
    r = client.get("/ui/targets/fake-1/cpu/long")
    assert r.status_code == 200
    # 'long' defaults to 168h lookback
    assert "168h of past actuals" in r.text or 'value="168"' in r.text


# ---- run detail Cancel button -----------------------------------------

def test_run_detail_shows_cancel_button_for_pending(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    run_id = repo.create_run(
        instance="fake-1", metric="cpu", horizon="medium", config_snapshot={},
    )
    repo.record_celery_task_id(run_id, "task-x")
    client = _client()
    body = client.get(f"/ui/runs/{run_id}").text
    assert "Cancel this run" in body

    # Now mark completed → button disappears
    repo.mark_completed(run_id, duration_seconds=1.0)
    body2 = client.get(f"/ui/runs/{run_id}").text
    assert "Cancel this run" not in body2
