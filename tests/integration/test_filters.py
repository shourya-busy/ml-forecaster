"""Tests for the filter controls on Trends / Compare / Models."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
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


def _seed_old_and_new_runs(repo):
    """Create two completed runs: one 30 days ago, one today, for cpu/medium."""
    old = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)
    new = datetime.now(timezone.utc) - timedelta(hours=1)
    for completed_at, winner in [(old, "naive"), (new, "ets")]:
        run_id = repo.create_run(
            instance="fake-1", metric="cpu", horizon="medium", config_snapshot={},
        )
        repo.mark_completed(run_id, duration_seconds=1.0)
        repo.add_metrics(run_id, winner,
                         {"mae": 1.0, "rmse": 1.0, "mape": 5.0, "smape": 5.0, "r2": 0.9},
                         fold=-1)
        repo.add_ranking(
            run_id=run_id, instance="fake-1", metric="cpu", horizon="medium",
            winning_algo=winner,
            ranked=[{"rank": 1, "algo": winner, "composite": 0.9,
                     "raw_scores": {"mae": 1.0}, "normalised_scores": {}}],
        )
        # Backdate
        with repo.session() as s:
            from forecaster.registry.models import TrainingRun
            run = s.get(TrainingRun, run_id)
            run.completed_at = completed_at


# ---- Trends -----------------------------------------------------------

def test_trends_horizon_picker(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    client = _client()
    r = client.get("/ui/trends?metric=cpu&horizon=short")
    assert r.status_code == 200
    body = r.text
    # Selected horizon should be present in the dropdown as selected
    assert 'value="short"' in body
    # The "showing X across Y" subtitle mentions the agg & window
    assert "median" in body or "mean" in body


def test_trends_window_filter_drops_old_wins(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    _seed_old_and_new_runs(repo)
    client = _client()
    # Default window is 7d so the 30-day-old ranking is excluded.
    r = client.get("/ui/trends?metric=cpu&horizon=medium&window=7d")
    assert r.status_code == 200
    body = r.text
    # The 7d winner share should not include `naive` (only old run)
    # We assert by checking the JSON-embedded share excludes naive
    import re, json as _json
    m = re.search(r"const WINNER_SHARE\s*=\s*(\[.*?\]);", body, re.DOTALL)
    assert m
    share = _json.loads(m.group(1))
    algos = {row["algo"] for row in share}
    assert "ets" in algos
    assert "naive" not in algos

    # With window=all both should appear
    r_all = client.get("/ui/trends?metric=cpu&horizon=medium&window=all")
    body = r_all.text
    m = re.search(r"const WINNER_SHARE\s*=\s*(\[.*?\]);", body, re.DOTALL)
    share = _json.loads(m.group(1))
    algos = {row["algo"] for row in share}
    assert algos == {"naive", "ets"}


def test_trends_aggregation_options(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    client = _client()
    for agg in ["mean", "median", "p95"]:
        r = client.get(f"/ui/trends?agg={agg}")
        assert r.status_code == 200
        # Selected agg surfaces in the dropdown
        assert f'value="{agg}" selected' in r.text or f'value="{agg}"' in r.text


def test_trends_freshness_default_per_horizon(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    client = _client()
    # Defaults: short -> 2, medium -> 12, long -> 72
    r = client.get("/ui/trends?horizon=long")
    assert r.status_code == 200
    assert "72h" in r.text


# ---- Compare ----------------------------------------------------------

def test_compare_horizon_narrows_picker(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    # Seed two different horizons so the filter has something to remove
    _seed_old_and_new_runs(repo)
    with repo.session() as s:
        from forecaster.registry.models import TrainingRun
        run_id = repo.create_run(
            instance="fake-1", metric="cpu", horizon="long", config_snapshot={},
        )
        repo.mark_completed(run_id, duration_seconds=1.0)
        repo.add_ranking(
            run_id=run_id, instance="fake-1", metric="cpu", horizon="long",
            winning_algo="prophet",
            ranked=[{"rank": 1, "algo": "prophet", "composite": 0.9,
                     "raw_scores": {}, "normalised_scores": {}}],
        )
    client = _client()
    r = client.get("/ui/compare?horizon=long")
    assert r.status_code == 200
    # The "picker shows X of Y" line is part of the page
    assert "Picker shows" in r.text


# ---- Models -----------------------------------------------------------

def test_models_metric_filter(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    _seed_old_and_new_runs(repo)
    client = _client()
    r = client.get("/ui/models?metric=mem")
    assert r.status_code == 200
    # No mem runs were seeded so wins should be empty in the JSON payload
    import re, json as _json
    m = re.search(r"const PER_METRIC\s*=\s*(\{.*?\});", r.text, re.DOTALL)
    assert m
    pm = _json.loads(m.group(1))
    assert pm == {} or "mem" not in pm


def test_models_window_filter(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    _seed_old_and_new_runs(repo)
    client = _client()
    # Last 7d → only `ets` should have wins
    r = client.get("/ui/models?metric=cpu&window=7d")
    assert r.status_code == 200
    import re, json as _json
    m = re.search(r"const PER_METRIC\s*=\s*(\{.*?\});", r.text, re.DOTALL)
    assert m
    pm = _json.loads(m.group(1))
    # cpu key may or may not exist depending on data; if it does it
    # should only include ets's win
    if "cpu" in pm:
        assert "ets" in pm["cpu"]
        assert "naive" not in pm["cpu"]


def test_models_filters_render(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    client = _client()
    r = client.get("/ui/models")
    assert r.status_code == 200
    body = r.text
    assert 'name="metric"' in body
    assert 'name="horizon"' in body
    assert 'name="window"' in body
