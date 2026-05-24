"""Smoke tests for the new /ui/graphs and /ui/explore pages."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest


def _seed(tmp_path: Path, monkeypatch):
    pytest.importorskip("jinja2")
    pytest.importorskip("fastapi")

    db = tmp_path / "graphs.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("FORECASTER__ARTIFACT_STORE__VOLUME_PATH", str(tmp_path / "art"))

    from forecaster.config import loader as cfg_loader
    cfg_loader._settings = None
    settings = cfg_loader.load_settings()
    settings.database_url = f"sqlite:///{db}"
    cfg_loader._settings = settings

    from forecaster.registry.repo import RegistryRepo
    repo = RegistryRepo(settings.database_url)
    repo.create_schema()

    base = datetime(2026, 5, 19, tzinfo=UTC)
    for instance in ("fake-1", "fake-2"):
        run_id = repo.create_run(
            instance=instance, metric="cpu", horizon="medium",
            config_snapshot={"x": 1},
        )
        repo.mark_completed(run_id, duration_seconds=1.0)
        repo.add_metrics(run_id, "naive",
                         {"mae": 1.0, "rmse": 1.0, "mape": 5.0, "smape": 5.0, "r2": 0.9},
                         fold=-1)
        repo.add_forecasts(
            run_id=run_id, instance=instance, metric="cpu", horizon="medium",
            algo="naive", is_best=True,
            timestamps=[base + timedelta(minutes=5*i) for i in range(3)],
            point=[10.0, 11.0, 12.0],
            lower=[9.0, 10.0, 11.0], upper=[11.0, 12.0, 13.0],
        )
        repo.add_ranking(
            run_id=run_id, instance=instance, metric="cpu", horizon="medium",
            winning_algo="naive",
            ranked=[
                {"rank": 1, "algo": "naive", "composite": 0.9,
                 "raw_scores": {"mae": 1.0, "rmse": 1.0, "mape": 5.0, "smape": 5.0, "r2": 0.9},
                 "normalised_scores": {}},
            ],
        )
        with repo.session() as s:
            from forecaster.registry.models import TrainingRun
            run = s.get(TrainingRun, run_id)
            run.completed_at = base


def _client():
    from fastapi.testclient import TestClient

    from forecaster.api import deps
    deps._repo.cache_clear()
    from forecaster.api.main import create_app
    return TestClient(create_app())


def test_graphs_page_renders(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    r = _client().get("/ui/graphs")
    assert r.status_code == 200
    body = r.text
    assert "Graphs" in body
    # Both seeded targets visible in tile placeholders
    assert "fake-1" in body
    assert "fake-2" in body
    # The HTMX lazy-load attribute is present
    assert "hx-get=" in body
    assert 'hx-trigger="revealed"' in body
    # Filter form is wired
    assert 'name="metric"' in body
    assert 'name="lookback_hours"' in body


def test_graphs_filter_narrows_tiles(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    r = _client().get("/ui/graphs?instance=fake-1")
    assert r.status_code == 200
    body = r.text
    # Tile URLs are inside an HTMX attribute, so '&' is rendered as '&amp;'.
    # fake-1 must appear in a tile URL; fake-2 must not.
    assert "instance=fake-1&amp;metric=cpu" in body
    assert "instance=fake-2&amp;metric=cpu" not in body


def test_graph_tile_fragment_renders(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    r = _client().get(
        "/ui/_/graph-tile",
        params={"instance": "fake-1", "metric": "cpu", "horizon": "medium",
                "lookback_hours": 1, "bands": 1, "overlay_algos": 0},
    )
    assert r.status_code == 200
    body = r.text
    assert "fake-1" in body
    assert "canvas" in body
    # Inline JS block embedded with forecast data
    assert "BY_ALGO" in body
    assert "ACTUALS" in body


def test_explore_page_renders(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    r = _client().get("/ui/explore")
    assert r.status_code == 200
    body = r.text
    assert "Explore" in body
    assert "Builder" in body
    assert "Raw PromQL" in body
    # Configured metric appears in the Builder dropdown
    assert "cpu" in body
    # Hidden custom range inputs present
    assert "data-custom-only" in body


def test_sidebar_contains_new_links(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    body = _client().get("/ui/").text
    assert ">Graphs<" in body
    assert ">Explore<" in body
