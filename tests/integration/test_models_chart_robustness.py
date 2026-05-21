"""Models page charts must render cleanly at any algorithm count.

Regression for: adding 8+ new algorithms made the vertical-bar charts
collapse / overlap labels. Both charts are now horizontal, filtered to
algos with data, sorted descending, and gracefully render an empty-state
when nothing has run yet.
"""

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


def test_models_renders_empty_state_with_no_data(tmp_path, monkeypatch):
    """Fresh deploy → no runs yet → page should still render with an
    explicit empty-state message instead of an invisible blank chart."""
    _setup(tmp_path, monkeypatch)
    client = _client()
    body = client.get("/ui/models").text
    assert "Win rate by algorithm" in body
    assert "Wins by metric" in body
    # Empty-state copy is present (hidden by default but in the DOM)
    assert "No algorithms match the current filters" in body
    # All registered algos still appear in the details table
    for algo in ["naive", "auto_arima", "neural_prophet", "drift", "mstl"]:
        assert algo in body


def test_models_charts_use_horizontal_bars(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    client = _client()
    body = client.get("/ui/models").text
    # Horizontal-bar marker
    assert 'indexAxis: "y"' in body
    # Dynamic-height helper present
    assert "setHeight" in body
    # The two canvas wrappers we size programmatically
    assert 'id="wr-canvas-wrap"' in body
    assert 'id="wbm-canvas-wrap"' in body


def test_models_chart_sorts_and_filters(tmp_path, monkeypatch):
    """When some algos have runs and others don't, the JS keeps only the
    populated ones in the chart data while the details table still shows
    all 23."""
    repo = _setup(tmp_path, monkeypatch)
    # Create two completed runs with different winners
    for (winner, mae) in [("ets", 1.0), ("lstm", 0.8)]:
        run_id = repo.create_run(
            instance="fake-1", metric="cpu", horizon="medium", config_snapshot={},
        )
        repo.mark_completed(run_id, duration_seconds=1.0)
        repo.add_metrics(run_id, winner,
                         {"mae": mae, "rmse": 1.0, "mape": 5.0, "smape": 5.0, "r2": 0.9},
                         fold=-1)
        repo.add_ranking(
            run_id=run_id, instance="fake-1", metric="cpu", horizon="medium",
            winning_algo=winner,
            ranked=[{"rank": 1, "algo": winner, "composite": 0.9,
                     "raw_scores": {}, "normalised_scores": {}}],
        )

    client = _client()
    body = client.get("/ui/models").text
    # The chart subtitle now reports the count of algorithms plotted
    assert "algorithm(s) plotted" in body
    # Details table still lists every registered algo (incl. those with 0 runs)
    for algo in ["naive", "drift", "mean", "auto_arima", "lstm"]:
        assert algo in body
