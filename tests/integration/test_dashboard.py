"""Smoke + content tests for the /ui/ dashboard.

Seeds two runs against SQLite (different winners), then hits every page
and asserts both an HTTP 200 and key DOM content. Also exercises the
HTMX fragments and the form-driven config-reload endpoint.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


def _seed(tmp_path: Path, monkeypatch):
    pytest.importorskip("jinja2")
    pytest.importorskip("fastapi")

    db = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("FORECASTER__ARTIFACT_STORE__VOLUME_PATH", str(tmp_path / "artifacts"))
    monkeypatch.setenv("FORECASTER_USE_CUDA", "0")

    from forecaster.config import loader as cfg_loader

    cfg_loader._settings = None
    settings = cfg_loader.load_settings()
    settings.algorithms.enabled = ["naive", "seasonal_naive", "ets"]
    settings.algorithms.per_metric = {"cpu": ["naive", "ets"]}
    settings.database_url = f"sqlite:///{db}"
    settings.artifact_store.volume_path = str(tmp_path / "artifacts")
    cfg_loader._settings = settings

    from forecaster.registry.repo import RegistryRepo
    repo = RegistryRepo(settings.database_url)
    repo.create_schema()

    instance, metric, horizon = "fake-1", "cpu", "medium"

    def _seed_one(winner: str, completed_at: datetime, *, score: float):
        run_id = repo.create_run(
            instance=instance, metric=metric, horizon=horizon, config_snapshot={"x": 1},
        )
        repo.mark_completed(run_id, duration_seconds=1.5)
        repo.add_metrics(run_id, "naive",
                         {"mae": 1.0 if winner == "naive" else 2.0,
                          "rmse": 1.0, "mape": 5.0, "smape": 5.0, "r2": 0.9},
                         fold=-1)
        repo.add_metrics(run_id, "ets",
                         {"mae": 1.0 if winner == "ets" else 2.0,
                          "rmse": 1.0, "mape": 5.0, "smape": 5.0, "r2": 0.9},
                         fold=-1)
        repo.add_artifact(run_id, "naive", str(tmp_path / "n.pkl"), 12, 0.5)
        repo.add_artifact(run_id, "ets", str(tmp_path / "e.pkl"), 12, 0.7)
        repo.add_forecasts(
            run_id=run_id, instance=instance, metric=metric, horizon=horizon, algo=winner,
            is_best=True,
            timestamps=[completed_at + timedelta(minutes=5*i) for i in range(3)],
            point=[10.0, 11.0, 12.0],
            lower=[9.0, 10.0, 11.0], upper=[11.0, 12.0, 13.0],
        )
        repo.add_ranking(
            run_id=run_id, instance=instance, metric=metric, horizon=horizon,
            winning_algo=winner,
            ranked=[
                {"rank": 1, "algo": winner, "composite": score,
                 "raw_scores": {"mae": 1.0, "rmse": 1.0, "mape": 5.0, "smape": 5.0, "r2": 0.9},
                 "normalised_scores": {}},
                {"rank": 2, "algo": "naive" if winner == "ets" else "ets",
                 "composite": score - 0.1,
                 "raw_scores": {"mae": 2.0, "rmse": 1.0, "mape": 5.0, "smape": 5.0, "r2": 0.9},
                 "normalised_scores": {}},
            ],
        )
        with repo.session() as s:
            from forecaster.registry.models import TrainingRun
            run = s.get(TrainingRun, run_id)
            run.completed_at = completed_at
        return run_id

    base = datetime(2026, 5, 19, tzinfo=timezone.utc)
    rid1 = _seed_one("naive", base, score=0.8)
    rid2 = _seed_one("ets", base + timedelta(hours=1), score=0.9)
    return rid1, rid2, instance, metric, horizon


def _client():
    from fastapi.testclient import TestClient
    from forecaster.api import deps
    deps._repo.cache_clear()
    from forecaster.api.main import create_app
    return TestClient(create_app(), follow_redirects=False)


def test_root_redirects_to_ui(tmp_path: Path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    client = _client()
    r = client.get("/")
    assert r.status_code in (302, 307)
    assert r.headers["location"].endswith("/ui/")


def test_overview_page_renders(tmp_path: Path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    client = _client()
    r = client.get("/ui/")
    assert r.status_code == 200
    body = r.text
    assert "System overview" in body
    assert "Recent training runs" in body
    assert "Targets needing attention" in body


def test_overview_fragments(tmp_path: Path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    client = _client()
    cards = client.get("/ui/_/overview/cards").text
    # 2 completed runs were seeded
    assert "Targets tracked" in cards
    assert "Distinct instances" in cards

    runs = client.get("/ui/_/overview/runs").text
    assert "fake-1" in runs

    attention = client.get("/ui/_/overview/attention").text
    # In the synthetic case unique winners == 2, not flapping → expect healthy
    assert "healthy" in attention or "fake-1" in attention


def test_targets_page(tmp_path: Path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    client = _client()
    r = client.get("/ui/targets")
    assert r.status_code == 200
    body = r.text
    assert "fake-1" in body
    assert "ets" in body  # current winner


def test_targets_filter(tmp_path: Path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    client = _client()
    r = client.get("/ui/targets?metric=cpu&horizon=medium")
    assert r.status_code == 200
    assert "fake-1" in r.text
    r2 = client.get("/ui/targets?metric=disk")
    assert "No targets match" in r2.text


def test_target_detail_renders_charts_payload(tmp_path: Path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    client = _client()
    r = client.get("/ui/targets/fake-1/cpu/medium")
    assert r.status_code == 200
    body = r.text
    # Three chart payloads + score history embedded as JS arrays
    assert "FORECAST_DATA" in body
    assert "RANKING_DATA" in body
    assert "WINNER_HISTORY" in body
    assert "SCORE_HISTORY" in body
    assert '"winning_algo": "naive"' in body
    assert '"winning_algo": "ets"' in body


def test_runs_page_and_detail(tmp_path: Path, monkeypatch):
    rid1, rid2, *_ = _seed(tmp_path, monkeypatch)
    client = _client()
    r = client.get("/ui/runs")
    assert r.status_code == 200
    assert f"#{rid1}" in r.text and f"#{rid2}" in r.text

    detail = client.get(f"/ui/runs/{rid2}")
    assert detail.status_code == 200
    body = detail.text
    assert "Per-algo scores" in body
    assert "Config snapshot" in body
    assert "RANK" in body and "DUR" in body
    # Winner pill present
    assert 'pill good">ets' in body


def test_run_detail_404(tmp_path: Path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    client = _client()
    assert client.get("/ui/runs/999999").status_code == 404


def test_models_page(tmp_path: Path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    client = _client()
    r = client.get("/ui/models")
    assert r.status_code == 200
    body = r.text
    # All 10 registered algos appear (even those with 0 runs)
    for algo in ["naive", "seasonal_naive", "arima", "ets", "holt_winters",
                 "prophet", "xgboost", "lightgbm", "lstm", "nbeats"]:
        assert algo in body


def test_config_page(tmp_path: Path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    client = _client()
    r = client.get("/ui/config")
    assert r.status_code == 200
    body = r.text
    assert "Effective configuration" in body
    assert "Per-metric shortlists" in body
    assert "Retrain schedule" in body
    assert "Ranking weights" in body
    # The shipped per_metric.cpu shortlist (we overrode it to ["naive","ets"])
    assert "naive" in body and "ets" in body


def test_config_reload_redirects(tmp_path: Path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    client = _client()
    r = client.post("/ui/config/reload")
    assert r.status_code in (302, 303, 307)
    assert "/ui/config" in r.headers["location"]


def test_static_assets_served(tmp_path: Path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    client = _client()
    r = client.get("/ui/static/css/app.css")
    assert r.status_code == 200
    assert "forecaster dashboard" in r.text


def test_instances_page(tmp_path: Path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    client = _client()
    r = client.get("/ui/instances")
    assert r.status_code == 200
    body = r.text
    assert "fake-1" in body
    assert "detail" in body


def test_instances_filter(tmp_path: Path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    client = _client()
    r = client.get("/ui/instances?q=fake-1")
    assert r.status_code == 200
    assert "fake-1" in r.text
    miss = client.get("/ui/instances?q=does-not-exist")
    assert "No instances match" in miss.text


def test_instance_detail(tmp_path: Path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    client = _client()
    r = client.get("/ui/instances/fake-1")
    assert r.status_code == 200
    body = r.text
    assert "Health grid" in body
    assert "Recent runs" in body
    assert "cpu" in body
    not_found = client.get("/ui/instances/no-such-server")
    assert not_found.status_code == 404


def test_runs_page_has_sort_and_date_filters(tmp_path: Path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    client = _client()
    r = client.get("/ui/runs")
    assert r.status_code == 200
    body = r.text
    assert 'name="since"' in body
    assert 'name="until"' in body
    assert "sort=started_at" in body


def test_runs_date_range_filter(tmp_path: Path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    client = _client()
    r = client.get("/ui/runs?since=2099-01-01T00:00")
    assert r.status_code == 200
    assert "No runs match" in r.text
    r2 = client.get("/ui/runs?since=2000-01-01T00:00")
    assert r2.status_code == 200
    assert "No runs match" not in r2.text


def test_runs_sort_alternate_column(tmp_path: Path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    client = _client()
    r = client.get("/ui/runs?sort=duration_seconds&direction=asc")
    assert r.status_code == 200
    assert "sort=duration_seconds" in r.text


def test_ist_format_visible_in_runs_table(tmp_path: Path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    client = _client()
    body = client.get("/ui/runs").text
    # Cells should render in IST, not raw UTC ISO with +00:00 suffix.
    assert "+00:00</td>" not in body
    # IST tz marker or +0530 offset must appear somewhere in the timestamps.
    assert "IST" in body or "+0530" in body
