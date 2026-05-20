"""Diagnostics REST + Prom exposition: seed two runs, hit the surface."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


def _seed_two_runs_with_different_winners(tmp_path: Path, monkeypatch):
    """Manufacture two completed runs where the winner changes between them."""
    pytest.importorskip("statsmodels")
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
    settings.training.backtest_folds = 2
    settings.training.parallelism.algos_per_job = 1
    settings.database_url = f"sqlite:///{db}"
    settings.artifact_store.volume_path = str(tmp_path / "artifacts")
    cfg_loader._settings = settings

    from forecaster.registry.repo import RegistryRepo
    from tests.fixtures.synthetic_series import synthetic_series

    repo = RegistryRepo(settings.database_url)
    repo.create_schema()

    # Manually insert two rankings with different winners.
    instance, metric, horizon = "fake-1", "cpu", "medium"

    def _add_ranking(repo: RegistryRepo, winner: str, completed_at: datetime, *, score: float):
        run_id = repo.create_run(
            instance=instance, metric=metric, horizon=horizon,
            config_snapshot={},
        )
        repo.mark_completed(run_id, duration_seconds=1.0)
        repo.add_metrics(
            run_id, "naive",
            {"mae": 1.0 if winner == "naive" else 2.0,
             "rmse": 1.0, "mape": 5.0, "smape": 5.0, "r2": 0.9},
            fold=-1,
        )
        repo.add_metrics(
            run_id, "ets",
            {"mae": 1.0 if winner == "ets" else 2.0,
             "rmse": 1.0, "mape": 5.0, "smape": 5.0, "r2": 0.9},
            fold=-1,
        )
        repo.add_ranking(
            run_id=run_id, instance=instance, metric=metric, horizon=horizon,
            winning_algo=winner,
            ranked=[
                {"rank": 1, "algo": winner, "composite": score,
                 "raw_scores": {}, "normalised_scores": {}},
                {"rank": 2, "algo": "naive" if winner == "ets" else "ets",
                 "composite": score - 0.1, "raw_scores": {}, "normalised_scores": {}},
            ],
        )
        # Backdate completed_at so the order is deterministic.
        with repo.session() as s:
            from forecaster.registry.models import TrainingRun
            run = s.get(TrainingRun, run_id)
            run.completed_at = completed_at
        return run_id

    base = datetime(2026, 5, 19, tzinfo=timezone.utc)
    _add_ranking(repo, "naive", base, score=0.8)
    _add_ranking(repo, "ets", base + timedelta(hours=1), score=0.9)
    return repo, instance, metric, horizon


def test_diagnostics_endpoints(tmp_path: Path, monkeypatch):
    repo, instance, metric, horizon = _seed_two_runs_with_different_winners(tmp_path, monkeypatch)

    from forecaster.api import deps
    deps._repo.cache_clear()

    from fastapi.testclient import TestClient

    from forecaster.api.main import create_app

    client = TestClient(create_app())

    winners = client.get("/diagnostics/winners").json()
    assert len(winners) == 1
    row = winners[0]
    assert row["instance"] == instance
    assert row["metric"] == metric
    assert row["horizon"] == horizon
    assert row["current_winner"] == "ets"
    assert row["previous_winner"] == "naive"
    assert row["unique_winners_recent"] == 2
    assert len(row["current_top3"]) >= 1

    wh = client.get("/diagnostics/winner-history", params={
        "instance": instance, "metric": metric, "horizon": horizon,
    }).json()
    assert [r["winning_algo"] for r in wh] == ["naive", "ets"]  # oldest-first

    sh = client.get("/diagnostics/score-history", params={
        "instance": instance, "metric": metric, "horizon": horizon,
        "algo": "ets", "score": "mae",
    }).json()
    assert len(sh) == 2
    assert all(r["algo"] == "ets" and r["score"] == "mae" for r in sh)
    # MAE drops between runs (ets is worse in run 1, best in run 2).
    assert sh[0]["value"] >= sh[1]["value"]


def test_diagnostics_prom_exposition(tmp_path: Path, monkeypatch):
    repo, instance, metric, horizon = _seed_two_runs_with_different_winners(tmp_path, monkeypatch)

    from forecaster.api import deps
    deps._repo.cache_clear()

    from fastapi.testclient import TestClient

    from forecaster.api.main import create_app

    client = TestClient(create_app())
    body = client.get("/metrics").text
    assert "forecaster_winner{" in body
    assert f'instance="{instance}"' in body
    # Current winner is `ets` per the seed function
    assert 'model="ets"' in body
    # unique_winners_recent should be 2
    assert "forecaster_winner_unique_recent{" in body
    assert "2.0" in body or " 2 " in body
