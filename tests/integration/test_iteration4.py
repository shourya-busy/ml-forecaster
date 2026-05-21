"""Tests for the iteration 4 additions:

- Forecast-vs-actual overlay on /ui/targets/.../detail
- Per-fold backtest visualization on /ui/runs/{id}
- Horizons editable from /ui/schedule (step / horizon / lookback)
- Algorithm library cards in /ui/manage/training
"""

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


def _seed_run_with_folds(repo, instance="fake-1", metric="cpu", horizon="medium"):
    run_id = repo.create_run(
        instance=instance, metric=metric, horizon=horizon,
        config_snapshot={"k": "v"},
    )
    repo.mark_completed(run_id, duration_seconds=1.0)
    # Averaged (fold=-1) — what the existing pages show
    repo.add_metrics(run_id, "naive",
                     {"mae": 1.0, "rmse": 1.2, "mape": 5.0, "smape": 5.0, "r2": 0.9},
                     fold=-1)
    repo.add_metrics(run_id, "ets",
                     {"mae": 0.8, "rmse": 1.0, "mape": 4.0, "smape": 4.0, "r2": 0.95},
                     fold=-1)
    # Per-fold (fold=0,1,2) — used by the new backtest section
    for i, mae in enumerate([1.1, 1.0, 0.9]):
        repo.add_metrics(run_id, "naive", {"mae": mae, "rmse": mae * 1.1}, fold=i)
    for i, mae in enumerate([0.9, 0.8, 0.7]):
        repo.add_metrics(run_id, "ets", {"mae": mae, "rmse": mae * 1.1}, fold=i)
    repo.add_artifact(run_id, "naive", str("/tmp/n.pkl"), 12, 0.5)
    repo.add_artifact(run_id, "ets", str("/tmp/e.pkl"), 12, 0.6)
    repo.add_ranking(
        run_id=run_id, instance=instance, metric=metric, horizon=horizon,
        winning_algo="ets",
        ranked=[
            {"rank": 1, "algo": "ets", "composite": 0.9,
             "raw_scores": {"mae": 0.8}, "normalised_scores": {}},
            {"rank": 2, "algo": "naive", "composite": 0.7,
             "raw_scores": {"mae": 1.0}, "normalised_scores": {}},
        ],
    )
    return run_id


def test_run_detail_shows_backtest_section(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    rid = _seed_run_with_folds(repo)
    client = _client()
    r = client.get(f"/ui/runs/{rid}")
    assert r.status_code == 200
    body = r.text
    assert "Backtest folds" in body
    assert "walk-forward cross-validation" in body
    # Per-fold JSON for the chart
    assert "PER_FOLD" in body
    # The MAE numbers I seeded should be present
    assert "0.900" in body or "0.9" in body


def test_run_full_detail_includes_per_fold(tmp_path, monkeypatch):
    """Repo helper returns oldest-first folds per algo."""
    repo = _setup(tmp_path, monkeypatch)
    rid = _seed_run_with_folds(repo)
    pf = repo.run_per_fold_scores(rid)
    assert set(pf.keys()) == {"naive", "ets"}
    assert [f["fold"] for f in pf["naive"]] == [0, 1, 2]
    assert pf["ets"][0]["mae"] == 0.9


def test_algo_info_metadata_present():
    from forecaster.models.registry import ALGO_INFO, algo_info
    # All 10 algorithms have descriptions
    for name in ["naive", "seasonal_naive", "arima", "ets", "holt_winters",
                 "prophet", "xgboost", "lightgbm", "lstm", "nbeats"]:
        info = algo_info(name)
        assert info["description"]
        assert info["when_to_use"]
        assert info["family"] in {"baseline", "statistical", "ml", "deep-learning"}


def test_manage_training_shows_algo_cards(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    client = _client()
    r = client.get("/ui/manage/training")
    assert r.status_code == 200
    body = r.text
    # Library copy
    assert "Algorithm library" in body
    # A description from ALGO_INFO surfaces
    assert "Sanity baseline" in body or "naive" in body
    # The 'why not add new' note
    assert "Why can't I add new algorithms" in body


def test_schedule_page_shows_step_and_horizon(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    client = _client()
    r = client.get("/ui/schedule")
    assert r.status_code == 200
    body = r.text
    # All three editable fields render
    assert 'name="step"' in body
    assert 'name="forecast_horizon"' in body
    assert 'name="lookback_days"' in body
    # Point-count surfaced
    assert "points predicted" in body


def test_schedule_save_step_and_lookback(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    client = _client()
    r = client.post("/ui/schedule/horizon", data={
        "horizon": "medium", "retrain": "*/30 * * * *",
        "step": "10min", "forecast_horizon": "12h", "lookback_days": "21",
    })
    assert r.status_code in (302, 303, 307)
    ov = repo.get_all_settings_overrides()
    assert ov["horizons.medium.retrain"] == "*/30 * * * *"
    assert ov["horizons.medium.step"] == "10min"
    assert ov["horizons.medium.horizon"] == "12h"
    assert ov["horizons.medium.lookback_days"] == 21


def test_schedule_save_invalid_step_rejected(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    client = _client()
    r = client.post("/ui/schedule/horizon", data={
        "horizon": "medium", "retrain": "0 * * * *",
        "step": "not a duration",
    })
    assert r.status_code == 400


def test_target_detail_includes_actuals_payload(tmp_path, monkeypatch):
    """The target detail page must always include the ACTUALS_DATA JS var,
    even when no actuals could be fetched (data source unreachable)."""
    repo = _setup(tmp_path, monkeypatch)
    # Seed a run so the target page has a forecast row to render
    run_id = repo.create_run(
        instance="fake-1", metric="cpu", horizon="medium", config_snapshot={},
    )
    repo.mark_completed(run_id, duration_seconds=1.0)
    repo.add_forecasts(
        run_id=run_id, instance="fake-1", metric="cpu", horizon="medium",
        algo="ets", is_best=True,
        timestamps=[datetime(2026, 5, 21, 6, i, tzinfo=timezone.utc) for i in [0, 5, 10]],
        point=[1.0, 1.1, 1.2],
        lower=[0.9, 1.0, 1.1], upper=[1.1, 1.2, 1.3],
    )
    repo.add_ranking(
        run_id=run_id, instance="fake-1", metric="cpu", horizon="medium",
        winning_algo="ets",
        ranked=[{"rank": 1, "algo": "ets", "composite": 0.9,
                 "raw_scores": {}, "normalised_scores": {}}],
    )
    client = _client()
    r = client.get("/ui/targets/fake-1/cpu/medium")
    assert r.status_code == 200
    body = r.text
    assert "ACTUALS_DATA" in body
    assert "actual (Prometheus)" in body  # legend label baked into the JS


def test_horizons_override_applies_to_settings(tmp_path, monkeypatch):
    """Settings overrides for horizons.X.step propagate through the loader."""
    repo = _setup(tmp_path, monkeypatch)
    repo.set_settings_override("horizons.medium.step", "10min")
    repo.set_settings_override("horizons.medium.horizon", "12h")
    repo.set_settings_override("horizons.medium.lookback_days", 21)

    from forecaster.config import loader as cfg_loader
    cfg_loader._settings = None
    cfg_loader._settings_loaded_at = 0.0
    settings = cfg_loader.load_settings()
    assert settings.horizons["medium"].step == "10min"
    assert settings.horizons["medium"].horizon == "12h"
    assert settings.horizons["medium"].lookback_days == 21
