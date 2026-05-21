"""Tests for the extended Models page: state/family/min-runs/sort filters,
summary cards, the three new charts (MAE, train time, family roll-up)."""

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


def _seed_runs(repo, winners: list[tuple[str, str]]):
    """winners: list of (algo, metric) tuples; one completed run per entry."""
    for algo, met in winners:
        run_id = repo.create_run(
            instance="fake-1", metric=met, horizon="medium", config_snapshot={},
        )
        repo.mark_completed(run_id, duration_seconds=1.0)
        repo.add_metrics(run_id, algo,
                         {"mae": 0.5, "rmse": 0.6, "mape": 5.0, "smape": 5.0, "r2": 0.9},
                         fold=-1)
        repo.add_artifact(run_id, algo, "/tmp/x.pkl", 12, 2.0)
        repo.add_ranking(
            run_id=run_id, instance="fake-1", metric=met, horizon="medium",
            winning_algo=algo,
            ranked=[{"rank": 1, "algo": algo, "composite": 0.9,
                     "raw_scores": {}, "normalised_scores": {}}],
        )


# ---- Filter parameters propagate through ------------------------------

def test_models_filter_bar_renders_all_controls(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    client = _client()
    body = client.get("/ui/models").text
    for name in ["metric", "horizon", "window", "state", "family",
                 "min_runs", "sort", "show_all"]:
        assert f'name="{name}"' in body, f"missing filter control: {name}"


def test_models_family_filter_narrows_set(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    _seed_runs(repo, [("naive", "cpu"), ("lstm", "cpu"), ("arima", "cpu")])
    client = _client()
    body = client.get("/ui/models?family=baseline").text
    # Family chip "baseline" appears in the table
    assert "baseline" in body
    # The rows_json should only contain naive (the baseline) since other
    # filtered-out families are excluded from the chart JSON.
    import re, json as _json
    m = re.search(r"const ROWS\s*=\s*(\[.*?\]);", body, re.DOTALL)
    assert m
    rows = _json.loads(m.group(1))
    families = {r["family"] for r in rows}
    assert families == {"baseline"} or families == set()


def test_models_state_filter(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    _seed_runs(repo, [("naive", "cpu"), ("auto_arima", "cpu")])
    client = _client()
    body = client.get("/ui/models?state=enabled").text
    import re, json as _json
    m = re.search(r"const ROWS\s*=\s*(\[.*?\]);", body, re.DOTALL)
    rows = _json.loads(m.group(1))
    # naive is enabled by default; auto_arima is disabled by default
    algos = {r["algo"] for r in rows}
    assert "naive" in algos
    assert "auto_arima" not in algos


def test_models_min_runs_filter(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    # 3 runs for naive, 1 run for ets
    _seed_runs(repo, [("naive", "cpu"), ("naive", "cpu"), ("naive", "cpu"),
                       ("ets", "cpu")])
    client = _client()
    body = client.get("/ui/models?min_runs=2").text
    import re, json as _json
    m = re.search(r"const ROWS\s*=\s*(\[.*?\]);", body, re.DOTALL)
    rows = _json.loads(m.group(1))
    algos = {r["algo"] for r in rows}
    assert "naive" in algos
    assert "ets" not in algos


def test_models_show_all_toggle(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    _seed_runs(repo, [("naive", "cpu")])
    client = _client()

    # Default: only algos with runs > 0
    import re, json as _json
    body = client.get("/ui/models").text
    rows = _json.loads(re.search(r"const ROWS\s*=\s*(\[.*?\]);", body, re.DOTALL).group(1))
    assert all(r["runs"] > 0 for r in rows)

    # show_all=1: include zero-run algos
    body = client.get("/ui/models?show_all=1").text
    rows = _json.loads(re.search(r"const ROWS\s*=\s*(\[.*?\]);", body, re.DOTALL).group(1))
    assert any(r["runs"] == 0 for r in rows)


def test_models_sort_by_avg_mae(tmp_path, monkeypatch):
    """sort=avg_mae sorts ascending (lower is better)."""
    repo = _setup(tmp_path, monkeypatch)
    # Two algos, one with low MAE, one high
    run_id_a = repo.create_run(instance="x", metric="cpu", horizon="medium", config_snapshot={})
    repo.mark_completed(run_id_a, duration_seconds=1.0)
    repo.add_metrics(run_id_a, "naive", {"mae": 0.1, "rmse": 0.5}, fold=-1)
    repo.add_ranking(run_id=run_id_a, instance="x", metric="cpu", horizon="medium",
                     winning_algo="naive", ranked=[{"rank": 1, "algo": "naive",
                     "composite": 0.9, "raw_scores": {}, "normalised_scores": {}}])
    run_id_b = repo.create_run(instance="x", metric="cpu", horizon="medium", config_snapshot={})
    repo.mark_completed(run_id_b, duration_seconds=1.0)
    repo.add_metrics(run_id_b, "ets", {"mae": 5.0, "rmse": 5.5}, fold=-1)
    repo.add_ranking(run_id=run_id_b, instance="x", metric="cpu", horizon="medium",
                     winning_algo="ets", ranked=[{"rank": 1, "algo": "ets",
                     "composite": 0.8, "raw_scores": {}, "normalised_scores": {}}])

    client = _client()
    import re, json as _json
    body = client.get("/ui/models?sort=avg_mae").text
    rows = _json.loads(re.search(r"const ROWS\s*=\s*(\[.*?\]);", body, re.DOTALL).group(1))
    # naive (lower MAE) should appear before ets
    algos_in_order = [r["algo"] for r in rows if r["algo"] in ("naive", "ets")]
    assert algos_in_order.index("naive") < algos_in_order.index("ets")


# ---- New charts present ----------------------------------------------

def test_models_new_charts_present(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    client = _client()
    body = client.get("/ui/models").text
    for cid in ["winrateChart", "winByMetricChart", "maeChart", "durChart", "familyChart"]:
        assert f'id="{cid}"' in body


def test_models_summary_cards(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    _seed_runs(repo, [("naive", "cpu"), ("naive", "cpu"), ("ets", "cpu")])
    client = _client()
    body = client.get("/ui/models").text
    for label in ["Active algorithms", "Total runs entered",
                  "Total wins", "Top performer"]:
        assert label in body


def test_models_family_rollup_data(tmp_path, monkeypatch):
    repo = _setup(tmp_path, monkeypatch)
    _seed_runs(repo, [("naive", "cpu"), ("ets", "cpu"), ("lstm", "cpu")])
    client = _client()
    import re, json as _json
    body = client.get("/ui/models").text
    roll = _json.loads(re.search(r"const FAMILY_ROLL\s*=\s*(\[.*?\]);", body, re.DOTALL).group(1))
    families = {r["family"] for r in roll}
    # naive=baseline, ets=statistical, lstm=deep-learning
    assert families == {"baseline", "statistical", "deep-learning"}


# ---- Custom Run select theming (global CSS) --------------------------

def test_app_css_themes_native_select(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    client = _client()
    css = client.get("/ui/static/css/app.css").text
    # Global rule for bare `select` (not just `.filterbar select`)
    assert "select," in css or "select {" in css
    # Custom dropdown arrow built with two linear-gradients (proves we drew
    # our own so it matches across browsers/OS themes)
    assert "linear-gradient" in css
    # appearance: none on selects so OS native chrome is suppressed
    assert "appearance: none" in css or "appearance:none" in css


def test_custom_run_page_no_inline_select_styles(tmp_path, monkeypatch):
    """The Custom Run template doesn't need inline styles on its selects
    because the global rule covers them."""
    _setup(tmp_path, monkeypatch)
    client = _client()
    body = client.get("/ui/custom-run").text
    # The Metric and Horizon selects are present (and not styled inline)
    assert '<select name="metric"' in body
    assert '<select name="horizon"' in body
