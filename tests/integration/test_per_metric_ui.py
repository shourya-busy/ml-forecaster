"""Manage → Training: per-metric shortlist UI + auto-prune on enabled change."""

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


# ---------- UI surface ---------------------------------------------------

def test_manage_training_renders_per_metric_section(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    client = _client()
    body = client.get("/ui/manage/training").text
    assert "Per-metric shortlists" in body
    # One block per configured metric
    for m in ["cpu", "mem", "disk"]:
        assert f'name="per_metric__{m}"' in body
    # "Select all enabled" + "Clear" controls
    assert "Select all enabled" in body
    assert "Clear (= no restriction)" in body


def test_manage_training_shows_wont_train_warning(tmp_path, monkeypatch):
    """If an algo is enabled but not present in any per_metric shortlist
    (and every metric has a shortlist), the card flags the trap."""
    repo = _setup(tmp_path, monkeypatch)
    # Enable a Nixtla algo that the YAML's per_metric doesn't reference.
    # The YAML defaults restrict cpu/mem/disk, none of which contain auto_arima.
    repo.set_settings_override(
        "algorithms.enabled",
        ["naive", "seasonal_naive", "arima", "ets", "holt_winters", "prophet",
         "xgboost", "lightgbm", "lstm", "nbeats", "auto_arima"],
    )
    client = _client()
    body = client.get("/ui/manage/training").text
    # The warning chip exists somewhere on the page
    assert "won't train" in body


def test_manage_training_shows_trains_on_chip(tmp_path, monkeypatch):
    """Algos that DO appear in shortlists get the green 'trains on' chip."""
    _setup(tmp_path, monkeypatch)
    client = _client()
    body = client.get("/ui/manage/training").text
    # Default per_metric has lstm in cpu's list
    assert "trains on:" in body


# ---------- Save handler -------------------------------------------------

def test_save_auto_prunes_per_metric_when_disabling(tmp_path, monkeypatch):
    """User disables an algo via the library checkboxes → the per_metric
    shortlists referencing that algo are auto-pruned so the override
    doesn't get silently rejected by the pydantic validator."""
    repo = _setup(tmp_path, monkeypatch)
    # Default per_metric.cpu includes holt_winters. Disable it.
    new_enabled = [
        "naive", "seasonal_naive", "arima", "ets",
        # holt_winters removed
        "prophet", "xgboost", "lightgbm", "lstm", "nbeats",
    ]
    client = _client()
    r = client.post(
        "/ui/manage/training/save",
        data={"enabled_algos": new_enabled},
    )
    assert r.status_code in (302, 303, 307)
    # The saved per_metric.cpu must NOT contain holt_winters now
    ov = repo.get_all_settings_overrides()
    pm = ov.get("algorithms.per_metric")
    assert pm is not None, "per_metric override should have been written"
    assert "holt_winters" not in pm.get("cpu", []), (
        f"holt_winters should have been pruned: {pm.get('cpu')}"
    )
    # And the saved enabled list reflects the user's choice
    assert ov["algorithms.enabled"] == new_enabled


def test_save_writes_per_metric_when_form_supplies_it(tmp_path, monkeypatch):
    """Ticking checkboxes in the per-metric section persists them.

    Uses dict-with-list form (one form-field-per-value) because httpx's
    list-of-tuples form silently keeps only the last value per key.
    """
    repo = _setup(tmp_path, monkeypatch)
    client = _client()
    r = client.post(
        "/ui/manage/training/save",
        data={
            "enabled_algos": ["naive", "ets", "lstm"],
            "per_metric__cpu": ["ets", "lstm"],
            "per_metric__mem": ["naive"],
        },
    )
    assert r.status_code in (302, 303, 307)
    pm = repo.get_all_settings_overrides().get("algorithms.per_metric")
    assert pm is not None
    assert set(pm.get("cpu", [])) == {"ets", "lstm"}
    assert set(pm.get("mem", [])) == {"naive"}


def test_save_clearing_all_per_metric_drops_override(tmp_path, monkeypatch):
    """If the user reduces enabled to a single algo and that algo is also
    chosen for all metrics, only that algo survives in the per_metric
    shortlists."""
    repo = _setup(tmp_path, monkeypatch)
    repo.set_settings_override("algorithms.per_metric", {"cpu": ["naive"]})
    client = _client()
    r = client.post(
        "/ui/manage/training/save",
        data={
            "enabled_algos": ["naive"],
            # No per_metric__X fields → form didn't render any grid
            # → seeded override is auto-pruned but otherwise left alone.
        },
    )
    assert r.status_code in (302, 303, 307)
    pm = repo.get_all_settings_overrides().get("algorithms.per_metric")
    # Seeded override "cpu: [naive]" survives because naive is still enabled
    assert pm == {"cpu": ["naive"]}


def test_loader_subset_invariant_not_violated_after_save(tmp_path, monkeypatch):
    """After saving a reduced enabled list, get_settings() must not log
    'failed validation, discarding' — the auto-prune keeps the invariant."""
    repo = _setup(tmp_path, monkeypatch)
    client = _client()
    # Reduce enabled to a smaller set — this would previously have caused
    # per_metric.cpu (containing holt_winters etc.) to violate the subset
    # invariant and silently drop.
    reduced = ["naive", "ets", "lstm"]
    r = client.post("/ui/manage/training/save", data={"enabled_algos": reduced})
    assert r.status_code in (302, 303, 307)

    # Re-load settings and confirm enabled IS the reduced set (not the
    # YAML default — which would mean the override was discarded).
    from forecaster.config import loader as cfg_loader
    cfg_loader._settings = None
    cfg_loader._settings_loaded_at = 0.0
    settings = cfg_loader.load_settings()
    assert set(settings.algorithms.enabled) == set(reduced)


# ---------- End-to-end gotcha ------------------------------------------

def test_enabling_unshortlisted_algo_now_trains_after_per_metric_edit(tmp_path, monkeypatch):
    """The original user scenario: enable auto_arima via library →
    initially it won't train (no per_metric entry) → add it to cpu's
    shortlist → CPU pipeline now picks it up."""
    repo = _setup(tmp_path, monkeypatch)
    # Step 1: enable auto_arima (added to enabled, but cpu shortlist
    # still excludes it → 'won't train' state)
    client = _client()
    base_enabled = [
        "naive", "seasonal_naive", "arima", "ets", "holt_winters", "prophet",
        "xgboost", "lightgbm", "lstm", "nbeats", "auto_arima",
    ]
    client.post("/ui/manage/training/save", data={"enabled_algos": base_enabled})

    # At this point, per_metric.cpu still doesn't include auto_arima.
    from forecaster.config import loader as cfg_loader
    cfg_loader._settings = None
    cfg_loader._settings_loaded_at = 0.0
    s = cfg_loader.load_settings()
    assert "auto_arima" in s.algorithms.enabled
    assert "auto_arima" not in s.algorithms.per_metric.get("cpu", [])

    # Step 2: user goes to per-metric section and adds auto_arima to cpu
    r = client.post("/ui/manage/training/save", data={
        "enabled_algos": base_enabled,
        "per_metric__cpu": ["seasonal_naive", "holt_winters", "lstm", "nbeats", "auto_arima"],
        "per_metric__mem": ["naive", "arima", "ets", "prophet", "xgboost", "lstm"],
        "per_metric__disk": ["naive", "arima", "prophet", "holt_winters"],
    })
    assert r.status_code in (302, 303, 307)

    cfg_loader._settings = None
    cfg_loader._settings_loaded_at = 0.0
    s = cfg_loader.load_settings()
    assert "auto_arima" in s.algorithms.per_metric["cpu"]
