"""Unit tests for /explore REST routes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


@pytest.fixture
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    db = tmp_path / "explore.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("FORECASTER__ARTIFACT_STORE__VOLUME_PATH", str(tmp_path / "art"))

    from fastapi.testclient import TestClient

    from forecaster.api import deps
    deps._repo.cache_clear()
    from forecaster.api.main import create_app
    from forecaster.config.loader import get_settings
    from forecaster.registry.repo import RegistryRepo
    repo = RegistryRepo(get_settings().database_url)
    repo.create_schema()
    return TestClient(create_app())


def _stub_ds(monkeypatch, *, series=None, raises=None, discover=None):
    """Replace make_data_source everywhere it's referenced."""
    from forecaster.data.base import TSDataSource

    class _Stub(TSDataSource):
        def fetch_range(self, *a, **kw):
            if raises:
                raise raises
            return series or []
        def discover_instances(self, *a, **kw):
            if raises:
                raise raises
            return discover or []
        def close(self):
            pass

    from forecaster.api.routes import explore as explore_mod
    from forecaster.data import actuals as actuals_mod
    monkeypatch.setattr(actuals_mod, "make_data_source", lambda cfg: _Stub())
    monkeypatch.setattr(explore_mod, "make_data_source", lambda cfg: _Stub())


def _ts(instance: str, n: int = 4) -> object:
    from forecaster.data.base import TimeSeries
    idx = pd.date_range("2026-05-20T00:00:00Z", periods=n, freq="1min")
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0][:n]}, index=idx)
    return TimeSeries(instance=instance, metric="cpu", step="1min", df=df)


def test_catalog_lists_metrics_horizons(client, monkeypatch):
    _stub_ds(monkeypatch, discover=["fake-1", "fake-2"])
    r = client.get("/explore/catalog")
    assert r.status_code == 200
    body = r.json()
    assert "cpu" in [m["name"] for m in body["metrics"]]
    assert {"short", "medium", "long"} <= set(h["name"] for h in body["horizons"])
    # Static fallback merged with discovery results
    assert "fake-1" in body["instances"]
    assert body["instance_label"] == "instance"
    assert body["discovery_error"] is None


def test_catalog_falls_back_to_static_on_discovery_error(client, monkeypatch):
    _stub_ds(monkeypatch, raises=RuntimeError("upstream down"))
    r = client.get("/explore/catalog")
    assert r.status_code == 200
    body = r.json()
    assert "fake-1" in body["instances"]  # from static_instances
    assert "upstream down" in (body.get("discovery_error") or "")


def test_timeseries_returns_values(client, monkeypatch):
    _stub_ds(monkeypatch, series=[_ts("fake-1")])
    r = client.get("/explore/timeseries", params={
        "instance": "fake-1", "metric": "cpu", "horizon": "short",
        "lookback_hours": 1,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["instance"] == "fake-1"
    assert body["step"] == "1min"
    # values may be empty if the (synthetic) data window doesn't overlap now,
    # but the response shape must always be intact
    assert isinstance(body["values"], list)


def test_timeseries_404_for_unknown_metric(client, monkeypatch):
    _stub_ds(monkeypatch, series=[])
    r = client.get("/explore/timeseries", params={
        "instance": "fake-1", "metric": "nope", "horizon": "short",
    })
    assert r.status_code == 404


def test_timeseries_404_for_unknown_horizon(client, monkeypatch):
    _stub_ds(monkeypatch, series=[])
    r = client.get("/explore/timeseries", params={
        "instance": "fake-1", "metric": "cpu", "horizon": "bogus",
    })
    assert r.status_code == 404


def test_query_basic_passthrough(client, monkeypatch):
    _stub_ds(monkeypatch, series=[_ts("a"), _ts("b")])
    now = datetime.now(tz=UTC)
    r = client.get("/explore/query", params={
        "query": "up",
        "start": (now - timedelta(hours=1)).isoformat(),
        "end": now.isoformat(),
        "step": "60s",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["request"]["query"] == "up"
    assert body["request"]["step_seconds"] == 60
    instances = {s["instance"] for s in body["series"]}
    assert instances == {"a", "b"}


def test_query_rejects_invalid_step(client, monkeypatch):
    _stub_ds(monkeypatch, series=[])
    r = client.get("/explore/query", params={"query": "up", "step": "0s"})
    assert r.status_code == 422


def test_query_rejects_too_many_points(client, monkeypatch):
    _stub_ds(monkeypatch, series=[])
    # 30 days at 1s step = 2.6M points → must 422
    now = datetime.now(tz=UTC)
    r = client.get("/explore/query", params={
        "query": "up",
        "start": (now - timedelta(days=30)).isoformat(),
        "end": now.isoformat(),
        "step": "1s",
    })
    assert r.status_code == 422
    assert "exceeds" in r.json()["detail"].lower()


def test_query_rejects_inverted_range(client, monkeypatch):
    _stub_ds(monkeypatch, series=[])
    now = datetime.now(tz=UTC)
    r = client.get("/explore/query", params={
        "query": "up",
        "start": now.isoformat(),
        "end": (now - timedelta(hours=1)).isoformat(),
        "step": "60s",
    })
    assert r.status_code == 422


def test_query_502_on_ds_error(client, monkeypatch):
    _stub_ds(monkeypatch, raises=RuntimeError("prom unreachable"))
    r = client.get("/explore/query", params={"query": "up", "step": "60s"})
    assert r.status_code == 502
    assert "prom unreachable" in r.json()["detail"]
