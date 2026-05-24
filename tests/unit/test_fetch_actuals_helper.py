"""Unit tests for forecaster.data.actuals helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from forecaster.config.loader import load_settings
from forecaster.data import actuals as actuals_mod
from forecaster.data.actuals import fetch_actuals_for_target, fetch_raw_series
from forecaster.data.base import TimeSeries, TSDataSource

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


class _StubDS(TSDataSource):
    """In-memory data source that records calls and replays fixtures."""

    def __init__(self, *, series: list[TimeSeries] | None = None, raises: Exception | None = None):
        self._series = series or []
        self._raises = raises
        self.calls: list[dict] = []
        self.closed = False

    def fetch_range(self, query, start, end, step, *, instance_label="instance", metric_name=None):
        self.calls.append({
            "query": query, "start": start, "end": end, "step": step,
            "instance_label": instance_label, "metric_name": metric_name,
        })
        if self._raises:
            raise self._raises
        return self._series

    def discover_instances(self, query, instance_label="instance"):
        return []

    def close(self):
        self.closed = True


def _settings():
    s = load_settings(CONFIG_DIR)
    # Ensure 'cpu' is configured (it is in the shipped default.yaml).
    assert "cpu" in s.metrics_to_forecast.queries
    return s


def _make_series(instance: str, n: int = 6, step: str = "1min") -> TimeSeries:
    idx = pd.date_range("2026-05-20T00:00:00Z", periods=n, freq=step)
    df = pd.DataFrame({"value": [float(i) for i in range(n)]}, index=idx)
    return TimeSeries(instance=instance, metric="cpu", step=step, df=df)


def test_returns_empty_when_metric_not_configured(monkeypatch):
    s = _settings()
    monkeypatch.setattr(actuals_mod, "make_data_source", lambda cfg: _StubDS())
    out = fetch_actuals_for_target(
        s, instance="fake-1", metric="not_a_real_metric",
        horizon="short", lookback_hours=1,
    )
    assert out == []


def test_returns_empty_when_horizon_not_configured(monkeypatch):
    s = _settings()
    monkeypatch.setattr(actuals_mod, "make_data_source", lambda cfg: _StubDS())
    out = fetch_actuals_for_target(
        s, instance="fake-1", metric="cpu",
        horizon="bogus", lookback_hours=1,
    )
    assert out == []


def test_returns_empty_when_ds_raises(monkeypatch):
    s = _settings()
    monkeypatch.setattr(
        actuals_mod, "make_data_source",
        lambda cfg: _StubDS(raises=RuntimeError("boom")),
    )
    out = fetch_actuals_for_target(
        s, instance="fake-1", metric="cpu",
        horizon="short", lookback_hours=2,
    )
    assert out == []


def test_returns_only_matching_instance(monkeypatch):
    s = _settings()
    series = [_make_series("fake-1"), _make_series("fake-2")]
    stub = _StubDS(series=series)
    monkeypatch.setattr(actuals_mod, "make_data_source", lambda cfg: stub)

    out = fetch_actuals_for_target(
        s, instance="fake-2", metric="cpu",
        horizon="short", lookback_hours=24,
    )
    assert out, "expected non-empty actuals"
    assert all("ts" in p and "value" in p for p in out)
    assert stub.closed, "data source should be closed even on success"


def test_window_uses_forecast_bounds_when_provided(monkeypatch):
    s = _settings()
    stub = _StubDS(series=[_make_series("fake-1")])
    monkeypatch.setattr(actuals_mod, "make_data_source", lambda cfg: stub)

    # Forecast first/last in the future shouldn't extend `end` past now.
    fc_first = pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=1)
    fc_last = pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=2)
    out = fetch_actuals_for_target(
        s, instance="fake-1", metric="cpu",
        horizon="short", lookback_hours=3,
        forecast_first_ts=fc_first, forecast_last_ts=fc_last,
    )
    assert out
    call = stub.calls[0]
    # end clipped to "now"
    assert call["end"] <= datetime.now(tz=UTC) + pd.Timedelta(minutes=1).to_pytimedelta()


def test_fetch_raw_series_returns_per_instance(monkeypatch):
    s = _settings()
    series = [_make_series("a"), _make_series("b")]
    monkeypatch.setattr(actuals_mod, "make_data_source", lambda cfg: _StubDS(series=series))

    out = fetch_raw_series(
        s, query="up", start=datetime(2026, 5, 20, tzinfo=UTC),
        end=datetime(2026, 5, 20, 1, tzinfo=UTC), step="1min",
    )
    assert {r["instance"] for r in out} == {"a", "b"}
    assert all(isinstance(r["values"], list) and r["values"] for r in out)
    # Each value is [iso_ts, float]
    first = out[0]["values"][0]
    assert isinstance(first[0], str) and isinstance(first[1], float)
