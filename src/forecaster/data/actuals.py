"""Live actuals overlay helper.

Pulls historical Prometheus/Mimir values for a configured (instance, metric)
target, snapped to the horizon's step, intended to be overlaid against a
forecast on a chart. Used by the target_detail page, the graphs gallery
tiles, and the /explore/timeseries endpoint.

Fails open: any data-source error logs a warning and returns an empty list,
so a flaky Prometheus never bricks a chart.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from ..config.schema import Settings
from .factory import make_data_source

log = logging.getLogger(__name__)


def _safe(v: Any) -> Any:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def fetch_actuals_for_target(
    settings: Settings,
    *,
    instance: str,
    metric: str,
    horizon: str,
    lookback_hours: int,
    forecast_first_ts: datetime | pd.Timestamp | None = None,
    forecast_last_ts: datetime | pd.Timestamp | None = None,
) -> list[dict[str, Any]]:
    """Fetch actual values for the (instance, metric) over a recent window.

    The window is `[start, end]` where:
        start = min(forecast_first_ts, now - lookback_hours)
        end   = min(forecast_last_ts,  now)
    so the result covers from `lookback_hours` ago through the present, plus
    any leading forecast timestamps that have already elapsed (the
    meaningful overlap with a forecast band).

    If the forecast timestamps are omitted, the window is simply
    `[now - lookback_hours, now]`.

    Returns: list of `{"ts": iso8601, "value": float}`, oldest first.
    """
    if horizon not in settings.horizons:
        return []
    query = settings.metrics_to_forecast.queries.get(metric)
    if not query:
        return []

    def _as_utc(t: Any) -> pd.Timestamp:
        ts = pd.Timestamp(t)
        # SQLite round-trips datetimes without tzinfo. Treat any naive
        # timestamp here as UTC (storage is always UTC in this project).
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

    now = pd.Timestamp.now(tz="UTC")
    fc_first = _as_utc(forecast_first_ts) if forecast_first_ts is not None else now
    fc_last = _as_utc(forecast_last_ts) if forecast_last_ts is not None else now
    start = min(fc_first, now - pd.Timedelta(hours=max(0, lookback_hours)))
    end = min(fc_last, now)
    if end <= start:
        return []

    step = settings.horizons[horizon].step
    try:
        ds = make_data_source(settings.data_sources)
        try:
            series_list = ds.fetch_range(
                query,
                start.to_pydatetime(),
                end.to_pydatetime(),
                step,
                instance_label=settings.targets.instance_label,
                metric_name=metric,
            )
        finally:
            ds.close()
    except Exception as exc:  # noqa: BLE001 — fail open
        log.warning("actuals fetch failed for %s/%s: %s", instance, metric, exc)
        return []

    for s in series_list:
        if s.instance != instance or s.df.empty:
            continue
        snapped = s.df["value"].asfreq(pd.Timedelta(step)).interpolate("time")
        return [
            {"ts": idx.isoformat(), "value": _safe(float(v))}
            for idx, v in snapped.dropna().items()
        ]
    return []


def fetch_raw_series(
    settings: Settings,
    *,
    query: str,
    start: datetime,
    end: datetime,
    step: str,
) -> list[dict[str, Any]]:
    """Run an arbitrary PromQL query and return all returned instances.

    Used by /explore/query. Returns a list of:
        {"instance": str, "values": [[iso_ts, value], ...]}
    Raises on data-source errors (callers map to HTTP 502).
    """
    ds = make_data_source(settings.data_sources)
    try:
        series_list = ds.fetch_range(
            query,
            start,
            end,
            step,
            instance_label=settings.targets.instance_label,
        )
    finally:
        ds.close()

    out: list[dict[str, Any]] = []
    for s in series_list:
        if s.df.empty:
            continue
        out.append({
            "instance": s.instance,
            "values": [
                [idx.isoformat(), _safe(float(v))]
                for idx, v in s.df["value"].items()
            ],
        })
    return out


def utc_now() -> datetime:
    """Tz-aware UTC now — extracted so tests can monkeypatch."""
    return datetime.now(tz=UTC)


# Re-export so callers do not need to import timedelta separately.
__all__ = [
    "fetch_actuals_for_target",
    "fetch_raw_series",
    "utc_now",
    "timedelta",
]
