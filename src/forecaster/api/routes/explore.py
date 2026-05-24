"""Explore endpoints — catalog + ad-hoc timeseries fetch.

These power the Graphs gallery (`/ui/graphs`) and the Explore page
(`/ui/explore`). The catalog lists the configured instances, metrics, and
horizons. /timeseries returns actual values for a configured (instance,
metric) pair; /query is an arbitrary PromQL passthrough.

All three reuse the existing TSDataSource layer via fetch_actuals_for_target
and fetch_raw_series in forecaster.data.actuals.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from ...config.schema import Settings
from ...data.actuals import fetch_actuals_for_target, fetch_raw_series
from ...data.factory import make_data_source
from ...models.registry import ALGO_INFO
from ..deps import settings_dep

log = logging.getLogger(__name__)

router = APIRouter(prefix="/explore", tags=["explore"])

# Hard caps so a runaway query can't OOM the browser. /api/v1/query_range
# itself caps at 11 000 points per call (the PrometheusClient chunks
# transparently above that). We mirror that here so we can return a clean
# 422 instead of waiting on a multi-minute chunked fetch.
MAX_POINTS_PER_REQUEST = 11_000
MAX_POINTS_PER_SERIES = 10_000
MAX_SERIES_RETURNED = 50
MIN_STEP_SECONDS = 1
MAX_STEP_SECONDS = 86_400


def _step_seconds(step: str) -> int:
    """Parse a pandas-style step ('30s', '1min', '5m', '1h') to seconds."""
    try:
        return int(pd.Timedelta(step).total_seconds())
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, f"invalid step '{step}': {exc}") from None


def _parse_iso(name: str, value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(422, f"invalid {name} '{value}': {exc}") from None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


@router.get("/catalog")
def catalog(
    refresh: bool = False,
    settings: Settings = Depends(settings_dep),
) -> dict[str, Any]:
    """Return everything the Graphs and Explore pages need for their dropdowns.

    `instances` is discovered from the data source when `targets.discovery`
    is `promql`; on any error it falls back to `static_instances` so the
    page still renders.
    """
    instances: list[str] = list(settings.targets.static_instances)
    discovery_error: str | None = None
    if settings.targets.discovery == "promql" and settings.targets.discovery_query:
        try:
            ds = make_data_source(settings.data_sources)
            try:
                discovered = ds.discover_instances(
                    settings.targets.discovery_query,
                    instance_label=settings.targets.instance_label,
                )
            finally:
                ds.close()
            if discovered:
                instances = sorted(set(discovered) | set(instances))
        except Exception as exc:  # noqa: BLE001
            log.warning("instance discovery failed: %s", exc)
            discovery_error = str(exc)

    metrics = [
        {"name": name, "query": q}
        for name, q in sorted(settings.metrics_to_forecast.queries.items())
    ]
    horizons = [
        {"name": h, "step": spec.step, "horizon": spec.horizon}
        for h, spec in settings.horizons.items()
    ]
    return {
        "instances": instances,
        "metrics": metrics,
        "horizons": horizons,
        "algorithms": {
            "enabled": list(settings.algorithms.enabled),
            "info": ALGO_INFO,
        },
        "instance_label": settings.targets.instance_label,
        "discovery_error": discovery_error,
    }


@router.get("/timeseries")
def timeseries(
    instance: str,
    metric: str,
    horizon: str,
    lookback_hours: int = Query(24, ge=0, le=720),
    settings: Settings = Depends(settings_dep),
) -> dict[str, Any]:
    """Return actual values for a configured (instance, metric, horizon).

    Time window: [now - lookback_hours, now]. The horizon's step is used as
    the resample step. Fails open: errors return an empty `values` array
    rather than 5xx.
    """
    if metric not in settings.metrics_to_forecast.queries:
        raise HTTPException(404, f"metric '{metric}' not in metrics_to_forecast")
    if horizon not in settings.horizons:
        raise HTTPException(404, f"horizon '{horizon}' not configured")

    values = fetch_actuals_for_target(
        settings,
        instance=instance,
        metric=metric,
        horizon=horizon,
        lookback_hours=lookback_hours,
    )
    return {
        "instance": instance,
        "metric": metric,
        "horizon": horizon,
        "lookback_hours": lookback_hours,
        "step": settings.horizons[horizon].step,
        "values": values,
    }


@router.get("/query")
def query(
    query: str = Query(..., min_length=1, alias="query"),
    start: str | None = None,
    end: str | None = None,
    step: str = "60s",
    settings: Settings = Depends(settings_dep),
) -> dict[str, Any]:
    """Run an arbitrary PromQL / MetricsQL query.

    `start` and `end` are ISO-8601; if omitted, defaults to the last hour.
    `step` accepts pandas-style ('60s', '5min', '1h').
    Returns one entry per series; up to MAX_SERIES_RETURNED series, each
    truncated to MAX_POINTS_PER_SERIES with a `truncated` warning if needed.
    """
    now = datetime.now(tz=UTC)
    start_dt = _parse_iso("start", start) if start else now - timedelta(hours=1)
    end_dt = _parse_iso("end", end) if end else now
    if end_dt <= start_dt:
        raise HTTPException(422, "end must be after start")

    step_sec = _step_seconds(step)
    if step_sec < MIN_STEP_SECONDS or step_sec > MAX_STEP_SECONDS:
        raise HTTPException(
            422, f"step must be between {MIN_STEP_SECONDS}s and {MAX_STEP_SECONDS}s"
        )

    total_seconds = (end_dt - start_dt).total_seconds()
    expected_points = total_seconds / step_sec
    if expected_points > MAX_POINTS_PER_REQUEST:
        raise HTTPException(
            422,
            f"requested {int(expected_points)} points exceeds the {MAX_POINTS_PER_REQUEST} "
            "limit — widen the step or shorten the time range",
        )

    try:
        series = fetch_raw_series(
            settings, query=query, start=start_dt, end=end_dt, step=step,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("explore query failed: %s", exc)
        raise HTTPException(502, f"data source error: {exc}") from None

    warnings: list[str] = []
    if len(series) > MAX_SERIES_RETURNED:
        warnings.append(
            f"returned {len(series)} series, truncated to {MAX_SERIES_RETURNED}"
        )
        series = series[:MAX_SERIES_RETURNED]
    for s in series:
        if len(s["values"]) > MAX_POINTS_PER_SERIES:
            warnings.append(
                f"series {s['instance']} truncated from {len(s['values'])} to "
                f"{MAX_POINTS_PER_SERIES} points"
            )
            s["values"] = s["values"][:MAX_POINTS_PER_SERIES]

    return {
        "request": {
            "query": query,
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "step": step,
            "step_seconds": step_sec,
        },
        "series": series,
        "warnings": warnings,
    }
