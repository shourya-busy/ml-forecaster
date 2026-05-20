"""Prometheus HTTP API client.

Uses /api/v1/query_range. Splits long ranges into chunks if the result
would exceed Prometheus' default 11000-sample point cap.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config.schema import DataSourceEndpoint
from .base import FetchError, TSDataSource, TimeSeries

log = logging.getLogger(__name__)

# Prometheus rejects query_range with more than 11000 points.
# We keep a safety margin.
_MAX_POINTS_PER_REQUEST = 10_000


def _step_to_seconds(step: str) -> int:
    """Convert '1min', '5min', '1h' etc. to seconds."""
    td = pd.Timedelta(step.replace("min", "min"))
    return int(td.total_seconds())


def _ensure_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class PrometheusClient(TSDataSource):
    """Thin wrapper around the Prometheus HTTP API."""

    def __init__(self, endpoint: DataSourceEndpoint) -> None:
        self._endpoint = endpoint
        self._client = httpx.Client(
            base_url=endpoint.base_url.rstrip("/"),
            timeout=endpoint.timeout_seconds,
            verify=endpoint.verify_tls,
            headers=self._build_headers(endpoint),
            auth=(
                (endpoint.basic_auth_user, endpoint.basic_auth_password)
                if endpoint.basic_auth_user
                else None
            ),
        )

    @staticmethod
    def _build_headers(endpoint: DataSourceEndpoint) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/json"}
        if endpoint.bearer_token:
            headers["Authorization"] = f"Bearer {endpoint.bearer_token}"
        if endpoint.tenant_id:
            headers["X-Scope-OrgID"] = endpoint.tenant_id
        return headers

    @retry(
        wait=wait_exponential(multiplier=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((httpx.HTTPError,)),
        reraise=True,
    )
    def _query_range(
        self, query: str, start: datetime, end: datetime, step_s: int
    ) -> list[dict]:
        resp = self._client.get(
            "/api/v1/query_range",
            params={
                "query": query,
                "start": start.timestamp(),
                "end": end.timestamp(),
                "step": step_s,
            },
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") != "success":
            raise FetchError(f"prometheus error: {payload.get('error')}")
        return payload["data"]["result"]

    @retry(
        wait=wait_exponential(multiplier=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((httpx.HTTPError,)),
        reraise=True,
    )
    def _query_instant(self, query: str) -> list[dict]:
        resp = self._client.get("/api/v1/query", params={"query": query})
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") != "success":
            raise FetchError(f"prometheus error: {payload.get('error')}")
        return payload["data"]["result"]

    def fetch_range(
        self,
        query: str,
        start: datetime,
        end: datetime,
        step: str,
        *,
        instance_label: str = "instance",
        metric_name: str | None = None,
    ) -> list[TimeSeries]:
        start_u = _ensure_utc(start)
        end_u = _ensure_utc(end)
        step_s = _step_to_seconds(step)
        chunk_span = timedelta(seconds=step_s * _MAX_POINTS_PER_REQUEST)

        merged: dict[str, list[tuple[float, str]]] = {}

        cur = start_u
        while cur < end_u:
            chunk_end = min(cur + chunk_span, end_u)
            results = self._query_range(query, cur, chunk_end, step_s)
            for r in results:
                inst = r.get("metric", {}).get(instance_label, "")
                if not inst:
                    log.debug("skip series missing %s label: %s", instance_label, r["metric"])
                    continue
                merged.setdefault(inst, []).extend(r["values"])
            cur = chunk_end

        out: list[TimeSeries] = []
        for inst, raw in merged.items():
            if not raw:
                continue
            # de-dup overlapping chunk boundaries
            df = (
                pd.DataFrame(raw, columns=["ts", "value"])
                .drop_duplicates(subset="ts")
                .assign(
                    ts=lambda d: pd.to_datetime(d["ts"].astype(float), unit="s", utc=True),
                    value=lambda d: pd.to_numeric(d["value"], errors="coerce"),
                )
                .dropna()
                .set_index("ts")
                .sort_index()
            )
            if df.empty:
                continue
            out.append(
                TimeSeries(instance=inst, metric=metric_name or "", step=step, df=df)
            )
        return out

    def discover_instances(self, query: str, instance_label: str = "instance") -> list[str]:
        results = self._query_instant(query)
        seen: set[str] = set()
        for r in results:
            v = r.get("metric", {}).get(instance_label)
            if v:
                seen.add(v)
        return sorted(seen)

    def close(self) -> None:
        self._client.close()
