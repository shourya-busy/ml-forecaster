"""Pluggable time-series data-source interface.

A TSDataSource fetches a single series for a given PromQL expression in a
time window at a given step. Implementations: PrometheusClient,
MimirClient (which is just Prometheus + a tenant header).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import datetime

import pandas as pd


class FetchError(RuntimeError):
    """Raised when a data-source fetch fails for non-transient reasons."""


@dataclass(slots=True)
class TimeSeries:
    """Per-instance series result.

    The DataFrame has a tz-aware DatetimeIndex (UTC) and a single column
    'value'. Missing samples are dropped; gaps may exist.
    """

    instance: str
    metric: str
    step: str
    df: pd.DataFrame


class TSDataSource(abc.ABC):
    """ABC for any time-series fetcher."""

    @abc.abstractmethod
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
        """Run query in [start, end] at `step`, return one TimeSeries per instance."""

    @abc.abstractmethod
    def discover_instances(self, query: str, instance_label: str = "instance") -> list[str]:
        """Run an instant query and extract distinct values of `instance_label`."""

    def close(self) -> None:  # pragma: no cover - default no-op
        return
