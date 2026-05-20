"""Build a TSDataSource from configuration."""

from __future__ import annotations

from ..config.schema import DataSourcesConfig
from .base import TSDataSource
from .mimir_client import MimirClient
from .prometheus_client import PrometheusClient


def make_data_source(cfg: DataSourcesConfig) -> TSDataSource:
    endpoint = cfg.endpoints.get(cfg.active)
    if endpoint is None:
        raise KeyError(f"data_sources.active='{cfg.active}' not in endpoints")
    if endpoint.kind == "mimir":
        return MimirClient(endpoint)
    return PrometheusClient(endpoint)
