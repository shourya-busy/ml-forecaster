"""Mimir client.

Mimir exposes the Prometheus HTTP API verbatim under a configurable
prefix. The only meaningful difference is the X-Scope-OrgID tenant header,
already handled by PrometheusClient when tenant_id is set on the
endpoint. This class exists so the factory can pick a clear class name
and so that any future Mimir-specific divergence has a home.
"""

from __future__ import annotations

from .prometheus_client import PrometheusClient


class MimirClient(PrometheusClient):
    """Identical wire protocol to Prometheus; relies on tenant_id header."""

    pass
