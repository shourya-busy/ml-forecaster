"""Target discovery + fan-out into Celery.

A scheduler tick for horizon "medium" produces N tasks where N =
|targets| × |metrics|. We stagger them across `fetch_jitter_seconds` to
avoid hammering Prometheus.
"""

from __future__ import annotations

import logging
import random

from ..config.loader import get_settings
from ..data.factory import make_data_source
from ..training.tasks import train_task

log = logging.getLogger(__name__)


def discover_targets() -> list[str]:
    """Return the list of instance identifiers to forecast."""
    settings = get_settings()
    t = settings.targets
    if t.discovery == "static":
        return list(t.static_instances)
    if not t.discovery_query:
        raise ValueError("targets.discovery=promql requires discovery_query")
    ds = make_data_source(settings.data_sources)
    try:
        return ds.discover_instances(t.discovery_query, instance_label=t.instance_label)
    finally:
        ds.close()


def fan_out(horizon: str) -> int:
    """Enqueue training tasks for one (horizon) tick. Returns the count."""
    settings = get_settings()
    if horizon not in settings.horizons:
        raise KeyError(f"horizon '{horizon}' not configured")
    targets = discover_targets()
    metrics = list(settings.metrics_to_forecast.queries.keys())
    jitter = settings.training.parallelism.fetch_jitter_seconds

    n = 0
    for instance in targets:
        for metric in metrics:
            countdown = random.randint(0, max(0, jitter))
            train_task.apply_async(
                args=[instance, metric, horizon],
                countdown=countdown,
            )
            n += 1
    log.info("scheduler: fan-out horizon=%s tasks=%d targets=%d metrics=%d",
             horizon, n, len(targets), len(metrics))
    return n
