"""Target discovery + fan-out into Celery.

A scheduler tick for horizon "medium" produces N tasks where N =
|targets| × |metrics| minus anything disabled via the `target_overrides`
DB table. We stagger them across `fetch_jitter_seconds` to avoid
hammering Prometheus.
"""

from __future__ import annotations

import logging
import random

from croniter import croniter

from ..config.loader import get_settings
from ..data.factory import make_data_source
from ..registry.repo import RegistryRepo
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


def _repo() -> RegistryRepo:
    """Built fresh per call so each scheduler tick sees latest DB state."""
    return RegistryRepo(get_settings().database_url)


def fan_out(horizon: str) -> int:
    """Enqueue training tasks for one (horizon) tick. Returns the count.

    Respects per-target overrides:
      - Targets with `enabled=False` are skipped.
      - Targets with a non-null `schedule_cron` are skipped here (they
        are scheduled by their own per-target cron job in scheduler.py).
    """
    settings = get_settings()
    if horizon not in settings.horizons:
        raise KeyError(f"horizon '{horizon}' not configured")
    targets = discover_targets()
    metrics = list(settings.metrics_to_forecast.queries.keys())
    jitter = settings.training.parallelism.fetch_jitter_seconds

    overrides = _repo().get_target_overrides_map()

    n = 0
    skipped_disabled = 0
    skipped_custom_cron = 0
    for instance in targets:
        for metric in metrics:
            key = (instance, metric, horizon)
            ov = overrides.get(key)
            if ov:
                if not ov["enabled"]:
                    skipped_disabled += 1
                    continue
                if ov.get("schedule_cron"):
                    # Handled by its own per-target cron job; don't double-fire here
                    skipped_custom_cron += 1
                    continue
            countdown = random.randint(0, max(0, jitter))
            train_task.apply_async(
                args=[instance, metric, horizon],
                countdown=countdown,
            )
            n += 1
    log.info(
        "scheduler: fan-out horizon=%s enqueued=%d skipped_disabled=%d skipped_custom_cron=%d targets=%d metrics=%d",
        horizon, n, skipped_disabled, skipped_custom_cron, len(targets), len(metrics),
    )
    return n


def fire_single_target(instance: str, metric: str, horizon: str) -> str | None:
    """Enqueue one training task for a per-target cron job.

    Returns the Celery task id, or None if the target is disabled.
    """
    repo = _repo()
    ov_map = repo.get_target_overrides_map()
    ov = ov_map.get((instance, metric, horizon))
    if ov and not ov["enabled"]:
        log.info("per-target cron: skipping disabled %s/%s/%s", instance, metric, horizon)
        return None
    result = train_task.apply_async(args=[instance, metric, horizon])
    return result.id


def next_fires(cron_expr: str, count: int = 5, tz: str | None = None) -> list[str]:
    """Compute the next `count` ISO timestamps a cron expression will fire at.

    Used by the Schedule page to preview upcoming runs.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    tz_obj = ZoneInfo(tz or get_settings().display_timezone)
    base = datetime.now(tz_obj)
    it = croniter(cron_expr, base)
    return [it.get_next(datetime).isoformat() for _ in range(count)]
