"""Celery task entrypoints."""

from __future__ import annotations

import logging
import os
import sys

from celery import Celery

from ..config.loader import get_settings
from ..observability.logging import configure_logging
from .pipeline import run_pipeline

log = logging.getLogger(__name__)


def make_app() -> Celery:
    settings = get_settings()
    app = Celery(
        "forecaster",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
    )
    app.conf.update(
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        task_default_queue="forecaster",
        timezone="UTC",
        result_expires=3600 * 24 * 7,
        task_track_started=True,
    )
    return app


celery_app = make_app()


@celery_app.task(name="forecaster.train", bind=True, max_retries=2, default_retry_delay=60)
def train_task(
    self, instance: str, metric: str, horizon: str,
    overrides: dict | None = None,
) -> int:
    """Celery wrapper around run_pipeline.

    `overrides` is an optional JSON-safe dict; the Custom Run panel uses
    it to pass `algorithms` and `anomaly_filter` per single run without
    touching global settings. Existing scheduler callers don't pass it.
    """
    try:
        return run_pipeline(
            instance=instance, metric=metric, horizon=horizon,
            celery_task_id=self.request.id,
            overrides=overrides,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("train_task failed; retrying")
        raise self.retry(exc=exc)


def revoke_task(task_id: str | None) -> bool:
    """Cancel a Celery task. Returns True if a revoke was issued.

    Safe to call with a None / empty / unknown id (returns False).
    """
    if not task_id:
        return False
    try:
        celery_app.control.revoke(task_id, terminate=True, signal="SIGTERM")
    except Exception as exc:  # noqa: BLE001
        log.warning("revoke failed for %s: %s", task_id, exc)
        return False
    return True


def run_worker() -> None:
    configure_logging()
    concurrency = int(os.environ.get("CELERY_CONCURRENCY", "2"))
    argv = [
        "worker",
        "--loglevel=INFO",
        f"--concurrency={concurrency}",
        "-Q", "forecaster",
        "-n", f"worker@%h",
    ]
    sys.exit(celery_app.worker_main(argv=argv))


if __name__ == "__main__":  # pragma: no cover
    run_worker()
