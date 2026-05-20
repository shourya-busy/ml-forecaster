"""APScheduler entrypoint.

Two layers of cron jobs:

1. **Per-horizon** (global) — one job per configured horizon, firing
   `fan_out(horizon)`. Reads cron from `horizons.<name>.retrain`,
   override-able via `settings_overrides[horizons.<name>.retrain]`.

2. **Per-target** (DB) — for each TargetOverride row that has a
   non-null `schedule_cron`, register an additional job that fires a
   single training task. These coexist with the global jobs;
   `fan_out` skips any target with a per-target cron so we don't
   double-fire.

Jobs are reconciled at startup, on SIGHUP, and once per minute
(via an internal housekeeping job) so UI edits propagate within a
minute without restarts.
"""

from __future__ import annotations

import logging
import signal

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from ..config.loader import get_settings, reload_settings
from ..observability.logging import configure_logging
from ..registry.repo import RegistryRepo
from .jobs import fan_out, fire_single_target

log = logging.getLogger(__name__)

_RECONCILE_JOB_ID = "__forecaster_reconcile__"


def _per_horizon_job_id(name: str) -> str:
    return f"fan_out_{name}"


def _per_target_job_id(instance: str, metric: str, horizon: str) -> str:
    return f"target::{instance}::{metric}::{horizon}"


def _reconcile_jobs(sched: BlockingScheduler) -> None:
    """Sync the scheduler's job list with the current effective config."""
    settings = get_settings()
    tz = settings.display_timezone

    desired: set[str] = set()

    # Per-horizon jobs
    for name, spec in settings.horizons.items():
        job_id = _per_horizon_job_id(name)
        desired.add(job_id)
        trigger = CronTrigger.from_crontab(spec.retrain, timezone=tz)
        sched.add_job(
            fan_out, trigger=trigger,
            id=job_id, name=f"fan-out horizon={name}",
            kwargs={"horizon": name},
            replace_existing=True, max_instances=1,
            coalesce=True, misfire_grace_time=600,
        )

    # Per-target jobs
    try:
        repo = RegistryRepo(settings.database_url)
        overrides = repo.get_target_overrides()
    except Exception as exc:  # noqa: BLE001 - DB may be unreachable briefly
        log.warning("could not read target_overrides: %s", exc)
        overrides = []

    for ov in overrides:
        if not ov["enabled"] or not ov.get("schedule_cron"):
            continue
        job_id = _per_target_job_id(ov["instance"], ov["metric"], ov["horizon"])
        desired.add(job_id)
        try:
            trigger = CronTrigger.from_crontab(ov["schedule_cron"], timezone=tz)
        except ValueError as exc:
            log.warning("invalid cron for %s: %s — skipping", job_id, exc)
            continue
        sched.add_job(
            fire_single_target, trigger=trigger,
            id=job_id, name=f"per-target {ov['instance']}/{ov['metric']}/{ov['horizon']}",
            kwargs={
                "instance": ov["instance"],
                "metric": ov["metric"],
                "horizon": ov["horizon"],
            },
            replace_existing=True, max_instances=1,
            coalesce=True, misfire_grace_time=600,
        )

    desired.add(_RECONCILE_JOB_ID)

    # Drop any jobs the user has removed
    for job in list(sched.get_jobs()):
        if job.id not in desired:
            sched.remove_job(job.id)

    log.info("scheduler: reconciled — %d jobs total", len(desired))


def _build_scheduler() -> BlockingScheduler:
    settings = get_settings()
    sched = BlockingScheduler(timezone=settings.display_timezone)
    _reconcile_jobs(sched)
    # Periodic reconciliation so UI edits propagate without SIGHUP.
    sched.add_job(
        _reconcile_jobs, "interval", seconds=60,
        id=_RECONCILE_JOB_ID, args=[sched],
        replace_existing=True, max_instances=1, coalesce=True,
    )
    log.info("scheduler: built with timezone=%s", settings.display_timezone)
    return sched


def _install_sighup_reload(sched: BlockingScheduler) -> None:
    def handler(_signum, _frame):
        log.info("SIGHUP — reloading config and reconciling jobs")
        reload_settings()
        _reconcile_jobs(sched)
    signal.signal(signal.SIGHUP, handler)


def run() -> None:
    configure_logging()
    sched = _build_scheduler()
    _install_sighup_reload(sched)
    log.info("scheduler: starting")
    sched.start()


if __name__ == "__main__":  # pragma: no cover
    run()
