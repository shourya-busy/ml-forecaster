"""APScheduler entrypoint.

Each configured horizon gets one cron job whose handler calls fan_out.
"""

from __future__ import annotations

import logging
import signal

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from ..config.loader import get_settings, reload_settings
from ..observability.logging import configure_logging
from .jobs import fan_out

log = logging.getLogger(__name__)


def _build_scheduler() -> BlockingScheduler:
    settings = get_settings()
    sched = BlockingScheduler(timezone="UTC")
    for name, spec in settings.horizons.items():
        trigger = CronTrigger.from_crontab(spec.retrain, timezone="UTC")
        sched.add_job(
            fan_out,
            trigger=trigger,
            id=f"fan_out_{name}",
            name=f"fan-out horizon={name}",
            kwargs={"horizon": name},
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=600,
        )
        log.info("scheduler: registered horizon=%s cron='%s'", name, spec.retrain)
    return sched


def _install_sighup_reload(sched: BlockingScheduler) -> None:
    def handler(_signum, _frame):
        log.info("SIGHUP — reloading config")
        reload_settings()
        # Re-build jobs to pick up changed cron expressions.
        for job in list(sched.get_jobs()):
            sched.remove_job(job.id)
        settings = get_settings()
        for name, spec in settings.horizons.items():
            trigger = CronTrigger.from_crontab(spec.retrain, timezone="UTC")
            sched.add_job(
                fan_out, trigger=trigger, id=f"fan_out_{name}",
                kwargs={"horizon": name}, replace_existing=True,
                max_instances=1, coalesce=True, misfire_grace_time=600,
            )

    signal.signal(signal.SIGHUP, handler)


def run() -> None:
    configure_logging()
    sched = _build_scheduler()
    _install_sighup_reload(sched)
    log.info("scheduler: starting")
    sched.start()


if __name__ == "__main__":  # pragma: no cover
    run()
