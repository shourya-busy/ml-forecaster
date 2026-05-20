"""Run a synthetic pipeline pass without needing a real Prometheus.

Usage: forecaster-seed --instances fake-1,fake-2 --days 30
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main() -> None:
    # Make src importable even when run from inside the container without
    # installation (rare; the entrypoint script normally handles this).
    here = Path(__file__).resolve().parent.parent
    src = here / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from forecaster.config.loader import get_settings
    from forecaster.observability.logging import configure_logging
    from forecaster.training import pipeline as pl

    sys.path.insert(0, str(here))  # so tests/fixtures resolves when present
    try:
        from tests.fixtures.synthetic_series import synthetic_series  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover
        from scripts._fallback_synth import synthetic_series  # type: ignore[import-not-found]

    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", default="fake-1,fake-2")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--horizons", default="medium")
    parser.add_argument("--metrics", default="cpu,mem,disk")
    args = parser.parse_args()

    configure_logging()
    log = logging.getLogger("seed")
    settings = get_settings()

    instances = [s.strip() for s in args.instances.split(",") if s.strip()]
    horizons = [s.strip() for s in args.horizons.split(",") if s.strip()]
    metrics = [s.strip() for s in args.metrics.split(",") if s.strip()]

    # Monkeypatch the pipeline's fetch to return synthetic data.
    def fake_fetch(*, instance: str, metric: str, horizon_step: str, **_kw):
        return synthetic_series(days=args.days, step=horizon_step, seed=abs(hash((instance, metric))) % 100000)

    pl._fetch_series = fake_fetch  # type: ignore[assignment]

    for inst in instances:
        for h in horizons:
            if h not in settings.horizons:
                log.warning("skip unknown horizon=%s", h)
                continue
            for m in metrics:
                if m not in settings.metrics_to_forecast.queries:
                    log.warning("skip unknown metric=%s", m)
                    continue
                log.info("seeding instance=%s metric=%s horizon=%s", inst, m, h)
                run_id = pl.run_pipeline(instance=inst, metric=m, horizon=h)
                log.info("  → run_id=%s", run_id)


if __name__ == "__main__":
    main()
