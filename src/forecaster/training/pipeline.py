"""End-to-end per-job orchestration.

Inputs:  (instance, metric_name, horizon_name)
Outputs: a row of TrainingRun + per-algo metrics + forecasts + ranking,
         and N artifacts on the model-store volume.

The pipeline is callable from Celery, from the API for synchronous test
triggers, and from the CLI for one-off debugging.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from ..config.loader import get_settings
from ..data.factory import make_data_source
from ..evaluation.ranking import rank_models
from ..registry.repo import RegistryRepo
from ..registry.store import VolumeArtifactStore
from .parallel import train_all_algos

log = logging.getLogger(__name__)


def _step_to_timedelta(step: str) -> timedelta:
    return pd.Timedelta(step).to_pytimedelta()


def _horizon_steps(horizon: str, step: str) -> int:
    return int(pd.Timedelta(horizon) / pd.Timedelta(step))


def _fetch_series(
    *, instance: str, metric: str, query: str, horizon_step: str, lookback_days: int,
    instance_label: str,
) -> pd.Series:
    settings = get_settings()
    ds = make_data_source(settings.data_sources)
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days)
        results = ds.fetch_range(
            query, start, end, horizon_step,
            instance_label=instance_label, metric_name=metric,
        )
    finally:
        ds.close()
    for ts in results:
        if ts.instance == instance and not ts.df.empty:
            return ts.df["value"]
    raise RuntimeError(f"no data returned for {instance=} {metric=}")


def run_pipeline(*, instance: str, metric: str, horizon: str) -> int:
    """Run the full pipeline for one (instance, metric, horizon).

    Returns the TrainingRun id.
    """
    settings = get_settings()
    if horizon not in settings.horizons:
        raise KeyError(f"horizon '{horizon}' not configured")
    h = settings.horizons[horizon]
    if metric not in settings.metrics_to_forecast.queries:
        raise KeyError(f"metric '{metric}' not configured")
    query = settings.metrics_to_forecast.queries[metric]

    repo = RegistryRepo(settings.database_url)
    artifact_store = VolumeArtifactStore(settings.artifact_store.volume_path)

    config_snapshot = {
        "horizon": h.model_dump(),
        "training": settings.training.model_dump(),
        "ranking": settings.ranking.model_dump(),
        "algorithms": settings.algorithms.model_dump(),
    }
    run_id = repo.create_run(
        instance=instance, metric=metric, horizon=horizon,
        config_snapshot=config_snapshot,
    )

    t0 = time.perf_counter()
    error: str | None = None
    try:
        lookback = h.lookback_days or settings.training.lookback_days
        series = _fetch_series(
            instance=instance,
            metric=metric,
            query=query,
            horizon_step=h.step,
            lookback_days=lookback,
            instance_label=settings.targets.instance_label,
        )
        # Ensure regular spacing
        series = series.asfreq(pd.Timedelta(h.step)).interpolate("time").dropna()
        if len(series) < 50:
            raise RuntimeError(f"too few points after fetch+resample: {len(series)}")

        horizon_steps = _horizon_steps(h.horizon, h.step)
        shortlist = settings.algorithms.per_metric.get(metric) or settings.algorithms.enabled
        results = train_all_algos(
            series,
            algorithms=shortlist,
            defaults=settings.algorithms.defaults,
            horizon_steps=horizon_steps,
            folds=settings.training.backtest_folds,
            holdout_fraction=settings.training.backtest_holdout_fraction,
            alpha=settings.training.confidence_alpha,
            max_workers=settings.training.parallelism.algos_per_job,
        )
        successful = [r for r in results if r.ok]
        if not successful:
            raise RuntimeError("all algorithms failed during training")

        # Rank
        scored = {r.algo: r.scores for r in successful if r.scores}
        ranking = rank_models(scored, settings.ranking) if scored else []
        winner = ranking[0].algo if ranking else successful[0].algo

        # Persist per-algo metrics + artifacts + forecasts
        for r in successful:
            if r.scores:
                repo.add_metrics(run_id, r.algo, r.scores, fold=-1)
            for i, fold_scores in enumerate(r.per_fold_scores):
                repo.add_metrics(run_id, r.algo, fold_scores, fold=i)

            # Save artifact
            if r.artifact_bytes is not None:
                path = artifact_store._path_for(  # noqa: SLF001 - access ok within package
                    instance=instance, metric=metric, horizon=horizon,
                    algo=r.algo, run_id=run_id,
                )
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_bytes(r.artifact_bytes)
                repo.add_artifact(run_id, r.algo, str(path), len(r.artifact_bytes), r.train_duration_seconds)

            if r.forecast_point is not None and r.forecast_timestamps is not None:
                repo.add_forecasts(
                    run_id=run_id, instance=instance, metric=metric, horizon=horizon,
                    algo=r.algo, is_best=(r.algo == winner),
                    timestamps=list(r.forecast_timestamps),
                    point=[_finite(v) for v in r.forecast_point],
                    lower=[_finite(v) for v in (r.forecast_lower if r.forecast_lower is not None else [None] * len(r.forecast_point))],
                    upper=[_finite(v) for v in (r.forecast_upper if r.forecast_upper is not None else [None] * len(r.forecast_point))],
                )

        # Persist ranking row
        if ranking:
            repo.add_ranking(
                run_id=run_id, instance=instance, metric=metric, horizon=horizon,
                winning_algo=winner,
                ranked=[
                    {
                        "rank": rm.rank,
                        "algo": rm.algo,
                        "composite": _finite(rm.composite),
                        "raw_scores": {k: _finite(v) for k, v in rm.raw_scores.items()},
                        "normalised_scores": {k: _finite(v) for k, v in rm.normalised_scores.items()},
                    }
                    for rm in ranking
                ],
            )

    except Exception as exc:  # noqa: BLE001
        log.exception("training pipeline failed")
        error = f"{exc.__class__.__name__}: {exc}"
    finally:
        repo.mark_completed(run_id, time.perf_counter() - t0, error=error)
    return run_id


def _finite(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f
