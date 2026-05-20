"""Prometheus exposition endpoint.

Emits gauges so Prometheus can scrape and Grafana can overlay. Series
families are toggled by exposition.yaml so you can dial cardinality
without changing code.

Series families:
    forecast_best_value{instance, metric, horizon, bound}
    forecast_best_model_info{instance, metric, horizon, model} = 1
    forecast_value{instance, metric, horizon, model, bound}
    forecast_model_score{instance, metric, horizon, model, score}
    forecaster_training_run_timestamp_seconds{instance, metric, horizon}
    forecaster_training_duration_seconds{instance, metric, horizon, model}
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from fastapi import APIRouter, Depends, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Gauge, generate_latest
from sqlalchemy import select

from ..config.schema import Settings
from ..registry.models import Forecast, ModelArtifact, Ranking, RunMetric, TrainingRun
from ..registry.repo import RegistryRepo
from .deps import repo_dep, settings_dep

router = APIRouter(tags=["metrics"])


def _build_registry(repo: RegistryRepo, settings: Settings) -> CollectorRegistry:
    """Build a fresh registry per scrape.

    We deliberately rebuild on every /metrics call because the underlying
    forecasts in the DB change with every training run. Cardinality is
    controlled via settings.exposition.emit toggles.
    """
    reg = CollectorRegistry()
    emit = settings.exposition.emit
    include_horizon = settings.exposition.labels.include_horizon
    series_per = max(1, int(settings.exposition.series_per_forecast))

    # Common label sets
    target_labels = ["instance", "metric"] + (["horizon"] if include_horizon else [])

    best_value = Gauge(
        "forecast_best_value",
        "Point forecast (best ranked model) at a future timestamp.",
        [*target_labels, "bound", "ts"],
        registry=reg,
    ) if emit.best_model_forecast else None

    best_info = Gauge(
        "forecast_best_model_info",
        "Marker series; value=1 indicates which model is currently best for the target.",
        [*target_labels, "model"],
        registry=reg,
    ) if emit.best_model_forecast else None

    per_model_value = Gauge(
        "forecast_value",
        "Per-model point forecast at a future timestamp.",
        [*target_labels, "model", "bound", "ts"],
        registry=reg,
    ) if emit.per_model_forecast else None

    model_score = Gauge(
        "forecast_model_score",
        "Backtest score (raw value) for a model.",
        [*target_labels, "model", "score"],
        registry=reg,
    ) if emit.ranking_scores else None

    run_ts = Gauge(
        "forecaster_training_run_timestamp_seconds",
        "Unix time of the latest completed training run.",
        target_labels,
        registry=reg,
    ) if emit.training_run_timestamps else None

    train_duration = Gauge(
        "forecaster_training_duration_seconds",
        "Per-algorithm training wall-clock duration of the latest run.",
        [*target_labels, "model"],
        registry=reg,
    ) if emit.training_durations else None

    winner_gauge = Gauge(
        "forecaster_winner",
        "Marker (=1) for the current winning algorithm per target. Mirrors the "
        "winners table in /diagnostics/winners.",
        [*target_labels, "model"],
        registry=reg,
    ) if emit.diagnostics else None

    winner_unique_gauge = Gauge(
        "forecaster_winner_unique_recent",
        "Number of distinct winning algos across the last K completed runs "
        "(K = exposition.diagnostics.recent_window_runs). 1 = stable, K = flapping.",
        target_labels,
        registry=reg,
    ) if emit.diagnostics else None

    # ---- Pull data once per scrape ----
    with repo.session() as s:
        # Latest completed run per (instance, metric, horizon)
        latest_runs_q = (
            select(TrainingRun)
            .where(TrainingRun.status == "completed")
            .order_by(
                TrainingRun.instance, TrainingRun.metric, TrainingRun.horizon,
                TrainingRun.completed_at.desc(),
            )
            .distinct(TrainingRun.instance, TrainingRun.metric, TrainingRun.horizon)
        )
        latest_runs = list(s.scalars(latest_runs_q))
        run_ids = [r.id for r in latest_runs]
        run_meta = {r.id: r for r in latest_runs}

        if run_ids:
            forecasts = list(s.scalars(select(Forecast).where(Forecast.run_id.in_(run_ids))))
            metrics = list(s.scalars(select(RunMetric).where(RunMetric.run_id.in_(run_ids)).where(RunMetric.fold == -1)))
            durations = list(s.scalars(select(ModelArtifact).where(ModelArtifact.run_id.in_(run_ids))))
            rankings = list(s.scalars(select(Ranking).where(Ranking.run_id.in_(run_ids))))
        else:
            forecasts, metrics, durations, rankings = [], [], [], []

    # Group forecasts by (run_id, algo) so we can cap series_per_forecast.
    grouped: dict[tuple[int, str], list[Forecast]] = {}
    for f in forecasts:
        grouped.setdefault((f.run_id, f.algo), []).append(f)
    for key, lst in grouped.items():
        lst.sort(key=lambda r: r.ts)
        grouped[key] = lst[:series_per]

    def labels_of(inst: str, met: str, hor: str) -> dict[str, str]:
        out = {"instance": inst, "metric": met}
        if include_horizon:
            out["horizon"] = hor
        return out

    # ---- Emit ----
    for (_run_id, _algo), rows in grouped.items():
        for f in rows:
            base = labels_of(f.instance, f.metric, f.horizon)
            ts_str = f.ts.isoformat()
            if f.is_best and best_value is not None:
                best_value.labels(**base, bound="point", ts=ts_str).set(f.point)
                if emit.best_model_bounds and f.lower is not None and f.upper is not None:
                    best_value.labels(**base, bound="lower", ts=ts_str).set(f.lower)
                    best_value.labels(**base, bound="upper", ts=ts_str).set(f.upper)
            if per_model_value is not None:
                per_model_value.labels(**base, model=f.algo, bound="point", ts=ts_str).set(f.point)
                if emit.per_model_bounds and f.lower is not None and f.upper is not None:
                    per_model_value.labels(**base, model=f.algo, bound="lower", ts=ts_str).set(f.lower)
                    per_model_value.labels(**base, model=f.algo, bound="upper", ts=ts_str).set(f.upper)

    # forecast_best_model_info & training run timestamps come from the ranking row
    for r in rankings:
        base = labels_of(r.instance, r.metric, r.horizon)
        if best_info is not None:
            best_info.labels(**base, model=r.winning_algo).set(1)

    for r in latest_runs:
        base = labels_of(r.instance, r.metric, r.horizon)
        if run_ts is not None and r.completed_at:
            run_ts.labels(**base).set(r.completed_at.timestamp())

    for m in metrics:
        if model_score is None:
            break
        run = run_meta.get(m.run_id)
        if not run:
            continue
        base = labels_of(run.instance, run.metric, run.horizon)
        if m.value is None:
            continue
        model_score.labels(**base, model=m.algo, score=m.score_metric).set(m.value)

    for a in durations:
        if train_duration is None:
            break
        run = run_meta.get(a.run_id)
        if not run:
            continue
        base = labels_of(run.instance, run.metric, run.horizon)
        train_duration.labels(**base, model=a.algo).set(a.train_duration_seconds)

    if winner_gauge is not None or winner_unique_gauge is not None:
        summary = repo.winners_summary(
            recent_window=settings.exposition.diagnostics.recent_window_runs
        )
        for row in summary:
            base = labels_of(row["instance"], row["metric"], row["horizon"])
            if winner_gauge is not None:
                winner_gauge.labels(**base, model=row["current_winner"]).set(1)
            if winner_unique_gauge is not None:
                winner_unique_gauge.labels(**base).set(row["unique_winners_recent"])

    return reg


@router.get("/metrics")
def metrics(
    settings: Settings = Depends(settings_dep),
    repo: RegistryRepo = Depends(repo_dep),
) -> Response:
    reg = _build_registry(repo, settings)
    return Response(generate_latest(reg), media_type=CONTENT_TYPE_LATEST)
