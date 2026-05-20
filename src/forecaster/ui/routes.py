"""Server-rendered dashboard pages.

All pages live under /ui. Static assets under /ui/static (mounted by
forecaster.api.main). HTMX powers auto-refresh fragments; Chart.js powers
the plots. No JS build pipeline.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles

from ..api.deps import repo_dep, settings_dep
from ..config.schema import Settings
from ..models import REGISTRY
from ..registry.repo import RegistryRepo

_HERE = Path(__file__).resolve().parent
_TEMPLATES_DIR = _HERE / "templates"
_STATIC_DIR = _HERE / "static"

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

router = APIRouter(prefix="/ui", tags=["ui"])

# Exported so the app factory can mount static files at /ui/static
# with a named route resolvable via url_for('ui-static', path=...).
STATIC_DIR = _STATIC_DIR
static_app = StaticFiles(directory=str(_STATIC_DIR))


# Top-level redirect from /ui → /ui/ for tidiness.
@router.get("", include_in_schema=False)
def _slash_redirect() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


# ----- helpers -----

SCORE_METRICS = ["mae", "rmse", "mape", "smape", "r2"]


def _safe(v):
    """Strip NaN/Inf for JSON embed."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _safe_dict(d: dict) -> dict:
    return {k: _safe(v) for k, v in d.items()}


# ============================================================
# Overview
# ============================================================

@router.get("/", response_class=HTMLResponse, name="ui_overview")
def overview_page(request: Request, repo: RegistryRepo = Depends(repo_dep),
                  settings: Settings = Depends(settings_dep)) -> HTMLResponse:
    stats = _build_overview_stats(repo, settings)
    return templates.TemplateResponse(
        request, "overview.html", {"active": "overview", "stats": stats},
    )


@router.get("/_/overview/cards", response_class=HTMLResponse, name="ui_overview_cards")
def overview_cards(request: Request, repo: RegistryRepo = Depends(repo_dep),
                   settings: Settings = Depends(settings_dep)) -> HTMLResponse:
    stats = _build_overview_stats(repo, settings)
    return templates.TemplateResponse(
        request, "_overview_cards.html", {"stats": stats},
    )


@router.get("/_/overview/runs", response_class=HTMLResponse, name="ui_recent_runs_fragment")
def overview_recent_runs(request: Request, repo: RegistryRepo = Depends(repo_dep)) -> HTMLResponse:
    runs = repo.list_runs(limit=15)
    return templates.TemplateResponse(
        request, "_recent_runs.html",
        {"runs": [_run_dict(r) for r in runs]},
    )


@router.get("/_/overview/attention", response_class=HTMLResponse, name="ui_attention_fragment")
def overview_attention(request: Request, repo: RegistryRepo = Depends(repo_dep),
                       settings: Settings = Depends(settings_dep)) -> HTMLResponse:
    attention = repo.attention_targets(
        recent_window=settings.exposition.diagnostics.recent_window_runs,
    )
    return templates.TemplateResponse(
        request, "_attention.html", {"attention": attention},
    )


def _build_overview_stats(repo: RegistryRepo, settings: Settings) -> dict[str, Any]:
    o = repo.system_overview()
    summary = repo.winners_summary(
        recent_window=settings.exposition.diagnostics.recent_window_runs
    )
    flapping = sum(1 for s in summary if s["unique_winners_recent"] >= 3)
    return {
        **o,
        "flapping": flapping,
        "models_registered": len(REGISTRY),
        "models_enabled": len(settings.algorithms.enabled),
    }


def _run_dict(r) -> dict[str, Any]:
    return {
        "id": r.id, "instance": r.instance, "metric": r.metric, "horizon": r.horizon,
        "status": r.status,
        "started_at": r.started_at.isoformat() if r.started_at else None,
        "completed_at": r.completed_at.isoformat() if r.completed_at else None,
        "duration_seconds": r.duration_seconds,
        "error": r.error,
    }


# ============================================================
# Targets
# ============================================================

@router.get("/targets", response_class=HTMLResponse, name="ui_targets")
def targets_page(
    request: Request,
    instance: str | None = None,
    metric: str | None = None,
    horizon: str | None = None,
    winner: str | None = None,
    repo: RegistryRepo = Depends(repo_dep),
    settings: Settings = Depends(settings_dep),
) -> HTMLResponse:
    summary = repo.winners_summary(
        recent_window=settings.exposition.diagnostics.recent_window_runs
    )
    all_metrics = sorted({s["metric"] for s in summary})
    all_horizons = sorted({s["horizon"] for s in summary})
    all_winners = sorted({s["current_winner"] for s in summary if s["current_winner"]})

    filtered = [
        s for s in summary
        if (not instance or instance in s["instance"])
        and (not metric or s["metric"] == metric)
        and (not horizon or s["horizon"] == horizon)
        and (not winner or s["current_winner"] == winner)
    ]
    return templates.TemplateResponse(
        request, "targets.html",
        {
            "active": "targets",
            "targets": filtered,
            "total_targets": len(summary),
            "all_metrics": all_metrics, "all_horizons": all_horizons, "all_winners": all_winners,
            "filters": {"instance": instance, "metric": metric, "horizon": horizon, "winner": winner},
        },
    )


@router.get("/targets/{instance}/{metric}/{horizon}", response_class=HTMLResponse, name="ui_target_detail")
def target_detail_page(
    request: Request, instance: str, metric: str, horizon: str,
    repo: RegistryRepo = Depends(repo_dep),
    settings: Settings = Depends(settings_dep),
) -> HTMLResponse:
    summary_all = repo.winners_summary(
        recent_window=settings.exposition.diagnostics.recent_window_runs
    )
    summary = next(
        (s for s in summary_all
         if s["instance"] == instance and s["metric"] == metric and s["horizon"] == horizon),
        None,
    )
    forecasts = repo.latest_forecasts(
        instance=instance, metric=metric, horizon=horizon, only_best=True,
    )
    forecasts_data = [
        {"ts": f.ts.isoformat(), "point": _safe(f.point),
         "lower": _safe(f.lower), "upper": _safe(f.upper)}
        for f in sorted(forecasts, key=lambda x: x.ts)
    ]

    rankings = repo.latest_rankings(instance=instance, metric=metric, horizon=horizon)
    ranking_list = rankings[0].ranked if rankings else []
    # Sanitize NaN/inf for embed
    ranking_list = [
        {**r, "composite": _safe(r.get("composite")),
         "raw_scores": {k: _safe(v) for k, v in (r.get("raw_scores") or {}).items()},
         "normalised_scores": {k: _safe(v) for k, v in (r.get("normalised_scores") or {}).items()}}
        for r in ranking_list
    ]

    score_history = repo.score_history(
        instance=instance, metric=metric, horizon=horizon, limit=50,
    )
    winner_history = repo.winner_history(
        instance=instance, metric=metric, horizon=horizon, limit=50,
    )
    algos_history = sorted({r["winning_algo"] for r in winner_history})

    return templates.TemplateResponse(
        request, "target_detail.html",
        {
            "active": "targets",
            "instance": instance, "metric": metric, "horizon": horizon,
            "summary": summary,
            "forecasts": forecasts_data,
            "forecasts_json": json.dumps(forecasts_data),
            "ranking": ranking_list,
            "ranking_json": json.dumps(ranking_list),
            "score_metrics": SCORE_METRICS,
            "score_history_json": json.dumps(score_history),
            "winner_history_json": json.dumps(winner_history),
            "algos_history_json": json.dumps(algos_history),
        },
    )


@router.post("/targets/_trigger", name="ui_trigger_run")
def trigger_run(
    request: Request,
    instance: str = Form(...),
    metric: str = Form(...),
    horizon: str = Form(...),
) -> RedirectResponse:
    # Imported lazily so the UI module doesn't pull Celery on import.
    from ..training.tasks import train_task

    train_task.apply_async(args=[instance, metric, horizon])
    target_url = request.url_for(
        "ui_target_detail", instance=instance, metric=metric, horizon=horizon
    )
    return RedirectResponse(url=str(target_url), status_code=303)


# ============================================================
# Runs
# ============================================================

@router.get("/runs", response_class=HTMLResponse, name="ui_runs")
def runs_page(
    request: Request,
    instance: str | None = None,
    metric: str | None = None,
    horizon: str | None = None,
    status: str | None = None,
    limit: int = 100,
    repo: RegistryRepo = Depends(repo_dep),
) -> HTMLResponse:
    runs = repo.list_runs(instance=instance, metric=metric, horizon=horizon, limit=limit)
    if status:
        runs = [r for r in runs if r.status == status]
    all_metrics = sorted({r.metric for r in runs})
    all_horizons = sorted({r.horizon for r in runs})
    return templates.TemplateResponse(
        request, "runs.html",
        {
            "active": "runs",
            "runs": [_run_dict(r) for r in runs],
            "all_metrics": all_metrics, "all_horizons": all_horizons,
            "filters": {"instance": instance, "metric": metric, "horizon": horizon, "status": status},
        },
    )


@router.get("/runs/{run_id}", response_class=HTMLResponse, name="ui_run_detail")
def run_detail_page(
    request: Request, run_id: int, repo: RegistryRepo = Depends(repo_dep),
) -> HTMLResponse:
    detail = repo.run_full_detail(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    run = detail["run"]
    rows = detail["rows"]
    rank_chart = [
        {"algo": r["algo"], "composite": _safe(r["composite"])}
        for r in rows if r["composite"] is not None
    ]
    dur_chart = [
        {"algo": r["algo"], "duration": _safe(r["duration"])}
        for r in rows if r["duration"] is not None
    ]
    return templates.TemplateResponse(
        request, "run_detail.html",
        {
            "active": "runs",
            "run": run, "rows": rows, "score_metrics": SCORE_METRICS,
            "rank_chart_json": json.dumps(rank_chart),
            "dur_chart_json": json.dumps(dur_chart),
            "config_snapshot_json": json.dumps(run["config_snapshot"], indent=2, default=str),
        },
    )


# ============================================================
# Models
# ============================================================

@router.get("/models", response_class=HTMLResponse, name="ui_models")
def models_page(
    request: Request,
    repo: RegistryRepo = Depends(repo_dep),
    settings: Settings = Depends(settings_dep),
) -> HTMLResponse:
    stats = repo.model_stats()
    enabled = set(settings.algorithms.enabled)
    per_metric_eligibility: dict[str, list[str]] = {}
    for algo in REGISTRY:
        eligible = [
            m for m, shortlist in settings.algorithms.per_metric.items()
            if algo in shortlist
        ]
        per_metric_eligibility[algo] = eligible

    # Ensure every registered algo appears, even if zero runs
    by_algo = {row["algo"]: row for row in stats}
    rows: list[dict[str, Any]] = []
    for algo in sorted(REGISTRY.keys()):
        row = by_algo.get(algo, {
            "algo": algo, "wins": 0, "runs": 0, "win_rate": 0.0,
            "avg_mae": None, "avg_rmse": None, "avg_train_duration": None,
        })
        rows.append({
            **row,
            "state": "enabled" if algo in enabled else "disabled",
            "eligible_metrics": per_metric_eligibility.get(algo, []),
        })

    return templates.TemplateResponse(
        request, "models.html",
        {
            "active": "models",
            "models": rows,
            "rows_json": json.dumps([
                {"algo": r["algo"], "wins": r["wins"], "runs": r["runs"],
                 "win_rate": r["win_rate"]}
                for r in rows
            ]),
            "per_metric_wins_json": json.dumps(repo.wins_by_metric()),
        },
    )


# ============================================================
# Config
# ============================================================

@router.get("/config", response_class=HTMLResponse, name="ui_config")
def config_page(
    request: Request,
    settings: Settings = Depends(settings_dep),
) -> HTMLResponse:
    effective = settings.model_dump()
    active_endpoint = settings.data_sources.endpoints[settings.data_sources.active]
    yaml_str = yaml.safe_dump(effective, sort_keys=False)
    return templates.TemplateResponse(
        request, "config.html",
        {
            "active": "config",
            "settings": settings,
            "active_endpoint": active_endpoint,
            "effective_yaml": yaml_str,
            "models_registered": len(REGISTRY),
        },
    )


@router.post("/config/reload", name="ui_reload_config")
def reload_config_action(request: Request) -> RedirectResponse:
    from ..config.loader import reload_settings
    reload_settings()
    return RedirectResponse(url=str(request.url_for("ui_config")), status_code=303)
