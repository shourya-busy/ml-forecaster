"""Server-rendered dashboard pages.

All pages live under /ui. Static assets under /ui/static (mounted by
forecaster.api.main). HTMX powers auto-refresh fragments; Chart.js powers
the plots. No JS build pipeline.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

log = logging.getLogger(__name__)
from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles

from ..api.deps import repo_dep, settings_dep
from ..config.loader import get_settings
from ..config.schema import Settings
from ..models import REGISTRY
from ..models.registry import ALGO_INFO, algo_info
from ..registry.repo import RegistryRepo

_HERE = Path(__file__).resolve().parent
_TEMPLATES_DIR = _HERE / "templates"
_STATIC_DIR = _HERE / "static"

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


# ----- timezone-aware Jinja filters -----

def _resolve_tz(tz_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        return ZoneInfo("UTC")


def to_local(value: Any, fmt: str = "%Y-%m-%d %H:%M:%S %Z") -> str:
    """Render a UTC ISO 8601 string / datetime in the configured timezone.

    Storage is always UTC; this is presentation only. Reads the timezone
    on every call so a runtime `POST /config/reload` is reflected
    immediately without re-importing the template environment.
    """
    if value is None or value == "":
        return "—"
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
    else:
        return str(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_resolve_tz(get_settings().display_timezone)).strftime(fmt)


def to_local_short(value: Any) -> str:
    """Compact form for table cells: 'May 20 18:00:32 IST'."""
    return to_local(value, fmt="%b %d %H:%M:%S %Z")


def to_local_date(value: Any) -> str:
    return to_local(value, fmt="%Y-%m-%d")


templates.env.filters["to_local"] = to_local
templates.env.filters["to_local_short"] = to_local_short
templates.env.filters["to_local_date"] = to_local_date

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
        "training_paused": bool(settings.training.paused),
        "active_runs": len(repo.list_active_runs()),
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
    lookback_hours: int | None = None,
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

    # Default lookback per horizon: enough past actual context that the
    # comparison is meaningful without flooding Prometheus.
    if lookback_hours is None:
        defaults = {"short": 6, "medium": 24, "long": 168}
        effective_lookback_hours = defaults.get(horizon, 24)
    else:
        effective_lookback_hours = max(0, int(lookback_hours))

    # ----- live forecast vs actual -----
    # Re-fetch the actual metric values for [now - lookback_hours, forecast_end]
    # so the user sees both the recent past *and* the forecast band on one
    # chart. Fails open: any data-source error just skips the overlay.
    actuals_data: list[dict[str, Any]] = []
    live_mae: float | None = None
    live_count = 0
    if forecasts_data and horizon in settings.horizons:
        try:
            import pandas as _pd
            from ..data.factory import make_data_source

            query = settings.metrics_to_forecast.queries.get(metric)
            if query:
                fc_first = _pd.Timestamp(forecasts_data[0]["ts"])
                fc_last = _pd.Timestamp(forecasts_data[-1]["ts"])
                now = _pd.Timestamp.now(tz="UTC")
                # Pull from (now - lookback) to whichever forecast point
                # has already occurred — that's the meaningful overlap.
                start = min(fc_first, now - _pd.Timedelta(hours=effective_lookback_hours))
                end = min(fc_last, now)
                if end > start:
                    step = settings.horizons[horizon].step
                    ds = make_data_source(settings.data_sources)
                    try:
                        series_list = ds.fetch_range(
                            query,
                            start.to_pydatetime(),
                            end.to_pydatetime(),
                            step,
                            instance_label=settings.targets.instance_label,
                            metric_name=metric,
                        )
                    finally:
                        ds.close()
                    for s in series_list:
                        if s.instance != instance or s.df.empty:
                            continue
                        snapped = s.df["value"].asfreq(_pd.Timedelta(step)).interpolate("time")
                        actuals_data = [
                            {"ts": idx.isoformat(), "value": _safe(float(v))}
                            for idx, v in snapped.dropna().items()
                        ]
                        break
                    # Compute MAE only over the overlap with the forecast
                    # window (we're not scoring the past, just the future).
                    if actuals_data:
                        fc_map = {p["ts"]: p["point"] for p in forecasts_data}
                        ac_map = {p["ts"]: p["value"] for p in actuals_data}
                        common = set(fc_map.keys()) & set(ac_map.keys())
                        diffs = [
                            abs(fc_map[t] - ac_map[t])
                            for t in common
                            if fc_map[t] is not None and ac_map[t] is not None
                        ]
                        if diffs:
                            live_mae = sum(diffs) / len(diffs)
                            live_count = len(diffs)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not fetch actuals for overlay: %s", exc)

    return templates.TemplateResponse(
        request, "target_detail.html",
        {
            "active": "targets",
            "instance": instance, "metric": metric, "horizon": horizon,
            "summary": summary,
            "forecasts": forecasts_data,
            "forecasts_json": json.dumps(forecasts_data),
            "actuals_json": json.dumps(actuals_data),
            "live_mae": live_mae,
            "live_count": live_count,
            "lookback_hours": effective_lookback_hours,
            "ranking": ranking_list,
            "ranking_json": json.dumps(ranking_list),
            "score_metrics": SCORE_METRICS,
            "score_history_json": json.dumps(score_history),
            "winner_history_json": json.dumps(winner_history),
            "algos_history_json": json.dumps(algos_history),
            "display_timezone": settings.display_timezone,
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
# Instances
# ============================================================

@router.get("/instances", response_class=HTMLResponse, name="ui_instances")
def instances_page(
    request: Request,
    q: str | None = None,
    repo: RegistryRepo = Depends(repo_dep),
    settings: Settings = Depends(settings_dep),
) -> HTMLResponse:
    rows = repo.instance_summary(
        recent_window=settings.exposition.diagnostics.recent_window_runs
    )
    if q:
        rows = [r for r in rows if q.lower() in r["instance"].lower()]
    return templates.TemplateResponse(
        request, "instances.html",
        {"active": "instances", "instances": rows, "filters": {"q": q}},
    )


@router.get("/instances/{instance}", response_class=HTMLResponse, name="ui_instance_detail")
def instance_detail_page(
    request: Request, instance: str,
    repo: RegistryRepo = Depends(repo_dep),
    settings: Settings = Depends(settings_dep),
) -> HTMLResponse:
    detail = repo.instance_detail(
        instance,
        recent_window=settings.exposition.diagnostics.recent_window_runs,
    )
    if detail is None:
        raise HTTPException(status_code=404, detail=f"instance '{instance}' not found")
    return templates.TemplateResponse(
        request, "instance_detail.html",
        {
            "active": "instances",
            "instance": instance,
            "targets": detail["targets"],
            "recent_runs": detail["recent_runs"],
        },
    )


# ============================================================
# Runs
# ============================================================

_SORT_COLUMNS = {"id", "instance", "metric", "horizon", "status",
                 "started_at", "completed_at", "duration_seconds"}


def _parse_when(s: str | None) -> datetime | None:
    if not s:
        return None
    # Accept 'YYYY-MM-DD', 'YYYY-MM-DDTHH:MM', full ISO; treat as IST input.
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        tz = _resolve_tz(get_settings().display_timezone)
        dt = dt.replace(tzinfo=tz)
    return dt.astimezone(timezone.utc)


@router.get("/runs", response_class=HTMLResponse, name="ui_runs")
def runs_page(
    request: Request,
    instance: str | None = None,
    metric: str | None = None,
    horizon: str | None = None,
    status: str | None = None,
    since: str | None = None,
    until: str | None = None,
    sort: str = "started_at",
    direction: str = "desc",
    limit: int = 200,
    repo: RegistryRepo = Depends(repo_dep),
) -> HTMLResponse:
    sort = sort if sort in _SORT_COLUMNS else "started_at"
    direction = "asc" if direction == "asc" else "desc"
    since_dt = _parse_when(since)
    until_dt = _parse_when(until)
    runs = repo.runs_filtered(
        instance=instance, metric=metric, horizon=horizon, status=status,
        since=since_dt, until=until_dt, sort=sort, direction=direction, limit=limit,
    )
    all_metrics = sorted({r.metric for r in runs})
    all_horizons = sorted({r.horizon for r in runs})
    return templates.TemplateResponse(
        request, "runs.html",
        {
            "active": "runs",
            "runs": [_run_dict(r) for r in runs],
            "all_metrics": all_metrics, "all_horizons": all_horizons,
            "error_groups": repo.error_groups(hours=24),
            "filters": {
                "instance": instance, "metric": metric, "horizon": horizon,
                "status": status, "since": since, "until": until,
                "sort": sort, "direction": direction, "limit": limit,
            },
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
    per_fold = repo.run_per_fold_scores(run_id)
    return templates.TemplateResponse(
        request, "run_detail.html",
        {
            "active": "runs",
            "run": run, "rows": rows, "score_metrics": SCORE_METRICS,
            "rank_chart_json": json.dumps(rank_chart),
            "dur_chart_json": json.dumps(dur_chart),
            "per_fold": per_fold,
            "per_fold_json": json.dumps(per_fold),
            "config_snapshot_json": json.dumps(run["config_snapshot"], indent=2, default=str),
        },
    )


# ============================================================
# Cancellation endpoints
# ============================================================

@router.post("/runs/{run_id}/cancel", name="ui_cancel_run")
def cancel_run(
    request: Request, run_id: int,
    repo: RegistryRepo = Depends(repo_dep),
) -> RedirectResponse:
    from ..training.tasks import revoke_task

    with repo.session() as s:
        from ..registry.models import TrainingRun
        run = s.get(TrainingRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"run {run_id} not found")
        task_id = run.celery_task_id
        status = run.status
    if status in ("completed", "failed", "cancelled"):
        # Idempotent — already terminal, nothing to do.
        return RedirectResponse(
            url=str(request.url_for("ui_run_detail", run_id=run_id)),
            status_code=303,
        )
    revoke_task(task_id)
    repo.mark_cancelled(run_id, reason="cancelled via UI")
    return RedirectResponse(
        url=str(request.url_for("ui_run_detail", run_id=run_id)),
        status_code=303,
    )


@router.post("/training/pause", name="ui_training_pause")
def training_pause(
    request: Request,
    repo: RegistryRepo = Depends(repo_dep),
) -> RedirectResponse:
    repo.set_training_paused(True, updated_by="ui")
    _invalidate()
    log.info("training paused via UI")
    return RedirectResponse(
        url=request.headers.get("referer") or str(request.url_for("ui_overview")),
        status_code=303,
    )


@router.post("/training/resume", name="ui_training_resume")
def training_resume(
    request: Request,
    repo: RegistryRepo = Depends(repo_dep),
) -> RedirectResponse:
    repo.set_training_paused(False, updated_by="ui")
    _invalidate()
    log.info("training resumed via UI")
    return RedirectResponse(
        url=request.headers.get("referer") or str(request.url_for("ui_overview")),
        status_code=303,
    )


@router.post("/runs/cancel-active", name="ui_cancel_active_runs")
def cancel_active_runs(
    request: Request,
    repo: RegistryRepo = Depends(repo_dep),
) -> RedirectResponse:
    from ..training.tasks import revoke_task

    active = repo.list_active_runs()
    for run in active:
        revoke_task(run.celery_task_id)
        repo.mark_cancelled(run.id, reason="bulk cancel via UI")
    log.info("bulk cancel: revoked %d active run(s)", len(active))
    return RedirectResponse(
        url=str(request.url_for("ui_runs")), status_code=303,
    )


# ============================================================
# Custom Run panel
# ============================================================

def _algorithms_grouped(settings: Settings) -> list[tuple[str, list[str]]]:
    info_map = {name: algo_info(name) for name in REGISTRY}
    family_order = ["baseline", "statistical", "ml", "deep-learning"]
    grouped: dict[str, list[str]] = {f: [] for f in family_order}
    for name in sorted(REGISTRY):
        fam = info_map[name].get("family") or "other"
        grouped.setdefault(fam, []).append(name)
    return [(f, grouped[f]) for f in [*family_order, "other"] if grouped.get(f)]


def _build_overrides(
    *, algorithms: list[str] | None,
    anomaly_enabled: bool, anomaly_contamination: float | None,
    anomaly_window: int | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if algorithms:
        out["algorithms"] = list(algorithms)
    af: dict[str, Any] = {"enabled": bool(anomaly_enabled)}
    if anomaly_contamination is not None:
        af["contamination"] = float(anomaly_contamination)
    if anomaly_window is not None:
        af["window"] = int(anomaly_window)
    out["anomaly_filter"] = af
    return out


@router.get("/custom-run", response_class=HTMLResponse, name="ui_custom_run")
def custom_run_page(
    request: Request,
    repo: RegistryRepo = Depends(repo_dep),
    settings: Settings = Depends(settings_dep),
) -> HTMLResponse:
    saved = repo.list_custom_configs()
    # Filter active runs that were probably triggered from this panel,
    # but we don't actually mark them — surface all active so user has one
    # cancel surface for everything they kicked off.
    active = repo.list_active_runs()
    try:
        from ..scheduling.jobs import discover_targets
        instances = discover_targets()
    except Exception:  # noqa: BLE001
        instances = sorted({r.instance for r in repo.list_runs(limit=200)})
    return templates.TemplateResponse(
        request, "custom_run.html",
        {
            "active": "custom_run",
            "instances": instances,
            "metrics": list(settings.metrics_to_forecast.queries.keys()),
            "horizons": list(settings.horizons.keys()),
            "grouped_algos": _algorithms_grouped(settings),
            "algo_info": {name: algo_info(name) for name in REGISTRY},
            "default_enabled": set(settings.algorithms.enabled),
            "anomaly_default": settings.training.anomaly_filter.model_dump(),
            "saved": [
                {
                    "id": c.id, "name": c.name, "instance": c.instance,
                    "metric": c.metric, "horizon": c.horizon,
                    "algorithms": c.algorithms or [],
                    "anomaly_filter": c.anomaly_filter or {},
                    "note": c.note or "",
                    "run_count": c.run_count or 0,
                    "last_used_at": c.last_used_at.isoformat() if c.last_used_at else None,
                }
                for c in saved
            ],
            "active_runs": [_run_dict(r) for r in active],
            "training_paused": settings.training.paused,
        },
    )


def _parse_custom_form(form: dict) -> dict[str, Any]:
    """Shared form-extraction logic for save / run."""
    name = (form.get("name") or "").strip()
    instance = (form.get("instance") or "").strip()
    metric = (form.get("metric") or "").strip()
    horizon = (form.get("horizon") or "").strip()
    if not instance or not metric or not horizon:
        raise HTTPException(status_code=400, detail="instance, metric, and horizon are required")
    algos_raw = form.getlist("algorithms") if hasattr(form, "getlist") else form.get("algorithms", [])
    if isinstance(algos_raw, str):
        algos = [algos_raw] if algos_raw else []
    else:
        algos = list(algos_raw)
    algos = [a for a in algos if a in REGISTRY]
    anomaly_enabled = (form.get("anomaly_enabled") or "") not in ("", "0", "false", "no")
    anomaly_contamination = form.get("anomaly_contamination") or ""
    anomaly_window = form.get("anomaly_window") or ""
    try:
        contamination = float(anomaly_contamination) if anomaly_contamination else None
        window = int(anomaly_window) if anomaly_window else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"anomaly knobs: {exc}") from exc
    note = (form.get("note") or "").strip() or None
    return {
        "name": name, "instance": instance, "metric": metric, "horizon": horizon,
        "algorithms": algos,
        "anomaly_enabled": anomaly_enabled,
        "anomaly_contamination": contamination,
        "anomaly_window": window,
        "note": note,
    }


async def _form_dict(request: Request) -> Any:
    """Returns a starlette FormData object (has .getlist)."""
    return await request.form()


@router.post("/custom-run/run", name="ui_custom_run_run")
async def custom_run_run(
    request: Request,
    repo: RegistryRepo = Depends(repo_dep),
) -> RedirectResponse:
    form = await _form_dict(request)
    p = _parse_custom_form(form)
    overrides = _build_overrides(
        algorithms=p["algorithms"] or None,
        anomaly_enabled=p["anomaly_enabled"],
        anomaly_contamination=p["anomaly_contamination"],
        anomaly_window=p["anomaly_window"],
    )
    from ..training.tasks import train_task
    train_task.apply_async(
        args=[p["instance"], p["metric"], p["horizon"]],
        kwargs={"overrides": overrides},
    )
    log.info("custom-run: triggered %s/%s/%s with overrides=%s",
             p["instance"], p["metric"], p["horizon"], overrides)
    return RedirectResponse(url=str(request.url_for("ui_custom_run")), status_code=303)


@router.post("/custom-run/save", name="ui_custom_run_save")
async def custom_run_save(
    request: Request,
    repo: RegistryRepo = Depends(repo_dep),
) -> RedirectResponse:
    form = await _form_dict(request)
    p = _parse_custom_form(form)
    if not p["name"]:
        raise HTTPException(status_code=400, detail="name is required to save")
    repo.upsert_custom_config(
        name=p["name"], instance=p["instance"], metric=p["metric"], horizon=p["horizon"],
        algorithms=p["algorithms"] or None,
        anomaly_filter={
            "enabled": p["anomaly_enabled"],
            "contamination": p["anomaly_contamination"],
            "window": p["anomaly_window"],
        },
        note=p["note"],
    )
    return RedirectResponse(url=str(request.url_for("ui_custom_run")), status_code=303)


@router.post("/custom-run/save-and-run", name="ui_custom_run_save_and_run")
async def custom_run_save_and_run(
    request: Request,
    repo: RegistryRepo = Depends(repo_dep),
) -> RedirectResponse:
    """Save then immediately fire."""
    form = await _form_dict(request)
    p = _parse_custom_form(form)
    if not p["name"]:
        raise HTTPException(status_code=400, detail="name is required to save")
    cfg = repo.upsert_custom_config(
        name=p["name"], instance=p["instance"], metric=p["metric"], horizon=p["horizon"],
        algorithms=p["algorithms"] or None,
        anomaly_filter={
            "enabled": p["anomaly_enabled"],
            "contamination": p["anomaly_contamination"],
            "window": p["anomaly_window"],
        },
        note=p["note"],
    )
    overrides = _build_overrides(
        algorithms=p["algorithms"] or None,
        anomaly_enabled=p["anomaly_enabled"],
        anomaly_contamination=p["anomaly_contamination"],
        anomaly_window=p["anomaly_window"],
    )
    from ..training.tasks import train_task
    train_task.apply_async(
        args=[p["instance"], p["metric"], p["horizon"]],
        kwargs={"overrides": overrides},
    )
    repo.touch_custom_config(cfg.id)
    return RedirectResponse(url=str(request.url_for("ui_custom_run")), status_code=303)


@router.post("/custom-run/run-saved/{config_id}", name="ui_custom_run_run_saved")
def custom_run_run_saved(
    request: Request, config_id: int,
    repo: RegistryRepo = Depends(repo_dep),
) -> RedirectResponse:
    cfg = repo.get_custom_config(config_id)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"config {config_id} not found")
    overrides: dict[str, Any] = {}
    if cfg.algorithms:
        overrides["algorithms"] = list(cfg.algorithms)
    if cfg.anomaly_filter:
        overrides["anomaly_filter"] = dict(cfg.anomaly_filter)
    from ..training.tasks import train_task
    train_task.apply_async(
        args=[cfg.instance, cfg.metric, cfg.horizon],
        kwargs={"overrides": overrides},
    )
    repo.touch_custom_config(cfg.id)
    log.info("custom-run: fired saved config '%s' (%s/%s/%s)",
             cfg.name, cfg.instance, cfg.metric, cfg.horizon)
    return RedirectResponse(url=str(request.url_for("ui_custom_run")), status_code=303)


@router.post("/custom-run/delete/{config_id}", name="ui_custom_run_delete")
def custom_run_delete(
    request: Request, config_id: int,
    repo: RegistryRepo = Depends(repo_dep),
) -> RedirectResponse:
    repo.delete_custom_config(config_id)
    return RedirectResponse(url=str(request.url_for("ui_custom_run")), status_code=303)


# ============================================================
# Models
# ============================================================

@router.get("/models", response_class=HTMLResponse, name="ui_models")
def models_page(
    request: Request,
    metric: str | None = None,
    horizon: str | None = None,
    window: str = "all",
    repo: RegistryRepo = Depends(repo_dep),
    settings: Settings = Depends(settings_dep),
) -> HTMLResponse:
    since = _since_for_window(window) if window != "all" else None
    stats = repo.model_stats(metric=metric, horizon=horizon, since=since)
    enabled = set(settings.algorithms.enabled)
    per_metric_eligibility: dict[str, list[str]] = {}
    for algo in REGISTRY:
        eligible = [
            m for m, shortlist in settings.algorithms.per_metric.items()
            if algo in shortlist
        ]
        per_metric_eligibility[algo] = eligible

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
            "metrics": list(settings.metrics_to_forecast.queries.keys()),
            "horizons": list(settings.horizons.keys()),
            "window_presets": list(_WINDOW_PRESETS.keys()),
            "selected_metric": metric, "selected_horizon": horizon,
            "selected_window": window,
            "rows_json": json.dumps([
                {"algo": r["algo"], "wins": r["wins"], "runs": r["runs"],
                 "win_rate": r["win_rate"]}
                for r in rows
            ]),
            "per_metric_wins_json": json.dumps(
                repo.wins_by_metric(metric=metric, horizon=horizon, since=since),
            ),
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


# ============================================================
# Schedule (per-horizon cron + upcoming runs)
# ============================================================

def _invalidate():
    from ..config.loader import invalidate_settings_cache
    invalidate_settings_cache()


@router.get("/schedule", response_class=HTMLResponse, name="ui_schedule")
def schedule_page(
    request: Request,
    settings: Settings = Depends(settings_dep),
    repo: RegistryRepo = Depends(repo_dep),
) -> HTMLResponse:
    from ..scheduling.jobs import next_fires

    horizons_data = []
    upcoming: list[dict[str, Any]] = []
    for name, spec in settings.horizons.items():
        try:
            fires = next_fires(spec.retrain, count=5)
        except Exception as e:  # noqa: BLE001
            fires = []
            log_err = str(e)
        else:
            log_err = ""
        try:
            import pandas as _pd
            # Pandas 4.x prefers capital 'D'; normalise quietly.
            n_points = int(
                _pd.Timedelta(spec.horizon.replace("d", "D"))
                / _pd.Timedelta(spec.step.replace("d", "D"))
            )
        except Exception:  # noqa: BLE001
            n_points = 0
        horizons_data.append({
            "name": name, "step": spec.step, "horizon": spec.horizon,
            "lookback_days": spec.lookback_days or settings.training.lookback_days,
            "points_predicted": n_points,
            "retrain": spec.retrain, "next_fires": fires, "error": log_err,
        })
        for ts in fires:
            upcoming.append({
                "when": ts, "type": "fan-out",
                "horizon": name, "target": "all enabled targets",
                "cron": spec.retrain,
            })

    # Per-target cron jobs
    target_ovs = [o for o in repo.get_target_overrides()
                  if o["enabled"] and o.get("schedule_cron")]
    for ov in target_ovs:
        try:
            fires = next_fires(ov["schedule_cron"], count=3)
        except Exception:  # noqa: BLE001
            fires = []
        for ts in fires:
            upcoming.append({
                "when": ts, "type": "per-target",
                "horizon": ov["horizon"],
                "target": f"{ov['instance']} · {ov['metric']}",
                "cron": ov["schedule_cron"],
            })

    upcoming.sort(key=lambda r: r["when"])
    return templates.TemplateResponse(
        request, "schedule.html",
        {
            "active": "schedule",
            "horizons": horizons_data,
            "upcoming": upcoming[:30],
            "display_tz": settings.display_timezone,
        },
    )


@router.post("/schedule/horizon", name="ui_schedule_save_horizon")
def schedule_save_horizon(
    request: Request,
    horizon: str = Form(...),
    retrain: str = Form(...),
    step: str = Form(""),
    forecast_horizon: str = Form(""),
    lookback_days: str = Form(""),
    repo: RegistryRepo = Depends(repo_dep),
) -> RedirectResponse:
    """Save changes to one horizon block.

    Edits to `step` or `forecast_horizon` invalidate previously trained
    models (they were fit at the old step). The UI warns about this; the
    *next* training run will re-fit on the new step + horizon.
    """
    from croniter import croniter
    import pandas as _pd

    if not croniter.is_valid(retrain):
        raise HTTPException(status_code=400, detail=f"invalid cron: {retrain!r}")
    repo.set_settings_override(f"horizons.{horizon}.retrain", retrain)

    def _validate_duration(s: str, name: str) -> str:
        try:
            td = _pd.Timedelta(s)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"{name}: not a valid pandas Timedelta: {s!r}") from exc
        if td.total_seconds() <= 0:
            raise HTTPException(status_code=400, detail=f"{name}: must be positive")
        return s

    if step.strip():
        repo.set_settings_override(f"horizons.{horizon}.step", _validate_duration(step.strip(), "step"))
    if forecast_horizon.strip():
        repo.set_settings_override(f"horizons.{horizon}.horizon", _validate_duration(forecast_horizon.strip(), "horizon"))
    if lookback_days.strip():
        try:
            v = int(lookback_days)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"lookback_days: {exc}") from exc
        if v < 1:
            raise HTTPException(status_code=400, detail="lookback_days must be ≥ 1")
        repo.set_settings_override(f"horizons.{horizon}.lookback_days", v)
    _invalidate()
    return RedirectResponse(url=str(request.url_for("ui_schedule")), status_code=303)


# ============================================================
# Manage Targets — enable/disable + per-target cron
# ============================================================

@router.get("/manage", response_class=HTMLResponse, name="ui_manage_index")
def manage_index(request: Request) -> RedirectResponse:
    return RedirectResponse(url=str(request.url_for("ui_manage_targets")), status_code=303)


@router.get("/manage/targets", response_class=HTMLResponse, name="ui_manage_targets")
def manage_targets_page(
    request: Request,
    repo: RegistryRepo = Depends(repo_dep),
    settings: Settings = Depends(settings_dep),
) -> HTMLResponse:
    # Build the cross-product of (discovered instance × metric × horizon)
    # so the user can flip any of them from a single view.
    try:
        from ..scheduling.jobs import discover_targets
        instances = discover_targets()
    except Exception as exc:  # noqa: BLE001
        log.warning("manage/targets: discovery failed: %s", exc)
        instances = sorted({s["instance"] for s in repo.winners_summary()})

    metrics = list(settings.metrics_to_forecast.queries.keys())
    horizons = list(settings.horizons.keys())
    ov_map = repo.get_target_overrides_map()

    rows: list[dict[str, Any]] = []
    for inst in instances:
        for metric in metrics:
            for horizon in horizons:
                ov = ov_map.get((inst, metric, horizon))
                rows.append({
                    "instance": inst, "metric": metric, "horizon": horizon,
                    "enabled": True if ov is None else ov["enabled"],
                    "schedule_cron": (ov or {}).get("schedule_cron") or "",
                    "note": (ov or {}).get("note") or "",
                    "updated_at": (ov or {}).get("updated_at"),
                })
    return templates.TemplateResponse(
        request, "manage_targets.html",
        {
            "active": "manage",
            "subnav": "targets",
            "rows": rows,
            "instances": instances,
            "metrics": metrics,
            "horizons": horizons,
        },
    )


@router.post("/manage/targets/save", name="ui_manage_targets_save")
def manage_targets_save(
    request: Request,
    instance: str = Form(...),
    metric: str = Form(...),
    horizon: str = Form(...),
    enabled: str | None = Form(None),
    schedule_cron: str = Form(""),
    note: str = Form(""),
    repo: RegistryRepo = Depends(repo_dep),
) -> RedirectResponse:
    cron = schedule_cron.strip() or None
    if cron:
        from croniter import croniter
        if not croniter.is_valid(cron):
            raise HTTPException(status_code=400, detail=f"invalid cron: {cron!r}")
    repo.upsert_target_override(
        instance=instance, metric=metric, horizon=horizon,
        enabled=(enabled is not None),
        schedule_cron=cron,
        note=note.strip() or None,
    )
    _invalidate()
    return RedirectResponse(url=str(request.url_for("ui_manage_targets")), status_code=303)


@router.post("/manage/targets/bulk", name="ui_manage_targets_bulk")
def manage_targets_bulk(
    request: Request,
    action: str = Form(...),       # "enable" or "disable"
    metric: str = Form(""),
    horizon: str = Form(""),
    repo: RegistryRepo = Depends(repo_dep),
    settings: Settings = Depends(settings_dep),
) -> RedirectResponse:
    """Bulk enable/disable across a filter (metric and/or horizon)."""
    from ..scheduling.jobs import discover_targets
    try:
        instances = discover_targets()
    except Exception:
        instances = sorted({s["instance"] for s in repo.winners_summary()})
    metrics = [metric] if metric else list(settings.metrics_to_forecast.queries.keys())
    horizons = [horizon] if horizon else list(settings.horizons.keys())
    flag = (action == "enable")
    for i in instances:
        for m in metrics:
            for h in horizons:
                repo.upsert_target_override(
                    instance=i, metric=m, horizon=h, enabled=flag,
                )
    _invalidate()
    return RedirectResponse(url=str(request.url_for("ui_manage_targets")), status_code=303)


# ============================================================
# Manage Metrics — PromQL CRUD + test
# ============================================================

@router.get("/manage/metrics", response_class=HTMLResponse, name="ui_manage_metrics")
def manage_metrics_page(
    request: Request,
    settings: Settings = Depends(settings_dep),
) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "manage_metrics.html",
        {
            "active": "manage",
            "subnav": "metrics",
            "queries": settings.metrics_to_forecast.queries,
        },
    )


@router.post("/manage/metrics/save", name="ui_manage_metrics_save")
def manage_metrics_save(
    request: Request,
    name: str = Form(...),
    query: str = Form(...),
    repo: RegistryRepo = Depends(repo_dep),
) -> RedirectResponse:
    name = name.strip()
    if not name or not query.strip():
        raise HTTPException(status_code=400, detail="name and query are required")
    if "." in name:
        raise HTTPException(status_code=400, detail="metric name must not contain dots")
    repo.set_settings_override(f"metrics_to_forecast.queries.{name}", query.strip())
    _invalidate()
    return RedirectResponse(url=str(request.url_for("ui_manage_metrics")), status_code=303)


@router.post("/manage/metrics/delete", name="ui_manage_metrics_delete")
def manage_metrics_delete(
    request: Request,
    name: str = Form(...),
    repo: RegistryRepo = Depends(repo_dep),
) -> RedirectResponse:
    # We can only "delete" overrides we stored. Built-in YAML metrics can't
    # be removed via UI — clear the override which restores the YAML value.
    repo.delete_settings_override(f"metrics_to_forecast.queries.{name}")
    _invalidate()
    return RedirectResponse(url=str(request.url_for("ui_manage_metrics")), status_code=303)


@router.post("/manage/metrics/test", name="ui_manage_metrics_test")
def manage_metrics_test(
    query: str = Form(...),
    settings: Settings = Depends(settings_dep),
) -> dict[str, Any]:
    """Run an instant query against the active data source; return summary."""
    from ..data.factory import make_data_source

    ds = make_data_source(settings.data_sources)
    try:
        instances = ds.discover_instances(query, instance_label=settings.targets.instance_label)
        return {"ok": True, "instance_count": len(instances),
                "sample_instances": instances[:10]}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}
    finally:
        ds.close()


# ============================================================
# Manage Training — limits + ranking weights
# ============================================================

@router.get("/manage/training", response_class=HTMLResponse, name="ui_manage_training")
def manage_training_page(
    request: Request,
    settings: Settings = Depends(settings_dep),
) -> HTMLResponse:
    # Group by family so the library renders as sectioned cards; selection
    # is still a flat checkbox group, so cross-category mixing works.
    info_map = {name: algo_info(name) for name in REGISTRY}
    family_order = ["baseline", "statistical", "ml", "deep-learning"]
    grouped: dict[str, list[str]] = {f: [] for f in family_order}
    for name in sorted(REGISTRY):
        fam = info_map[name].get("family") or "other"
        grouped.setdefault(fam, []).append(name)
    grouped_sections = [
        (fam, grouped[fam])
        for fam in [*family_order, "other"]
        if grouped.get(fam)
    ]
    return templates.TemplateResponse(
        request, "manage_training.html",
        {
            "active": "manage",
            "subnav": "training",
            "training": settings.training,
            "ranking": settings.ranking,
            "algorithms": settings.algorithms,
            "all_algos": sorted(REGISTRY.keys()),
            "algo_info": info_map,
            "grouped_sections": grouped_sections,
            "total_registered": len(REGISTRY),
            "total_enabled": len(set(settings.algorithms.enabled) & set(REGISTRY)),
        },
    )


def _set_or_delete(repo: RegistryRepo, key: str, raw: str, parser):
    s = raw.strip()
    if not s:
        repo.delete_settings_override(key)
        return
    try:
        repo.set_settings_override(key, parser(s))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{key}: {exc}") from exc


@router.post("/manage/training/save", name="ui_manage_training_save")
def manage_training_save(
    request: Request,
    lookback_days: str = Form(""),
    backtest_folds: str = Form(""),
    workers: str = Form(""),
    algos_per_job: str = Form(""),
    confidence_alpha: str = Form(""),
    weight_mae: str = Form(""),
    weight_rmse: str = Form(""),
    weight_mape: str = Form(""),
    weight_smape: str = Form(""),
    weight_r2: str = Form(""),
    enabled_algos: list[str] = Form(default=[]),
    anomaly_filter_enabled: str | None = Form(None),
    anomaly_contamination: str = Form(""),
    anomaly_window: str = Form(""),
    repo: RegistryRepo = Depends(repo_dep),
) -> RedirectResponse:
    _set_or_delete(repo, "training.lookback_days", lookback_days, int)
    _set_or_delete(repo, "training.backtest_folds", backtest_folds, int)
    _set_or_delete(repo, "training.parallelism.workers", workers, int)
    _set_or_delete(repo, "training.parallelism.algos_per_job", algos_per_job, int)
    _set_or_delete(repo, "training.confidence_alpha", confidence_alpha, float)
    weights = {
        "mae": weight_mae, "rmse": weight_rmse, "mape": weight_mape,
        "smape": weight_smape, "r2": weight_r2,
    }
    if any(v.strip() for v in weights.values()):
        try:
            parsed = {k: float(v) for k, v in weights.items() if v.strip()}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"weights: {exc}") from exc
        if parsed:
            # Persist as a single dotted-path-per-key set so we can clear individually.
            for k, v in parsed.items():
                repo.set_settings_override(f"ranking.weights.{k}", v)
    # Enabled algorithms
    if enabled_algos:
        # Restrict to registered names
        unknown = [a for a in enabled_algos if a not in REGISTRY]
        if unknown:
            raise HTTPException(status_code=400, detail=f"unknown algos: {unknown}")
        repo.set_settings_override("algorithms.enabled", list(enabled_algos))

    # Anomaly-filter preprocessing toggle
    repo.set_settings_override(
        "training.anomaly_filter.enabled", anomaly_filter_enabled is not None
    )
    _set_or_delete(repo, "training.anomaly_filter.contamination",
                   anomaly_contamination, float)
    _set_or_delete(repo, "training.anomaly_filter.window",
                   anomaly_window, int)

    _invalidate()
    return RedirectResponse(url=str(request.url_for("ui_manage_training")), status_code=303)


# ============================================================
# Compare — overlay two targets
# ============================================================

@router.get("/compare", response_class=HTMLResponse, name="ui_compare")
def compare_page(
    request: Request,
    a: str | None = None,   # instance::metric::horizon
    b: str | None = None,
    horizon: str | None = None,    # narrow the A/B picker
    metric: str | None = None,     # narrow the A/B picker
    limit: int = 50,
    repo: RegistryRepo = Depends(repo_dep),
    settings: Settings = Depends(settings_dep),
) -> HTMLResponse:
    horizons = list(settings.horizons.keys())
    metrics = list(settings.metrics_to_forecast.queries.keys())

    summary = repo.winners_summary(
        recent_window=settings.exposition.diagnostics.recent_window_runs,
    )
    # Filter the picker options by horizon and metric if provided, so the
    # dropdowns don't become a giant unscannable list at scale.
    filtered = [
        s for s in summary
        if (not horizon or s["horizon"] == horizon)
        and (not metric or s["metric"] == metric)
    ]
    options = [
        {
            "key": f"{s['instance']}::{s['metric']}::{s['horizon']}",
            "label": f"{s['instance']} · {s['metric']} · {s['horizon']}",
        }
        for s in filtered
    ]

    def _data_for(key: str | None) -> dict[str, Any] | None:
        if not key or "::" not in key:
            return None
        try:
            inst, met, hor = key.split("::")
        except ValueError:
            return None
        forecasts = repo.latest_forecasts(
            instance=inst, metric=met, horizon=hor, only_best=True,
        )
        score_history = repo.score_history(
            instance=inst, metric=met, horizon=hor, score="mae",
            limit=max(1, min(500, int(limit))),
        )
        rankings = repo.latest_rankings(instance=inst, metric=met, horizon=hor)
        winner = rankings[0].winning_algo if rankings else None
        return {
            "key": key, "label": f"{inst} · {met} · {hor}",
            "instance": inst, "metric": met, "horizon": hor,
            "winner": winner,
            "forecast": [
                {"ts": f.ts.isoformat(), "point": _safe(f.point),
                 "lower": _safe(f.lower), "upper": _safe(f.upper)}
                for f in sorted(forecasts, key=lambda x: x.ts)
            ],
            "score_history": score_history,
        }

    return templates.TemplateResponse(
        request, "compare.html",
        {
            "active": "compare",
            "options": options,
            "all_options_count": len(summary),
            "horizons": horizons, "metrics": metrics,
            "selected_horizon": horizon, "selected_metric": metric,
            "limit": limit,
            "a": _data_for(a), "b": _data_for(b),
            "a_key": a, "b_key": b,
            "display_timezone": settings.display_timezone,
            "a_json": json.dumps(_data_for(a)),
            "b_json": json.dumps(_data_for(b)),
        },
    )


# ============================================================
# Trends — aggregate forecast + accuracy drift
# ============================================================

_WINDOW_PRESETS = {
    "1d": 24, "3d": 72, "7d": 168, "14d": 336, "30d": 720, "90d": 2160,
}


def _since_for_window(window: str | None) -> datetime | None:
    """Convert '7d', '1d', '30d' presets or a custom ISO into a UTC datetime."""
    if not window or window == "all":
        return None
    if window in _WINDOW_PRESETS:
        return datetime.now(timezone.utc) - timedelta(hours=_WINDOW_PRESETS[window])
    # Try parsing as ISO (datetime-local in IST)
    try:
        dt = datetime.fromisoformat(window)
    except ValueError:
        return None
    if dt.tzinfo is None:
        tz = _resolve_tz(get_settings().display_timezone)
        dt = dt.replace(tzinfo=tz)
    return dt.astimezone(timezone.utc)


def _aggregate(values: list[float], agg: str) -> float:
    if not values:
        return 0.0
    if agg == "median":
        sv = sorted(values)
        n = len(sv)
        return sv[n // 2] if n % 2 else (sv[n // 2 - 1] + sv[n // 2]) / 2
    if agg == "p95":
        sv = sorted(values)
        idx = max(0, min(len(sv) - 1, int(0.95 * (len(sv) - 1))))
        return sv[idx]
    # default: mean
    return sum(values) / len(values)


@router.get("/trends", response_class=HTMLResponse, name="ui_trends")
def trends_page(
    request: Request,
    metric: str | None = None,
    horizon: str | None = None,
    window: str = "7d",
    agg: str = "median",
    fresh_hours: int | None = None,
    repo: RegistryRepo = Depends(repo_dep),
    settings: Settings = Depends(settings_dep),
) -> HTMLResponse:
    metrics = list(settings.metrics_to_forecast.queries.keys())
    horizons = list(settings.horizons.keys())
    selected_metric = metric or (metrics[0] if metrics else "cpu")
    selected_horizon = horizon if horizon in horizons else (
        "medium" if "medium" in horizons else (horizons[0] if horizons else "medium")
    )
    selected_window = window if (window in _WINDOW_PRESETS or window == "all") else "7d"
    selected_agg = agg if agg in {"mean", "median", "p95"} else "median"

    # If user explicitly set fresh_hours, use it. Otherwise default to
    # something sane per horizon: short -> 2h, medium -> 12h, long -> 72h.
    if fresh_hours is None:
        defaults = {"short": 2, "medium": 12, "long": 72}
        fresh_hours_eff = defaults.get(selected_horizon, 24)
    else:
        fresh_hours_eff = max(0, int(fresh_hours))
    fresh_since = (
        datetime.now(timezone.utc) - timedelta(hours=fresh_hours_eff)
        if fresh_hours_eff > 0 else None
    )
    window_since = _since_for_window(selected_window)

    # 1) Aggregate forecast curve across instances at each future timestamp,
    # restricted to instances whose latest run is "fresh" enough.
    forecasts = repo.latest_forecasts(
        metric=selected_metric, horizon=selected_horizon, only_best=True,
        fresh_since=fresh_since,
    )
    by_ts: dict[str, list[float]] = {}
    for f in forecasts:
        if f.point is None:
            continue
        by_ts.setdefault(f.ts.isoformat(), []).append(float(f.point))
    forecast_curve = [
        {"ts": ts, "value": _aggregate(vs, selected_agg), "n": len(vs)}
        for ts, vs in sorted(by_ts.items())
    ]
    unique_instances = len({f.instance for f in forecasts})

    # 2) Accuracy drift: for each completed run within the window on the
    # selected (metric, horizon), use the winning algo's MAE.
    from sqlalchemy import text as _text
    with repo.session() as s:
        sql = """
            SELECT tr.completed_at, r.winning_algo, m.value AS mae
            FROM training_runs tr
            JOIN rankings r ON r.run_id = tr.id
            JOIN run_metrics m ON m.run_id = tr.id AND m.algo = r.winning_algo
                              AND m.score_metric = 'mae' AND m.fold = -1
            WHERE tr.status = 'completed'
              AND tr.metric = :metric
              AND tr.horizon = :horizon
        """
        params: dict[str, Any] = {"metric": selected_metric, "horizon": selected_horizon}
        if window_since is not None:
            sql += " AND tr.completed_at >= :since"
            params["since"] = window_since
        sql += " ORDER BY tr.completed_at ASC"
        rows = list(s.execute(_text(sql), params))
    def _iso(v):
        # text() queries on SQLite return strings; psycopg returns datetimes
        if v is None:
            return None
        if isinstance(v, str):
            return v
        return v.isoformat()

    drift_curve = [
        {
            "completed_at": _iso(r.completed_at),
            "winner": r.winning_algo,
            "mae": float(r.mae) if r.mae is not None else None,
        }
        for r in rows
    ]

    # 3) Winner share — within the time window, for the selected metric+horizon
    wins_map = repo.wins_by_metric(
        metric=selected_metric, horizon=selected_horizon, since=window_since,
    )
    winner_share = [
        {"algo": algo, "wins": cnt}
        for algo, cnt in wins_map.get(selected_metric, {}).items()
    ]
    winner_share.sort(key=lambda r: -r["wins"])

    return templates.TemplateResponse(
        request, "trends.html",
        {
            "active": "trends",
            "metrics": metrics,
            "horizons": horizons,
            "selected_metric": selected_metric,
            "selected_horizon": selected_horizon,
            "selected_window": selected_window,
            "selected_agg": selected_agg,
            "window_presets": list(_WINDOW_PRESETS.keys()),
            "fresh_hours": fresh_hours_eff,
            "instance_count": unique_instances,
            "drift_sample_size": len(drift_curve),
            "forecast_curve_json": json.dumps(forecast_curve),
            "drift_curve_json": json.dumps(drift_curve),
            "winner_share_json": json.dumps(winner_share),
            "display_timezone": settings.display_timezone,
        },
    )
