"""Training-run endpoints.

POST /runs              — kick off a training run (async, Celery)
POST /runs/sync         — kick off a run synchronously (debugging)
GET  /runs              — list recent runs
GET  /runs/{id}         — fetch a single run summary
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ...registry.repo import RegistryRepo
from ...training.pipeline import run_pipeline
from ...training.tasks import train_task
from ..deps import repo_dep

router = APIRouter(prefix="/runs", tags=["runs"])


class RunRequest(BaseModel):
    instance: str
    metric: str
    horizon: str


@router.post("")
def trigger(req: RunRequest) -> dict[str, Any]:
    async_result = train_task.apply_async(args=[req.instance, req.metric, req.horizon])
    return {"task_id": async_result.id, **req.model_dump()}


@router.post("/sync")
def trigger_sync(req: RunRequest) -> dict[str, Any]:
    """Run the pipeline in-process. Useful in dev / for tests."""
    run_id = run_pipeline(instance=req.instance, metric=req.metric, horizon=req.horizon)
    return {"run_id": run_id, **req.model_dump()}


@router.get("")
def list_runs(
    instance: str | None = None,
    metric: str | None = None,
    horizon: str | None = None,
    limit: int = 50,
    repo: RegistryRepo = Depends(repo_dep),
) -> list[dict[str, Any]]:
    rows = repo.list_runs(instance=instance, metric=metric, horizon=horizon, limit=limit)
    return [
        {
            "id": r.id,
            "instance": r.instance,
            "metric": r.metric,
            "horizon": r.horizon,
            "status": r.status,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "completed_at": r.completed_at.isoformat() if r.completed_at else None,
            "duration_seconds": r.duration_seconds,
            "error": r.error,
        }
        for r in rows
    ]


@router.get("/{run_id}")
def get_run(run_id: int, repo: RegistryRepo = Depends(repo_dep)) -> dict[str, Any]:
    r = repo.get_run(run_id)
    if r is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {
        "id": r.id,
        "instance": r.instance,
        "metric": r.metric,
        "horizon": r.horizon,
        "status": r.status,
        "started_at": r.started_at.isoformat() if r.started_at else None,
        "completed_at": r.completed_at.isoformat() if r.completed_at else None,
        "duration_seconds": r.duration_seconds,
        "error": r.error,
        "config_snapshot": r.config_snapshot,
    }
