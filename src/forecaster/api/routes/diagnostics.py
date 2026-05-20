"""Diagnostics endpoints — tools for trusting / debugging the auto-ranking.

Three endpoints:

- GET /diagnostics/winners
    One row per (instance, metric, horizon); current winner, previous
    winner, when the current streak started, and how stable the recent
    history is. Use this to spot flapping rankings at a glance.

- GET /diagnostics/score-history?instance=&metric=&horizon=&algo=&score=&limit=
    Per-run averaged-fold backtest scores, sorted oldest-first. Use this
    to plot how a model's MAE / RMSE / etc. trend across retrains.

- GET /diagnostics/winner-history?instance=&metric=&horizon=&limit=
    Per-run winning algo + the full ranked list, sorted oldest-first.
    Use this when the score-history shows churn and you want to see
    exactly which model held the crown at each retrain.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from ...config.schema import Settings
from ...registry.repo import RegistryRepo
from ..deps import repo_dep, settings_dep

router = APIRouter(prefix="/diagnostics", tags=["diagnostics"])


@router.get("/winners")
def winners(
    settings: Settings = Depends(settings_dep),
    repo: RegistryRepo = Depends(repo_dep),
) -> list[dict[str, Any]]:
    return repo.winners_summary(
        recent_window=settings.exposition.diagnostics.recent_window_runs
    )


@router.get("/score-history")
def score_history(
    instance: str,
    metric: str,
    horizon: str,
    algo: str | None = None,
    score: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    repo: RegistryRepo = Depends(repo_dep),
) -> list[dict[str, Any]]:
    return repo.score_history(
        instance=instance, metric=metric, horizon=horizon,
        algo=algo, score=score, limit=limit,
    )


@router.get("/winner-history")
def winner_history(
    instance: str,
    metric: str,
    horizon: str,
    limit: int = Query(50, ge=1, le=500),
    repo: RegistryRepo = Depends(repo_dep),
) -> list[dict[str, Any]]:
    return repo.winner_history(
        instance=instance, metric=metric, horizon=horizon, limit=limit,
    )
