"""Ranking endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from ...registry.repo import RegistryRepo
from ..deps import repo_dep

router = APIRouter(prefix="/rankings", tags=["rankings"])


@router.get("")
def latest(
    instance: str | None = None,
    metric: str | None = None,
    horizon: str | None = None,
    repo: RegistryRepo = Depends(repo_dep),
) -> list[dict[str, Any]]:
    rows = repo.latest_rankings(instance=instance, metric=metric, horizon=horizon)
    return [
        {
            "instance": r.instance,
            "metric": r.metric,
            "horizon": r.horizon,
            "winning_algo": r.winning_algo,
            "ranked": r.ranked,
            "run_id": r.run_id,
        }
        for r in rows
    ]
