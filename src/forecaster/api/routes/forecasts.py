"""Forecast endpoints — full JSON payload with bands."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from ...registry.repo import RegistryRepo
from ..deps import repo_dep

router = APIRouter(prefix="/forecasts", tags=["forecasts"])


@router.get("")
def latest(
    instance: str | None = None,
    metric: str | None = None,
    horizon: str | None = None,
    only_best: bool = True,
    algo: str | None = None,
    repo: RegistryRepo = Depends(repo_dep),
) -> list[dict[str, Any]]:
    rows = repo.latest_forecasts(
        instance=instance, metric=metric, horizon=horizon,
        only_best=only_best, algo=algo,
    )
    return [
        {
            "instance": f.instance,
            "metric": f.metric,
            "horizon": f.horizon,
            "algo": f.algo,
            "ts": f.ts.isoformat(),
            "point": f.point,
            "lower": f.lower,
            "upper": f.upper,
            "is_best": f.is_best,
            "run_id": f.run_id,
        }
        for f in rows
    ]
