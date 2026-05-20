"""Health endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ...registry.repo import RegistryRepo
from ..deps import repo_dep

router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
def readyz(repo: RegistryRepo = Depends(repo_dep)) -> dict[str, str]:
    # Touch the DB to confirm we can reach it.
    with repo.engine.connect() as c:
        c.exec_driver_sql("SELECT 1")
    return {"status": "ready"}
