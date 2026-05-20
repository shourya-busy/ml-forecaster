"""Effective-config introspection + reload trigger."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from ...config.loader import reload_settings
from ...config.schema import Settings
from ..deps import settings_dep

router = APIRouter(prefix="/config", tags=["config"])


@router.get("")
def get_effective(settings: Settings = Depends(settings_dep)) -> dict[str, Any]:
    return settings.model_dump()


@router.post("/reload")
def reload() -> dict[str, str]:
    reload_settings()
    return {"status": "reloaded"}
