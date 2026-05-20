"""Algorithm-registry introspection."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from ...config.schema import Settings
from ...models import REGISTRY
from ..deps import settings_dep

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
def list_models(settings: Settings = Depends(settings_dep)) -> dict[str, Any]:
    enabled = set(settings.algorithms.enabled)
    return {
        "registered": sorted(REGISTRY.keys()),
        "enabled": sorted(enabled),
        "disabled_but_registered": sorted(set(REGISTRY) - enabled),
        "defaults": settings.algorithms.defaults,
    }
