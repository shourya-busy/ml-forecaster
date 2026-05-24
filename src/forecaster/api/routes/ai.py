"""AI explainer endpoints.

POST /ai/explain  → returns a plain-English explanation of the supplied chart.
GET  /ai/status   → cheap probe so the UI can hide the button when Ollama
                    is unreachable, instead of letting users click into an
                    error message.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ...ai.explainer import (
    ExplainContext,
    ForecastPoint,
    SeriesPoint,
    explain,
)
from ...ai.ollama_client import OllamaClient, OllamaError
from ...config.schema import Settings
from ..deps import settings_dep

log = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["ai"])


# ----- request / response shapes -----

class _SeriesIn(BaseModel):
    ts: str
    value: float | None = None


class _ForecastIn(BaseModel):
    ts: str
    point: float | None = None
    lower: float | None = None
    upper: float | None = None


class ExplainRequest(BaseModel):
    kind: Literal["forecast", "explore"]
    title: str = "chart"
    instance: str | None = None
    metric: str | None = None
    horizon: str | None = None
    best_algo: str | None = None
    actuals: list[_SeriesIn] = Field(default_factory=list)
    forecast: list[_ForecastIn] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class ExplainResponse(BaseModel):
    explanation: str
    model: str


@router.get("/status")
def status(settings: Settings = Depends(settings_dep)) -> dict[str, Any]:
    if not settings.ai.enabled:
        return {"enabled": False, "reachable": False, "reason": "disabled in config"}
    client = OllamaClient(settings.ai.ollama)
    ok, err = client.is_reachable()
    return {
        "enabled": True,
        "reachable": ok,
        "model": settings.ai.ollama.model,
        "base_url": settings.ai.ollama.base_url,
        "reason": err,
    }


@router.get("/models")
def models(
    base_url: str | None = None,
    settings: Settings = Depends(settings_dep),
) -> dict[str, Any]:
    """List models available on the configured (or supplied) Ollama server.

    The UI Settings page passes `?base_url=…` to preview models for a URL
    the user has typed in but not yet saved. Returns an empty list with an
    `error` field rather than 5xx so the form can render a hint.
    """
    cfg = settings.ai.ollama
    if base_url:
        cfg = cfg.model_copy(update={"base_url": base_url})
    client = OllamaClient(cfg)
    try:
        names = client.list_models()
        return {"models": names, "base_url": cfg.base_url, "error": None}
    except OllamaError as exc:
        return {"models": [], "base_url": cfg.base_url, "error": str(exc)}


@router.post("/explain", response_model=ExplainResponse)
def explain_route(
    body: ExplainRequest,
    settings: Settings = Depends(settings_dep),
) -> ExplainResponse:
    if not settings.ai.enabled:
        raise HTTPException(503, "AI explainer is disabled in config")
    # No actuals AND no forecast → nothing to explain
    if not body.actuals and not body.forecast:
        raise HTTPException(422, "request must include actuals or forecast")

    ctx = ExplainContext(
        kind=body.kind, title=body.title,
        instance=body.instance, metric=body.metric, horizon=body.horizon,
        best_algo=body.best_algo,
        actuals=[
            SeriesPoint(ts=p.ts, value=float(p.value))
            for p in body.actuals if p.value is not None
        ],
        forecast=[
            ForecastPoint(ts=p.ts, point=p.point, lower=p.lower, upper=p.upper)
            for p in body.forecast
        ],
        extra=body.extra,
    )

    try:
        text = explain(ctx, settings)
    except OllamaError as exc:
        log.warning("ai explain failed: %s", exc)
        raise HTTPException(502, f"ollama error: {exc}") from None

    return ExplainResponse(explanation=text, model=settings.ai.ollama.model)
