"""Build prompts for the Ollama explainer and invoke it.

The UI passes a structured `ExplainContext` describing the chart (kind,
identifiers, series, optional forecast). This module shapes that into a
prompt and returns the model's free-form explanation.

The prompt format is deliberately deterministic — same chart in, same
prompt out — so behavior changes are reproducible and easy to audit.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Literal

from ..config.schema import Settings
from .ollama_client import OllamaClient, OllamaError

log = logging.getLogger(__name__)


# ----- request shape (mirrors what the UI sends) -----

@dataclass
class SeriesPoint:
    ts: str
    value: float


@dataclass
class ForecastPoint:
    ts: str
    point: float | None = None
    lower: float | None = None
    upper: float | None = None


@dataclass
class ExplainContext:
    """All the chart metadata + data needed to ground the model."""

    kind: Literal["forecast", "explore"]
    title: str
    instance: str | None = None
    metric: str | None = None
    horizon: str | None = None
    best_algo: str | None = None
    actuals: list[SeriesPoint] = field(default_factory=list)
    forecast: list[ForecastPoint] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


SYSTEM_PROMPT = (
    "You are a senior site-reliability engineer reviewing a server-metric "
    "forecast chart. Be concise (4-7 sentences max). Cover, in order: "
    "(1) what the actual values are doing recently — trend, level, any "
    "noteworthy spikes; (2) whether the forecast looks reasonable given "
    "those actuals — call out divergence, over/under-shoot, or band width "
    "that seems off; (3) one concrete thing the operator should watch for "
    "or do next. Use plain English. Quote specific numbers from the data. "
    "Do not invent metrics that are not present. If there is no forecast "
    "(explore mode), just summarize the series and any notable patterns."
)


# ----- helpers -----

def _downsample(points: list[Any], cap: int) -> list[Any]:
    """Evenly-spaced subsample down to `cap` items (keeps first + last)."""
    n = len(points)
    if n <= cap or cap < 2:
        return list(points)
    step = (n - 1) / (cap - 1)
    return [points[round(i * step)] for i in range(cap)]


def _fmt_num(v: float | None) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    if abs(v) >= 1000:
        return f"{v:.0f}"
    return f"{v:.3f}"


def _series_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "first": values[0],
        "last": values[-1],
    }


def _format_series_block(label: str, series: list[tuple[str, float | None]], cap: int) -> str:
    sampled = _downsample(series, cap)
    lines = [f"{label} ({len(series)} pts, showing {len(sampled)}):"]
    for ts, v in sampled:
        lines.append(f"  {ts}  {_fmt_num(v)}")
    return "\n".join(lines)


def build_prompt(ctx: ExplainContext, *, cap: int = 60) -> str:
    """Compose the structured prompt body sent as the `prompt` field."""
    parts: list[str] = []
    header = f"Chart: {ctx.title}"
    ident = " · ".join(p for p in (ctx.instance, ctx.metric, ctx.horizon) if p)
    if ident:
        header += f"\nTarget: {ident}"
    if ctx.best_algo:
        header += f"\nBest model: {ctx.best_algo}"
    parts.append(header)

    if ctx.actuals:
        actual_values = [p.value for p in ctx.actuals if p.value is not None]
        stats = _series_stats(actual_values)
        if stats:
            parts.append(
                "Actual stats: "
                + ", ".join(f"{k}={_fmt_num(v)}" for k, v in stats.items())
            )
        parts.append(_format_series_block(
            "Actuals",
            [(p.ts, p.value) for p in ctx.actuals],
            cap,
        ))

    if ctx.forecast:
        point_values = [p.point for p in ctx.forecast if p.point is not None]
        stats = _series_stats(point_values)
        if stats:
            parts.append(
                "Forecast stats: "
                + ", ".join(f"{k}={_fmt_num(v)}" for k, v in stats.items())
            )
        # Compare overlap if any
        if ctx.actuals:
            ac_map = {p.ts: p.value for p in ctx.actuals if p.value is not None}
            diffs = [
                abs(fp.point - ac_map[fp.ts])
                for fp in ctx.forecast
                if fp.point is not None and fp.ts in ac_map
            ]
            if diffs:
                parts.append(
                    f"Overlap MAE: {_fmt_num(statistics.fmean(diffs))} "
                    f"across {len(diffs)} point(s)"
                )
        # Compact forecast block: ts, point, [lower-upper]
        sampled = _downsample(ctx.forecast, cap)
        lines = [f"Forecast ({len(ctx.forecast)} pts, showing {len(sampled)}):"]
        for fp in sampled:
            band = ""
            if fp.lower is not None and fp.upper is not None:
                band = f"  [{_fmt_num(fp.lower)} – {_fmt_num(fp.upper)}]"
            lines.append(f"  {fp.ts}  {_fmt_num(fp.point)}{band}")
        parts.append("\n".join(lines))

    if ctx.extra:
        extras = ", ".join(f"{k}={v}" for k, v in ctx.extra.items())
        parts.append(f"Extra context: {extras}")

    parts.append(
        "Explain the chart for an operator who can see it on a dashboard. "
        "Be specific, cite numbers, and end with one actionable next step."
    )
    return "\n\n".join(parts)


# ----- public entrypoint -----

def explain(ctx: ExplainContext, settings: Settings) -> str:
    """Build the prompt and call Ollama. Raises OllamaError on failure."""
    if not settings.ai.enabled:
        raise OllamaError("AI explainer is disabled in config (ai.enabled=false)")
    client = OllamaClient(settings.ai.ollama)
    prompt = build_prompt(ctx, cap=settings.ai.ollama.max_points_in_prompt)
    return client.generate(prompt, system=SYSTEM_PROMPT)
