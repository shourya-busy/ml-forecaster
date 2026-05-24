"""Unit tests for the AI explainer prompt builder + Ollama wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

from forecaster.ai import explainer as exp_mod
from forecaster.ai.explainer import (
    ExplainContext,
    ForecastPoint,
    SeriesPoint,
    build_prompt,
    explain,
)
from forecaster.ai.ollama_client import OllamaError
from forecaster.config.loader import load_settings

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def _settings():
    return load_settings(CONFIG_DIR)


def _ctx(**overrides):
    base = ExplainContext(
        kind="forecast", title="cpu chart",
        instance="srv-1", metric="cpu", horizon="medium",
        best_algo="ets",
        actuals=[SeriesPoint(ts=f"2026-05-20T00:{i:02d}:00Z", value=10.0 + i)
                 for i in range(5)],
        forecast=[ForecastPoint(ts=f"2026-05-20T00:{i:02d}:00Z",
                                point=12.0 + i, lower=11.0 + i, upper=13.0 + i)
                  for i in range(3)],
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def test_build_prompt_contains_identifiers_and_data():
    p = build_prompt(_ctx())
    assert "Target: srv-1 · cpu · medium" in p
    assert "Best model: ets" in p
    assert "Actuals" in p
    assert "Forecast" in p
    assert "Actual stats" in p
    assert "Forecast stats" in p


def test_build_prompt_includes_overlap_mae_when_aligned():
    # Make actuals and forecast share timestamps so the overlap MAE block fires
    common_ts = [f"2026-05-20T00:{i:02d}:00Z" for i in range(3)]
    actuals = [SeriesPoint(ts=ts, value=10.0) for ts in common_ts]
    forecast = [ForecastPoint(ts=ts, point=12.0, lower=11.0, upper=13.0) for ts in common_ts]
    ctx = ExplainContext(kind="forecast", title="x",
                         instance="i", metric="cpu", horizon="short",
                         actuals=actuals, forecast=forecast)
    p = build_prompt(ctx)
    assert "Overlap MAE" in p


def test_build_prompt_handles_explore_with_no_forecast():
    ctx = ExplainContext(
        kind="explore", title='up{job="netdata"}',
        actuals=[SeriesPoint(ts="2026-05-20T00:00:00Z", value=1.0)],
    )
    p = build_prompt(ctx)
    assert "Actuals" in p
    assert "Forecast" not in p


def test_downsample_keeps_first_and_last():
    big_actuals = [SeriesPoint(ts=f"TS{i:04d}", value=float(i)) for i in range(200)]
    ctx = ExplainContext(kind="explore", title="x", actuals=big_actuals)
    p = build_prompt(ctx, cap=10)
    # First and last entries always survive the even-spaced downsample.
    assert "TS0000" in p
    assert "TS0199" in p
    # And we should NOT have crammed all 200 in
    assert "TS0100" not in p or p.count("\n  TS") <= 10


def test_explain_calls_ollama_and_returns_text(monkeypatch):
    settings = _settings()
    settings.ai.enabled = True

    calls = {}
    class _StubClient:
        def __init__(self, cfg):
            calls["init_url"] = cfg.base_url
        def generate(self, prompt, *, system=None, model=None, options=None):
            calls["prompt_len"] = len(prompt)
            calls["system_present"] = bool(system)
            return "well, the actuals are flat and the forecast follows."
    monkeypatch.setattr(exp_mod, "OllamaClient", _StubClient)

    out = explain(_ctx(), settings)
    assert "flat" in out
    assert calls["system_present"] is True
    assert calls["prompt_len"] > 100


def test_explain_raises_when_disabled():
    settings = _settings()
    settings.ai.enabled = False
    with pytest.raises(OllamaError, match="disabled"):
        explain(_ctx(), settings)
