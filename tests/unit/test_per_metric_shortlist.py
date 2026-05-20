"""Validation tests for algorithms.per_metric shortlists."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

# Importing forecaster.models triggers algorithm registration so the
# validator's REGISTRY check has something to match against.
import forecaster.models  # noqa: F401
from forecaster.config.schema import AlgorithmConfig


def _base_kwargs(**overrides):
    base = {
        "enabled": ["naive", "seasonal_naive", "ets"],
        "defaults": {},
    }
    base.update(overrides)
    return base


def test_per_metric_subset_accepted():
    cfg = AlgorithmConfig(**_base_kwargs(per_metric={"cpu": ["naive", "ets"]}))
    assert cfg.per_metric["cpu"] == ["naive", "ets"]


def test_per_metric_missing_metric_falls_back():
    cfg = AlgorithmConfig(**_base_kwargs(per_metric={"cpu": ["naive"]}))
    # other metrics aren't in per_metric → caller falls back to `enabled`
    assert "disk" not in cfg.per_metric


def test_per_metric_non_subset_rejected():
    # `arima` is registered but not in enabled → should fail
    with pytest.raises(ValidationError) as excinfo:
        AlgorithmConfig(**_base_kwargs(per_metric={"cpu": ["naive", "arima"]}))
    assert "must also appear in algorithms.enabled" in str(excinfo.value)


def test_per_metric_typo_rejected():
    with pytest.raises(ValidationError) as excinfo:
        AlgorithmConfig(
            **_base_kwargs(
                enabled=["naive", "ets", "totally_made_up"],
                per_metric={"cpu": ["totally_made_up"]},
            )
        )
    assert "unregistered" in str(excinfo.value)


def test_per_metric_duplicate_rejected():
    with pytest.raises(ValidationError) as excinfo:
        AlgorithmConfig(**_base_kwargs(per_metric={"cpu": ["naive", "naive"]}))
    assert "duplicates" in str(excinfo.value)


def test_per_metric_empty_list_rejected():
    with pytest.raises(ValidationError) as excinfo:
        AlgorithmConfig(**_base_kwargs(per_metric={"cpu": []}))
    assert "empty" in str(excinfo.value)


def test_shipped_defaults_validate():
    """The defaults shipped in config/default.yaml must be valid."""
    from pathlib import Path

    from forecaster.config.loader import load_settings

    config_dir = Path(__file__).resolve().parents[2] / "config"
    settings = load_settings(config_dir)
    assert "cpu" in settings.algorithms.per_metric
    assert "mem" in settings.algorithms.per_metric
    assert "disk" in settings.algorithms.per_metric
    # Each is a subset of enabled
    for metric, shortlist in settings.algorithms.per_metric.items():
        assert set(shortlist).issubset(set(settings.algorithms.enabled)), metric
