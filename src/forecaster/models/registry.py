"""Algorithm registry.

Each Forecaster subclass decorates itself with @register("name") so the
training pipeline can build it from a config string.
"""

from __future__ import annotations

from typing import Any

from .base import Forecaster

REGISTRY: dict[str, type[Forecaster]] = {}


def register(name: str):
    """Decorator to register a forecaster class by name."""

    def deco(cls: type[Forecaster]) -> type[Forecaster]:
        if name in REGISTRY:
            raise ValueError(f"duplicate forecaster name: {name}")
        cls.name = name  # type: ignore[assignment]
        REGISTRY[name] = cls
        return cls

    return deco


def build(name: str, **hyperparams: Any) -> Forecaster:
    if name not in REGISTRY:
        raise KeyError(f"unknown algorithm: {name}. registered: {sorted(REGISTRY)}")
    return REGISTRY[name](**hyperparams)
