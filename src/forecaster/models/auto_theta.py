"""Nixtla AutoTheta — auto-selects the best Theta variant."""

from __future__ import annotations

from typing import Any

from ._nixtla_base import NixtlaForecasterBase, _common_init
from .registry import register


@register("auto_theta")
class AutoThetaForecaster(NixtlaForecasterBase):
    def __init__(self, season_length: int = 288, **hp: Any) -> None:
        _common_init(self, season_length, **hp)

    def _make_model(self):
        from statsforecast.models import AutoTheta
        return AutoTheta(season_length=self._effective_season)
