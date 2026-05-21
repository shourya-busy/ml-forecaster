"""Nixtla AutoARIMA — numba-compiled, auto (p,d,q,P,D,Q,s)."""

from __future__ import annotations

from typing import Any

from ._nixtla_base import NixtlaForecasterBase, _common_init
from .registry import register


@register("auto_arima")
class AutoARIMAForecaster(NixtlaForecasterBase):
    def __init__(self, season_length: int = 288, **hp: Any) -> None:
        _common_init(self, season_length, **hp)

    def _make_model(self):
        from statsforecast.models import AutoARIMA
        return AutoARIMA(season_length=self._effective_season)
