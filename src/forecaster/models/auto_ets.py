"""Nixtla AutoETS — auto-selects the best ETS spec for the series."""

from __future__ import annotations

from typing import Any

from ._nixtla_base import NixtlaForecasterBase, _common_init
from .registry import register


@register("auto_ets")
class AutoETSForecaster(NixtlaForecasterBase):
    def __init__(self, season_length: int = 288, **hp: Any) -> None:
        _common_init(self, season_length, **hp)

    def _make_model(self):
        from statsforecast.models import AutoETS
        return AutoETS(season_length=self._effective_season)
