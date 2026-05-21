"""Nixtla MSTL — multiple-seasonality STL decomposition + ETS on residual.

Built for series with overlapping seasonalities (e.g. daily + weekly):
decompose, fit each season independently, fit ETS to what's left.
"""

from __future__ import annotations

from typing import Any

from ._nixtla_base import NixtlaForecasterBase
from .base import BaseForecaster
from .registry import register


@register("mstl")
class MSTLForecaster(NixtlaForecasterBase):
    def __init__(
        self,
        season_lengths: list | tuple = (288, 2016),   # daily + weekly @ 5min step
        **hp: Any,
    ) -> None:
        BaseForecaster.__init__(self, season_lengths=list(season_lengths), **hp)
        self.season_lengths = [int(x) for x in season_lengths]
        # The base class uses a single season_length for its
        # "shrink-if-too-short" logic; use the longest period.
        self.season_length = max(self.season_lengths) if self.season_lengths else 1

    def _make_model(self):
        from statsforecast.models import MSTL
        # Shrink the season list to what fits the available data
        # (the base class already computed self._effective_season).
        seasons = [s for s in self.season_lengths if s <= self._effective_season]
        if not seasons:
            seasons = [self._effective_season]
        return MSTL(season_length=seasons)

    def lookback_required(self) -> int:
        return max(50, 2 * (self.season_length or 1))
