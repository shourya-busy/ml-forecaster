"""Weighted composite ranking of candidate models.

Each ranking metric is min-max normalised across the candidates, then
direction-flipped (so larger = better for everyone), then a weighted sum
is taken. The model with the highest composite score wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config.schema import RankingConfig


@dataclass(slots=True)
class RankedModel:
    algo: str
    composite: float
    rank: int
    raw_scores: dict[str, float]
    normalised_scores: dict[str, float] = field(default_factory=dict)


def _normalise(values: list[float], direction: str) -> list[float]:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return [float("nan")] * len(values)
    lo = float(arr[finite].min())
    hi = float(arr[finite].max())
    if hi - lo < 1e-12:
        normalised = np.where(finite, 1.0, np.nan)
    else:
        normalised = (arr - lo) / (hi - lo)
        if direction == "min":
            normalised = 1.0 - normalised
        normalised = np.where(finite, normalised, np.nan)
    return [float(x) for x in normalised]


def rank_models(
    scored: dict[str, dict[str, float]],
    config: RankingConfig,
) -> list[RankedModel]:
    """Rank candidates from a {algo: {metric: score}} mapping."""
    if not scored:
        return []

    algos = list(scored.keys())
    norm_per_metric: dict[str, list[float]] = {}
    for metric in config.metrics:
        raw = [scored[a].get(metric, float("nan")) for a in algos]
        norm_per_metric[metric] = _normalise(raw, config.direction[metric])

    composites: list[float] = []
    norm_score_per_algo: dict[str, dict[str, float]] = {a: {} for a in algos}
    for i, algo in enumerate(algos):
        total = 0.0
        weight_used = 0.0
        for metric in config.metrics:
            v = norm_per_metric[metric][i]
            w = config.weights.get(metric, 0.0)
            norm_score_per_algo[algo][metric] = v
            if not np.isnan(v):
                total += v * w
                weight_used += w
        composites.append(total / weight_used if weight_used > 0 else float("nan"))

    order = sorted(
        range(len(algos)),
        key=lambda i: (-composites[i] if not np.isnan(composites[i]) else 1e9),
    )
    out: list[RankedModel] = []
    for rank, idx in enumerate(order, start=1):
        out.append(
            RankedModel(
                algo=algos[idx],
                composite=composites[idx],
                rank=rank,
                raw_scores=scored[algos[idx]],
                normalised_scores=norm_score_per_algo[algos[idx]],
            )
        )
    return out
