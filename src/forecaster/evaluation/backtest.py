"""Walk-forward cross-validation.

We split a 1-D series into K expanding-window folds. For each fold we
fit a fresh model on the train slice and predict the holdout. Scores are
averaged across folds.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..models.base import Forecaster
from .metrics import all_metrics


@dataclass(slots=True)
class BacktestResult:
    algo: str
    scores: dict[str, float] = field(default_factory=dict)
    per_fold_scores: list[dict[str, float]] = field(default_factory=list)
    error: str | None = None
    duration_seconds: float = 0.0


def _fold_indices(n: int, folds: int, holdout_fraction: float) -> list[tuple[int, int]]:
    """Return list of (train_end_inclusive, test_end_exclusive) indices."""
    holdout_size = max(1, int(n * holdout_fraction))
    test_total = holdout_size * folds
    if test_total >= n:
        # Can't fit this many folds; collapse to one.
        return [(n - holdout_size, n)]
    starts = [n - test_total + i * holdout_size for i in range(folds)]
    return [(s, s + holdout_size) for s in starts]


def walk_forward(
    series: pd.Series,
    factory: Callable[[], Forecaster],
    *,
    folds: int = 5,
    holdout_fraction: float = 0.1,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Run walk-forward CV; return (averaged_scores, per_fold_scores)."""
    n = len(series)
    fold_idx = _fold_indices(n, folds, holdout_fraction)

    per_fold: list[dict[str, float]] = []
    for train_end, test_end in fold_idx:
        train = series.iloc[:train_end]
        test = series.iloc[train_end:test_end]
        if len(train) < 5 or test.empty:
            continue
        model = factory()
        model.fit(train)
        pred = model.predict(len(test))
        per_fold.append(all_metrics(test.to_numpy(), pred))

    if not per_fold:
        return {}, []

    avg = {
        k: float(np.nanmean([f[k] for f in per_fold if not np.isnan(f.get(k, np.nan))]))
        for k in per_fold[0]
    }
    return avg, per_fold
