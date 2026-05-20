"""Per-algorithm parallelism inside a single training job.

A worker picks one (instance, metric, horizon) task off the queue and
trains all configured algorithms for it concurrently. Each algorithm
runs in a child process (ProcessPoolExecutor) so a misbehaving model
can't bring down its siblings.

We deliberately avoid sharing the Forecaster object across processes —
the child returns lightweight artifacts (scores, forecast arrays,
pickled bytes) which the parent persists.
"""

from __future__ import annotations

import logging
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..evaluation.backtest import walk_forward
from ..evaluation.metrics import all_metrics
from ..models import build

log = logging.getLogger(__name__)


@dataclass(slots=True)
class AlgoResult:
    algo: str
    ok: bool
    scores: dict[str, float] = field(default_factory=dict)
    per_fold_scores: list[dict[str, float]] = field(default_factory=list)
    forecast_point: np.ndarray | None = None
    forecast_lower: np.ndarray | None = None
    forecast_upper: np.ndarray | None = None
    forecast_timestamps: pd.DatetimeIndex | None = None
    artifact_bytes: bytes | None = None
    train_duration_seconds: float = 0.0
    error: str | None = None


def _train_one(
    algo: str,
    hyperparams: dict[str, Any],
    series_pickle: bytes,
    horizon_steps: int,
    folds: int,
    holdout_fraction: float,
    alpha: float,
) -> AlgoResult:
    """Train + backtest + forecast a single algorithm. Runs in child proc."""
    import pickle

    from ..models import build as _build  # re-import in child to register

    t0 = time.perf_counter()
    try:
        series: pd.Series = pickle.loads(series_pickle)  # noqa: S301 - trusted source
        # 1) Backtest
        avg, per_fold = walk_forward(
            series,
            factory=lambda: _build(algo, **hyperparams),
            folds=folds,
            holdout_fraction=holdout_fraction,
        )
        # 2) Refit on full history
        model = _build(algo, **hyperparams)
        model.fit(series)
        point = model.predict(horizon_steps).astype(float)
        lower, upper = model.predict_interval(horizon_steps, alpha=alpha)
        # 3) Future timestamps
        step = series.index[1] - series.index[0]
        ts = pd.date_range(start=series.index[-1] + step, periods=horizon_steps, freq=step)

        artifact = pickle.dumps(model)
        # If backtest produced no scores (tiny series), score on a tail holdout
        if not avg:
            tail = max(1, int(len(series) * holdout_fraction))
            if len(series) > tail + 5:
                train = series.iloc[:-tail]
                test = series.iloc[-tail:]
                m = _build(algo, **hyperparams)
                m.fit(train)
                pred = m.predict(len(test))
                avg = all_metrics(test.to_numpy(), pred)
                per_fold = [avg]

        return AlgoResult(
            algo=algo,
            ok=True,
            scores=avg,
            per_fold_scores=per_fold,
            forecast_point=point,
            forecast_lower=np.asarray(lower, dtype=float),
            forecast_upper=np.asarray(upper, dtype=float),
            forecast_timestamps=ts,
            artifact_bytes=artifact,
            train_duration_seconds=time.perf_counter() - t0,
        )
    except Exception as exc:  # noqa: BLE001 - we want to surface but not crash
        return AlgoResult(
            algo=algo,
            ok=False,
            error=f"{exc.__class__.__name__}: {exc}\n{traceback.format_exc()}",
            train_duration_seconds=time.perf_counter() - t0,
        )


def train_all_algos(
    series: pd.Series,
    *,
    algorithms: list[str],
    defaults: dict[str, dict[str, Any]],
    horizon_steps: int,
    folds: int,
    holdout_fraction: float,
    alpha: float,
    max_workers: int,
) -> list[AlgoResult]:
    """Train every configured algorithm in parallel; return per-algo results."""
    import pickle

    series_pickle = pickle.dumps(series)
    results: list[AlgoResult] = []
    # ProcessPool with max_workers=1 still gives us isolation.
    with ProcessPoolExecutor(max_workers=max(1, max_workers)) as ex:
        future_to_algo = {
            ex.submit(
                _train_one,
                algo,
                defaults.get(algo, {}),
                series_pickle,
                horizon_steps,
                folds,
                holdout_fraction,
                alpha,
            ): algo
            for algo in algorithms
        }
        for fut in as_completed(future_to_algo):
            res = fut.result()
            if not res.ok:
                log.warning("algo %s failed: %s", res.algo, res.error)
            results.append(res)
    return results
