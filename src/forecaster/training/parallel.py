"""Per-algorithm parallelism inside a single training job.

A worker picks one (instance, metric, horizon) task off the queue and
trains all configured algorithms for it concurrently using threads.

Why threads, not processes? Celery's default `prefork` pool runs each task
inside a *daemonic* child process, and Python's `multiprocessing` (and by
extension `ProcessPoolExecutor`) refuses to let a daemonic process spawn
its own children — you get `AssertionError: daemonic processes are not
allowed to have children`. Threads have no such restriction.

Trade-offs vs. ProcessPoolExecutor:
- We lose hard isolation: a C-level segfault in one algo can take down the
  whole worker process. In practice the libraries we depend on
  (statsmodels, lightgbm, xgboost, torch) are mature enough that this is
  rare; if it happens, Celery's master restarts the worker and the failed
  task is retried.
- We gain compatibility with Celery's default pool *and* keep parallelism:
  most algo work is in C extensions that release the GIL (numpy, BLAS,
  torch, lightgbm, xgboost), so threads still parallelise well.

A worker-process Python exception is still caught here and surfaced as
`AlgoResult.ok=False`, so a misbehaving model's Python failure can't crash
the worker.
"""

from __future__ import annotations

import logging
import pickle
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    series: pd.Series,
    horizon_steps: int,
    folds: int,
    holdout_fraction: float,
    alpha: float,
) -> AlgoResult:
    """Train + backtest + forecast a single algorithm in a worker thread."""
    t0 = time.perf_counter()
    try:
        # 1) Backtest
        avg, per_fold = walk_forward(
            series,
            factory=lambda: build(algo, **hyperparams),
            folds=folds,
            holdout_fraction=holdout_fraction,
        )
        # 2) Refit on full history
        model = build(algo, **hyperparams)
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
                m = build(algo, **hyperparams)
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
    except Exception as exc:  # noqa: BLE001 - surface but never crash siblings
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
    """Train every configured algorithm in parallel threads; return per-algo results."""
    results: list[AlgoResult] = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers),
                            thread_name_prefix="train") as ex:
        future_to_algo = {
            ex.submit(
                _train_one,
                algo,
                defaults.get(algo, {}),
                series,
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
