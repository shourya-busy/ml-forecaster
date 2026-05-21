"""Algorithm registry.

Each Forecaster subclass decorates itself with @register("name") so the
training pipeline can build it from a config string.

`ALGO_INFO` is the surface the UI consumes to render an "algorithm
library" — short descriptions and "when to use" hints so non-ML
operators can intelligently enable/disable algorithms without reading
the code.
"""

from __future__ import annotations

from typing import Any

from .base import Forecaster

REGISTRY: dict[str, type[Forecaster]] = {}


# Curated metadata. The training pipeline ignores this; only the UI reads it.
# Keep entries terse — they render as table cells in /ui/manage/training.
ALGO_INFO: dict[str, dict[str, str]] = {
    "naive": {
        "family": "baseline",
        "complexity": "trivial",
        "speed": "instant",
        "description": "Predicts the last observed value indefinitely. Sanity baseline — if a fancier model can't beat naive, something is wrong with your data or your features.",
        "when_to_use": "Always enable as a control. Wins surprisingly often on noisy short-horizon forecasts.",
    },
    "seasonal_naive": {
        "family": "baseline",
        "complexity": "trivial",
        "speed": "instant",
        "description": "y_{t+h} = y_{t+h-S}: repeats the last full season. Default season is 288 steps (one day at 5-minute step).",
        "when_to_use": "Strong daily/weekly seasonality (e.g. CPU, traffic). Skip for monotonic series like disk usage.",
    },
    "arima": {
        "family": "statistical",
        "complexity": "medium",
        "speed": "seconds",
        "description": "Autoregressive Integrated Moving Average. Grid-search over (p,d,q) to find the lowest AIC fit.",
        "when_to_use": "Stationary or near-stationary series. Memory / disk growth where you want a parametric trend.",
    },
    "ets": {
        "family": "statistical",
        "complexity": "low",
        "speed": "fast",
        "description": "Exponential smoothing state-space. Captures level, trend, and additive/multiplicative seasonality.",
        "when_to_use": "Smooth, slowly-evolving signals. A solid statistical default that's hard to beat without DL.",
    },
    "holt_winters": {
        "family": "statistical",
        "complexity": "low",
        "speed": "fast",
        "description": "Triple exponential smoothing with explicit seasonal_periods. Needs at least 2 full seasons of history.",
        "when_to_use": "Repeating daily/weekly patterns on long-enough series. Falls back to non-seasonal on short series.",
    },
    "prophet": {
        "family": "statistical",
        "complexity": "medium",
        "speed": "slow (~10s)",
        "description": "Facebook's piecewise-trend + Fourier seasonality model. Handles holidays / changepoints out of the box.",
        "when_to_use": "Business metrics with multiple seasonalities. Worth the cost when you have ≥30 days of history.",
    },
    "xgboost": {
        "family": "ml",
        "complexity": "medium",
        "speed": "fast (~2s)",
        "description": "Gradient-boosted trees on lag + calendar features (hour, dow, minute). Recursive multi-step forecasts.",
        "when_to_use": "Non-linear patterns, regime changes. Pairs well with seasonality when calendar features matter.",
    },
    "lightgbm": {
        "family": "ml",
        "complexity": "medium",
        "speed": "fast (~2s)",
        "description": "LightGBM on the same lag features as XGBoost. Often slightly faster and similar accuracy.",
        "when_to_use": "Same niche as XGBoost; keep one of the two unless you want an ensemble.",
    },
    "lstm": {
        "family": "deep-learning",
        "complexity": "high",
        "speed": "slow (~5-10s)",
        "description": "Small (32-unit, 1-layer) PyTorch LSTM. Trains 20 epochs on normalized lag windows.",
        "when_to_use": "Long-memory non-linear dynamics. Bumps accuracy when statistical models stall.",
    },
    "nbeats": {
        "family": "deep-learning",
        "complexity": "high",
        "speed": "slow (~5-10s)",
        "description": "Generic-block N-BEATS architecture: stacked MLPs with backcast residuals. Universally strong on M4-style benchmarks.",
        "when_to_use": "When LSTM wins and you want a competing DL model. Often #1 on noisy seasonal series.",
    },
    # ----- additional baselines -----
    "drift": {
        "family": "baseline",
        "complexity": "trivial",
        "speed": "instant",
        "description": "Random walk with constant slope (y_n - y_1)/(n-1). Extrapolates a straight line through the endpoints.",
        "when_to_use": "Slowly growing or shrinking series (disk, slow memory leaks). A trend-aware control next to naive.",
    },
    "mean": {
        "family": "baseline",
        "complexity": "trivial",
        "speed": "instant",
        "description": "Predicts the historical arithmetic mean indefinitely. No trend, no seasonality, no memory.",
        "when_to_use": "Stationary, mean-reverting series. Sanity baseline for 'is there structure I can exploit?'",
    },
    "median": {
        "family": "baseline",
        "complexity": "trivial",
        "speed": "instant",
        "description": "Like mean but uses the median — robust to outliers and spikes.",
        "when_to_use": "Outlier-heavy series where mean over-weights anomalies.",
    },
    # ----- additional statistical -----
    "theta": {
        "family": "statistical",
        "complexity": "low",
        "speed": "fast",
        "description": "Theta method (Assimakopoulos & Nikolopoulos 2000) — decomposes into two θ-lines, recombines. Famously beat all comers on M3.",
        "when_to_use": "Strong simple baseline. Often within 1-2% of more complex models for a fraction of the cost.",
    },
    "sarima": {
        "family": "statistical",
        "complexity": "high",
        "speed": "slow (~10-30s)",
        "description": "Seasonal ARIMA via state-space form. Captures (p,d,q) × seasonal (P,D,Q,S) dynamics.",
        "when_to_use": "When ARIMA wins but you suspect strong residual seasonality. Expensive; weigh against holt_winters / prophet.",
    },
    # ----- additional ML -----
    "linear_lag": {
        "family": "ml",
        "complexity": "low",
        "speed": "fast",
        "description": "L2-regularised (ridge) linear regression on lag + calendar features.",
        "when_to_use": "Interpretable, hard-to-overfit baseline. Often surprisingly close to xgboost on benign series.",
    },
    "random_forest": {
        "family": "ml",
        "complexity": "medium",
        "speed": "fast (~2-3s)",
        "description": "Bagging of decision trees on lag + calendar features. Less prone to overfitting than boosting.",
        "when_to_use": "When you want an ML model without the hyperparameter drama of xgboost / lightgbm.",
    },
    "knn": {
        "family": "ml",
        "complexity": "low",
        "speed": "fast",
        "description": "K-Nearest Neighbors on standardised lag windows — finds historical windows that look like 'now' and averages their next-step values.",
        "when_to_use": "Series with repeating local motifs (workload patterns, scheduled spikes). Bad fit for monotonic series.",
    },
    # ----- additional deep learning -----
    "gru": {
        "family": "deep-learning",
        "complexity": "high",
        "speed": "slow (~3-6s)",
        "description": "Gated Recurrent Unit — similar architecture to LSTM with fewer parameters (~30% faster, slightly worse on very long-memory signals).",
        "when_to_use": "When you want LSTM-like flexibility with shorter training time. Try alongside lstm; keep whichever wins.",
    },
    # ----- Nixtla statsforecast — numba-compiled auto-models -----
    "auto_arima": {
        "family": "statistical",
        "complexity": "low",
        "speed": "fast (~1s)",
        "description": "Nixtla's numba-compiled AutoARIMA. Same model class as `arima` but searches (p,d,q,P,D,Q) automatically and runs 10-100× faster.",
        "when_to_use": "Drop-in replacement for `arima` once you trust the new stack. Faster and at least as accurate.",
    },
    "auto_ets": {
        "family": "statistical",
        "complexity": "low",
        "speed": "fast",
        "description": "Nixtla's AutoETS — auto-selects level / trend / seasonality components. Numba-compiled counterpart to statsmodels' ETS.",
        "when_to_use": "Always worth enabling alongside auto_arima; one of the two typically reaches the Pareto frontier on smooth series.",
    },
    "auto_theta": {
        "family": "statistical",
        "complexity": "low",
        "speed": "fast",
        "description": "Nixtla's AutoTheta — auto-picks between Theta variants (STM / OTM / DSTM / DOTM).",
        "when_to_use": "Drop-in replacement for `theta` with auto-variant selection. Strong M-competition baseline.",
    },
    "mstl": {
        "family": "statistical",
        "complexity": "medium",
        "speed": "fast",
        "description": "Multiple-seasonality STL decomposition + ETS on the residual. Handles overlapping daily + weekly (default 288 + 2016 steps at 5-min cadence).",
        "when_to_use": "Series with more than one seasonality (traffic with day-of-week × hour-of-day). Beats single-period models on these.",
    },
    # ----- NeuralProphet -----
    "neural_prophet": {
        "family": "deep-learning",
        "complexity": "high",
        "speed": "slow (~15-30s)",
        "description": "NeuralProphet — Prophet's decomposable trend + a neural AR-Net for short-term dynamics. Successor to Prophet by the same author.",
        "when_to_use": "When Prophet wins but you suspect autoregressive structure it's missing (deployment cadences, periodic spikes).",
    },
}


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


def algo_info(name: str) -> dict[str, str]:
    """Return descriptive metadata for an algorithm, or an empty stub."""
    return ALGO_INFO.get(name, {
        "family": "unknown", "complexity": "unknown", "speed": "unknown",
        "description": "(no description registered)",
        "when_to_use": "—",
    })
