"""All built-in forecasters auto-register here."""

from .base import BaseForecaster, ForecastResult, Forecaster
from .registry import REGISTRY, build, register

# Importing the implementation modules registers them.
from . import (  # noqa: F401
    arima,
    auto_arima,
    auto_ets,
    auto_theta,
    drift,
    ets,
    gru,
    holt_winters,
    knn,
    lightgbm_model,
    linear_lag,
    lstm,
    mean,
    median,
    mstl,
    naive,
    nbeats,
    neural_prophet,
    prophet_model,
    random_forest,
    sarima,
    seasonal_naive,
    theta,
    xgboost_model,
)

__all__ = ["BaseForecaster", "ForecastResult", "Forecaster", "REGISTRY", "build", "register"]
