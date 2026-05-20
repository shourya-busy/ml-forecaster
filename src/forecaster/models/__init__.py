"""All built-in forecasters auto-register here."""

from .base import BaseForecaster, ForecastResult, Forecaster
from .registry import REGISTRY, build, register

# Importing the implementation modules registers them.
from . import (  # noqa: F401
    arima,
    ets,
    holt_winters,
    lightgbm_model,
    lstm,
    naive,
    nbeats,
    prophet_model,
    seasonal_naive,
    xgboost_model,
)

__all__ = ["BaseForecaster", "ForecastResult", "Forecaster", "REGISTRY", "build", "register"]
