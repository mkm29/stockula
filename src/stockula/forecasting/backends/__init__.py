"""Forecasting backends for time series prediction."""

from .autogluon import AutoGluonBackend
from .autots import AutoTSBackend
from .base import ForecastBackend, ForecastResult

__all__ = [
    "ForecastBackend",
    "ForecastResult",
    "AutoTSBackend",
    "AutoGluonBackend",
]
