"""Forecasting module with AutoGluon backend (falls back to simple if unavailable)."""

from .backends import AUTOGLUON_AVAILABLE, AutoGluonBackend, ForecastBackend, ForecastResult, SimpleForecastBackend
from .factory import create_forecast_backend
from .manager import ForecastingManager

__all__ = [
    # Core components
    "ForecastingManager",
    # Backend abstraction
    "ForecastBackend",
    "ForecastResult",
    "AutoGluonBackend",
    "SimpleForecastBackend",
    "create_forecast_backend",
    "AUTOGLUON_AVAILABLE",
]
