"""Forecasting module with AutoGluon and Chronos backends (falls back to simple)."""

from .backends import (
    AUTOGLUON_AVAILABLE,
    CHRONOS_AVAILABLE,
    AutoGluonBackend,
    ChronosBackend,
    ForecastBackend,
    ForecastResult,
    SimpleForecastBackend,
)
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
    "ChronosBackend",
    "create_forecast_backend",
    "AUTOGLUON_AVAILABLE",
    "CHRONOS_AVAILABLE",
]
