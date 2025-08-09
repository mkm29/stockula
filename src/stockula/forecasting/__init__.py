"""Forecasting module with swappable backends (AutoTS and AutoGluon)."""

from .backends import AutoGluonBackend, AutoTSBackend, ForecastBackend, ForecastResult
from .factory import create_forecast_backend
from .forecaster import StockForecaster
from .manager import ForecastingManager

__all__ = [
    # Core components
    "StockForecaster",
    "ForecastingManager",
    # Backend abstraction
    "ForecastBackend",
    "ForecastResult",
    "AutoTSBackend",
    "AutoGluonBackend",
    "create_forecast_backend",
]
