"""Forecasting backends for time series prediction."""

from .base import ForecastBackend, ForecastResult
from .simple import SimpleForecastBackend

# Try to import AutoGluon, fall back to simple backend if not available
try:
    from .autogluon import AutoGluonBackend

    AUTOGLUON_AVAILABLE = True
except ImportError:
    AutoGluonBackend = SimpleForecastBackend  # Use simple backend as fallback
    AUTOGLUON_AVAILABLE = False

__all__ = [
    "ForecastBackend",
    "ForecastResult",
    "AutoGluonBackend",
    "SimpleForecastBackend",
    "AUTOGLUON_AVAILABLE",
]
