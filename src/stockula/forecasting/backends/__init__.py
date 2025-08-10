"""Forecasting backends for time series prediction."""

from .base import ForecastBackend, ForecastResult
from .simple import SimpleForecastBackend

# Try to import AutoGluon (real dependency), else fall back to simple backend
try:
    import importlib

    importlib.import_module("autogluon.timeseries")
    from .autogluon import AutoGluonBackend

    AUTOGLUON_AVAILABLE = True
except Exception:
    AutoGluonBackend = SimpleForecastBackend  # type: ignore[assignment,misc] # Use simple backend as fallback
    AUTOGLUON_AVAILABLE = False

# Try to import Chronos backend (zero-shot forecasting)
try:
    from .chronos import ChronosBackend

    CHRONOS_AVAILABLE = True
except Exception:  # ImportError or runtime import issues
    ChronosBackend = SimpleForecastBackend  # type: ignore[assignment,misc] # Fallback to simple backend
    CHRONOS_AVAILABLE = False

__all__ = [
    "ForecastBackend",
    "ForecastResult",
    "AutoGluonBackend",
    "SimpleForecastBackend",
    "ChronosBackend",
    "AUTOGLUON_AVAILABLE",
    "CHRONOS_AVAILABLE",
]
