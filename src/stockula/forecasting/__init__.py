"""Forecasting module with AutoGluon and Chronos backends (falls back to simple)."""

from .manager import ForecastingManager

__all__ = [
    "ForecastingManager",
]
