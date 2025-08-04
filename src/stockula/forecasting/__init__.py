"""Forecasting module using AutoTS."""

from .forecaster import StockForecaster
from .manager import ForecastingManager

__all__ = ["StockForecaster", "ForecastingManager"]
