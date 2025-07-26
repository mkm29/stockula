"""Stockula - Financial trading and analysis library"""

from .data import DataFetcher
from .technical_analysis import TechnicalIndicators
from .backtesting import BaseStrategy, SMACrossStrategy, RSIStrategy, BacktestRunner
from .forecasting import StockForecaster

__version__ = "0.1.0"

__all__ = [
    "DataFetcher",
    "TechnicalIndicators",
    "BaseStrategy",
    "SMACrossStrategy",
    "RSIStrategy",
    "BacktestRunner",
    "StockForecaster",
]
