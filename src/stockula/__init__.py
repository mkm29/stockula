"""Stockula - Financial trading and analysis library"""

from .data import DataFetcher
from .technical_analysis import TechnicalIndicators
from .backtesting import (
    BaseStrategy,
    SMACrossStrategy,
    RSIStrategy,
    MACDStrategy,
    BacktestRunner,
)
from .forecasting import StockForecaster
from .config import StockulaConfig, load_config

__version__ = "0.1.0"

__all__ = [
    "DataFetcher",
    "TechnicalIndicators",
    "BaseStrategy",
    "SMACrossStrategy",
    "RSIStrategy",
    "MACDStrategy",
    "BacktestRunner",
    "StockForecaster",
    "StockulaConfig",
    "load_config",
]
