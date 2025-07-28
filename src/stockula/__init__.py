"""Stockula - Financial trading and analysis library"""

from .backtesting import (
    BacktestRunner,
    BaseStrategy,
    MACDStrategy,
    RSIStrategy,
    SMACrossStrategy,
)
from .config import StockulaConfig, load_config
from .data import DataFetcher
from .forecasting import StockForecaster
from .technical_analysis import TechnicalIndicators

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
