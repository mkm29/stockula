"""Stockula - Financial trading and analysis library"""

# Suppress warnings at package import time
import logging

logging.getLogger("alembic.runtime.migration").setLevel(logging.WARNING)
logging.getLogger("alembic").setLevel(logging.WARNING)

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import os

os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

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

__version__ = "0.4.0"

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
