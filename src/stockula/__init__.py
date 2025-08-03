"""Stockula - Financial trading and analysis library"""

# All module-level imports must be at the top
import logging
import os
import warnings

# Configure logging
logging.getLogger("alembic.runtime.migration").setLevel(logging.WARNING)
logging.getLogger("alembic").setLevel(logging.WARNING)

# Configure warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure environment
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

# Package imports
from .backtesting import (
    BacktestRunner,
    BaseStrategy,
    MACDStrategy,
    RSIStrategy,
    SMACrossStrategy,
)
from .config import StockulaConfig, load_config
from .data import DataFetcher
from .display import ResultsDisplay
from .forecasting import StockForecaster
from .manager import StockulaManager
from .technical_analysis import TechnicalIndicators

__version__ = "0.7.0"

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
    "StockulaManager",
    "ResultsDisplay",
]
