"""Stockula - Financial trading and analysis library.

This package avoids global side effects on import (e.g., mutating logging,
warnings, or environment variables). Runtime configuration is handled in
`stockula.main.setup_logging` and the CLI entrypoint.
"""

# Package imports
from .backtesting import BacktestRunner, BaseStrategy, MACDStrategy, RSIStrategy, SMACrossStrategy
from .config import StockulaConfig, load_config
from .data import DataFetcher
from .display import ResultsDisplay
from .forecasting import ForecastingManager
from .manager import StockulaManager
from .technical_analysis import TechnicalIndicators

# x-release-please-start-version
__version__ = "0.17.0"
# x-release-please-end

__all__ = [
    "DataFetcher",
    "TechnicalIndicators",
    "BaseStrategy",
    "SMACrossStrategy",
    "RSIStrategy",
    "MACDStrategy",
    "BacktestRunner",
    "ForecastingManager",
    "StockulaConfig",
    "load_config",
    "StockulaManager",
    "ResultsDisplay",
]
