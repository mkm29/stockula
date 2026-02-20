"""Backtesting module using backtesting.py library."""

from ..data import strategy_repository
from .manager import BacktestingManager
from .metrics import calculate_rolling_sharpe_ratio
from .runner import BacktestRunner
from .strategies import (
    BaseStrategy,
    DoubleEMACrossStrategy,
    KaufmanEfficiencyStrategy,
    MACDStrategy,
    RSIStrategy,
    SMACrossStrategy,
    TRIMACrossStrategy,
    TripleEMACrossStrategy,
)

# For backward compatibility
strategy_registry = strategy_repository  # Alias for backward compatibility

__all__ = [
    "BaseStrategy",
    "SMACrossStrategy",
    "RSIStrategy",
    "MACDStrategy",
    "DoubleEMACrossStrategy",
    "TripleEMACrossStrategy",
    "TRIMACrossStrategy",
    "KaufmanEfficiencyStrategy",
    "BacktestingManager",
    "BacktestRunner",
    "strategy_registry",  # Keep for backward compatibility
    "calculate_rolling_sharpe_ratio",
]
