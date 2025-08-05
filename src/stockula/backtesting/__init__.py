"""Backtesting module using backtesting.py library."""

from ..data import StrategyRepository, strategy_repository
from .manager import BacktestingManager
from .metrics import (
    calculate_dynamic_sharpe_ratio,
    calculate_rolling_sharpe_ratio,
    calculate_sortino_ratio_dynamic,
    enhance_backtest_metrics,
)
from .runner import BacktestRunner
from .strategies import (
    BaseStrategy,
    DoubleEMACrossStrategy,
    FRAMAStrategy,
    KAMAStrategy,
    KaufmanEfficiencyStrategy,
    MACDStrategy,
    RSIStrategy,
    SMACrossStrategy,
    TRIMACrossStrategy,
    TripleEMACrossStrategy,
    VAMAStrategy,
    VIDYAStrategy,
)

# strategy_repository is the singleton instance

# For backward compatibility
StrategyRegistry = StrategyRepository  # Alias for backward compatibility
strategy_registry = strategy_repository  # Alias for backward compatibility

__all__ = [
    "BaseStrategy",
    "SMACrossStrategy",
    "RSIStrategy",
    "MACDStrategy",
    "DoubleEMACrossStrategy",
    "TripleEMACrossStrategy",
    "TRIMACrossStrategy",
    "VIDYAStrategy",
    "KAMAStrategy",
    "FRAMAStrategy",
    "VAMAStrategy",
    "KaufmanEfficiencyStrategy",
    "BacktestingManager",
    "BacktestRunner",
    "StrategyRepository",
    "strategy_repository",
    "StrategyRegistry",  # Keep for backward compatibility
    "strategy_registry",  # Keep for backward compatibility
    "calculate_dynamic_sharpe_ratio",
    "calculate_rolling_sharpe_ratio",
    "calculate_sortino_ratio_dynamic",
    "enhance_backtest_metrics",
]
