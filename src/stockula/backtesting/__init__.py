"""Backtesting module using backtesting.py library."""

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
    "BacktestRunner",
    "calculate_dynamic_sharpe_ratio",
    "calculate_rolling_sharpe_ratio",
    "calculate_sortino_ratio_dynamic",
    "enhance_backtest_metrics",
]
