"""Backtesting module using backtesting.py library."""

from .strategies import (
    BaseStrategy,
    SMACrossStrategy,
    RSIStrategy,
    MACDStrategy,
    DoubleEMACrossStrategy,
    TripleEMACrossStrategy,
    TRIMACrossStrategy,
    VIDYAStrategy,
    KAMAStrategy,
    FRAMAStrategy,
    VAMAStrategy,
    KaufmanEfficiencyStrategy,
)
from .runner import BacktestRunner
from .metrics import (
    calculate_dynamic_sharpe_ratio,
    calculate_rolling_sharpe_ratio,
    calculate_sortino_ratio_dynamic,
    enhance_backtest_metrics,
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
