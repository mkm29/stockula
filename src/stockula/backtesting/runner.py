"""Backtesting runner and utilities."""

from backtesting import Backtest
import pandas as pd
from typing import Type, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..interfaces import IDataFetcher


class BacktestRunner:
    """Runner for executing backtests."""

    def __init__(
        self,
        cash: float = 10000,
        commission: float = 0.002,
        margin: float = 1.0,
        data_fetcher: Optional["IDataFetcher"] = None,
    ):
        """Initialize backtest runner.

        Args:
            cash: Starting cash amount
            commission: Commission per trade (0.002 = 0.2%)
            margin: Margin requirement for leveraged trading
            data_fetcher: Injected data fetcher instance
        """
        self.cash = cash
        self.commission = commission
        self.margin = margin
        self.results = None
        self.data_fetcher = data_fetcher

    def run(self, data: pd.DataFrame, strategy: Type, **kwargs) -> Dict[str, Any]:
        """Run backtest with given data and strategy.

        Args:
            data: OHLCV DataFrame
            strategy: Strategy class to test
            **kwargs: Additional parameters for the strategy

        Returns:
            Backtest results dictionary
        """
        # Validate data sufficiency for strategies with period requirements
        if hasattr(strategy, "slow_period") and hasattr(
            strategy, "min_trading_days_buffer"
        ):
            total_days = len(data)
            required_days = strategy.slow_period + getattr(
                strategy, "min_trading_days_buffer", 20
            )

            if total_days < required_days:
                print(
                    f"Warning: {strategy.__name__} requires at least {required_days} days of data "
                    f"({strategy.slow_period} for indicators + {getattr(strategy, 'min_trading_days_buffer', 20)} buffer), "
                    f"but only {total_days} days available."
                )

        bt = Backtest(
            data,
            strategy,
            cash=self.cash,
            commission=self.commission,
            margin=self.margin,
        )

        self.results = bt.run(**kwargs)
        return self.results

    def optimize(
        self, data: pd.DataFrame, strategy: Type, **param_ranges
    ) -> Dict[str, Any]:
        """Optimize strategy parameters.

        Args:
            data: OHLCV DataFrame
            strategy: Strategy class to optimize
            **param_ranges: Parameter ranges for optimization

        Returns:
            Optimal parameters and results
        """
        bt = Backtest(
            data,
            strategy,
            cash=self.cash,
            commission=self.commission,
            margin=self.margin,
        )

        return bt.optimize(**param_ranges)

    def run_from_symbol(
        self,
        symbol: str,
        strategy: Type,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run backtest by fetching data for a symbol.

        Args:
            symbol: Stock symbol to test
            strategy: Strategy class to test
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (YYYY-MM-DD)
            **kwargs: Additional parameters for the strategy

        Returns:
            Backtest results
        """
        if not self.data_fetcher:
            raise ValueError(
                "Data fetcher not configured. Ensure DI container is properly set up."
            )

        data = self.data_fetcher.get_stock_data(symbol, start_date, end_date)
        return self.run(data, strategy, **kwargs)

    def get_stats(self) -> pd.Series:
        """Get detailed statistics from last backtest.

        Returns:
            Series with backtest statistics
        """
        if self.results is None:
            raise ValueError("No backtest results available. Run a backtest first.")
        return self.results

    def plot(self, **kwargs):
        """Plot backtest results.

        Args:
            **kwargs: Additional plotting parameters
        """
        if self.results is None:
            raise ValueError("No backtest results available. Run a backtest first.")
        self.results.plot(**kwargs)
