"""Backtesting runner and utilities."""

from backtesting import Backtest
import pandas as pd
from typing import Type, Dict, Any, Optional, TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ..interfaces import IDataFetcher
    from ..config.models import BrokerConfig


class BacktestRunner:
    """Runner for executing backtests."""

    def __init__(
        self,
        cash: float = 10000,
        commission: float = 0.002,
        margin: float = 1.0,
        data_fetcher: Optional["IDataFetcher"] = None,
        broker_config: Optional["BrokerConfig"] = None,
    ):
        """Initialize backtest runner.

        Args:
            cash: Starting cash amount
            commission: Commission per trade (0.002 = 0.2%) - deprecated
            margin: Margin requirement for leveraged trading
            data_fetcher: Injected data fetcher instance
            broker_config: Broker-specific fee configuration
        """
        self.cash = cash
        self.margin = margin
        self.results = None
        self.data_fetcher = data_fetcher
        self.broker_config = broker_config

        # If broker_config is provided, use it to create commission function
        if broker_config:
            self.commission = self._create_commission_func(broker_config)
        else:
            # Use legacy simple commission
            self.commission = commission

    def _create_commission_func(self, broker_config: "BrokerConfig") -> Callable:
        """Create commission function based on broker configuration.

        Args:
            broker_config: Broker configuration with fee structure

        Returns:
            Commission function for backtesting.py
        """

        def commission_func(quantity: float, price: float) -> float:
            """Calculate commission for a trade.

            Args:
                quantity: Number of shares
                price: Price per share

            Returns:
                Total commission for the trade
            """
            trade_value = abs(quantity * price)
            commission = 0.0

            # Calculate base commission based on type
            if broker_config.commission_type == "percentage":
                commission = trade_value * broker_config.commission_value
            elif broker_config.commission_type == "fixed":
                commission = broker_config.commission_value
            elif broker_config.commission_type == "per_share":
                per_share = (
                    broker_config.per_share_commission or broker_config.commission_value
                )
                commission = abs(quantity) * per_share
            elif broker_config.commission_type == "tiered":
                # For tiered commissions, we need to track total volume
                # For simplicity, using the lowest tier rate
                if isinstance(broker_config.commission_value, dict):
                    tiers = sorted(
                        [(int(k), v) for k, v in broker_config.commission_value.items()]
                    )
                    # Use first tier rate (could be enhanced to track monthly volume)
                    commission = abs(quantity) * tiers[0][1]

            # Apply min/max constraints
            if broker_config.min_commission is not None:
                commission = max(commission, broker_config.min_commission)
            if broker_config.max_commission is not None:
                commission = min(commission, broker_config.max_commission)

            # Add regulatory and exchange fees
            regulatory_fee = trade_value * broker_config.regulatory_fees

            # Handle exchange fees (e.g., TAF for Robinhood)
            if broker_config.name == "robinhood" and broker_config.exchange_fees > 0:
                # Robinhood TAF: only on sells, waived for 50 shares or less
                # For backtesting, we'll apply it to all trades but waive for small trades
                if abs(quantity) > 50:
                    exchange_fee = abs(quantity) * broker_config.exchange_fees
                    # TAF maximum is $8.30 per trade
                    exchange_fee = min(exchange_fee, 8.30)
                else:
                    exchange_fee = 0.0
            else:
                # For other brokers, simple exchange fee calculation
                exchange_fee = broker_config.exchange_fees

            total_fee = commission + regulatory_fee + exchange_fee

            return total_fee

        return commission_func

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
