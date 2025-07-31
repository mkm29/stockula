"""Backtesting runner and utilities."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd
from backtesting import Backtest

if TYPE_CHECKING:
    from ..config.models import BrokerConfig
    from ..interfaces import IDataFetcher


class BacktestRunner:
    """Runner for executing backtests."""

    def __init__(
        self,
        cash: float = 10000,
        commission: float = 0.002,
        margin: float = 1.0,
        data_fetcher: Optional["IDataFetcher"] = None,
        broker_config: Optional["BrokerConfig"] = None,
        risk_free_rate: float | pd.Series | None = None,
        trade_on_close: bool = True,
        exclusive_orders: bool = True,
    ):
        """Initialize backtest runner.

        Args:
            cash: Starting cash amount
            commission: Commission per trade (0.002 = 0.2%) - deprecated
            margin: Margin requirement for leveraged trading
            data_fetcher: Injected data fetcher instance
            broker_config: Broker-specific fee configuration
            risk_free_rate: Risk-free rate (float for static, pd.Series for dynamic)
            trade_on_close: Execute trades on close prices (more realistic)
            exclusive_orders: Whether orders are exclusive (prevents margin issues)
        """
        self.cash = cash
        self.margin = margin
        self.trade_on_close = trade_on_close
        self.exclusive_orders = exclusive_orders
        self.results: Any = None
        self.data_fetcher = data_fetcher
        self.broker_config = broker_config
        self.risk_free_rate = risk_free_rate
        self._equity_curve = None
        self._treasury_rates = None
        self.commission: float | Callable[[float, float], float]

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
                if isinstance(broker_config.commission_value, int | float):
                    commission = trade_value * broker_config.commission_value
            elif broker_config.commission_type == "fixed":
                if isinstance(broker_config.commission_value, int | float):
                    commission = broker_config.commission_value
            elif broker_config.commission_type == "per_share":
                per_share = broker_config.per_share_commission
                if per_share is None and isinstance(broker_config.commission_value, int | float):
                    per_share = broker_config.commission_value
                if per_share is not None:
                    commission = abs(quantity) * per_share
            elif broker_config.commission_type == "tiered":
                # For tiered commissions, we need to track total volume
                # For simplicity, using the lowest tier rate
                if isinstance(broker_config.commission_value, dict):
                    tiers = sorted([(int(k), v) for k, v in broker_config.commission_value.items()])
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

    def run(self, data: pd.DataFrame, strategy: type, **kwargs) -> dict[str, Any]:
        """Run backtest with given data and strategy.

        Args:
            data: OHLCV DataFrame
            strategy: Strategy class to test
            **kwargs: Additional parameters for the strategy

        Returns:
            Backtest results dictionary with enhanced metrics if dynamic rates provided
        """
        # Validate data sufficiency for strategies with period requirements
        if hasattr(strategy, "slow_period") and hasattr(strategy, "min_trading_days_buffer"):
            total_days = len(data)
            required_days = strategy.slow_period + getattr(strategy, "min_trading_days_buffer", 20)

            if total_days < required_days:
                print(
                    f"Warning: {strategy.__name__} requires at least {required_days} days of data "
                    f"({strategy.slow_period} for indicators + "
                    f"{getattr(strategy, 'min_trading_days_buffer', 20)} buffer), "
                    f"but only {total_days} days available."
                )

        # Store treasury rates if dynamic rates provided
        if isinstance(self.risk_free_rate, pd.Series):
            self._treasury_rates = self.risk_free_rate

        bt = Backtest(
            data,
            strategy,
            cash=self.cash,
            commission=self.commission,
            margin=self.margin,
            trade_on_close=self.trade_on_close,
            exclusive_orders=self.exclusive_orders,
        )

        # Run backtest with static risk-free rate
        # Suppress progress output by redirecting stderr
        import os
        import sys

        # Save current stderr
        old_stderr = sys.stderr
        try:
            # Redirect stderr to devnull to suppress progress bars
            sys.stderr = open(os.devnull, "w")
            self.results = bt.run(**kwargs)
        finally:
            # Restore stderr
            sys.stderr.close()
            sys.stderr = old_stderr

        # Store equity curve for dynamic metrics calculation
        # The results object has an _equity_curve attribute
        self._equity_curve = getattr(self.results, "_equity_curve", None)

        # Add portfolio information to results (only if results is a dict, not a mock)
        if hasattr(self.results, "__setitem__"):
            self.results["Initial Cash"] = self.cash

            # Safely extract dates from index
            if len(data) > 0:
                try:
                    # Check if index has datetime-like objects with strftime method
                    if hasattr(data.index[0], "strftime"):
                        self.results["Start Date"] = data.index[0].strftime("%Y-%m-%d")
                        self.results["End Date"] = data.index[-1].strftime("%Y-%m-%d")

                        # Calculate trading period in days
                        trading_days = len(data)
                        calendar_days = (data.index[-1] - data.index[0]).days
                        self.results["Trading Days"] = trading_days
                        self.results["Calendar Days"] = calendar_days
                    elif hasattr(data.index[0], "date"):
                        # Pandas datetime index
                        self.results["Start Date"] = data.index[0].date().strftime("%Y-%m-%d")
                        self.results["End Date"] = data.index[-1].date().strftime("%Y-%m-%d")

                        # Calculate trading period in days
                        trading_days = len(data)
                        calendar_days = (data.index[-1] - data.index[0]).days
                        self.results["Trading Days"] = trading_days
                        self.results["Calendar Days"] = calendar_days
                except (AttributeError, TypeError):
                    # If date extraction fails, just set trading days
                    self.results["Trading Days"] = len(data)

        # If dynamic rates provided, enhance results with dynamic metrics
        if isinstance(self.risk_free_rate, pd.Series) and self._equity_curve is not None:
            self._enhance_results_with_dynamic_metrics()

        return self.results

    def optimize(self, data: pd.DataFrame, strategy: type, **param_ranges) -> dict[str, Any]:
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
            trade_on_close=self.trade_on_close,
            exclusive_orders=self.exclusive_orders,
        )

        # Suppress progress output by redirecting stderr
        import os
        import sys

        # Save current stderr
        old_stderr = sys.stderr
        try:
            # Redirect stderr to devnull to suppress progress bars
            sys.stderr = open(os.devnull, "w")
            result = bt.optimize(**param_ranges)
        finally:
            # Restore stderr
            sys.stderr.close()
            sys.stderr = old_stderr

        return result

    def run_with_train_test_split(
        self,
        symbol: str,
        strategy: type,
        train_start_date: str | None = None,
        train_end_date: str | None = None,
        test_start_date: str | None = None,
        test_end_date: str | None = None,
        optimize_on_train: bool = True,
        treasury_duration: str = "3_month",
        use_dynamic_risk_free_rate: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """Run backtest with train/test split for out-of-sample validation.

        Args:
            symbol: Stock symbol to test
            strategy: Strategy class to test
            train_start_date: Start date for training data (YYYY-MM-DD)
            train_end_date: End date for training data (YYYY-MM-DD)
            test_start_date: Start date for testing data (YYYY-MM-DD)
            test_end_date: End date for testing data (YYYY-MM-DD)
            optimize_on_train: Whether to optimize parameters on training data
            treasury_duration: Treasury duration to use ('3_month', '13_week', etc.)
            use_dynamic_risk_free_rate: Whether to automatically fetch dynamic T-bill rates
            **kwargs: Additional parameters for the strategy or optimization

        Returns:
            Dictionary with both training and testing results
        """
        if not self.data_fetcher:
            raise ValueError("Data fetcher not configured. Ensure DI container is properly set up.")

        # Fetch data for the entire period
        all_start_date = train_start_date or test_start_date
        all_end_date = test_end_date or train_end_date

        all_data = self.data_fetcher.get_stock_data(symbol, all_start_date, all_end_date)

        # Split data into train and test sets
        train_data = None
        test_data = None

        if train_start_date and train_end_date:
            train_mask = (all_data.index >= pd.to_datetime(train_start_date)) & (
                all_data.index <= pd.to_datetime(train_end_date)
            )
            train_data = all_data[train_mask]

        if test_start_date and test_end_date:
            test_mask = (all_data.index >= pd.to_datetime(test_start_date)) & (
                all_data.index <= pd.to_datetime(test_end_date)
            )
            test_data = all_data[test_mask]

        # If no explicit split provided, use the entire dataset
        if train_data is None and test_data is None:
            train_data = all_data
            test_data = all_data
        elif train_data is None:
            train_data = test_data  # Use test data for both if no train data
        elif test_data is None:
            test_data = train_data  # Use train data for both if no test data

        # Fetch Treasury rates if requested
        if use_dynamic_risk_free_rate and not isinstance(self.risk_free_rate, pd.Series):
            if all_start_date and all_end_date:
                treasury_rates = self.data_fetcher.get_treasury_rates(all_start_date, all_end_date, treasury_duration)
                if not treasury_rates.empty:
                    self.risk_free_rate = treasury_rates

        results: dict[str, Any] = {
            "symbol": symbol,
            "strategy": strategy.__name__,
            "train_period": {
                "start": train_data.index[0].strftime("%Y-%m-%d")
                if train_data is not None and len(train_data) > 0
                else None,
                "end": train_data.index[-1].strftime("%Y-%m-%d")
                if train_data is not None and len(train_data) > 0
                else None,
                "days": len(train_data) if train_data is not None else 0,
            },
            "test_period": {
                "start": test_data.index[0].strftime("%Y-%m-%d")
                if test_data is not None and len(test_data) > 0
                else None,
                "end": test_data.index[-1].strftime("%Y-%m-%d")
                if test_data is not None and len(test_data) > 0
                else None,
                "days": len(test_data) if test_data is not None else 0,
            },
        }

        # Run on training data
        if optimize_on_train and "param_ranges" in kwargs:
            # Optimize parameters on training data
            param_ranges = kwargs.pop("param_ranges")
            print(f"Optimizing {strategy.__name__} parameters on training data...")
            optimized_result = self.optimize(train_data, strategy, **param_ranges)

            # Extract optimized parameters
            if hasattr(optimized_result, "items"):
                optimized_params = {
                    k: v
                    for k, v in optimized_result.items()
                    if k
                    not in [
                        "Start",
                        "End",
                        "Duration",
                        "Exposure Time [%]",
                        "Equity Final [$]",
                        "Equity Peak [$]",
                        "Return [%]",
                        "Buy & Hold Return [%]",
                        "Max. Drawdown [%]",
                        "Avg. Drawdown [%]",
                        "Max. Drawdown Duration",
                        "Avg. Drawdown Duration",
                        "# Trades",
                        "Win Rate [%]",
                        "Best Trade [%]",
                        "Worst Trade [%]",
                        "Avg. Trade [%]",
                        "Max. Trade Duration",
                        "Avg. Trade Duration",
                        "Profit Factor",
                        "Expectancy [%]",
                        "SQN",
                        "Sharpe Ratio",
                        "Sortino Ratio",
                        "Calmar Ratio",
                        "_strategy",
                        "_equity_curve",
                        "_trades",
                    ]
                }
            else:
                optimized_params = {}

            results["optimized_parameters"] = optimized_params

            # Run backtest on training data with optimized parameters
            for param_name, param_value in optimized_params.items():
                setattr(strategy, param_name, param_value)

            train_result = self.run(train_data, strategy, **kwargs)
            results["train_results"] = self._extract_key_metrics(train_result)
        else:
            # Run backtest on training data without optimization
            train_result = self.run(train_data, strategy, **kwargs)
            results["train_results"] = self._extract_key_metrics(train_result)
            results["optimized_parameters"] = kwargs

        # Run backtest on test data with same parameters
        test_result = self.run(test_data, strategy, **kwargs)
        results["test_results"] = self._extract_key_metrics(test_result)

        # Calculate performance degradation
        if results["train_results"]["return_pct"] != 0:
            results["performance_degradation"] = {
                "return_pct": (
                    (results["test_results"]["return_pct"] - results["train_results"]["return_pct"])
                    / abs(results["train_results"]["return_pct"])
                    * 100
                ),
                "sharpe_ratio": (
                    (results["test_results"]["sharpe_ratio"] - results["train_results"]["sharpe_ratio"])
                    / abs(results["train_results"]["sharpe_ratio"])
                    * 100
                    if results["train_results"]["sharpe_ratio"] != 0
                    else 0
                ),
            }
        else:
            results["performance_degradation"] = {"return_pct": 0, "sharpe_ratio": 0}

        return results

    def _extract_key_metrics(self, backtest_result: dict[str, Any]) -> dict[str, Any]:
        """Extract key metrics from backtest results.

        Args:
            backtest_result: Raw backtest result

        Returns:
            Dictionary with key metrics
        """
        return {
            "return_pct": backtest_result.get("Return [%]", 0),
            "sharpe_ratio": backtest_result.get("Sharpe Ratio", 0),
            "max_drawdown_pct": backtest_result.get("Max. Drawdown [%]", 0),
            "num_trades": backtest_result.get("# Trades", 0),
            "win_rate": backtest_result.get("Win Rate [%]", 0),
            "equity_final": backtest_result.get("Equity Final [$]", 0),
            "buy_hold_return_pct": backtest_result.get("Buy & Hold Return [%]", 0),
        }

    def run_from_symbol(
        self,
        symbol: str,
        strategy: type,
        start_date: str | None = None,
        end_date: str | None = None,
        treasury_duration: str = "3_month",
        use_dynamic_risk_free_rate: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """Run backtest by fetching data for a symbol.

        Args:
            symbol: Stock symbol to test
            strategy: Strategy class to test
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (YYYY-MM-DD)
            treasury_duration: Treasury duration to use ('3_month', '13_week', etc.)
            use_dynamic_risk_free_rate: Whether to automatically fetch dynamic T-bill rates
            **kwargs: Additional parameters for the strategy

        Returns:
            Backtest results with dynamic risk-free rate metrics by default
        """
        if not self.data_fetcher:
            raise ValueError("Data fetcher not configured. Ensure DI container is properly set up.")

        # Fetch stock data
        stock_data = self.data_fetcher.get_stock_data(symbol, start_date, end_date)

        # Automatically fetch dynamic Treasury rates if enabled and not already provided
        if use_dynamic_risk_free_rate and not isinstance(self.risk_free_rate, pd.Series):
            # Determine date range from stock data if not provided
            if start_date is None and hasattr(stock_data.index[0], "strftime"):
                start_date = stock_data.index[0].strftime("%Y-%m-%d")
            if end_date is None and hasattr(stock_data.index[-1], "strftime"):
                end_date = stock_data.index[-1].strftime("%Y-%m-%d")

            # Only fetch Treasury rates if we have valid dates
            if start_date and end_date:
                # Fetch Treasury rates for the same period
                treasury_rates = self.data_fetcher.get_treasury_rates(start_date, end_date, treasury_duration)

                # Set dynamic risk-free rates
                if not treasury_rates.empty:
                    self.risk_free_rate = treasury_rates

        return self.run(stock_data, strategy, **kwargs)

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

    def _enhance_results_with_dynamic_metrics(self):
        """Enhance backtest results with dynamic risk-free rate metrics."""
        from .metrics import enhance_backtest_metrics

        if self._equity_curve is None or self._treasury_rates is None:
            return

        # Get the equity curve as a Series with appropriate index
        if isinstance(self._equity_curve, np.ndarray):
            # Handle multi-dimensional arrays - extract the equity values
            if self._equity_curve.ndim > 1:
                # For multi-dimensional arrays, take the first column (equity values)
                equity_values = (
                    self._equity_curve[:, 0] if self._equity_curve.shape[1] > 0 else self._equity_curve.flatten()
                )
            else:
                equity_values = self._equity_curve

            # Create date index based on treasury rates index
            equity_series = pd.Series(
                equity_values,
                index=self._treasury_rates.index[: len(equity_values)],
            )
        else:
            # Convert whatever type to pandas Series
            if isinstance(self._equity_curve, pd.DataFrame):
                # It's a DataFrame - extract the first column (equity values)
                equity_series = pd.Series(self._equity_curve.iloc[:, 0], index=self._equity_curve.index)
            elif isinstance(self._equity_curve, pd.Series):
                # It's already a Series
                equity_series = self._equity_curve
            elif hasattr(self._equity_curve, "values") and hasattr(self._equity_curve, "index"):
                # It's some other pandas object - try to convert
                if hasattr(self._equity_curve, "iloc"):
                    # Has iloc, probably DataFrame-like
                    equity_series = pd.Series(self._equity_curve.iloc[:, 0], index=self._equity_curve.index)
                else:
                    # Try direct conversion
                    equity_series = pd.Series(self._equity_curve.values, index=self._equity_curve.index)
            elif hasattr(self._equity_curve, "__len__"):
                # It's array-like, convert to Series with treasury rates index
                try:
                    equity_series = pd.Series(
                        list(self._equity_curve),
                        index=self._treasury_rates.index[: len(self._equity_curve)],
                    )
                except Exception as e:
                    print(
                        f"Warning: Could not convert equity curve to pandas Series. "
                        f"Type: {type(self._equity_curve)}, Error: {e}"
                    )
                    return
            else:
                # Unknown type, skip dynamic metrics
                print(f"Warning: Unknown equity curve type: {type(self._equity_curve)}")
                return

        # Ensure it's definitely a pandas Series
        if not isinstance(equity_series, pd.Series):
            print(f"Warning: Could not convert equity curve to pandas Series. Type: {type(equity_series)}")
            return

        # Enhance results with dynamic metrics
        try:
            enhanced_stats = enhance_backtest_metrics(self.results, equity_series, self._treasury_rates)
        except Exception as e:
            print(f"Warning: Could not calculate dynamic metrics: {e}")
            return

        # Update results with enhanced metrics
        for key, value in enhanced_stats.items():
            if key not in self.results:
                self.results[key] = value

    def run_with_dynamic_risk_free_rate(
        self,
        symbol: str,
        strategy: type,
        start_date: str | None = None,
        end_date: str | None = None,
        treasury_duration: str = "3_month",
        **kwargs,
    ) -> dict[str, Any]:
        """Run backtest with dynamic Treasury rates for risk-free rate calculation.

        Args:
            symbol: Stock symbol to test
            strategy: Strategy class to test
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (YYYY-MM-DD)
            treasury_duration: Treasury duration to use ('3_month', '13_week', etc.)
            **kwargs: Additional parameters for the strategy

        Returns:
            Backtest results with enhanced dynamic metrics
        """
        if not self.data_fetcher:
            raise ValueError("Data fetcher not configured. Ensure DI container is properly set up.")

        # Fetch stock data
        stock_data = self.data_fetcher.get_stock_data(symbol, start_date, end_date)

        # Fetch Treasury rates for the same period
        if start_date is None:
            start_date = stock_data.index[0].strftime("%Y-%m-%d")
        if end_date is None:
            end_date = stock_data.index[-1].strftime("%Y-%m-%d")

        treasury_rates = self.data_fetcher.get_treasury_rates(start_date, end_date, treasury_duration)

        # Set dynamic risk-free rates
        self.risk_free_rate = treasury_rates

        # Run backtest
        return self.run(stock_data, strategy, **kwargs)
