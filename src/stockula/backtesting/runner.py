"""Backtesting runner and utilities."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np
import pandas as pd
from backtesting import Backtest

if TYPE_CHECKING:
    from ..config.models import BrokerConfig
    from ..interfaces import IDataFetcher


class BacktestRunner:
    """Runner for executing backtests."""

    DATA_FETCHER_NOT_CONFIGURED_MSG = "Data fetcher not configured. Ensure DI container is properly set up."

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
        """Create commission function based on broker configuration."""

        def _get_commission(trade_value: float, quantity: float) -> float:
            commission_type = broker_config.commission_type
            if commission_type == "percentage":
                return self._percentage_commission(broker_config, trade_value)
            elif commission_type == "fixed":
                return self._fixed_commission(broker_config)
            elif commission_type == "per_share":
                return self._per_share_commission(broker_config, quantity)
            elif commission_type == "tiered":
                return self._tiered_commission(broker_config, quantity)
            return 0.0

        def _apply_min_max_commission(commission: float) -> float:
            min_comm = broker_config.min_commission
            max_comm = broker_config.max_commission
            if min_comm is not None:
                commission = max(commission, min_comm)
            if max_comm is not None:
                commission = min(commission, max_comm)
            return commission

        def _regulatory_fee(trade_value: float) -> float:
            return trade_value * broker_config.regulatory_fees

        def _exchange_fee(quantity: float) -> float:
            if broker_config.name == "robinhood" and broker_config.exchange_fees > 0:
                if abs(quantity) > 50:
                    fee = abs(quantity) * broker_config.exchange_fees
                    return min(fee, 8.30)
                return 0.0
            return broker_config.exchange_fees

        def commission_func(quantity: float, price: float) -> float:
            trade_value = abs(quantity * price)
            commission = _get_commission(trade_value, quantity)
            commission = _apply_min_max_commission(commission)
            regulatory_fee = _regulatory_fee(trade_value)
            exchange_fee = _exchange_fee(quantity)
            return commission + regulatory_fee + exchange_fee

        return commission_func

    @staticmethod
    def _percentage_commission(broker_config, trade_value: float) -> float:
        if isinstance(broker_config.commission_value, (int, float)):
            return trade_value * broker_config.commission_value
        return 0.0

    @staticmethod
    def _fixed_commission(broker_config) -> float:
        if isinstance(broker_config.commission_value, (int, float)):
            return broker_config.commission_value
        return 0.0

    @staticmethod
    def _per_share_commission(broker_config, quantity: float) -> float:
        per_share = broker_config.per_share_commission
        if per_share is None and isinstance(broker_config.commission_value, (int, float)):
            per_share = broker_config.commission_value
        if per_share is not None:
            return abs(quantity) * per_share
        return 0.0

    @staticmethod
    def _tiered_commission(broker_config, quantity: float) -> float:
        if isinstance(broker_config.commission_value, dict):
            tiers = sorted([(int(k), v) for k, v in broker_config.commission_value.items()])
            if tiers:
                return abs(quantity) * tiers[0][1]
        return 0.0

    def run(self, data: pd.DataFrame, strategy: type, **kwargs) -> dict[str, Any]:
        """Run backtest with given data and strategy.

        Args:
            data: OHLCV DataFrame
            strategy: Strategy class to test
            **kwargs: Additional parameters for the strategy

        Returns:
            Backtest results dictionary with enhanced metrics if dynamic rates provided
        """
        self._check_strategy_data_sufficiency(data, strategy)

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

        # Suppress progress output by redirecting stderr
        import os
        import sys

        old_stderr = sys.stderr
        try:
            sys.stderr = open(os.devnull, "w")
            self.results = bt.run(**kwargs)
        finally:
            sys.stderr.close()
            sys.stderr = old_stderr

        self._equity_curve = getattr(self.results, "_equity_curve", None)

        if hasattr(self.results, "__setitem__"):
            self.results["Initial Cash"] = self.cash
            self._extract_and_set_dates(data)

        if isinstance(self.risk_free_rate, pd.Series) and self._equity_curve is not None:
            self._enhance_results_with_dynamic_metrics()

        return cast(dict[str, Any], self.results)  # type: ignore[arg-type]

    def _check_strategy_data_sufficiency(self, data: pd.DataFrame, strategy: type):
        """Check if the data is sufficient for the strategy's requirements."""
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

    def _extract_and_set_dates(self, data: pd.DataFrame):
        """Safely extract and set date-related results."""
        if len(data) > 0:
            try:
                idx0 = data.index[0]
                idxN = data.index[-1]
                if hasattr(idx0, "strftime"):
                    self.results["Start Date"] = idx0.strftime("%Y-%m-%d")
                    self.results["End Date"] = idxN.strftime("%Y-%m-%d")
                    self.results["Trading Days"] = len(data)
                    self.results["Calendar Days"] = (idxN - idx0).days
                elif hasattr(idx0, "date"):
                    self.results["Start Date"] = idx0.date().strftime("%Y-%m-%d")
                    self.results["End Date"] = idxN.date().strftime("%Y-%m-%d")
                    self.results["Trading Days"] = len(data)
                    self.results["Calendar Days"] = (idxN - idx0).days
                else:
                    self.results["Trading Days"] = len(data)
            except (AttributeError, TypeError):
                self.results["Trading Days"] = len(data)

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

        return cast(dict[str, Any], result)

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
        if not self.data_fetcher:
            raise ValueError(self.DATA_FETCHER_NOT_CONFIGURED_MSG)

        all_start_date = train_start_date or test_start_date
        all_end_date = test_end_date or train_end_date
        all_data = self.data_fetcher.get_stock_data(symbol, all_start_date, all_end_date)

        train_data, test_data = self._split_train_test_data(
            all_data, train_start_date, train_end_date, test_start_date, test_end_date
        )

        if use_dynamic_risk_free_rate and not isinstance(self.risk_free_rate, pd.Series):
            self._fetch_and_set_treasury_rates(all_start_date, all_end_date, treasury_duration)

        results = self._init_train_test_results(symbol, strategy, train_data, test_data)

        if optimize_on_train and "param_ranges" in kwargs:
            param_ranges = kwargs.pop("param_ranges")
            optimized_params = self._optimize_on_train(train_data, strategy, param_ranges)
            results["optimized_parameters"] = optimized_params
            self._set_strategy_params(strategy, optimized_params)
            train_result = self.run(train_data, strategy, **kwargs)
        else:
            train_result = self.run(train_data, strategy, **kwargs)
            results["optimized_parameters"] = kwargs

        results["train_results"] = self._extract_key_metrics(train_result)
        test_result = self.run(test_data, strategy, **kwargs)
        results["test_results"] = self._extract_key_metrics(test_result)
        results["performance_degradation"] = self._calculate_performance_degradation(
            results["train_results"], results["test_results"]
        )

        return results

    def _split_train_test_data(
        self,
        all_data: pd.DataFrame,
        train_start_date: str | None,
        train_end_date: str | None,
        test_start_date: str | None,
        test_end_date: str | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
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

        if train_data is None and test_data is None:
            train_data = all_data
            test_data = all_data
        elif train_data is None:
            train_data = test_data
        elif test_data is None:
            test_data = train_data

        return train_data, test_data

    def _fetch_and_set_treasury_rates(self, start_date, end_date, treasury_duration):
        if start_date and end_date:
            treasury_rates = self.data_fetcher.get_treasury_rates(start_date, end_date, treasury_duration)
            if not treasury_rates.empty:
                self.risk_free_rate = treasury_rates

    def _init_train_test_results(
        self, symbol, strategy, train_data, test_data
    ) -> dict[str, Any]:
        return {
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

    def _optimize_on_train(self, train_data, strategy, param_ranges):
        print(f"Optimizing {strategy.__name__} parameters on training data...")
        optimized_result = self.optimize(train_data, strategy, **param_ranges)
        if hasattr(optimized_result, "items"):
            return {
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
        return {}

    def _set_strategy_params(self, strategy, optimized_params):
        for param_name, param_value in optimized_params.items():
            setattr(strategy, param_name, param_value)

    def _calculate_performance_degradation(self, train_results, test_results):
        if train_results["return_pct"] != 0:
            return {
                "return_pct": (
                    (test_results["return_pct"] - train_results["return_pct"])
                    / abs(train_results["return_pct"])
                    * 100
                ),
                "sharpe_ratio": (
                    (test_results["sharpe_ratio"] - train_results["sharpe_ratio"])
                    / abs(train_results["sharpe_ratio"])
                    * 100
                    if train_results["sharpe_ratio"] != 0
                    else 0
                ),
            }
        else:
            return {"return_pct": 0, "sharpe_ratio": 0}

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

        equity_series = self._convert_equity_curve_to_series(self._equity_curve, self._treasury_rates)

        if equity_series is None or not isinstance(equity_series, pd.Series):
            print(f"Warning: Could not convert equity curve to pandas Series. Type: {type(equity_series)}")
            return

        try:
            enhanced_stats = enhance_backtest_metrics(self.results, equity_series, self._treasury_rates)
        except Exception as e:
            print(f"Warning: Could not calculate dynamic metrics: {e}")
            return

        for key, value in enhanced_stats.items():
            if key not in self.results:
                self.results[key] = value

    def _convert_equity_curve_to_series(self, equity_curve, treasury_rates):
        """Convert various equity curve types to pandas Series."""
        if isinstance(equity_curve, np.ndarray):
            return self._convert_ndarray_equity_curve(equity_curve, treasury_rates)
        if isinstance(equity_curve, pd.DataFrame):
            return self._convert_dataframe_equity_curve(equity_curve)
        if isinstance(equity_curve, pd.Series):
            return equity_curve
        if hasattr(equity_curve, "values") and hasattr(equity_curve, "index"):
            return self._convert_indexed_equity_curve(equity_curve)
        if hasattr(equity_curve, "__len__"):
            return self._convert_iterable_equity_curve(equity_curve, treasury_rates)
        print(f"Warning: Unknown equity curve type: {type(equity_curve)}")
        return None

    def _convert_ndarray_equity_curve(self, equity_curve, treasury_rates):
        if equity_curve.ndim > 1:
            equity_values = equity_curve[:, 0] if equity_curve.shape[1] > 0 else equity_curve.flatten()
        else:
            equity_values = equity_curve
        return pd.Series(equity_values, index=treasury_rates.index[: len(equity_values)])

    def _convert_dataframe_equity_curve(self, equity_curve):
        return pd.Series(equity_curve.iloc[:, 0], index=equity_curve.index)

    def _convert_indexed_equity_curve(self, equity_curve):
        if hasattr(equity_curve, "iloc"):
            return pd.Series(equity_curve.iloc[:, 0], index=equity_curve.index)
        return pd.Series(equity_curve.values, index=equity_curve.index)

    def _convert_iterable_equity_curve(self, equity_curve, treasury_rates):
        try:
            return pd.Series(list(equity_curve), index=treasury_rates.index[: len(equity_curve)])
        except Exception as e:
            print(
                f"Warning: Could not convert equity curve to pandas Series. "
                f"Type: {type(equity_curve)}, Error: {e}"
            )
            return None

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
