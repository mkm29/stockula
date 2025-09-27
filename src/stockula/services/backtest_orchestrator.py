"""
Backtest Orchestrator following SRP.
Single Responsibility: Orchestrating backtesting workflows.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from rich.console import Console

from ..config import StockulaConfig
from ..container import Container
from ..utils import get_console

logger = logging.getLogger(__name__)


class BacktestOrchestrator:
    """Orchestrates backtesting workflows - Single Responsibility: Backtest Orchestration."""

    def __init__(
        self,
        config: StockulaConfig,
        container: Container,
        console: Optional[Console] = None,
    ):
        """Initialize backtest orchestrator.

        Args:
            config: Configuration object
            container: Dependency injection container
            console: Rich console for output (optional)
        """
        self.config = config
        self.container = container
        self.console = get_console(console)
        self.log_manager = container.logging_manager()

    def run_backtest(self, ticker: str) -> List[Dict[str, Any]]:
        """Run backtesting for a ticker using BacktestingManager.

        Args:
            ticker: Stock symbol

        Returns:
            List of backtest results
        """
        backtesting_manager = self.container.backtesting_manager()
        runner = self.container.backtest_runner()

        # Set the runner in the manager
        backtesting_manager.set_runner(runner)

        results = []

        # Check if we should use train/test split for backtesting
        use_train_test_split = self._should_use_train_test_split()

        for strategy_config in self.config.backtest.strategies:
            try:
                if use_train_test_split:
                    result_entry = self._run_train_test_backtest(ticker, strategy_config, backtesting_manager)
                else:
                    result_entry = self._run_standard_backtest(ticker, strategy_config, backtesting_manager)

                # Only append if result_entry is not None (i.e., backtest succeeded)
                if result_entry is not None:
                    results.append(result_entry)

            except Exception as e:
                self.console.print(f"[red]Error backtesting {strategy_config.name} on {ticker}: {e}[/red]")
                logger.error(f"Backtest error for {strategy_config.name} on {ticker}: {e}", exc_info=True)

        return results

    def _should_use_train_test_split(self) -> bool:
        """Check if train/test split should be used.

        Returns:
            True if train/test split should be used
        """
        return (
            self.config.forecast.train_start_date is not None
            and self.config.forecast.train_end_date is not None
            and self.config.forecast.test_start_date is not None
            and self.config.forecast.test_end_date is not None
        )

    def _run_train_test_backtest(self, ticker: str, strategy_config, backtesting_manager) -> Optional[Dict[str, Any]]:
        """Run backtest with train/test split.

        Args:
            ticker: Stock symbol
            strategy_config: Strategy configuration
            backtesting_manager: Backtesting manager instance

        Returns:
            Result entry or None if failed
        """
        # Calculate train ratio from dates
        train_ratio = 0.7  # Default fallback

        # Run with train/test split using BacktestingManager
        backtest_result = backtesting_manager.run_with_train_test_split(
            ticker=ticker,
            strategy_name=strategy_config.name,
            train_ratio=train_ratio,
            config=self.config,
            strategy_params=strategy_config.parameters,
            optimize_on_train=self.config.backtest.optimize,
            param_ranges=self.config.backtest.optimization_params
            if self.config.backtest.optimize and self.config.backtest.optimization_params
            else None,
        )

        # Check if there was an error
        if "error" in backtest_result:
            # Log the error and skip this strategy
            self.log_manager.error(
                f"Error backtesting {strategy_config.name} on {ticker}: {backtest_result.get('error')}"
            )
            return None

        # Create result entry with train/test results
        return self._create_train_test_result(ticker, strategy_config, backtest_result)

    def _run_standard_backtest(self, ticker: str, strategy_config, backtesting_manager) -> Optional[Dict[str, Any]]:
        """Run standard backtest without train/test split.

        Args:
            ticker: Stock symbol
            strategy_config: Strategy configuration
            backtesting_manager: Backtesting manager instance

        Returns:
            Result entry or None if failed
        """
        # Run traditional backtest without train/test split
        backtest_start, backtest_end = self._get_backtest_dates()

        # Use BacktestingManager for single strategy backtest
        backtest_result = backtesting_manager.run_single_strategy(
            ticker=ticker,
            strategy_name=strategy_config.name,
            config=self.config,
            strategy_params=strategy_config.parameters,
            start_date=backtest_start,
            end_date=backtest_end,
        )

        return self._create_standard_result(ticker, strategy_config, backtest_result)

    def _get_backtest_dates(self) -> Tuple[Optional[str], Optional[str]]:
        """Get backtest date range from configuration.

        Returns:
            Tuple of (start_date, end_date) as strings or None
        """
        backtest_start = None
        backtest_end = None

        # First check if backtest has specific dates
        if self.config.backtest.start_date and self.config.backtest.end_date:
            backtest_start = self._date_to_string(self.config.backtest.start_date)
            backtest_end = self._date_to_string(self.config.backtest.end_date)
        # Fall back to general data dates
        elif self.config.data.start_date and self.config.data.end_date:
            backtest_start = self._date_to_string(self.config.data.start_date)
            backtest_end = self._date_to_string(self.config.data.end_date)

        return backtest_start, backtest_end

    def _create_train_test_result(
        self, ticker: str, strategy_config, backtest_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create result entry for train/test split backtest.

        Args:
            ticker: Stock symbol
            strategy_config: Strategy configuration
            backtest_result: Raw backtest result

        Returns:
            Formatted result entry
        """
        result_entry = {
            "ticker": ticker,
            "strategy": strategy_config.name,
            "parameters": backtest_result.get("optimized_parameters", strategy_config.parameters),
            "train_period": backtest_result["train_period"],
            "test_period": backtest_result["test_period"],
            "train_results": backtest_result["train_results"],
            "test_results": backtest_result["test_results"],
            "performance_degradation": backtest_result.get("performance_degradation", {}),
        }

        # For backward compatibility, also include test results as top-level metrics
        result_entry.update(
            {
                "return_pct": backtest_result["test_results"]["return_pct"],
                "sharpe_ratio": backtest_result["test_results"]["sharpe_ratio"],
                "max_drawdown_pct": backtest_result["test_results"]["max_drawdown_pct"],
                "num_trades": backtest_result["test_results"]["num_trades"],
                "win_rate": backtest_result["test_results"]["win_rate"],
            }
        )

        return result_entry

    def _create_standard_result(
        self, ticker: str, strategy_config, backtest_result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Create result entry for standard backtest.

        Args:
            ticker: Stock symbol
            strategy_config: Strategy configuration
            backtest_result: Raw backtest result

        Returns:
            Formatted result entry or None if backtest failed
        """
        # Check if this is an error result
        if "error" in backtest_result:
            self.console.print(
                f"[red]Error backtesting {strategy_config.name} on {ticker}: {backtest_result['error']}[/red]"
            )
            return None

        # Check if required keys are present
        required_keys = ["Return [%]", "Sharpe Ratio", "Max. Drawdown [%]", "# Trades"]
        missing_keys = [key for key in required_keys if key not in backtest_result]
        if missing_keys:
            self.console.print(
                f"[red]Backtest result for {strategy_config.name} on {ticker} "
                f"missing required keys: {missing_keys}[/red]"
            )
            return None

        # Handle NaN values for win rate when there are no trades
        win_rate = backtest_result.get("Win Rate [%]", 0)
        if pd.isna(win_rate):
            win_rate = None if backtest_result["# Trades"] == 0 else 0

        result_entry = {
            "ticker": ticker,
            "strategy": strategy_config.name,
            "parameters": strategy_config.parameters,
            "return_pct": backtest_result["Return [%]"],
            "sharpe_ratio": backtest_result["Sharpe Ratio"],
            "max_drawdown_pct": backtest_result["Max. Drawdown [%]"],
            "num_trades": backtest_result["# Trades"],
            "win_rate": win_rate,
        }

        # Add portfolio information from the raw backtest result
        portfolio_fields = {
            "Initial Cash": "initial_cash",
            "Start Date": "start_date",
            "End Date": "end_date",
            "Trading Days": "trading_days",
            "Calendar Days": "calendar_days",
        }

        for bt_key, result_key in portfolio_fields.items():
            if bt_key in backtest_result:
                result_entry[result_key] = backtest_result[bt_key]

        return result_entry

    def _date_to_string(self, date_value) -> Optional[str]:
        """Convert date or string to string format.

        Args:
            date_value: Date value to convert

        Returns:
            String representation or None
        """
        if date_value is None:
            return None
        if isinstance(date_value, str):
            return date_value
        return str(date_value.strftime("%Y-%m-%d"))

    def run_multiple_backtests(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """Run backtests for multiple tickers.

        Args:
            tickers: List of stock symbols

        Returns:
            List of all backtest results
        """
        all_results = []

        for ticker in tickers:
            results = self.run_backtest(ticker)
            all_results.extend(results)

        return all_results
