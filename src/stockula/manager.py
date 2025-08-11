"""Stockula Manager - Main business logic orchestrator."""

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from .backtesting import BaseStrategy, strategy_registry
from .config import StockulaConfig
from .config.models import BacktestResult, PortfolioBacktestResults, StrategyBacktestSummary
from .container import Container
from .display import ResultsDisplay
from .domain import Category, Portfolio
from .technical_analysis import TechnicalIndicators


class StockulaManager:
    """Manages the main business logic for Stockula."""

    def __init__(
        self,
        config: StockulaConfig,
        container: Container,
        console: Console | None = None,
    ):
        """Initialize the manager.

        Args:
            config: Configuration object
            container: Dependency injection container
            console: Rich console for output (optional)
        """
        self.config = config
        self.container = container
        self.console = console or Console()
        self.log_manager = container.logging_manager()

        # Strategy registry provides centralized strategy management
        self.strategy_registry = strategy_registry

    def get_strategy_class(self, strategy_name: str) -> type[BaseStrategy] | None:
        """Get strategy class by name.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Strategy class or None if not found
        """
        return self.strategy_registry.get_strategy_class(strategy_name)

    def date_to_string(self, date_value: str | date | None) -> str | None:
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
        return date_value.strftime("%Y-%m-%d")

    def run_optimize_allocation(self, save_path: str | None = None) -> int:
        """Run backtest optimization for allocation.

        Args:
            save_path: Path to save optimized config (optional)

        Returns:
            Exit code (0 for success, 1 for error)
        """
        self.console.print("\n[bold cyan]Running Backtest Optimization for Allocation[/bold cyan]")

        # Check if allocation method is backtest_optimized
        if self.config.portfolio.allocation_method != "backtest_optimized":
            self.console.print(
                "[yellow]Warning: allocation_method is not set to 'backtest_optimized' in config.[/yellow]"
            )
            self.console.print("[yellow]Setting it to 'backtest_optimized' for this run.[/yellow]")
            self.config.portfolio.allocation_method = "backtest_optimized"

        # Check if we have the necessary date configuration
        if not self.config.backtest_optimization:
            self.console.print(
                "[red]Error: backtest_optimization configuration is required for optimize-allocation mode.[/red]"
            )
            self.console.print("[red]Please add backtest_optimization section to your config file.[/red]")
            return 1

        try:
            # Get the allocator manager
            allocator_manager = self.container.allocator_manager()

            # Calculate optimized quantities using the manager
            self.console.print("\n[blue]Calculating optimized quantities...[/blue]")
            optimized_quantities = allocator_manager.calculate_backtest_optimized_quantities(
                config=self.config,
                tickers=self.config.portfolio.tickers,
            )

            # Display results
            display = ResultsDisplay(self.console)
            display.show_allocation_optimization(
                optimized_quantities=optimized_quantities,
                config=self.config,
                data_fetcher=self.container.data_fetcher(),
            )

            # Update the config with optimized quantities (even if not saving to file)
            for ticker_config in self.config.portfolio.tickers:
                if ticker_config.symbol in optimized_quantities:
                    # Convert numpy types to native Python types
                    quantity = optimized_quantities[ticker_config.symbol]
                    if hasattr(quantity, "item"):
                        # Convert numpy scalar to Python type
                        ticker_config.quantity = float(quantity.item())
                    else:
                        # Keep as integer if it's already an integer (from backtest_optimized)
                        if isinstance(quantity, int):
                            ticker_config.quantity = float(quantity)
                        else:
                            ticker_config.quantity = float(quantity)
                    # Clear allocation_pct and allocation_amount since we now have quantities
                    ticker_config.allocation_pct = None
                    ticker_config.allocation_amount = None

            # Save optimized config if requested
            if save_path:
                self._save_optimized_config(save_path, optimized_quantities)

            return 0

        except Exception as e:
            self.console.print(f"[red]Error during optimization: {e}[/red]")
            self.log_manager.error(f"Optimization error: {e}", exc_info=True)
            return 1

    def _normalize_strategy_name(self, strategy_name: str) -> str:
        """Normalize strategy name to snake_case format.

        Args:
            strategy_name: Strategy name in any format

        Returns:
            Normalized strategy name in snake_case
        """
        return self.strategy_registry.normalize_strategy_name(strategy_name)

    def _save_optimized_config(self, save_path: str, optimized_quantities: dict[str, float]) -> None:
        """Save optimized configuration to file.

        Args:
            save_path: Path to save the configuration
            optimized_quantities: Dictionary of symbol to quantity
        """
        # Update the config with optimized quantities
        for ticker_config in self.config.portfolio.tickers:
            if ticker_config.symbol in optimized_quantities:
                # Convert numpy types to native Python types
                quantity = optimized_quantities[ticker_config.symbol]
                if hasattr(quantity, "item"):
                    # Convert numpy scalar to Python type
                    ticker_config.quantity = float(quantity.item())
                else:
                    # Keep as integer if it's already an integer (from backtest_optimized)
                    if isinstance(quantity, int):
                        ticker_config.quantity = float(quantity)
                    else:
                        ticker_config.quantity = float(quantity)
                # Clear allocation_pct and allocation_amount since we now have quantities
                ticker_config.allocation_pct = None
                ticker_config.allocation_amount = None

        # Change allocation method to custom since we now have fixed quantities
        self.config.portfolio.allocation_method = "custom"
        self.config.portfolio.dynamic_allocation = False
        self.config.portfolio.auto_allocate = False

        # Save to file
        config_dict = self.config.model_dump(exclude_none=True)

        # Normalize strategy names in backtest configuration
        if "backtest" in config_dict and "strategies" in config_dict["backtest"]:
            for strategy in config_dict["backtest"]["strategies"]:
                if "name" in strategy:
                    strategy["name"] = self._normalize_strategy_name(strategy["name"])

        # Convert dates to strings for YAML serialization
        config_dict = self._convert_dates(config_dict)

        with open(save_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        self.console.print(f"\n[green]✓ Optimized configuration saved to: {save_path}[/green]")
        self.console.print(
            f"[dim]You can now run backtest with: uv run python -m stockula --config {save_path} --mode backtest[/dim]"
        )

    def _convert_dates(self, obj: Any) -> Any:
        """Recursively convert date objects to strings.

        Args:
            obj: Object to convert

        Returns:
            Converted object
        """
        if isinstance(obj, dict):
            return {k: self._convert_dates(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_dates(item) for item in obj]
        elif isinstance(obj, date):
            return obj.strftime("%Y-%m-%d")
        return obj

    def create_portfolio(self) -> Portfolio:
        """Create portfolio from configuration.

        Returns:
            Portfolio instance
        """
        factory = self.container.domain_factory()
        portfolio = factory.create_portfolio(self.config)
        from .domain import Portfolio

        return cast(Portfolio, portfolio)

    # Presentation methods have moved to ResultsDisplay.

    def run_technical_analysis(
        self,
        ticker: str,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Run technical analysis for a ticker.

        Args:
            ticker: Stock symbol
            show_progress: Whether to show progress bars

        Returns:
            Dictionary with indicator results
        """
        # Get the technical analysis manager
        ta_manager = self.container.technical_analysis_manager()

        # Determine which indicators to use based on configuration
        ta_config = self.config.technical_analysis
        custom_indicators = []

        # Build custom indicators list based on config
        if "sma" in ta_config.indicators:
            custom_indicators.append("sma")
        if "ema" in ta_config.indicators:
            custom_indicators.append("ema")
        if "rsi" in ta_config.indicators:
            custom_indicators.append("rsi")
        if "macd" in ta_config.indicators:
            custom_indicators.append("macd")
        if "bbands" in ta_config.indicators:
            custom_indicators.append("bbands")
        if "atr" in ta_config.indicators:
            custom_indicators.append("atr")
        if "adx" in ta_config.indicators:
            custom_indicators.append("adx")
        if "stoch" in ta_config.indicators:
            custom_indicators.append("stoch")
        if "williams_r" in ta_config.indicators:
            custom_indicators.append("williams_r")
        if "cci" in ta_config.indicators:
            custom_indicators.append("cci")
        if "obv" in ta_config.indicators:
            custom_indicators.append("obv")
        if "ichimoku" in ta_config.indicators:
            custom_indicators.append("ichimoku")

        # Use the manager to analyze the symbol
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=self.console,
                transient=True,
            ) as progress:
                task = progress.add_task(
                    f"[cyan]Analyzing technical indicators for {ticker}...",
                    total=1,
                )

                result = ta_manager.analyze_symbol(
                    ticker,
                    self.config,
                    analysis_type="custom" if custom_indicators else "comprehensive",
                    custom_indicators=custom_indicators if custom_indicators else None,
                )

                progress.advance(task)
        else:
            result = ta_manager.analyze_symbol(
                ticker,
                self.config,
                analysis_type="custom" if custom_indicators else "comprehensive",
                custom_indicators=custom_indicators if custom_indicators else None,
            )

        # If we need to maintain backward compatibility with the old format,
        # we can still compute the specific period values
        if "indicators" in result and not result.get("error"):
            # Get the data for period-specific calculations
            data_fetcher = self.container.data_fetcher()
            data = data_fetcher.get_stock_data(
                ticker,
                start=self.date_to_string(self.config.data.start_date),
                end=self.date_to_string(self.config.data.end_date),
                interval=self.config.data.interval,
            )

            if not data.empty:
                ta = TechnicalIndicators(data)

                # Add period-specific calculations if needed
                if "sma" in ta_config.indicators and "sma" in result["indicators"]:
                    for period in ta_config.sma_periods:
                        result["indicators"][f"SMA_{period}"] = ta.sma(period).iloc[-1]

                if "ema" in ta_config.indicators and "ema" in result["indicators"]:
                    for period in ta_config.ema_periods:
                        result["indicators"][f"EMA_{period}"] = ta.ema(period).iloc[-1]

                # Add simple indicator values for backward compatibility
                if "rsi" in ta_config.indicators and "rsi" in result["indicators"]:
                    result["indicators"]["RSI"] = result["indicators"]["rsi"]["current"]

                if "macd" in ta_config.indicators and "macd" in result["indicators"]:
                    macd_data = result["indicators"]["macd"]["current"]
                    if isinstance(macd_data, dict):
                        result["indicators"]["MACD"] = macd_data.get("MACD")

                if "bbands" in ta_config.indicators and "bbands" in result["indicators"]:
                    result["indicators"]["BBands"] = result["indicators"]["bbands"]["current"]

                if "atr" in ta_config.indicators and "atr" in result["indicators"]:
                    result["indicators"]["ATR"] = result["indicators"]["atr"]["current"]

                if "adx" in ta_config.indicators and "adx" in result["indicators"]:
                    result["indicators"]["ADX"] = result["indicators"]["adx"]["current"]

        return cast(dict[str, Any], result)

    def _compute_indicators(
        self,
        ta: TechnicalIndicators,
        ta_config: Any,
        results: dict[str, Any],
        progress: Progress | None = None,
        task: Any | None = None,
        ticker: str | None = None,
    ) -> None:
        """Compute technical indicators.

        Args:
            ta: TechnicalIndicators instance
            ta_config: Technical analysis configuration
            results: Results dictionary to populate
            progress: Progress instance (optional)
            task: Progress task (optional)
            ticker: Ticker symbol (optional, for progress display)
        """
        indicators_dict = results["indicators"]
        assert isinstance(indicators_dict, dict)

        if "sma" in ta_config.indicators:
            for period in ta_config.sma_periods:
                if progress and task:
                    progress.update(
                        task,
                        description=f"[cyan]Computing SMA({period}) for {ticker}...",
                    )
                indicators_dict[f"SMA_{period}"] = ta.sma(period).iloc[-1]
                if progress and task:
                    progress.advance(task)

        if "ema" in ta_config.indicators:
            for period in ta_config.ema_periods:
                if progress and task:
                    progress.update(
                        task,
                        description=f"[cyan]Computing EMA({period}) for {ticker}...",
                    )
                indicators_dict[f"EMA_{period}"] = ta.ema(period).iloc[-1]
                if progress and task:
                    progress.advance(task)

        if "rsi" in ta_config.indicators:
            if progress and task:
                progress.update(task, description=f"[cyan]Computing RSI for {ticker}...")
            indicators_dict["RSI"] = ta.rsi(ta_config.rsi_period).iloc[-1]
            if progress and task:
                progress.advance(task)

        if "macd" in ta_config.indicators:
            if progress and task:
                progress.update(task, description=f"[cyan]Computing MACD for {ticker}...")
            macd_data = ta.macd(**ta_config.macd_params)
            indicators_dict["MACD"] = macd_data.iloc[-1].to_dict()
            if progress and task:
                progress.advance(task)

        if "bbands" in ta_config.indicators:
            if progress and task:
                progress.update(task, description=f"[cyan]Computing Bollinger Bands for {ticker}...")
            bbands_data = ta.bbands(**ta_config.bbands_params)
            indicators_dict["BBands"] = bbands_data.iloc[-1].to_dict()
            if progress and task:
                progress.advance(task)

        if "atr" in ta_config.indicators:
            if progress and task:
                progress.update(task, description=f"[cyan]Computing ATR for {ticker}...")
            indicators_dict["ATR"] = ta.atr(ta_config.atr_period).iloc[-1]
            if progress and task:
                progress.advance(task)

        if "adx" in ta_config.indicators:
            if progress and task:
                progress.update(task, description=f"[cyan]Computing ADX for {ticker}...")
            indicators_dict["ADX"] = ta.adx(14).iloc[-1]
            if progress and task:
                progress.advance(task)

    def run_backtest(self, ticker: str) -> list[dict[str, Any]]:
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
        use_train_test_split = (
            self.config.forecast.train_start_date is not None
            and self.config.forecast.train_end_date is not None
            and self.config.forecast.test_start_date is not None
            and self.config.forecast.test_end_date is not None
        )

        for strategy_config in self.config.backtest.strategies:
            try:
                if use_train_test_split:
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
                        continue

                    # Create result entry with train/test results
                    result_entry: dict[str, Any] | None = self._create_train_test_result(
                        ticker, strategy_config, backtest_result
                    )

                else:
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

                    result_entry = self._create_standard_result(ticker, strategy_config, backtest_result)  # type: ignore[no-redef]

                # Only append if result_entry is not None (i.e., backtest succeeded)
                if result_entry is not None:
                    results.append(result_entry)
            except Exception as e:
                self.console.print(f"[red]Error backtesting {strategy_config.name} on {ticker}: {e}[/red]")

        return results

    def _get_backtest_dates(self) -> tuple[str | None, str | None]:
        """Get backtest date range from configuration.

        Returns:
            Tuple of (start_date, end_date) as strings or None
        """
        backtest_start = None
        backtest_end = None

        # First check if backtest has specific dates
        if self.config.backtest.start_date and self.config.backtest.end_date:
            backtest_start = self.date_to_string(self.config.backtest.start_date)
            backtest_end = self.date_to_string(self.config.backtest.end_date)
        # Fall back to general data dates
        elif self.config.data.start_date and self.config.data.end_date:
            backtest_start = self.date_to_string(self.config.data.start_date)
            backtest_end = self.date_to_string(self.config.data.end_date)

        return backtest_start, backtest_end

    def _create_train_test_result(
        self, ticker: str, strategy_config: Any, backtest_result: dict[str, Any]
    ) -> dict[str, Any]:
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
        self, ticker: str, strategy_config: Any, backtest_result: dict[str, Any]
    ) -> dict[str, Any] | None:
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
        if "Initial Cash" in backtest_result:
            result_entry["initial_cash"] = backtest_result["Initial Cash"]
        if "Start Date" in backtest_result:
            result_entry["start_date"] = backtest_result["Start Date"]
        if "End Date" in backtest_result:
            result_entry["end_date"] = backtest_result["End Date"]
        if "Trading Days" in backtest_result:
            result_entry["trading_days"] = backtest_result["Trading Days"]
        if "Calendar Days" in backtest_result:
            result_entry["calendar_days"] = backtest_result["Calendar Days"]

        return result_entry

    def run_forecast_with_evaluation(self, ticker: str) -> dict[str, Any]:
        """Run forecasting with train/test split and evaluation.

        Args:
            ticker: Stock symbol

        Returns:
            Dictionary with forecast results and evaluation metrics
        """
        self.log_manager.info(f"\nForecasting {ticker} with train/test evaluation...")

        forecasting_manager = self.container.forecasting_manager()

        try:
            # Determine if we should use evaluation
            use_evaluation = (
                self.config.forecast.train_start_date is not None
                and self.config.forecast.train_end_date is not None
                and self.config.forecast.test_start_date is not None
                and self.config.forecast.test_end_date is not None
            )

            result = forecasting_manager.forecast_symbol(
                ticker,
                self.config,
                use_evaluation=use_evaluation,
            )

            # Add additional info if evaluation was used
            if use_evaluation and "evaluation" in result:
                # Log MASE if available, otherwise log MAPE
                eval_metrics = result["evaluation"]
                if "mase" in eval_metrics:
                    self.log_manager.info(
                        f"Evaluation metrics for {ticker}: RMSE={eval_metrics['rmse']:.2f}, "
                        f"MASE={eval_metrics['mase']:.3f}"
                    )
                else:
                    # Fallback to MAPE for backward compatibility
                    self.log_manager.info(
                        f"Evaluation metrics for {ticker}: RMSE={eval_metrics['rmse']:.2f}, "
                        f"MAPE={eval_metrics.get('mape', 0):.2f}%"
                    )

            return cast(dict[str, Any], result)

        except KeyboardInterrupt:
            self.log_manager.warning(f"Forecast for {ticker} interrupted by user")
            return {"ticker": ticker, "error": "Interrupted by user"}
        except Exception as e:
            self.log_manager.error(f"Error forecasting {ticker}: {e}")
            return {"ticker": ticker, "error": str(e)}

    def run_forecast(self, ticker: str) -> dict[str, Any]:
        """Run forecasting for a ticker.

        Args:
            ticker: Stock symbol

        Returns:
            Dictionary with forecast results
        """
        self.log_manager.info(f"\nForecasting {ticker} for {self.config.forecast.forecast_length} days...")

        forecasting_manager = self.container.forecasting_manager()

        try:
            result = forecasting_manager.forecast_symbol(
                ticker,
                self.config,
                use_evaluation=False,  # Explicit no evaluation for standard forecast
            )

            return cast(dict[str, Any], result)
        except KeyboardInterrupt:
            self.log_manager.warning(f"Forecast for {ticker} interrupted by user")
            return {"ticker": ticker, "error": "Interrupted by user"}
        except Exception as e:
            self.log_manager.error(f"Error forecasting {ticker}: {e}")
            return {"ticker": ticker, "error": str(e)}

    def save_detailed_report(
        self,
        strategy_name: str,
        strategy_results: list[dict],
        results: dict[str, Any],
        portfolio_results: PortfolioBacktestResults | None = None,
    ) -> str:
        """Save detailed strategy report to file.

        Args:
            strategy_name: Name of the strategy
            strategy_results: List of backtest results for this strategy
            results: Overall results dictionary
            portfolio_results: Portfolio backtest results (optional)

        Returns:
            Path to the saved report file
        """
        # Create reports directory if it doesn't exist
        reports_dir = Path(self.config.output.get("results_dir", "./results")) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"strategy_report_{strategy_name}_{timestamp}.json"

        # Prepare detailed report data
        report_data = {
            "strategy": strategy_name,
            "timestamp": timestamp,
            "date_range": {
                "start": self.date_to_string(self.config.data.start_date),
                "end": self.date_to_string(self.config.data.end_date),
            },
            "portfolio": {
                "initial_value": results.get("initial_portfolio_value", 0),
                "initial_capital": results.get("initial_capital", 0),
            },
            "broker_config": self._get_broker_config_dict(),
            "detailed_results": strategy_results,
            "summary": {
                "total_trades": sum(r.get("num_trades", 0) for r in strategy_results),
                "winning_stocks": sum(1 for r in strategy_results if r.get("return_pct", 0) > 0),
                "losing_stocks": sum(1 for r in strategy_results if r.get("return_pct", 0) < 0),
                "average_return": sum(r.get("return_pct", 0) for r in strategy_results) / len(strategy_results)
                if strategy_results
                else 0,
                "average_sharpe": sum(r.get("sharpe_ratio", 0) for r in strategy_results) / len(strategy_results)
                if strategy_results
                else 0,
            },
        }

        # Save report
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        # Also save structured results if provided
        if portfolio_results:
            structured_file = reports_dir / f"portfolio_backtest_{timestamp}.json"
            with open(structured_file, "w") as f:
                # Convert to dict using model_dump
                json.dump(portfolio_results.model_dump(), f, indent=2, default=str)

        return str(report_file)

    def _get_broker_config_dict(self) -> dict[str, Any]:
        """Get broker configuration as dictionary.

        Returns:
            Dictionary with broker configuration
        """
        if self.config.backtest.broker_config:
            return {
                "name": self.config.backtest.broker_config.name,
                "commission_type": self.config.backtest.broker_config.commission_type,
                "commission_value": self.config.backtest.broker_config.commission_value,
                "min_commission": self.config.backtest.broker_config.min_commission,
                "regulatory_fees": self.config.backtest.broker_config.regulatory_fees,
            }
        else:
            return {
                "name": "legacy",
                "commission_type": "percentage",
                "commission_value": self.config.backtest.commission,
                "min_commission": None,
                "regulatory_fees": 0,
            }

    def create_portfolio_backtest_results(
        self,
        results: dict[str, Any],
        strategy_results: dict[str, list[dict]],
    ) -> PortfolioBacktestResults:
        """Create structured backtest results.

        Args:
            results: Main results dictionary with initial values
            strategy_results: Raw backtest results grouped by strategy

        Returns:
            Structured portfolio backtest results
        """
        # Build strategy summaries
        strategy_summaries = []

        for strategy_name, backtests in strategy_results.items():
            # Create BacktestResult objects
            detailed_results = []
            for backtest in backtests:
                detailed_results.append(
                    BacktestResult(
                        ticker=backtest["ticker"],
                        strategy=backtest["strategy"],
                        parameters=backtest.get("parameters", {}),
                        return_pct=backtest["return_pct"],
                        sharpe_ratio=backtest["sharpe_ratio"],
                        max_drawdown_pct=backtest["max_drawdown_pct"],
                        num_trades=backtest["num_trades"],
                        win_rate=backtest.get("win_rate"),
                    )
                )

            # Calculate summary metrics
            total_return = sum(r.return_pct for r in detailed_results)
            avg_return = total_return / len(detailed_results) if detailed_results else 0
            avg_sharpe = (
                sum(r.sharpe_ratio for r in detailed_results) / len(detailed_results) if detailed_results else 0
            )
            total_trades = sum(r.num_trades for r in detailed_results)
            winning_stocks = sum(1 for r in detailed_results if r.return_pct > 0)
            losing_stocks = sum(1 for r in detailed_results if r.return_pct < 0)

            # Calculate approximate final portfolio value
            final_value = results["initial_portfolio_value"] * (1 + avg_return / 100)

            # Get strategy parameters from first result
            strategy_params = detailed_results[0].parameters if detailed_results else {}

            # Create strategy summary
            summary = StrategyBacktestSummary(
                strategy_name=strategy_name,
                parameters=strategy_params,
                initial_portfolio_value=results["initial_portfolio_value"],
                final_portfolio_value=final_value,
                total_return_pct=avg_return,
                total_trades=total_trades,
                winning_stocks=winning_stocks,
                losing_stocks=losing_stocks,
                average_return_pct=avg_return,
                average_sharpe_ratio=avg_sharpe,
                detailed_results=detailed_results,
            )

            strategy_summaries.append(summary)

        # Create broker config dict
        broker_config = self._get_full_broker_config_dict()

        # Get date range from config or results
        date_start, date_end = self._get_date_range(results)

        portfolio_results = PortfolioBacktestResults(
            initial_portfolio_value=results.get("initial_portfolio_value", 0),
            initial_capital=results.get("initial_capital", 0),
            date_range={
                "start": date_start,
                "end": date_end,
            },
            broker_config=broker_config,
            strategy_summaries=strategy_summaries,
        )

        return portfolio_results

    def _get_full_broker_config_dict(self) -> dict[str, Any]:
        """Get full broker configuration as dictionary.

        Returns:
            Dictionary with full broker configuration
        """
        if self.config.backtest.broker_config:
            return {
                "name": self.config.backtest.broker_config.name,
                "commission_type": self.config.backtest.broker_config.commission_type,
                "commission_value": self.config.backtest.broker_config.commission_value,
                "min_commission": self.config.backtest.broker_config.min_commission,
                "regulatory_fees": self.config.backtest.broker_config.regulatory_fees,
                "exchange_fees": getattr(self.config.backtest.broker_config, "exchange_fees", 0),
            }
        else:
            return {
                "name": "legacy",
                "commission_type": "percentage",
                "commission_value": self.config.backtest.commission,
                "min_commission": None,
                "regulatory_fees": 0,
                "exchange_fees": 0,
            }

    def _get_date_range(self, results: dict[str, Any]) -> tuple[str, str]:
        """Get date range from configuration or results.

        Args:
            results: Results dictionary

        Returns:
            Tuple of (start_date, end_date) as strings
        """
        date_start: str = "N/A"
        date_end: str = "N/A"

        # First try backtest dates, then data dates
        if self.config.backtest.start_date:
            date_start_val = self.date_to_string(self.config.backtest.start_date)
            if date_start_val is not None:
                date_start = date_start_val
        elif self.config.data.start_date:
            date_start_val = self.date_to_string(self.config.data.start_date)
            if date_start_val is not None:
                date_start = date_start_val

        if self.config.backtest.end_date:
            date_end_val = self.date_to_string(self.config.backtest.end_date)
            if date_end_val is not None:
                date_end = date_end_val
        elif self.config.data.end_date:
            date_end_val = self.date_to_string(self.config.data.end_date)
            if date_end_val is not None:
                date_end = date_end_val

        # If dates not in config, try to get from backtest results
        if (
            (date_start == "N/A" or date_end == "N/A")
            and results.get("backtesting")
            and len(results["backtesting"]) > 0
        ):
            # Look through all results to find one with dates
            for backtest_result in results["backtesting"]:
                if date_start == "N/A" and "start_date" in backtest_result:
                    date_start = backtest_result["start_date"]
                if date_end == "N/A" and "end_date" in backtest_result:
                    date_end = backtest_result["end_date"]
                # Stop if we found both dates
                if date_start != "N/A" and date_end != "N/A":
                    break

        return date_start, date_end

    def get_portfolio_value_at_date(
        self, portfolio: Portfolio, start_date_str: str | None
    ) -> tuple[float, dict[str, float]]:
        """Get portfolio value at a specific date.

        Args:
            portfolio: Portfolio instance
            start_date_str: Date string or None

        Returns:
            Tuple of (portfolio_value, prices_dict)
        """
        fetcher = self.container.data_fetcher()
        symbols = [asset.symbol for asset in portfolio.get_all_assets()]

        if start_date_str:
            self.log_manager.debug(f"\nFetching prices at start date ({start_date_str})...")
            # Fetch one day of data at the start date to get opening prices
            start_prices = {}
            for symbol in symbols:
                try:
                    data = fetcher.get_stock_data(symbol, start=start_date_str, end=start_date_str)
                    if not data.empty:
                        start_prices[symbol] = data["Close"].iloc[0]
                    else:
                        # If no data on exact date, get the next available date
                        start_dt = pd.to_datetime(start_date_str)
                        end_date = (start_dt + timedelta(days=7)).strftime("%Y-%m-%d")
                        data = fetcher.get_stock_data(symbol, start=start_date_str, end=end_date)
                        if not data.empty:
                            start_prices[symbol] = data["Close"].iloc[0]
                except Exception as e:
                    self.log_manager.warning(f"Could not get start price for {symbol}: {e}")

            return portfolio.get_portfolio_value(start_prices), start_prices
        else:
            self.log_manager.debug("\nFetching current prices...")
            current_prices = fetcher.get_current_prices(symbols, show_progress=True)
            return portfolio.get_portfolio_value(current_prices), current_prices

    def categorize_assets(self, portfolio: Portfolio) -> tuple[list[Any], list[Any], set[Category]]:
        """Categorize assets into tradeable and hold-only.

        Args:
            portfolio: Portfolio instance

        Returns:
            Tuple of (tradeable_assets, hold_only_assets, hold_only_categories)
        """
        # Get hold-only categories from config
        hold_only_category_names = set(self.config.backtest.hold_only_categories)
        hold_only_categories = set()
        for category_name in hold_only_category_names:
            try:
                hold_only_categories.add(Category[category_name])
            except KeyError:
                self.log_manager.warning(f"Unknown category '{category_name}' in hold_only_categories")

        tradeable_assets = []
        hold_only_assets = []

        for asset in portfolio.get_all_assets():
            if asset.category in hold_only_categories:
                hold_only_assets.append(asset)
            else:
                tradeable_assets.append(asset)

        if hold_only_assets:
            self.log_manager.info("\nHold-only assets (excluded from backtesting):")
            for asset in hold_only_assets:
                self.log_manager.info(f"  {asset.symbol} ({asset.category})")

        return tradeable_assets, hold_only_assets, hold_only_categories

    def run_main_processing(
        self,
        mode: str,
        portfolio,
        show_forecast_warning: bool = True,
    ) -> dict[str, Any]:
        """Run the main processing loop for ticker analysis.

        Args:
            mode: Processing mode ('all', 'ta', 'backtest', 'forecast')
            portfolio: Portfolio instance
            show_forecast_warning: Whether to show forecast warning

        Returns:
            Dictionary containing all results
        """
        from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

        # Get portfolio value at start of backtest period
        start_date_str = self.date_to_string(self.config.data.start_date) if mode in ["all", "backtest"] else None
        initial_portfolio_value, _ = self.get_portfolio_value_at_date(portfolio, start_date_str)

        # Calculate returns
        initial_return = initial_portfolio_value - portfolio.initial_capital
        initial_return_pct = (initial_return / portfolio.initial_capital) * 100

        self.log_manager.info(f"Initial Capital: ${portfolio.initial_capital:,.2f}")
        self.log_manager.info(f"Return Since Inception: ${initial_return:,.2f} ({initial_return_pct:+.2f}%)")

        # Initialize results
        results = {
            "initial_portfolio_value": initial_portfolio_value,
            "initial_capital": portfolio.initial_capital,
        }

        # Categorize assets
        all_assets = portfolio.get_all_assets()
        tradeable_assets, hold_only_assets, hold_only_categories = self.categorize_assets(portfolio)

        # Get ticker symbols for processing
        ticker_symbols = [asset.symbol for asset in all_assets]

        # Determine what operations will be performed
        will_backtest = mode in ["all", "backtest"]
        will_forecast = mode in ["all", "forecast"]

        # Create appropriate progress display
        if will_backtest or will_forecast:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=self.console,
            ) as progress:
                # Show forecast warning if needed
                if will_forecast and show_forecast_warning:
                    from .display import ResultsDisplay

                    display = ResultsDisplay(self.console)
                    display.show_forecast_warning(self.config)

                # Create progress tasks
                backtest_task = None
                if will_backtest:
                    # Count tradeable assets for backtesting
                    tradeable_count = len([a for a in all_assets if a.category not in hold_only_categories])
                    if tradeable_count > 0:
                        num_strategies = len(self.config.backtest.strategies)
                        backtest_task = progress.add_task(
                            f"[green]Backtesting {num_strategies} strategies across {tradeable_count} stocks...",
                            total=tradeable_count * num_strategies,
                        )

                # Process each ticker with progress tracking
                for ticker in ticker_symbols:
                    self.log_manager.debug(f"\nProcessing {ticker}...")

                    # Get the asset to check its category
                    asset = next((a for a in all_assets if a.symbol == ticker), None)
                    is_hold_only = asset and asset.category in hold_only_categories

                    if mode in ["all", "ta"]:
                        if "technical_analysis" not in results:
                            results["technical_analysis"] = []
                        # Show progress for TA when it's the only operation
                        show_ta_progress = mode == "ta" or not will_backtest and not will_forecast
                        results["technical_analysis"].append(self.run_technical_analysis(ticker, show_ta_progress))

                    if will_backtest and not is_hold_only:
                        if "backtesting" not in results:
                            results["backtesting"] = []

                        # Run backtest and update progress
                        backtest_results = self.run_backtest(ticker)
                        results["backtesting"].extend(backtest_results)

                        # Update progress
                        if backtest_task is not None:
                            for _ in backtest_results:
                                progress.advance(backtest_task)

                # Run sequential forecasting if needed
                if will_forecast and ticker_symbols:
                    forecasting_manager = self.container.forecasting_manager()
                    forecast_results = forecasting_manager.forecast_multiple_symbols_with_progress(
                        ticker_symbols, self.config, self.console
                    )
                    results["forecasting"] = forecast_results
        else:
            # No progress bars needed for TA only
            for ticker in ticker_symbols:
                self.log_manager.debug(f"\nProcessing {ticker}...")

                if mode in ["all", "ta"]:
                    if "technical_analysis" not in results:
                        results["technical_analysis"] = []
                    # Always show progress for standalone TA mode
                    results["technical_analysis"].append(self.run_technical_analysis(ticker, show_progress=True))

        return results
