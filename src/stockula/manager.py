"""Stockula Manager - Main business logic orchestrator."""

from datetime import date
from typing import Any

from rich.console import Console

from .backtesting import BaseStrategy, strategy_registry
from .config import StockulaConfig
from .config.models import PortfolioBacktestResults
from .container import Container
from .display import ResultsDisplay
from .domain import Category, Portfolio
from .utils import get_console


class StockulaManager:
    """
    Simplified StockulaManager following SRP - delegates to specialized services.
    Single Responsibility: Coordinating high-level business workflows.
    """

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
        self.console = get_console(console)
        self.log_manager = container.logging_manager()

        # Strategy registry provides centralized strategy management
        self.strategy_registry = strategy_registry

        # Initialize specialized services
        from .services import (
            AnalysisOrchestrator,
            BacktestOrchestrator,
            ForecastOrchestrator,
            PortfolioService,
            ReportService,
        )

        self.analysis_orchestrator = AnalysisOrchestrator(config, container, console)
        self.backtest_orchestrator = BacktestOrchestrator(config, container, console)
        self.forecast_orchestrator = ForecastOrchestrator(config, container, console)
        self.portfolio_service = PortfolioService(config, container, console)
        self.report_service = ReportService(config, container, console)

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
            self._update_config_with_optimized_quantities(optimized_quantities)

            # Save optimized config if requested
            if save_path:
                self.report_service.save_optimized_config(save_path, optimized_quantities)

            return 0

        except Exception as e:
            self.console.print(f"[red]Error during optimization: {e}[/red]")
            self.log_manager.error(f"Optimization error: {e}", exc_info=True)
            return 1

    def _update_config_with_optimized_quantities(self, optimized_quantities: dict[str, float]) -> None:
        """Update config with optimized quantities.

        Args:
            optimized_quantities: Dictionary of symbol to quantity
        """
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

    def _normalize_strategy_name(self, strategy_name: str) -> str:
        """Normalize strategy name to snake_case format.

        Args:
            strategy_name: Strategy name in any format

        Returns:
            Normalized strategy name in snake_case
        """
        return self.strategy_registry.normalize_strategy_name(strategy_name)

    # Portfolio service delegation
    def create_portfolio(self) -> Portfolio:
        """Create portfolio from configuration."""
        return self.portfolio_service.create_portfolio()

    def get_portfolio_value_at_date(
        self, portfolio: Portfolio, start_date_str: str | None
    ) -> tuple[float, dict[str, float]]:
        """Get portfolio value at a specific date."""
        return self.portfolio_service.get_portfolio_value_at_date(portfolio, start_date_str)

    def categorize_assets(self, portfolio: Portfolio) -> tuple[list[Any], list[Any], set[Category]]:
        """Categorize assets into tradeable and hold-only."""
        return self.portfolio_service.categorize_assets(portfolio)

    # Analysis orchestrator delegation
    def run_technical_analysis(
        self,
        ticker: str,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Run technical analysis for a ticker."""
        return self.analysis_orchestrator.run_technical_analysis(ticker, show_progress)

    # Backtest orchestrator delegation
    def run_backtest(self, ticker: str) -> list[dict[str, Any]]:
        """Run backtesting for a ticker."""
        return self.backtest_orchestrator.run_backtest(ticker)

    # Forecast orchestrator delegation
    def run_forecast_with_evaluation(self, ticker: str) -> dict[str, Any]:
        """Run forecasting with train/test split and evaluation."""
        return self.forecast_orchestrator.run_forecast_with_evaluation(ticker)

    def run_forecast(self, ticker: str) -> dict[str, Any]:
        """Run forecasting for a ticker."""
        return self.forecast_orchestrator.run_forecast(ticker)

    # Report service delegation
    def save_detailed_report(
        self,
        strategy_name: str,
        strategy_results: list[dict],
        results: dict[str, Any],
        portfolio_results: PortfolioBacktestResults | None = None,
    ) -> str:
        """Save detailed strategy report to file."""
        return self.report_service.save_detailed_report(strategy_name, strategy_results, results, portfolio_results)

    def create_portfolio_backtest_results(
        self,
        results: dict[str, Any],
        strategy_results: dict[str, list[dict]],
    ) -> PortfolioBacktestResults:
        """Create structured backtest results."""
        return self.report_service.create_portfolio_backtest_results(results, strategy_results)

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

        # Calculate portfolio returns and setup
        start_date_str = self.date_to_string(self.config.data.start_date) if mode in ["all", "backtest"] else None
        initial_portfolio_value, initial_return, initial_return_pct = (
            self.portfolio_service.calculate_portfolio_returns(portfolio, start_date_str)
        )

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

        # Execute processing with appropriate progress display
        if will_backtest or will_forecast:
            self._run_processing_with_progress(
                mode,
                results,
                ticker_symbols,
                all_assets,
                hold_only_categories,
                will_backtest,
                will_forecast,
                show_forecast_warning,
            )
        else:
            # No progress bars needed for TA only
            self._run_processing_simple(mode, results, ticker_symbols)

        return results

    def _run_processing_with_progress(
        self,
        mode: str,
        results: dict[str, Any],
        ticker_symbols: list[str],
        all_assets,
        hold_only_categories,
        will_backtest: bool,
        will_forecast: bool,
        show_forecast_warning: bool,
    ) -> None:
        """Run processing with progress bars."""
        from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

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
                forecast_results = self.forecast_orchestrator.run_multiple_forecasts(ticker_symbols)
                results["forecasting"] = forecast_results

    def _run_processing_simple(
        self,
        mode: str,
        results: dict[str, Any],
        ticker_symbols: list[str],
    ) -> None:
        """Run simple processing without progress bars."""
        for ticker in ticker_symbols:
            self.log_manager.debug(f"\nProcessing {ticker}...")

            if mode in ["all", "ta"]:
                if "technical_analysis" not in results:
                    results["technical_analysis"] = []
                # Always show progress for standalone TA mode
                results["technical_analysis"].append(self.run_technical_analysis(ticker, show_progress=True))

    # Backward compatibility methods for tests - Delegate to appropriate services
    def _get_backtest_dates(self) -> tuple[str | None, str | None]:
        """Get backtest dates. Delegates to BacktestOrchestrator."""
        return self.backtest_orchestrator._get_backtest_dates()

    def _save_optimized_config(self, save_path: str, optimized_quantities: dict) -> None:
        """Save optimized config. Delegates to ReportService."""
        self.report_service.save_optimized_config(save_path, optimized_quantities)

    def _convert_dates(self, obj: Any) -> Any:
        """Convert dates in object. Delegates to ReportService."""
        return self.report_service._convert_dates(obj)

    def _create_train_test_result(self, ticker: str, strategy_config: dict, backtest_result: dict) -> dict:
        """Create train/test result. Delegates to BacktestOrchestrator."""
        return self.backtest_orchestrator._create_train_test_result(ticker, strategy_config, backtest_result)

    def _create_standard_result(self, ticker: str, strategy_config: dict, backtest_result: dict) -> dict[str, Any]:
        """Create standard result. Delegates to BacktestOrchestrator."""
        result = self.backtest_orchestrator._create_standard_result(ticker, strategy_config, backtest_result)
        return result if result is not None else {}

    def _compute_indicators(self, ta_instance, ta_config, results, progress, task, ticker: str) -> None:
        """Compute indicators. Delegates to AnalysisOrchestrator."""
        self.analysis_orchestrator._compute_indicators(ta_instance, ta_config, results, progress, task, ticker)

    def _get_broker_config_dict(self) -> dict:
        """Get broker config dict. Delegates to ReportService."""
        return self.report_service._get_broker_config_dict()

    def _get_full_broker_config_dict(self) -> dict:
        """Get full broker config dict. Delegates to ReportService."""
        return self.report_service._get_full_broker_config_dict()

    def _get_date_range(self, results: dict) -> tuple[str | None, str | None]:
        """Get date range from results. Delegates to ReportService."""
        return self.report_service._get_date_range(results)
