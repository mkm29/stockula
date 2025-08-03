"""Stockula main entry point."""

# Suppress warnings early - before any imports that might trigger them
import logging

logging.getLogger("alembic.runtime.migration").setLevel(logging.WARNING)
logging.getLogger("alembic").setLevel(logging.WARNING)

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

import os

os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

import argparse
from datetime import datetime
from typing import Any

from dependency_injector.wiring import Provide, inject
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from .config import StockulaConfig, TickerConfig
from .config.settings import save_config
from .container import Container, create_container
from .display import ResultsDisplay
from .interfaces import ILoggingManager
from .manager import StockulaManager

# Global logging manager and console instances
log_manager: ILoggingManager | None = None
console = Console()


@inject
def setup_logging(
    config: StockulaConfig,
    logging_manager: ILoggingManager = Provide[Container.logging_manager],
) -> None:
    """Configure logging based on configuration."""
    global log_manager
    log_manager = logging_manager
    log_manager.setup(config)


def print_results(results: dict[str, Any], output_format: str = "console", config=None, container=None):
    """Print results in specified format using ResultsDisplay.

    Args:
        results: Results dictionary
        output_format: Output format (console, json)
        config: Optional configuration object for portfolio composition
        container: Optional DI container for fetching data
    """
    display = ResultsDisplay(console)
    display.print_results(results, output_format, config, container)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Stockula Trading Platform")
    parser.add_argument("--config", "-c", type=str, help="Path to configuration file (YAML)")
    parser.add_argument("--ticker", "-t", type=str, help="Override ticker symbol (single ticker mode)")
    parser.add_argument(
        "--mode",
        "-m",
        choices=["all", "ta", "backtest", "forecast", "optimize-allocation"],
        default="all",
        help="Operation mode",
    )
    parser.add_argument(
        "--output",
        "-o",
        choices=["console", "json"],
        default="console",
        help="Output format",
    )
    parser.add_argument("--save-config", type=str, help="Save current configuration to file")
    parser.add_argument(
        "--save-optimized-config",
        type=str,
        help="Save optimized configuration to file (used with optimize-allocation mode)",
    )

    # Add date range arguments
    parser.add_argument("--train-start", type=str, help="Training start date (YYYY-MM-DD)")
    parser.add_argument("--train-end", type=str, help="Training end date (YYYY-MM-DD)")
    parser.add_argument("--test-start", type=str, help="Testing start date (YYYY-MM-DD)")
    parser.add_argument("--test-end", type=str, help="Testing end date (YYYY-MM-DD)")

    args = parser.parse_args()

    # Initialize DI container first
    container = create_container(args.config)

    # Load configuration - the container will handle this
    config = container.stockula_config()

    # Set up logging based on configuration
    setup_logging(config, logging_manager=container.logging_manager())

    # Override ticker if provided
    if args.ticker:
        config.portfolio.tickers = [TickerConfig(symbol=args.ticker, quantity=1.0)]
        # Disable auto-allocation for single ticker mode since we don't have categories
        config.portfolio.auto_allocate = False
        config.portfolio.dynamic_allocation = False
        config.portfolio.allocation_method = "equal_weight"
        # Allow 100% position for single ticker mode
        config.portfolio.max_position_size = 100.0

    # Override date ranges if provided
    if args.train_start:
        config.forecast.train_start_date = datetime.strptime(args.train_start, "%Y-%m-%d").date()
    if args.train_end:
        config.forecast.train_end_date = datetime.strptime(args.train_end, "%Y-%m-%d").date()
    if args.test_start:
        config.forecast.test_start_date = datetime.strptime(args.test_start, "%Y-%m-%d").date()
    if args.test_end:
        config.forecast.test_end_date = datetime.strptime(args.test_end, "%Y-%m-%d").date()

    # Create manager instance
    manager = StockulaManager(config, container, console)

    # Handle optimize-allocation mode early (before portfolio creation)
    if args.mode == "optimize-allocation":
        save_path = args.save_optimized_config or args.save_config
        return manager.run_optimize_allocation(save_path)

    # Save configuration if requested (for non-optimize-allocation modes)
    if args.save_config and args.mode != "optimize-allocation":
        save_config(config, args.save_config)
        print(f"Configuration saved to {args.save_config}")
        return

    # Create portfolio
    portfolio = manager.create_portfolio()

    # Display portfolio summary
    manager.display_portfolio_summary(portfolio)
    manager.display_portfolio_holdings(portfolio)

    # Get portfolio value at start of backtest period
    start_date_str = manager.date_to_string(config.data.start_date) if args.mode in ["all", "backtest"] else None
    initial_portfolio_value, _ = manager.get_portfolio_value_at_date(portfolio, start_date_str)

    # Calculate returns
    initial_return = initial_portfolio_value - portfolio.initial_capital
    initial_return_pct = (initial_return / portfolio.initial_capital) * 100

    log_manager.info(f"Initial Capital: ${portfolio.initial_capital:,.2f}")
    log_manager.info(f"Return Since Inception: ${initial_return:,.2f} ({initial_return_pct:+.2f}%)")

    # Initialize results
    results = {
        "initial_portfolio_value": initial_portfolio_value,
        "initial_capital": portfolio.initial_capital,
    }

    # Categorize assets
    all_assets = portfolio.get_all_assets()
    tradeable_assets, hold_only_assets, hold_only_categories = manager.categorize_assets(portfolio)

    # Get ticker symbols for processing
    ticker_symbols = [asset.symbol for asset in all_assets]

    # Determine what operations will be performed
    will_backtest = args.mode in ["all", "backtest"]
    will_forecast = args.mode in ["all", "forecast"]

    # Create appropriate progress display
    if will_backtest or will_forecast:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            # Show forecast warning if needed
            if will_forecast:
                display = ResultsDisplay(console)
                display.show_forecast_warning(config)

            # Create progress tasks
            backtest_task = None
            if will_backtest:
                # Count tradeable assets for backtesting
                tradeable_count = len([a for a in all_assets if a.category not in hold_only_categories])
                if tradeable_count > 0:
                    num_strategies = len(config.backtest.strategies)
                    backtest_task = progress.add_task(
                        f"[green]Backtesting {num_strategies} strategies across {tradeable_count} stocks...",
                        total=tradeable_count * num_strategies,
                    )

            # Process each ticker with progress tracking
            for ticker in ticker_symbols:
                log_manager.debug(f"\nProcessing {ticker}...")

                # Get the asset to check its category
                asset = next((a for a in all_assets if a.symbol == ticker), None)
                is_hold_only = asset and asset.category in hold_only_categories

                if args.mode in ["all", "ta"]:
                    if "technical_analysis" not in results:
                        results["technical_analysis"] = []
                    # Show progress for TA when it's the only operation
                    show_ta_progress = args.mode == "ta" or not will_backtest and not will_forecast
                    results["technical_analysis"].append(manager.run_technical_analysis(ticker, show_ta_progress))

                if will_backtest and not is_hold_only:
                    if "backtesting" not in results:
                        results["backtesting"] = []

                    # Run backtest and update progress
                    backtest_results = manager.run_backtest(ticker)
                    results["backtesting"].extend(backtest_results)

                    # Update progress
                    if backtest_task is not None:
                        for _ in backtest_results:
                            progress.advance(backtest_task)

            # Run sequential forecasting if needed
            if will_forecast and ticker_symbols:
                _run_sequential_forecasting(manager, config, ticker_symbols, results)
    else:
        # No progress bars needed for TA only
        for ticker in ticker_symbols:
            log_manager.debug(f"\nProcessing {ticker}...")

            if args.mode in ["all", "ta"]:
                if "technical_analysis" not in results:
                    results["technical_analysis"] = []
                # Always show progress for standalone TA mode
                results["technical_analysis"].append(manager.run_technical_analysis(ticker, show_progress=True))

    # Show current portfolio value for forecast mode
    if args.mode == "forecast":
        display = ResultsDisplay(console)
        display.show_portfolio_forecast_value(config, portfolio, results)

    # Output results
    output_format = args.output or config.output.get("format", "console")
    print_results(results, output_format, config, container)

    # Show strategy-specific summaries after backtesting
    if args.mode in ["all", "backtest"] and "backtesting" in results:
        display = ResultsDisplay(console)
        display.show_strategy_summaries(manager, config, results)


def _run_sequential_forecasting(
    manager: StockulaManager, config: StockulaConfig, ticker_symbols: list[str], results: dict[str, Any]
):
    """Run sequential forecasting for all tickers.

    Args:
        manager: StockulaManager instance
        config: Configuration object
        ticker_symbols: List of ticker symbols
        results: Results dictionary to populate
    """
    console.print("\n[bold blue]Starting sequential forecasting...[/bold blue]")
    console.print(
        f"[dim]Configuration: max_generations={config.forecast.max_generations}, "
        f"num_validations={config.forecast.num_validations}[/dim]"
    )

    # Create a separate progress display for sequential forecasting
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as forecast_progress:
        forecast_task = forecast_progress.add_task(
            f"[blue]Forecasting {len(ticker_symbols)} tickers...",
            total=len(ticker_symbols),
        )

        # Initialize results
        results["forecasting"] = []

        # Process each ticker sequentially
        for idx, symbol in enumerate(ticker_symbols, 1):
            forecast_progress.update(
                forecast_task,
                description=f"[blue]Forecasting {symbol} ({idx}/{len(ticker_symbols)})...",
            )

            try:
                # Check if test dates are provided for evaluation
                if config.forecast.test_start_date and config.forecast.test_end_date:
                    # Use the new evaluation method
                    forecast_result = manager.run_forecast_with_evaluation(symbol)
                else:
                    # Use the original method
                    forecast_result = manager.run_forecast(symbol)

                results["forecasting"].append(forecast_result)

                # Update progress to show completion
                forecast_progress.update(
                    forecast_task,
                    description=f"[green]✅ Forecasted {symbol}[/green] ({idx}/{len(ticker_symbols)})",
                )

            except KeyboardInterrupt:
                if log_manager:
                    log_manager.warning(f"Forecast for {symbol} interrupted by user")
                results["forecasting"].append({"ticker": symbol, "error": "Interrupted by user"})
                break
            except Exception as e:
                if log_manager:
                    log_manager.error(f"Error forecasting {symbol}: {e}")
                results["forecasting"].append({"ticker": symbol, "error": str(e)})

                # Update progress to show error
                forecast_progress.update(
                    forecast_task,
                    description=f"[red]❌ Failed {symbol}[/red] ({idx}/{len(ticker_symbols)})",
                )

            # Advance progress
            forecast_progress.advance(forecast_task)

        # Mark progress as complete
        forecast_progress.update(
            forecast_task,
            description="[green]Forecasting complete!",
        )


if __name__ == "__main__":
    main()
