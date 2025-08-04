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

    # Run main processing through StockulaManager
    results = manager.run_main_processing(args.mode, portfolio)

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


if __name__ == "__main__":
    main()
