"""Command-line interface for Stockula."""

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, cast

import typer
from pydantic import ValidationError
from rich.panel import Panel

from .cli_manager import cli_manager
from .config import TickerConfig
from .config.settings import save_config
from .container import create_container
from .display import ResultsDisplay
from .manager import StockulaManager

# Create Typer app
app = typer.Typer(
    name="stockula",
    help="Stockula Trading Platform - Analyze stocks with technical analysis, backtesting, and forecasting.",
    add_completion=False,
)

# Get console from CLI manager
console = cli_manager.get_console()


class Mode(str, Enum):
    """Operation modes for Stockula."""

    ALL = "all"
    TA = "ta"
    BACKTEST = "backtest"
    FORECAST = "forecast"
    OPTIMIZE_ALLOCATION = "optimize-allocation"


class OutputFormat(str, Enum):
    """Output format options."""

    CONSOLE = "console"
    JSON = "json"


def print_results(results: dict[str, Any], output_format: str = "console", config=None, container=None, portfolio=None):
    """Print results in specified format using ResultsDisplay.

    Args:
        results: Results dictionary
        output_format: Output format (console, json)
        config: Optional configuration object for portfolio composition
        container: Optional DI container for fetching data
        portfolio: Optional portfolio instance for forecast display
    """
    display = ResultsDisplay(cli_manager.get_console())
    display.print_results(results, output_format, config, container, portfolio)


def run_stockula(
    config: str | None = None,
    ticker: str | None = None,
    mode: str = "all",
    output: str = "console",
    save_config_path: str | None = None,
    save_optimized_config: str | None = None,
    train_start: str | None = None,
    train_end: str | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
):
    """Core logic for running Stockula."""
    # Initialize DI container first
    container = create_container(config)

    # Load configuration - the container will handle this
    try:
        stockula_config = container.stockula_config()
    except ValidationError as e:
        # Use the provided config path or default
        config_path = config or ".stockula.yaml"
        handle_validation_error(e, config_path)

    # Set up logging based on configuration
    from .interfaces import ILoggingManager
    from .main import setup_logging

    setup_logging(stockula_config, logging_manager=cast(ILoggingManager, container.logging_manager()))

    # Override ticker if provided
    if ticker:
        stockula_config.portfolio.tickers = [TickerConfig(symbol=ticker, quantity=1.0)]
        # Disable auto-allocation for single ticker mode since we don't have categories
        stockula_config.portfolio.auto_allocate = False
        stockula_config.portfolio.dynamic_allocation = False
        stockula_config.portfolio.allocation_method = "equal_weight"
        # Allow 100% position for single ticker mode
        stockula_config.portfolio.max_position_size = 100.0

    # Override date ranges if provided
    if train_start:
        stockula_config.forecast.train_start_date = datetime.strptime(train_start, "%Y-%m-%d").date()
    if train_end:
        stockula_config.forecast.train_end_date = datetime.strptime(train_end, "%Y-%m-%d").date()
    if test_start:
        stockula_config.forecast.test_start_date = datetime.strptime(test_start, "%Y-%m-%d").date()
    if test_end:
        stockula_config.forecast.test_end_date = datetime.strptime(test_end, "%Y-%m-%d").date()

    # Create manager instance
    manager = StockulaManager(stockula_config, container, console)

    # Handle optimize-allocation mode early (before portfolio creation)
    if mode == "optimize-allocation":
        save_path = save_optimized_config or save_config_path
        return manager.run_optimize_allocation(save_path)

    # Save configuration if requested (for non-optimize-allocation modes)
    if save_config_path and mode != "optimize-allocation":
        save_config(stockula_config, save_config_path)
        print(f"Configuration saved to {save_config_path}")
        return

    # Create portfolio
    try:
        portfolio = manager.create_portfolio()
    except ValueError as e:
        # Handle portfolio validation errors (e.g., insufficient capital)
        error_msg = str(e)
        if "insufficient" in error_msg.lower() and "capital" in error_msg.lower():
            console.print("\n[bold red]❌ Portfolio Configuration Error[/bold red]\n")
            console.print(f"[red]{error_msg}[/red]\n")
            console.print("[dim]💡 Suggestions:[/dim]")
            console.print("[dim]  • Increase the initial_capital in your configuration[/dim]")
            console.print("[dim]  • Reduce the quantities of some assets[/dim]")
            console.print("[dim]  • Enable fractional shares: allow_fractional_shares: true[/dim]")
        else:
            console.print(f"\n[bold red]❌ Portfolio Error:[/bold red] [red]{error_msg}[/red]\n")
        raise typer.Exit(1) from None

    # Display portfolio summary and holdings via display layer
    display = ResultsDisplay(cli_manager.get_console())
    display.show_portfolio_summary(portfolio)
    display.show_portfolio_holdings(portfolio, mode=mode, data_fetcher=container.data_fetcher())

    # Run main processing through StockulaManager
    try:
        results = manager.run_main_processing(mode, portfolio)
    except Exception as e:
        # Handle processing errors with clean output
        error_msg = str(e)
        if "insufficient" in error_msg.lower() and "data" in error_msg.lower():
            console.print(f"\n[bold red]❌ Data Error:[/bold red] [red]{error_msg}[/red]")
            console.print("[dim]💡 Try adjusting the date range in your configuration[/dim]\n")
        elif "network" in error_msg.lower() or "connection" in error_msg.lower():
            console.print(f"\n[bold red]❌ Network Error:[/bold red] [red]{error_msg}[/red]")
            console.print("[dim]💡 Check your internet connection and try again[/dim]\n")
        else:
            console.print(f"\n[bold red]❌ Processing Error:[/bold red] [red]{error_msg}[/red]\n")
        raise typer.Exit(1) from None

    # Show current portfolio value for forecast mode
    if mode == "forecast":
        display = ResultsDisplay(cli_manager.get_console())
        display.show_portfolio_forecast_value(stockula_config, portfolio, results)

    # Output results
    output_format = output or stockula_config.output.get("format", "console")
    print_results(results, output_format, stockula_config, container, portfolio)

    # Show strategy-specific summaries after backtesting
    if mode in ["all", "backtest"] and "backtesting" in results:
        display = ResultsDisplay(cli_manager.get_console())
        display.show_strategy_summaries(manager, stockula_config, results)


@app.command()
def main(
    config: Annotated[str | None, typer.Option("--config", "-c", help="Path to configuration file (YAML)")] = None,
    ticker: Annotated[
        str | None, typer.Option("--ticker", "-t", help="Override ticker symbol (single ticker mode)")
    ] = None,
    mode: Annotated[Mode, typer.Option("--mode", "-m", help="Operation mode")] = Mode.ALL,
    output: Annotated[OutputFormat, typer.Option("--output", "-o", help="Output format")] = OutputFormat.CONSOLE,
    save_config_path: Annotated[
        str | None, typer.Option("--save-config", help="Save current configuration to file")
    ] = None,
    save_optimized_config: Annotated[
        str | None,
        typer.Option(
            "--save-optimized-config", help="Save optimized configuration to file (used with optimize-allocation mode)"
        ),
    ] = None,
    train_start: Annotated[str | None, typer.Option("--train-start", help="Training start date (YYYY-MM-DD)")] = None,
    train_end: Annotated[str | None, typer.Option("--train-end", help="Training end date (YYYY-MM-DD)")] = None,
    test_start: Annotated[str | None, typer.Option("--test-start", help="Testing start date (YYYY-MM-DD)")] = None,
    test_end: Annotated[str | None, typer.Option("--test-end", help="Testing end date (YYYY-MM-DD)")] = None,
):
    """
    Run Stockula trading analysis with various modes.

    Analyze stocks using technical indicators, backtesting strategies,
    and machine learning forecasts. Supports portfolio optimization
    and multiple output formats.
    """
    run_stockula(
        config=config,
        ticker=ticker,
        mode=mode.value,
        output=output.value,
        save_config_path=save_config_path,
        save_optimized_config=save_optimized_config,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )


def handle_validation_error(error: ValidationError, config_path: str) -> None:
    """Handle validation errors with clean, user-friendly output.

    Args:
        error: The validation error
        config_path: Path to the configuration file
    """
    console.print("\n[bold red]Configuration Validation Error[/bold red]\n")
    console.print(f"Failed to load configuration from: [cyan]{config_path}[/cyan]\n")

    # Parse and display errors in a user-friendly format
    errors = []
    for err in error.errors():
        location = " → ".join(str(loc) for loc in err["loc"])
        message = err["msg"]

        # Clean up common error messages
        if "test_start_date must be before test_end_date" in message:
            errors.append("[yellow]Date Range Error:[/yellow] Test end date is before test start date")
        elif "train_start_date must be before train_end_date" in message:
            errors.append("[yellow]Date Range Error:[/yellow] Train end date is before train start date")
        elif "train_end_date must be before or equal to test_start_date" in message:
            errors.append("[yellow]Date Sequence Error:[/yellow] Training period must end before test period begins")
        else:
            errors.append(f"[yellow]{location}:[/yellow] {message}")

    # Display errors in a panel
    error_text = "\n".join(f"  • {err}" for err in errors)
    console.print(
        Panel(
            error_text,
            title="[bold]Validation Issues[/bold]",
            border_style="red",
            padding=(1, 2),
        )
    )

    # Show the problematic configuration section if possible
    if "backtest_optimization" in str(error):
        console.print("\n[dim]Check your backtest_optimization section in the config file.[/dim]")
        console.print("[dim]Ensure that:[/dim]")
        console.print("[dim]  • All dates are in YYYY-MM-DD format[/dim]")
        console.print("[dim]  • train_start_date < train_end_date[/dim]")
        console.print("[dim]  • test_start_date < test_end_date[/dim]")
        console.print("[dim]  • train_end_date ≤ test_start_date[/dim]")

    console.print()
    raise typer.Exit(1)


def parse_test_args():
    """Parse command line arguments for test compatibility."""
    import sys

    args = sys.argv[1:]  # Skip program name
    kwargs = {}

    # Parse arguments
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--config" or arg == "-c":
            kwargs["config"] = args[i + 1]
            i += 2
        elif arg == "--ticker" or arg == "-t":
            kwargs["ticker"] = args[i + 1]
            i += 2
        elif arg == "--mode" or arg == "-m":
            kwargs["mode"] = args[i + 1]
            i += 2
        elif arg == "--output" or arg == "-o":
            kwargs["output"] = args[i + 1]
            i += 2
        elif arg == "--save-config":
            kwargs["save_config_path"] = args[i + 1]
            i += 2
        elif arg == "--save-optimized-config":
            kwargs["save_optimized_config"] = args[i + 1]
            i += 2
        elif arg == "--train-start":
            kwargs["train_start"] = args[i + 1]
            i += 2
        elif arg == "--train-end":
            kwargs["train_end"] = args[i + 1]
            i += 2
        elif arg == "--test-start":
            kwargs["test_start"] = args[i + 1]
            i += 2
        elif arg == "--test-end":
            kwargs["test_end"] = args[i + 1]
            i += 2
        else:
            i += 1

    return kwargs
