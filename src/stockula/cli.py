"""Command-line interface for Stockula."""

from enum import Enum
from typing import Annotated, Any, cast

import typer
from pydantic import ValidationError

from .cli_manager import cli_manager
from .config import StockulaConfig, TickerConfig
from .config.settings import save_config
from .container import Container, create_container
from .display import ResultsDisplay
from .domain import Portfolio
from .manager import StockulaManager
from .pipeline import StockulaPipeline
from .utils import DateOverrides, PipelineConfig, RunConfig, SavePaths, error_handler

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


def print_results(
    results: dict[str, Any],
    output_format: str = "console",
    config: StockulaConfig | None = None,
    container: Container | None = None,
    portfolio: Portfolio | None = None,
) -> None:
    """Print results in specified format using ResultsDisplay.

    Args:
        results: Results dictionary
        output_format: Output format (console, json)
        config: Optional configuration object for portfolio composition
        container: Optional DI container for fetching data
        portfolio: Optional portfolio instance for forecast display
    """
    display = ResultsDisplay()
    display.print_results(results, output_format, config, container, portfolio)


def run_stockula(run_config: RunConfig) -> int | None:
    """Core logic for running Stockula with simplified parameters."""
    # Initialize DI container first
    container = create_container(run_config.config_file)

    # Load configuration - the container will handle this
    try:
        stockula_config = container.stockula_config()
    except ValidationError as e:
        # Use the provided config path or default
        config_path = run_config.config_file or ".stockula.yaml"
        error_handler.handle_validation_error(e, config_path)

    # Set up logging based on configuration
    from .interfaces import ILoggingManager
    from .main import setup_logging

    setup_logging(stockula_config, logging_manager=cast(ILoggingManager, container.logging_manager()))

    # Override ticker if provided
    if run_config.ticker:
        _apply_single_ticker_mode(stockula_config, run_config.ticker)

    # Override date ranges if provided
    if run_config.date_overrides is not None:
        run_config.date_overrides.apply_to_config(stockula_config)

    # Create manager instance
    manager = StockulaManager(stockula_config, container, console)

    # Handle optimize-allocation mode early (before portfolio creation)
    if run_config.is_optimization_mode():
        save_path = (
            run_config.save_paths.get_save_path_for_optimization() if run_config.save_paths is not None else None
        )
        return manager.run_optimize_allocation(save_path)

    # Save configuration if requested (for non-optimize-allocation modes)
    if run_config.should_save_config():
        if run_config.save_paths is not None and run_config.save_paths.config_path:
            save_config(stockula_config, run_config.save_paths.config_path)
            print(f"Configuration saved to {run_config.save_paths.config_path}")
        else:
            print("Error: No config path specified for saving")
        return None

    # Create portfolio
    try:
        portfolio = manager.create_portfolio()
    except ValueError as e:
        error_handler.handle_portfolio_error(e)

    # Display portfolio summary and holdings via display layer
    display = ResultsDisplay()
    display.show_portfolio_summary(portfolio)
    display.show_portfolio_holdings(portfolio, mode=run_config.mode, data_fetcher=container.data_fetcher())

    # Run main processing through StockulaManager
    try:
        results = manager.run_main_processing(run_config.mode, portfolio)
    except Exception as e:
        error_handler.handle_processing_error(e)

    # Handle mode-specific display and output
    _handle_results_display(run_config, stockula_config, portfolio, results, container, manager)
    return None


def _apply_single_ticker_mode(stockula_config: StockulaConfig, ticker: str) -> None:
    """Apply single ticker mode configuration."""
    stockula_config.portfolio.tickers = [TickerConfig(symbol=ticker, quantity=1.0)]
    # Disable auto-allocation for single ticker mode since we don't have categories
    stockula_config.portfolio.auto_allocate = False
    stockula_config.portfolio.dynamic_allocation = False
    stockula_config.portfolio.allocation_method = "equal_weight"
    # Allow 100% position for single ticker mode
    stockula_config.portfolio.max_position_size = 100.0


def _handle_results_display(
    run_config: RunConfig,
    stockula_config: StockulaConfig,
    portfolio: Portfolio,
    results: dict[str, Any],
    container: Container,
    manager: StockulaManager,
) -> None:
    """Handle mode-specific display and output operations."""
    # Show current portfolio value for forecast mode
    if run_config.mode == "forecast":
        display = ResultsDisplay()
        display.show_portfolio_forecast_value(stockula_config, portfolio, results)

    # Output results
    output_format = run_config.output or stockula_config.output.get("format", "console")
    print_results(results, output_format, stockula_config, container, portfolio)

    # Show strategy-specific summaries after backtesting
    if run_config.mode in ["all", "backtest"] and "backtesting" in results:
        display = ResultsDisplay()
        display.show_strategy_summaries(manager, stockula_config, results)


def _run_pipeline_with_config(pipeline: StockulaPipeline, pipeline_config: PipelineConfig) -> dict[str, Any]:
    """Run pipeline with given configuration and return results."""
    if pipeline_config.skip_optimization:
        # Run backtest only
        return pipeline.run_backtest()
    elif pipeline_config.skip_backtest:
        # Run optimization only
        _, results = pipeline.run_optimization()
        return results
    else:
        # Run full pipeline (optimization + backtest)
        return pipeline.run_full_pipeline()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
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
) -> None:
    """
    Run Stockula trading analysis with various modes.

    Analyze stocks using technical indicators, backtesting strategies,
    and machine learning forecasts. Supports portfolio optimization
    and multiple output formats.
    """
    # If a subcommand is invoked, don't run the main logic
    if ctx.invoked_subcommand is not None:
        return

    # Otherwise, run the main stockula logic
    run_config = RunConfig(
        config_file=config,
        ticker=ticker,
        mode=mode.value,
        output=output.value,
        save_paths=SavePaths(config_path=save_config_path, optimized_config_path=save_optimized_config),
        date_overrides=DateOverrides(
            train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end
        ),
    )
    run_stockula(run_config)


@app.command(name="pipeline")
def pipeline_command(
    base_config: Annotated[
        str,
        typer.Option(
            "--base-config",
            "-b",
            help="Path to base configuration file",
        ),
    ] = ".stockula.yaml",
    optimized_config: Annotated[
        str | None,
        typer.Option(
            "--optimized-config",
            "-o",
            help="Path to save optimized configuration",
        ),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option(
            "--output",
            help="Path to save pipeline results (json/yaml/csv)",
        ),
    ] = None,
    skip_optimization: Annotated[
        bool,
        typer.Option(
            "--skip-optimization",
            help="Skip optimization and use existing config",
        ),
    ] = False,
    skip_backtest: Annotated[
        bool,
        typer.Option(
            "--skip-backtest",
            help="Skip backtesting after optimization",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose output",
        ),
    ] = False,
) -> None:
    """
    Run the complete Stockula pipeline: optimization followed by backtesting.

    This command orchestrates the full workflow:
    1. Load base configuration
    2. Run portfolio optimization
    3. Save optimized configuration
    4. Run backtesting with optimized allocations
    5. Compare and report results

    Examples:
        # Run full pipeline
        stockula pipeline --base-config .stockula.yaml --optimized-config .stockula-opt.yaml

        # Run with results output
        stockula pipeline -b config.yaml -o optimized.yaml --output results.json

        # Skip optimization and use existing optimized config
        stockula pipeline -b optimized.yaml --skip-optimization
    """
    from rich.console import Console

    console = Console()

    try:
        pipeline_config = PipelineConfig(
            base_config=base_config,
            optimized_config=optimized_config,
            output=output,
            skip_optimization=skip_optimization,
            skip_backtest=skip_backtest,
            verbose=verbose,
        )

        # Validate configuration
        if pipeline_config.should_skip_both():
            console.print("[red]Error: Cannot skip both optimization and backtesting[/red]")
            raise typer.Exit(1)

        # Create and run pipeline
        pipeline = StockulaPipeline(
            base_config_path=pipeline_config.base_config,
            verbose=pipeline_config.verbose,
            console=console,
        )

        _run_pipeline_with_config(pipeline, pipeline_config)

        # Save results if output path provided
        if pipeline_config.output:
            pipeline.save_results(pipeline_config.output, format=pipeline_config.get_output_format())

        console.print("[bold green]✨ Pipeline completed successfully![/bold green]")

    except ValidationError as e:
        error_handler.handle_validation_error(e, base_config)
    except KeyboardInterrupt:
        console.print("[yellow]Pipeline cancelled by user[/yellow]")
    except (FileNotFoundError, Exception) as e:
        error_handler.handle_pipeline_error(e, verbose=verbose)


def parse_test_args() -> dict[str, str]:
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
