"""Stockula main entry point."""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from dependency_injector.wiring import inject, Provide
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)
from rich.table import Table
from rich.panel import Panel

from .container import create_container, Container
from .technical_analysis import TechnicalIndicators
from .backtesting import (
    SMACrossStrategy,
    RSIStrategy,
    MACDStrategy,
    DoubleEMACrossStrategy,
    TripleEMACrossStrategy,
    TRIMACrossStrategy,
    VIDYAStrategy,
    KAMAStrategy,
    FRAMAStrategy,
    VAMAStrategy,
    KaufmanEfficiencyStrategy,
)
from .config import StockulaConfig
from .config.models import (
    BacktestResult,
    StrategyBacktestSummary,
    PortfolioBacktestResults,
)
from .domain import Category
from .interfaces import (
    IDataFetcher,
    IBacktestRunner,
    IStockForecaster,
    ILoggingManager,
)

# Global logging manager and console instances
log_manager: Optional[ILoggingManager] = None
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


def get_strategy_class(strategy_name: str):
    """Get strategy class by name."""
    strategies = {
        "smacross": SMACrossStrategy,
        "rsi": RSIStrategy,
        "macd": MACDStrategy,
        "doubleemacross": DoubleEMACrossStrategy,
        "tripleemacross": TripleEMACrossStrategy,
        "trimacross": TRIMACrossStrategy,
        "vidya": VIDYAStrategy,
        "kama": KAMAStrategy,
        "frama": FRAMAStrategy,
        "vama": VAMAStrategy,
        "er": KaufmanEfficiencyStrategy,
    }
    return strategies.get(strategy_name.lower())


@inject
def run_technical_analysis(
    ticker: str,
    config: StockulaConfig,
    data_fetcher: IDataFetcher = Provide[Container.data_fetcher],
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Run technical analysis for a ticker.

    Args:
        ticker: Stock symbol
        config: Configuration object
        data_fetcher: Injected data fetcher
        show_progress: Whether to show progress bars

    Returns:
        Dictionary with indicator results
    """
    data = data_fetcher.get_stock_data(
        ticker,
        start=config.data.start_date.strftime("%Y-%m-%d")
        if config.data.start_date
        else None,
        end=config.data.end_date.strftime("%Y-%m-%d") if config.data.end_date else None,
        interval=config.data.interval,
    )

    ta = TechnicalIndicators(data)
    results = {"ticker": ticker, "indicators": {}}
    ta_config = config.technical_analysis

    # Count total indicators to calculate
    total_indicators = 0
    if "sma" in ta_config.indicators:
        total_indicators += len(ta_config.sma_periods)
    if "ema" in ta_config.indicators:
        total_indicators += len(ta_config.ema_periods)
    if "rsi" in ta_config.indicators:
        total_indicators += 1
    if "macd" in ta_config.indicators:
        total_indicators += 1
    if "bbands" in ta_config.indicators:
        total_indicators += 1
    if "atr" in ta_config.indicators:
        total_indicators += 1
    if "adx" in ta_config.indicators:
        total_indicators += 1

    if show_progress and total_indicators > 0:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
            transient=True,  # Remove progress bar when done
        ) as progress:
            task = progress.add_task(
                f"[cyan]Computing {total_indicators} technical indicators for {ticker}...",
                total=total_indicators,
            )

            current_step = 0

            if "sma" in ta_config.indicators:
                for period in ta_config.sma_periods:
                    progress.update(
                        task,
                        description=f"[cyan]Computing SMA({period}) for {ticker}...",
                    )
                    results["indicators"][f"SMA_{period}"] = ta.sma(period).iloc[-1]
                    current_step += 1
                    progress.advance(task)

            if "ema" in ta_config.indicators:
                for period in ta_config.ema_periods:
                    progress.update(
                        task,
                        description=f"[cyan]Computing EMA({period}) for {ticker}...",
                    )
                    results["indicators"][f"EMA_{period}"] = ta.ema(period).iloc[-1]
                    current_step += 1
                    progress.advance(task)

            if "rsi" in ta_config.indicators:
                progress.update(
                    task, description=f"[cyan]Computing RSI for {ticker}..."
                )
                results["indicators"]["RSI"] = ta.rsi(ta_config.rsi_period).iloc[-1]
                current_step += 1
                progress.advance(task)

            if "macd" in ta_config.indicators:
                progress.update(
                    task, description=f"[cyan]Computing MACD for {ticker}..."
                )
                macd_data = ta.macd(**ta_config.macd_params)
                results["indicators"]["MACD"] = macd_data.iloc[-1].to_dict()
                current_step += 1
                progress.advance(task)

            if "bbands" in ta_config.indicators:
                progress.update(
                    task, description=f"[cyan]Computing Bollinger Bands for {ticker}..."
                )
                bbands_data = ta.bbands(**ta_config.bbands_params)
                results["indicators"]["BBands"] = bbands_data.iloc[-1].to_dict()
                current_step += 1
                progress.advance(task)

            if "atr" in ta_config.indicators:
                progress.update(
                    task, description=f"[cyan]Computing ATR for {ticker}..."
                )
                results["indicators"]["ATR"] = ta.atr(ta_config.atr_period).iloc[-1]
                current_step += 1
                progress.advance(task)

            if "adx" in ta_config.indicators:
                progress.update(
                    task, description=f"[cyan]Computing ADX for {ticker}..."
                )
                results["indicators"]["ADX"] = ta.adx(14).iloc[-1]
                current_step += 1
                progress.advance(task)
    else:
        # Run without progress bars (for single indicators or when disabled)
        if "sma" in ta_config.indicators:
            for period in ta_config.sma_periods:
                results["indicators"][f"SMA_{period}"] = ta.sma(period).iloc[-1]

        if "ema" in ta_config.indicators:
            for period in ta_config.ema_periods:
                results["indicators"][f"EMA_{period}"] = ta.ema(period).iloc[-1]

        if "rsi" in ta_config.indicators:
            results["indicators"]["RSI"] = ta.rsi(ta_config.rsi_period).iloc[-1]

        if "macd" in ta_config.indicators:
            macd_data = ta.macd(**ta_config.macd_params)
            results["indicators"]["MACD"] = macd_data.iloc[-1].to_dict()

        if "bbands" in ta_config.indicators:
            bbands_data = ta.bbands(**ta_config.bbands_params)
            results["indicators"]["BBands"] = bbands_data.iloc[-1].to_dict()

        if "atr" in ta_config.indicators:
            results["indicators"]["ATR"] = ta.atr(ta_config.atr_period).iloc[-1]

        if "adx" in ta_config.indicators:
            results["indicators"]["ADX"] = ta.adx(14).iloc[-1]

    return results


@inject
def run_backtest(
    ticker: str,
    config: StockulaConfig,
    backtest_runner: IBacktestRunner = Provide[Container.backtest_runner],
) -> List[Dict[str, Any]]:
    """Run backtesting for a ticker.

    Args:
        ticker: Stock symbol
        config: Configuration object
        backtest_runner: Injected backtest runner

    Returns:
        List of backtest results
    """
    runner = backtest_runner

    results = []

    for strategy_config in config.backtest.strategies:
        strategy_class = get_strategy_class(strategy_config.name)
        if not strategy_class:
            print(f"Warning: Unknown strategy '{strategy_config.name}'")
            continue

        # Set strategy parameters if provided
        if strategy_config.parameters:
            # Set class attributes from parameters
            for key, value in strategy_config.parameters.items():
                setattr(strategy_class, key, value)

        try:
            backtest_result = runner.run_from_symbol(
                ticker,
                strategy_class,
                start_date=config.data.start_date.strftime("%Y-%m-%d")
                if config.data.start_date
                else None,
                end_date=config.data.end_date.strftime("%Y-%m-%d")
                if config.data.end_date
                else None,
            )

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

            results.append(result_entry)
        except Exception as e:
            print(f"Error backtesting {strategy_config.name} on {ticker}: {e}")

    return results


@inject
def run_forecast(
    ticker: str,
    config: StockulaConfig,
    stock_forecaster: IStockForecaster = Provide[Container.stock_forecaster],
) -> Dict[str, Any]:
    """Run forecasting for a ticker.

    Args:
        ticker: Stock symbol
        config: Configuration object
        stock_forecaster: Injected stock forecaster

    Returns:
        Dictionary with forecast results
    """
    log_manager.info(
        f"\nForecasting {ticker} for {config.forecast.forecast_length} days..."
    )

    forecaster = stock_forecaster

    try:
        predictions = forecaster.forecast_from_symbol(
            ticker,
            start_date=config.data.start_date.strftime("%Y-%m-%d")
            if config.data.start_date
            else None,
            end_date=config.data.end_date.strftime("%Y-%m-%d")
            if config.data.end_date
            else None,
            model_list=config.forecast.model_list,
            ensemble=config.forecast.ensemble,
            max_generations=config.forecast.max_generations,
        )

        model_info = forecaster.get_best_model()

        log_manager.info(
            f"Forecast completed for {ticker} using {model_info['model_name']}"
        )

        return {
            "ticker": ticker,
            "current_price": predictions["forecast"].iloc[0],
            "forecast_price": predictions["forecast"].iloc[-1],
            "lower_bound": predictions["lower_bound"].iloc[-1],
            "upper_bound": predictions["upper_bound"].iloc[-1],
            "forecast_length": config.forecast.forecast_length,
            "best_model": model_info["model_name"],
            "model_params": model_info.get("model_params", {}),
        }
    except KeyboardInterrupt:
        log_manager.warning(f"Forecast for {ticker} interrupted by user")
        return {"ticker": ticker, "error": "Interrupted by user"}
    except Exception as e:
        log_manager.error(f"Error forecasting {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}


def save_detailed_report(
    strategy_name: str,
    strategy_results: List[Dict],
    results: Dict[str, Any],
    config: StockulaConfig,
    portfolio_results: Optional[PortfolioBacktestResults] = None,
) -> str:
    """Save detailed strategy report to file.

    Args:
        strategy_name: Name of the strategy
        strategy_results: List of backtest results for this strategy
        results: Overall results dictionary
        config: Configuration object

    Returns:
        Path to the saved report file
    """
    from pathlib import Path
    import json

    # Create reports directory if it doesn't exist
    reports_dir = Path(config.output.get("results_dir", "./results")) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = reports_dir / f"strategy_report_{strategy_name}_{timestamp}.json"

    # Prepare detailed report data
    report_data = {
        "strategy": strategy_name,
        "timestamp": timestamp,
        "date_range": {
            "start": config.data.start_date.strftime("%Y-%m-%d")
            if config.data.start_date
            else None,
            "end": config.data.end_date.strftime("%Y-%m-%d")
            if config.data.end_date
            else None,
        },
        "portfolio": {
            "initial_value": results.get("initial_portfolio_value", 0),
            "initial_capital": results.get("initial_capital", 0),
        },
        "broker_config": {
            "name": config.backtest.broker_config.name
            if config.backtest.broker_config
            else "legacy",
            "commission_type": config.backtest.broker_config.commission_type
            if config.backtest.broker_config
            else "percentage",
            "commission_value": config.backtest.broker_config.commission_value
            if config.backtest.broker_config
            else config.backtest.commission,
            "min_commission": config.backtest.broker_config.min_commission
            if config.backtest.broker_config
            else None,
            "regulatory_fees": config.backtest.broker_config.regulatory_fees
            if config.backtest.broker_config
            else 0,
        },
        "detailed_results": strategy_results,
        "summary": {
            "total_trades": sum(r.get("num_trades", 0) for r in strategy_results),
            "winning_stocks": sum(
                1 for r in strategy_results if r.get("return_pct", 0) > 0
            ),
            "losing_stocks": sum(
                1 for r in strategy_results if r.get("return_pct", 0) < 0
            ),
            "average_return": sum(r.get("return_pct", 0) for r in strategy_results)
            / len(strategy_results)
            if strategy_results
            else 0,
            "average_sharpe": sum(r.get("sharpe_ratio", 0) for r in strategy_results)
            / len(strategy_results)
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


def create_portfolio_backtest_results(
    results: Dict[str, Any],
    config: StockulaConfig,
    strategy_results: Dict[str, List[Dict]],
) -> PortfolioBacktestResults:
    """Create structured backtest results.

    Args:
        results: Main results dictionary with initial values
        config: Configuration object
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
            sum(r.sharpe_ratio for r in detailed_results) / len(detailed_results)
            if detailed_results
            else 0
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
    broker_config = {}
    if config.backtest.broker_config:
        broker_config = {
            "name": config.backtest.broker_config.name,
            "commission_type": config.backtest.broker_config.commission_type,
            "commission_value": config.backtest.broker_config.commission_value,
            "min_commission": config.backtest.broker_config.min_commission,
            "regulatory_fees": config.backtest.broker_config.regulatory_fees,
            "exchange_fees": getattr(config.backtest.broker_config, "exchange_fees", 0),
        }
    else:
        broker_config = {
            "name": "legacy",
            "commission_type": "percentage",
            "commission_value": config.backtest.commission,
            "min_commission": None,
            "regulatory_fees": 0,
            "exchange_fees": 0,
        }

    # Create portfolio results
    portfolio_results = PortfolioBacktestResults(
        initial_portfolio_value=results.get("initial_portfolio_value", 0),
        initial_capital=results.get("initial_capital", 0),
        date_range={
            "start": config.data.start_date.strftime("%Y-%m-%d")
            if config.data.start_date
            else None,
            "end": config.data.end_date.strftime("%Y-%m-%d")
            if config.data.end_date
            else None,
        },
        broker_config=broker_config,
        strategy_summaries=strategy_summaries,
    )

    return portfolio_results


def print_results(
    results: Dict[str, Any], output_format: str = "console", config=None, container=None
):
    """Print results in specified format.

    Args:
        results: Results dictionary
        output_format: Output format (console, json)
        config: Optional configuration object for portfolio composition
        container: Optional DI container for fetching data
    """
    if output_format == "json":
        console.print_json(json.dumps(results, indent=2, default=str))
    else:
        # Console output with Rich formatting

        if "technical_analysis" in results:
            console.print(
                "\n[bold blue]Technical Analysis Results[/bold blue]", style="bold"
            )

            for ta_result in results["technical_analysis"]:
                table = Table(title=f"Technical Analysis - {ta_result['ticker']}")
                table.add_column("Indicator", style="cyan", no_wrap=True)
                table.add_column("Value", style="magenta")

                for indicator, value in ta_result["indicators"].items():
                    if isinstance(value, dict):
                        for k, v in value.items():
                            formatted_value = (
                                f"{v:.2f}" if isinstance(v, (int, float)) else str(v)
                            )
                            table.add_row(f"{indicator} - {k}", formatted_value)
                    else:
                        formatted_value = (
                            f"{value:.2f}"
                            if isinstance(value, (int, float))
                            else str(value)
                        )
                        table.add_row(indicator, formatted_value)

                console.print(table)

        if "backtesting" in results:
            # Check if we have multiple strategies
            strategies = set(b["strategy"] for b in results["backtesting"])

            # Display general portfolio information
            console.print("\n[bold green]=== Backtesting Results ===[/bold green]")

            # Create portfolio information panel
            portfolio_info = []

            # Extract portfolio info from results metadata
            if "portfolio" in results:
                portfolio_data = results["portfolio"]
                if "initial_capital" in portfolio_data:
                    portfolio_info.append(
                        f"[cyan]Initial Capital:[/cyan] ${portfolio_data['initial_capital']:,.2f}"
                    )
                if "start" in portfolio_data and portfolio_data["start"]:
                    portfolio_info.append(
                        f"[cyan]Start Date:[/cyan] {portfolio_data['start']}"
                    )
                if "end" in portfolio_data and portfolio_data["end"]:
                    portfolio_info.append(
                        f"[cyan]End Date:[/cyan] {portfolio_data['end']}"
                    )

            # If portfolio info not in metadata, try to extract from backtest results
            if not portfolio_info and results.get("backtesting"):
                # Get portfolio information from the first backtest result
                first_backtest = (
                    results["backtesting"][0] if results["backtesting"] else {}
                )

                if "initial_cash" in first_backtest:
                    portfolio_info.append(
                        f"[cyan]Initial Capital:[/cyan] ${first_backtest['initial_cash']:,.2f}"
                    )
                if "start_date" in first_backtest:
                    portfolio_info.append(
                        f"[cyan]Start Date:[/cyan] {first_backtest['start_date']}"
                    )
                if "end_date" in first_backtest:
                    portfolio_info.append(
                        f"[cyan]End Date:[/cyan] {first_backtest['end_date']}"
                    )
                if "trading_days" in first_backtest:
                    portfolio_info.append(
                        f"[cyan]Trading Days:[/cyan] {first_backtest['trading_days']:,}"
                    )
                if "calendar_days" in first_backtest:
                    portfolio_info.append(
                        f"[cyan]Calendar Days:[/cyan] {first_backtest['calendar_days']:,}"
                    )

                # Fallback: Look for cash/initial capital in other locations
                if not portfolio_info:
                    initial_capital = results.get("initial_capital")
                    if initial_capital:
                        portfolio_info.append(
                            f"[cyan]Initial Capital:[/cyan] ${initial_capital:,.2f}"
                        )

                    # Add date information if available
                    start_date = results.get("start_date") or results.get("start")
                    end_date = results.get("end_date") or results.get("end")
                    if start_date:
                        portfolio_info.append(f"[cyan]Start Date:[/cyan] {start_date}")
                    if end_date:
                        portfolio_info.append(f"[cyan]End Date:[/cyan] {end_date}")

            # Display portfolio information if available
            if portfolio_info:
                console.print("[bold blue]Portfolio Information:[/bold blue]")
                for info in portfolio_info:
                    console.print(f"  {info}")
                console.print()  # Add blank line

            # Display portfolio composition table (only if config and container are provided)
            if config and container:
                table = Table(title="Portfolio Composition")
                table.add_column("Ticker", style="cyan", no_wrap=True)
                table.add_column("Category", style="yellow")
                table.add_column("Quantity", style="white", justify="right")
                table.add_column("Allocation %", style="green", justify="right")
                table.add_column("Value", style="blue", justify="right")
                table.add_column("Status", style="magenta")

                # Get portfolio composition information
                portfolio = container.domain_factory().create_portfolio(config)
                all_assets = portfolio.get_all_assets()

                # Get hold-only categories from config
                hold_only_category_names = set(config.backtest.hold_only_categories)
                hold_only_categories = set()
                for category_name in hold_only_category_names:
                    try:
                        hold_only_categories.add(Category[category_name])
                    except KeyError:
                        pass  # Skip unknown categories

                # Get current prices for calculation
                fetcher = container.data_fetcher()
                symbols = [asset.symbol for asset in all_assets]
                try:
                    current_prices = fetcher.get_current_prices(
                        symbols, show_progress=False
                    )
                    total_portfolio_value = sum(
                        asset.quantity * current_prices.get(asset.symbol, 0)
                        for asset in all_assets
                    )

                    for asset in all_assets:
                        current_price = current_prices.get(asset.symbol, 0)
                        asset_value = asset.quantity * current_price
                        allocation_pct = (
                            (asset_value / total_portfolio_value * 100)
                            if total_portfolio_value > 0
                            else 0
                        )

                        # Determine status
                        status = (
                            "Hold Only"
                            if asset.category in hold_only_categories
                            else "Tradeable"
                        )
                        status_color = "yellow" if status == "Hold Only" else "green"

                        table.add_row(
                            asset.symbol,
                            asset.category.name
                            if hasattr(asset.category, "name")
                            else str(asset.category),
                            f"{asset.quantity:.2f}",
                            f"{allocation_pct:.1f}%",
                            f"${asset_value:,.2f}",
                            f"[{status_color}]{status}[/{status_color}]",
                        )
                except Exception as e:
                    # Fallback if we can't get prices
                    for asset in all_assets:
                        status = (
                            "Hold Only"
                            if asset.category in hold_only_categories
                            else "Tradeable"
                        )
                        status_color = "yellow" if status == "Hold Only" else "green"

                        table.add_row(
                            asset.symbol,
                            asset.category.name
                            if hasattr(asset.category, "name")
                            else str(asset.category),
                            f"{asset.quantity:.2f}",
                            "N/A",
                            "N/A",
                            f"[{status_color}]{status}[/{status_color}]",
                        )

                console.print(table)
                console.print()  # Add blank line

            # Show ticker-level backtest results in a table
            console.print("\n[bold green]Ticker-Level Backtest Results[/bold green]")

            table = Table(title="Ticker-Level Backtest Results")
            table.add_column("Ticker", style="cyan", no_wrap=True)
            table.add_column("Strategy", style="yellow", no_wrap=True)
            table.add_column("Return", style="green", justify="right")
            table.add_column("Sharpe Ratio", style="blue", justify="right")
            table.add_column("Max Drawdown", style="red", justify="right")
            table.add_column("Trades", style="white", justify="right")
            table.add_column("Win Rate", style="magenta", justify="right")

            for backtest in results["backtesting"]:
                return_str = f"{backtest['return_pct']:+.2f}%"
                sharpe_str = f"{backtest['sharpe_ratio']:.2f}"
                drawdown_str = f"{backtest['max_drawdown_pct']:.2f}%"
                trades_str = str(backtest["num_trades"])

                if backtest["win_rate"] is None:
                    win_rate_str = "N/A"
                else:
                    win_rate_str = f"{backtest['win_rate']:.1f}%"

                table.add_row(
                    backtest["ticker"],
                    backtest["strategy"].upper(),
                    return_str,
                    sharpe_str,
                    drawdown_str,
                    trades_str,
                    win_rate_str,
                )

            console.print(table)
            console.print()  # Add blank line

            # Show summary message about strategies and stocks
            console.print(
                f"Running [bold]{len(strategies)}[/bold] strategies across [bold]{len(set(b['ticker'] for b in results['backtesting']))}[/bold] stocks..."
            )
            if len(strategies) > 1:
                console.print("Detailed results will be shown per strategy below.")

        if "forecasting" in results:
            console.print("\n[bold purple]=== Forecasting Results ===[/bold purple]")

            table = Table(title="Price Forecasts")
            table.add_column("Ticker", style="cyan", no_wrap=True)
            table.add_column("Current Price", style="white", justify="right")
            table.add_column("Forecast Price", style="green", justify="right")
            table.add_column("Confidence Range", style="yellow", justify="center")
            table.add_column("Best Model", style="blue")

            for forecast in results["forecasting"]:
                if "error" in forecast:
                    table.add_row(
                        forecast["ticker"],
                        "[red]Error[/red]",
                        "[red]Error[/red]",
                        "[red]Error[/red]",
                        f"[red]{forecast['error']}[/red]",
                    )
                else:
                    current_price = forecast["current_price"]
                    forecast_price = forecast["forecast_price"]

                    # Color code forecast based on direction
                    forecast_color = (
                        "green"
                        if forecast_price > current_price
                        else "red"
                        if forecast_price < current_price
                        else "white"
                    )
                    forecast_str = (
                        f"[{forecast_color}]${forecast_price:.2f}[/{forecast_color}]"
                    )

                    table.add_row(
                        forecast["ticker"],
                        f"${current_price:.2f}",
                        forecast_str,
                        f"${forecast['lower_bound']:.2f} - ${forecast['upper_bound']:.2f}",
                        forecast["best_model"],
                    )

            console.print(table)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Stockula Trading Platform")
    parser.add_argument(
        "--config", "-c", type=str, help="Path to configuration file (YAML)"
    )
    parser.add_argument(
        "--ticker", "-t", type=str, help="Override ticker symbol (single ticker mode)"
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["all", "ta", "backtest", "forecast"],
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
    parser.add_argument(
        "--save-config", type=str, help="Save current configuration to file"
    )

    args = parser.parse_args()

    # Initialize DI container first
    container = create_container(args.config)

    # Load configuration - the container will handle this
    config = container.stockula_config()

    # Set up logging based on configuration
    setup_logging(config, logging_manager=container.logging_manager())

    # Override ticker if provided
    if args.ticker:
        from .config import TickerConfig

        config.portfolio.tickers = [TickerConfig(symbol=args.ticker, quantity=1.0)]

    # Save configuration if requested
    if args.save_config:
        from .config.settings import save_config

        save_config(config, args.save_config)
        print(f"Configuration saved to {args.save_config}")
        return

    # Get injected services from container
    factory = container.domain_factory()
    portfolio = factory.create_portfolio(config)

    # Display portfolio summary with Rich
    portfolio_table = Table(title="Portfolio Summary")
    portfolio_table.add_column("Property", style="cyan", no_wrap=True)
    portfolio_table.add_column("Value", style="white")

    portfolio_table.add_row("Name", portfolio.name)
    portfolio_table.add_row("Initial Capital", f"${portfolio.initial_capital:,.2f}")
    portfolio_table.add_row("Total Assets", str(len(portfolio.get_all_assets())))
    portfolio_table.add_row("Allocation Method", portfolio.allocation_method)

    console.print(portfolio_table)

    # Get portfolio value at start of backtest period
    fetcher = container.data_fetcher()
    symbols = [asset.symbol for asset in portfolio.get_all_assets()]

    # Get prices at the start date if backtesting
    if args.mode in ["all", "backtest"] and config.data.start_date:
        log_manager.debug(
            f"\nFetching prices at start date ({config.data.start_date})..."
        )
        start_date_str = config.data.start_date.strftime("%Y-%m-%d")
        # Fetch one day of data at the start date to get opening prices
        start_prices = {}
        for symbol in symbols:
            try:
                data = fetcher.get_stock_data(
                    symbol, start=start_date_str, end=start_date_str
                )
                if not data.empty:
                    start_prices[symbol] = data["Close"].iloc[0]
                else:
                    # If no data on exact date, get the next available date
                    end_date = (config.data.start_date + timedelta(days=7)).strftime(
                        "%Y-%m-%d"
                    )
                    data = fetcher.get_stock_data(
                        symbol, start=start_date_str, end=end_date
                    )
                    if not data.empty:
                        start_prices[symbol] = data["Close"].iloc[0]
            except Exception as e:
                log_manager.warning(f"Could not get start price for {symbol}: {e}")

        initial_portfolio_value = portfolio.get_portfolio_value(start_prices)
        log_manager.info(
            f"\nPortfolio Value at Start Date: ${initial_portfolio_value:,.2f}"
        )
    else:
        log_manager.debug("\nFetching current prices...")
        current_prices = fetcher.get_current_prices(symbols, show_progress=True)
        initial_portfolio_value = portfolio.get_portfolio_value(current_prices)

        # Show portfolio value in a nice table
        portfolio_value_table = Table(title="Current Portfolio Value")
        portfolio_value_table.add_column("Metric", style="cyan", no_wrap=True)
        portfolio_value_table.add_column("Value", style="green")
        portfolio_value_table.add_row(
            "Current Portfolio Value", f"${initial_portfolio_value:,.2f}"
        )
        console.print(portfolio_value_table)

    # Calculate returns (always needed, not just for logging)
    initial_return = initial_portfolio_value - portfolio.initial_capital
    initial_return_pct = (initial_return / portfolio.initial_capital) * 100

    log_manager.info(f"Initial Capital: ${portfolio.initial_capital:,.2f}")
    log_manager.info(
        f"Return Since Inception: ${initial_return:,.2f} ({initial_return_pct:+.2f}%)"
    )

    # Run operations
    results = {
        "initial_portfolio_value": initial_portfolio_value,
        "initial_capital": portfolio.initial_capital,
    }

    # Get all assets from portfolio
    all_assets = portfolio.get_all_assets()

    # Separate tradeable and hold-only assets
    # Get hold-only categories from config
    hold_only_category_names = set(config.backtest.hold_only_categories)
    hold_only_categories = set()
    for category_name in hold_only_category_names:
        try:
            hold_only_categories.add(Category[category_name])
        except KeyError:
            log_manager.warning(
                f"Unknown category '{category_name}' in hold_only_categories"
            )

    tradeable_assets = []
    hold_only_assets = []

    for asset in all_assets:
        if asset.category in hold_only_categories:
            hold_only_assets.append(asset)
        else:
            tradeable_assets.append(asset)

    if hold_only_assets:
        log_manager.info("\nHold-only assets (excluded from backtesting):")
        for asset in hold_only_assets:
            log_manager.info(f"  {asset.symbol} ({asset.category})")

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
                console.print(
                    Panel.fit(
                        "[bold yellow]FORECAST MODE - IMPORTANT NOTES:[/bold yellow]\n"
                        "• AutoTS will try multiple models to find the best fit\n"
                        "• This process may take several minutes per ticker\n"
                        "• Press Ctrl+C at any time to cancel\n"
                        "• Enable logging for more detailed progress information",
                        border_style="yellow",
                    )
                )

            # Create progress tasks
            if will_backtest:
                # Count tradeable assets for backtesting
                tradeable_count = len(
                    [a for a in all_assets if a.category not in hold_only_categories]
                )
                if tradeable_count > 0:
                    backtest_task = progress.add_task(
                        f"[green]Backtesting {len(config.backtest.strategies)} strategies across {tradeable_count} stocks...",
                        total=tradeable_count * len(config.backtest.strategies),
                    )
                else:
                    backtest_task = None

            if will_forecast:
                forecast_task = progress.add_task(
                    f"[blue]Forecasting {len(ticker_symbols)} tickers...",
                    total=len(ticker_symbols),
                )

            # Process each ticker with progress tracking
            for ticker_idx, ticker in enumerate(ticker_symbols):
                log_manager.debug(f"\nProcessing {ticker}...")

                # Get the asset to check its category
                asset = next((a for a in all_assets if a.symbol == ticker), None)
                is_hold_only = asset and asset.category in hold_only_categories

                if args.mode in ["all", "ta"]:
                    if "technical_analysis" not in results:
                        results["technical_analysis"] = []
                    # Show progress for TA when it's the only operation or when running all
                    show_ta_progress = (
                        args.mode == "ta" or not will_backtest and not will_forecast
                    )
                    results["technical_analysis"].append(
                        run_technical_analysis(
                            ticker,
                            config,
                            data_fetcher=container.data_fetcher(),
                            show_progress=show_ta_progress,
                        )
                    )

                if will_backtest and not is_hold_only:
                    if "backtesting" not in results:
                        results["backtesting"] = []

                    # Update progress for each strategy
                    for strategy_idx, strategy_config in enumerate(
                        config.backtest.strategies
                    ):
                        progress.update(
                            backtest_task,
                            description=f"[green]Backtesting {strategy_config.name.upper()} on {ticker}...",
                        )

                        # Run single strategy backtest
                        strategy_class = get_strategy_class(strategy_config.name)
                        if strategy_class:
                            # Set strategy parameters if provided
                            if strategy_config.parameters:
                                for key, value in strategy_config.parameters.items():
                                    setattr(strategy_class, key, value)

                            try:
                                runner = container.backtest_runner()
                                backtest_result = runner.run_from_symbol(
                                    ticker,
                                    strategy_class,
                                    start_date=config.data.start_date.strftime(
                                        "%Y-%m-%d"
                                    )
                                    if config.data.start_date
                                    else None,
                                    end_date=config.data.end_date.strftime("%Y-%m-%d")
                                    if config.data.end_date
                                    else None,
                                )

                                # Handle NaN values for win rate when there are no trades
                                win_rate = backtest_result.get("Win Rate [%]", 0)
                                if pd.isna(win_rate):
                                    win_rate = (
                                        None if backtest_result["# Trades"] == 0 else 0
                                    )

                                results["backtesting"].append(
                                    {
                                        "ticker": ticker,
                                        "strategy": strategy_config.name,
                                        "parameters": strategy_config.parameters,
                                        "return_pct": backtest_result["Return [%]"],
                                        "sharpe_ratio": backtest_result["Sharpe Ratio"],
                                        "max_drawdown_pct": backtest_result[
                                            "Max. Drawdown [%]"
                                        ],
                                        "num_trades": backtest_result["# Trades"],
                                        "win_rate": win_rate,
                                    }
                                )
                            except Exception as e:
                                console.print(
                                    f"[red]Error backtesting {strategy_config.name} on {ticker}: {e}[/red]"
                                )

                        # Advance progress
                        if backtest_task is not None:
                            progress.advance(backtest_task)

                if will_forecast:
                    if "forecasting" not in results:
                        results["forecasting"] = []

                    progress.update(
                        forecast_task, description=f"[blue]Forecasting {ticker}..."
                    )

                    forecast_result = run_forecast(
                        ticker, config, stock_forecaster=container.stock_forecaster()
                    )
                    results["forecasting"].append(forecast_result)
                    progress.advance(forecast_task)
    else:
        # No progress bars needed for TA only
        for ticker in ticker_symbols:
            log_manager.debug(f"\nProcessing {ticker}...")

            # Get the asset to check its category
            asset = next((a for a in all_assets if a.symbol == ticker), None)
            is_hold_only = asset and asset.category in hold_only_categories

            if args.mode in ["all", "ta"]:
                if "technical_analysis" not in results:
                    results["technical_analysis"] = []
                # Always show progress for standalone TA mode
                results["technical_analysis"].append(
                    run_technical_analysis(
                        ticker,
                        config,
                        data_fetcher=container.data_fetcher(),
                        show_progress=True,
                    )
                )

    # Output results
    output_format = args.output or config.output.get("format", "console")
    print_results(results, output_format, config, container)

    # Show strategy-specific summaries after backtesting
    if args.mode in ["all", "backtest"] and "backtesting" in results:
        # Group results by strategy
        from collections import defaultdict

        strategy_results = defaultdict(list)

        for backtest in results["backtesting"]:
            strategy_results[backtest["strategy"]].append(backtest)

        # Only proceed if we have results
        if not strategy_results:
            console.print("\n[red]No backtesting results to display.[/red]")
            return

        # Create structured backtest results
        portfolio_backtest_results = create_portfolio_backtest_results(
            results, config, strategy_results
        )

        # Show ticker-level results table first
        console.print("\n[bold green]Ticker-Level Backtest Results[/bold green]")

        # Create table for ticker-level results
        table = Table(title="Ticker-Level Backtest Results")
        table.add_column("Ticker", style="cyan", no_wrap=True)
        table.add_column("Strategy", style="yellow", no_wrap=True)
        table.add_column("Return", style="green", justify="right")
        table.add_column("Sharpe Ratio", style="blue", justify="right")
        table.add_column("Max Drawdown", style="red", justify="right")
        table.add_column("Trades", style="white", justify="right")
        table.add_column("Win Rate", style="magenta", justify="right")

        # Add rows for each backtest result
        for backtest in results["backtesting"]:
            return_str = f"{backtest['return_pct']:+.2f}%"
            sharpe_str = f"{backtest['sharpe_ratio']:.2f}"
            drawdown_str = f"{backtest['max_drawdown_pct']:.2f}%"
            trades_str = str(backtest["num_trades"])

            if backtest["win_rate"] is None:
                win_rate_str = "N/A"
            else:
                win_rate_str = f"{backtest['win_rate']:.1f}%"

            table.add_row(
                backtest["ticker"],
                backtest["strategy"].upper(),
                return_str,
                sharpe_str,
                drawdown_str,
                trades_str,
                win_rate_str,
            )

        console.print(table)
        console.print()  # Add blank line

        # Show summary for each strategy using structured data
        for strategy_summary in portfolio_backtest_results.strategy_summaries:
            # Get broker config info
            broker_info = ""
            if config.backtest.broker_config:
                broker_config = config.backtest.broker_config
                if broker_config.name in [
                    "td_ameritrade",
                    "etrade",
                    "robinhood",
                    "fidelity",
                    "schwab",
                ]:
                    broker_info = f"Broker: {broker_config.name} (zero-commission)"
                elif broker_config.commission_type == "percentage":
                    broker_info = f"Broker: {broker_config.name} ({broker_config.commission_value * 100:.1f}% commission"
                    if broker_config.min_commission:
                        broker_info += f", ${broker_config.min_commission:.2f} min"
                    broker_info += ")"
                elif broker_config.commission_type == "per_share":
                    broker_info = f"Broker: {broker_config.name} (${broker_config.per_share_commission or broker_config.commission_value:.3f}/share"
                    if broker_config.min_commission:
                        broker_info += f", ${broker_config.min_commission:.2f} min"
                    broker_info += ")"
                elif broker_config.commission_type == "tiered":
                    broker_info = f"Broker: {broker_config.name} (tiered pricing"
                    if broker_config.min_commission:
                        broker_info += f", ${broker_config.min_commission:.2f} min"
                    broker_info += ")"
                elif broker_config.commission_type == "fixed":
                    broker_info = f"Broker: {broker_config.name} (${broker_config.commission_value:.2f}/trade)"
                else:
                    broker_info = f"Broker: {broker_config.name} ({broker_config.commission_type})"
            else:
                broker_info = f"Commission: {config.backtest.commission * 100:.1f}%"

            # Create rich panel for strategy summary
            period_return = (
                strategy_summary.final_portfolio_value
                - strategy_summary.initial_portfolio_value
            )
            period_return_color = (
                "green"
                if period_return > 0
                else "red"
                if period_return < 0
                else "white"
            )

            summary_content = f"""[bold]Parameters:[/bold] {strategy_summary.parameters if strategy_summary.parameters else "Default"}
[bold]{broker_info}[/bold]

[bold]Portfolio Value at Start Date:[/bold] ${strategy_summary.initial_portfolio_value:,.2f}
[bold]Portfolio Value at End (Backtest):[/bold] ${strategy_summary.final_portfolio_value:,.2f}

[bold]Strategy Performance:[/bold]
  Average Return: [{period_return_color}]{strategy_summary.average_return_pct:+.2f}%[/{period_return_color}]
  Winning Stocks: [green]{strategy_summary.winning_stocks}[/green]
  Losing Stocks: [red]{strategy_summary.losing_stocks}[/red]
  Total Trades: {strategy_summary.total_trades}

[bold]Return During Period:[/bold] [{period_return_color}]${period_return:,.2f} ({strategy_summary.total_return_pct:+.2f}%)[/{period_return_color}]

[bold]Detailed report saved to:[/bold] {save_detailed_report(strategy_summary.strategy_name, [r.model_dump() for r in strategy_summary.detailed_results], results, config)}"""

            console.print(
                Panel(
                    summary_content,
                    title=f"[bold white]STRATEGY: {strategy_summary.strategy_name.upper()}[/bold white]",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        # Exit after showing strategy summaries
        return

        # Re-fetch current prices to get the most up-to-date values
        log_manager.debug("\nFetching latest prices...")
        final_prices = fetcher.get_current_prices(symbols, show_progress=True)
        final_value = portfolio.get_portfolio_value(final_prices)

        print(f"""
Portfolio Value at Start Date: ${results["initial_portfolio_value"]:,.2f}
Portfolio Value at End (Current): ${final_value:,.2f}""")

        period_return = final_value - results["initial_portfolio_value"]
        period_return_pct = (period_return / results["initial_portfolio_value"]) * 100

        # Show category breakdown if available
        category_allocations = portfolio.get_allocation_by_category(final_prices)
        if category_allocations:
            print("\nAllocation by Category:")
            for category, data in sorted(
                category_allocations.items(), key=lambda x: x[1]["value"], reverse=True
            ):
                print(
                    f"  {category}: ${data['value']:,.2f} ({data['percentage']:.1f}%)"
                )

        # Show performance breakdown by category
        if args.mode in ["all", "backtest"] and config.data.start_date:
            start_category_allocations = portfolio.get_allocation_by_category(
                start_prices
            )
            final_category_allocations = portfolio.get_allocation_by_category(
                final_prices
            )

            print("\nPerformance Breakdown By Category:")
            for category in final_category_allocations.keys():
                if category in start_category_allocations:
                    start_value = start_category_allocations[category]["value"]
                    final_value = final_category_allocations[category]["value"]
                    category_return = final_value - start_value
                    category_return_pct = (
                        (category_return / start_value) * 100 if start_value > 0 else 0
                    )

                    print(f"""  {category}:
    Start Value: ${start_value:,.2f}
    Current Value: ${final_value:,.2f}
    Return: ${category_return:,.2f} ({category_return_pct:+.2f}%)
    Assets: {", ".join(final_category_allocations[category]["assets"])}""")
                else:
                    # New category added during the period
                    final_value = final_category_allocations[category]["value"]
                    print(f"""  {category}:
    Start Value: $0.00 (new category)
    Current Value: ${final_value:,.2f}
    Assets: {", ".join(final_category_allocations[category]["assets"])}""")

        # Show performance breakdown by asset type
        if hold_only_assets and tradeable_assets:
            print("\nAsset Type Breakdown:")

            # Calculate hold-only assets value
            hold_only_value = sum(
                asset.get_value(final_prices.get(asset.symbol, 0))
                for asset in hold_only_assets
            )

            # Calculate tradeable assets value
            tradeable_value = sum(
                asset.get_value(final_prices.get(asset.symbol, 0))
                for asset in tradeable_assets
            )

            print(f"""  Hold-only Assets: ${hold_only_value:,.2f}
  Tradeable Assets: ${tradeable_value:,.2f}
  Total Portfolio: ${hold_only_value + tradeable_value:,.2f}""")

        # Calculate and show total trades from backtesting results
        total_trades = 0
        if "backtesting" in results:
            total_trades = sum(
                backtest.get("num_trades", 0) for backtest in results["backtesting"]
            )
            print(f"\nTotal Trades Executed: {total_trades}")

        # Show return during period at the very end
        print(
            f"Return During Period: ${period_return:,.2f} ({period_return_pct:+.2f}%)"
        )

    # Save results if configured
    if config.output.get("save_results", False):
        results_dir = Path(config.output.get("results_dir", "./results"))
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"stockula_results_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        log_manager.info(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
