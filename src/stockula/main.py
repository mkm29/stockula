"""Stockula main entry point."""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from dependency_injector.wiring import inject, Provide

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
)
from .config import StockulaConfig
from .domain import Category
from .interfaces import (
    IDataFetcher,
    IBacktestRunner,
    IStockForecaster,
    ILoggingManager,
)

# Global logging manager instance
log_manager: Optional[ILoggingManager] = None


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
    }
    return strategies.get(strategy_name.lower())


@inject
def run_technical_analysis(
    ticker: str,
    config: StockulaConfig,
    data_fetcher: IDataFetcher = Provide[Container.data_fetcher],
) -> Dict[str, Any]:
    """Run technical analysis for a ticker.

    Args:
        ticker: Stock symbol
        config: Configuration object
        data_fetcher: Injected data fetcher

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

            results.append(
                {
                    "ticker": ticker,
                    "strategy": strategy_config.name,
                    "parameters": strategy_config.parameters,
                    "return_pct": backtest_result["Return [%]"],
                    "sharpe_ratio": backtest_result["Sharpe Ratio"],
                    "max_drawdown_pct": backtest_result["Max. Drawdown [%]"],
                    "num_trades": backtest_result["# Trades"],
                    "win_rate": win_rate,
                }
            )
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

    return str(report_file)


def print_results(results: Dict[str, Any], output_format: str = "console"):
    """Print results in specified format.

    Args:
        results: Results dictionary
        output_format: Output format (console, json)
    """
    if output_format == "json":
        print(json.dumps(results, indent=2, default=str))
    else:
        # Console output

        if "technical_analysis" in results:
            print("\n=== Technical Analysis Results ===")
            for ta_result in results["technical_analysis"]:
                print(f"\n{ta_result['ticker']}:")
                for indicator, value in ta_result["indicators"].items():
                    if isinstance(value, dict):
                        print(f"  {indicator}:")
                        for k, v in value.items():
                            print(
                                f"    {k}: {v:.2f}"
                                if isinstance(v, (int, float))
                                else f"    {k}: {v}"
                            )
                    else:
                        print(
                            f"  {indicator}: {value:.2f}"
                            if isinstance(value, (int, float))
                            else f"  {indicator}: {value}"
                        )

        if "backtesting" in results:
            # Check if we have multiple strategies
            strategies = set(b["strategy"] for b in results["backtesting"])
            if len(strategies) > 1:
                # For multiple strategies, only show a brief message
                print("\n=== Backtesting Results ===")
                print(
                    f"Running {len(strategies)} strategies across {len(set(b['ticker'] for b in results['backtesting']))} stocks..."
                )
                print("Detailed results will be shown per strategy below.")
            else:
                # For single strategy, show the detailed results as before
                print("\n=== Backtesting Results ===")
                for backtest in results["backtesting"]:
                    print(f"""
{backtest["ticker"]} - {backtest["strategy"]}:
  Parameters: {backtest["parameters"]}
  Return: {backtest["return_pct"]:.2f}%
  Sharpe Ratio: {backtest["sharpe_ratio"]:.2f}
  Max Drawdown: {backtest["max_drawdown_pct"]:.2f}%
  Number of Trades: {backtest["num_trades"]}""")
                    if backtest.get("win_rate") is not None:
                        print(f"  Win Rate: {backtest['win_rate']:.2f}%")
                    elif backtest["num_trades"] == 0:
                        print("  Win Rate: N/A (no trades)")

        if "forecasting" in results:
            print("\n=== Forecasting Results ===")
            for forecast in results["forecasting"]:
                if "error" in forecast:
                    print(f"\n{forecast['ticker']}: Error - {forecast['error']}")
                else:
                    print(f"""
{forecast["ticker"]}:
  Current Price: ${forecast["current_price"]:.2f}
  {forecast["forecast_length"]}-Day Forecast: ${forecast["forecast_price"]:.2f}
  Confidence Range: ${forecast["lower_bound"]:.2f} - ${forecast["upper_bound"]:.2f}
  Best Model: {forecast["best_model"]}""")


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

    log_manager.info("\nPortfolio Summary:")
    log_manager.info(f"  Name: {portfolio.name}")
    log_manager.info(f"  Initial Capital: ${portfolio.initial_capital:,.2f}")
    log_manager.info(f"  Total Assets: {len(portfolio.get_all_assets())}")
    log_manager.info(f"  Allocation Method: {portfolio.allocation_method}")

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
        current_prices = fetcher.get_current_prices(symbols)
        initial_portfolio_value = portfolio.get_portfolio_value(current_prices)
        log_manager.info(f"\nCurrent Portfolio Value: ${initial_portfolio_value:,.2f}")

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

    for ticker in ticker_symbols:
        log_manager.debug(f"\nProcessing {ticker}...")

        # Get the asset to check its category
        asset = next((a for a in all_assets if a.symbol == ticker), None)
        is_hold_only = asset and asset.category in hold_only_categories

        if args.mode in ["all", "ta"]:
            if "technical_analysis" not in results:
                results["technical_analysis"] = []
            results["technical_analysis"].append(
                run_technical_analysis(
                    ticker, config, data_fetcher=container.data_fetcher()
                )
            )

        if args.mode in ["all", "backtest"]:
            if is_hold_only:
                log_manager.debug(f"  Skipping backtest for {ticker} (hold-only asset)")
            else:
                if "backtesting" not in results:
                    results["backtesting"] = []
                results["backtesting"].extend(
                    run_backtest(
                        ticker, config, backtest_runner=container.backtest_runner()
                    )
                )

        if args.mode in ["all", "forecast"]:
            if "forecasting" not in results:
                results["forecasting"] = []

            # Show warning about forecast mode
            if args.mode == "forecast" and ticker == config.portfolio.tickers[0].symbol:
                print(f"""
{"=" * 60}
FORECAST MODE - IMPORTANT NOTES:
{"=" * 60}
• AutoTS will try multiple models to find the best fit
• This process may take several minutes per ticker
• Press Ctrl+C at any time to cancel
• Enable logging for more detailed progress information
{"=" * 60}""")

            forecast_result = run_forecast(
                ticker, config, stock_forecaster=container.stock_forecaster()
            )
            results["forecasting"].append(forecast_result)

    # Output results
    output_format = args.output or config.output.get("format", "console")
    print_results(results, output_format)

    # Show strategy-specific summaries after backtesting
    if args.mode in ["all", "backtest"] and "backtesting" in results:
        # Group results by strategy
        from collections import defaultdict

        strategy_results = defaultdict(list)

        for backtest in results["backtesting"]:
            strategy_results[backtest["strategy"]].append(backtest)

        # Only proceed if we have results
        if not strategy_results:
            print("\nNo backtesting results to display.")
            return

        # Show summary for each strategy
        for strategy_name, strategy_backtests in strategy_results.items():
            # Calculate strategy-specific metrics
            strategy_trades = sum(b.get("num_trades", 0) for b in strategy_backtests)
            
            # Calculate strategy-specific portfolio performance
            strategy_total_return = 0.0
            strategy_winning_stocks = 0
            strategy_losing_stocks = 0
            
            for backtest in strategy_backtests:
                return_pct = backtest.get("return_pct", 0)
                strategy_total_return += return_pct
                if return_pct > 0:
                    strategy_winning_stocks += 1
                elif return_pct < 0:
                    strategy_losing_stocks += 1
            
            # Calculate average return (simple average, not weighted)
            strategy_avg_return = strategy_total_return / len(strategy_backtests) if strategy_backtests else 0
            
            # Calculate approximate final portfolio value based on average return
            # This is a simplification - actual portfolio value would depend on position sizing
            strategy_final_value = results["initial_portfolio_value"] * (1 + strategy_avg_return / 100)

            # Get the first backtest to extract parameters (assuming all use same parameters)
            strategy_params = (
                strategy_backtests[0].get("parameters", {})
                if strategy_backtests
                else {}
            )

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

            print(f"""
{"=" * 60}
STRATEGY: {strategy_name.upper()}
{"=" * 60}

Parameters: {strategy_params if strategy_params else "Default"}
{broker_info}

Portfolio Value at Start Date: ${results["initial_portfolio_value"]:,.2f}
Portfolio Value at End (Backtest): ${strategy_final_value:,.2f}

Strategy Performance:
  Average Return: {strategy_avg_return:+.2f}%
  Winning Stocks: {strategy_winning_stocks}
  Losing Stocks: {strategy_losing_stocks}
  Total Trades: {strategy_trades}

Return During Period: ${strategy_final_value - results["initial_portfolio_value"]:,.2f} ({strategy_avg_return:+.2f}%)

Detailed report saved to: {save_detailed_report(strategy_name, strategy_backtests, results, config)}
""")

        # Exit after showing strategy summaries
        return

        # Re-fetch current prices to get the most up-to-date values
        log_manager.debug("\nFetching latest prices...")
        final_prices = fetcher.get_current_prices(symbols)
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
