"""Stockula main entry point."""

import argparse
import json
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pandas as pd

from .data.fetcher import DataFetcher
from .technical_analysis import TechnicalIndicators
from .backtesting import (
    BacktestRunner,
    SMACrossStrategy,
    RSIStrategy,
    MACDStrategy,
    DoubleEMACrossStrategy,
    TripleEMACrossStrategy,
    TRIMACrossStrategy,
)
from .forecasting import StockForecaster
from .config import load_config, StockulaConfig
from .domain import DomainFactory, Category

# Create logger
logger = logging.getLogger(__name__)


def setup_logging(config: StockulaConfig) -> None:
    """Configure logging based on configuration."""
    # Clear any existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # List to store all handlers
    handlers = []

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)

    # Create formatters
    if config.logging.enabled:
        # Detailed format when logging is enabled
        detailed_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        simple_formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        log_level = getattr(logging, config.logging.level.upper(), logging.INFO)

        # Use simple formatter for console, detailed for file
        console_handler.setFormatter(simple_formatter)

        # Add file handler if requested
        if config.logging.log_to_file:
            file_handler = RotatingFileHandler(
                filename=config.logging.log_file,
                maxBytes=config.logging.max_log_size,
                backupCount=config.logging.backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(detailed_formatter)
            file_handler.setLevel(log_level)
            handlers.append(file_handler)
    else:
        # Simple format when logging is disabled (only warnings/errors)
        formatter = logging.Formatter(fmt="%(levelname)s: %(message)s")
        console_handler.setFormatter(formatter)
        log_level = logging.WARNING

    # Set level on console handler and add to handlers
    console_handler.setLevel(log_level)
    handlers.append(console_handler)

    # Configure root logger
    root_logger.setLevel(log_level)
    for handler in handlers:
        root_logger.addHandler(handler)

    # Configure stockula loggers with proper hierarchy
    stockula_logger = logging.getLogger("stockula")
    stockula_logger.setLevel(log_level)
    stockula_logger.propagate = True  # Propagate to root logger

    # Log startup message if enabled
    if config.logging.enabled:
        logger.info(f"Logging initialized - Level: {config.logging.level}")
        if config.logging.log_to_file:
            logger.info(f"Logging to file: {config.logging.log_file}")

    # Reduce noise from third-party libraries
    third_party_level = (
        logging.CRITICAL
        if not config.logging.enabled
        else (logging.WARNING if log_level != logging.DEBUG else logging.INFO)
    )

    for lib_name in [
        "yfinance",
        "urllib3",
        "requests",
        "apscheduler",
        "peewee",
        "backtesting",
    ]:
        logging.getLogger(lib_name).setLevel(third_party_level)


def get_strategy_class(strategy_name: str):
    """Get strategy class by name."""
    strategies = {
        "smacross": SMACrossStrategy,
        "rsi": RSIStrategy,
        "macd": MACDStrategy,
        "doubleemacross": DoubleEMACrossStrategy,
        "tripleemacross": TripleEMACrossStrategy,
        "trimacross": TRIMACrossStrategy,
    }
    return strategies.get(strategy_name.lower())


def run_technical_analysis(ticker: str, config: StockulaConfig) -> Dict[str, Any]:
    """Run technical analysis for a ticker.

    Args:
        ticker: Stock symbol
        config: Configuration object

    Returns:
        Dictionary with indicator results
    """
    fetcher = DataFetcher()
    data = fetcher.get_stock_data(
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


def run_backtest(ticker: str, config: StockulaConfig) -> List[Dict[str, Any]]:
    """Run backtesting for a ticker.

    Args:
        ticker: Stock symbol
        config: Configuration object

    Returns:
        List of backtest results
    """
    runner = BacktestRunner(
        cash=config.backtest.initial_cash,
        commission=config.backtest.commission,
        margin=config.backtest.margin,
    )

    results = []

    for strategy_config in config.backtest.strategies:
        strategy_class = get_strategy_class(strategy_config.name)
        if not strategy_class:
            print(f"Warning: Unknown strategy '{strategy_config.name}'")
            continue

        # Create strategy class with parameters
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


def run_forecast(ticker: str, config: StockulaConfig) -> Dict[str, Any]:
    """Run forecasting for a ticker.

    Args:
        ticker: Stock symbol
        config: Configuration object

    Returns:
        Dictionary with forecast results
    """
    forecaster = StockForecaster(
        forecast_length=config.forecast.forecast_length,
        frequency=config.forecast.frequency,
        prediction_interval=config.forecast.prediction_interval,
        num_validations=config.forecast.num_validations,
        validation_method=config.forecast.validation_method,
    )

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
    except Exception as e:
        print(f"Error forecasting {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}


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
            print("\n=== Backtesting Results ===")
            for backtest in results["backtesting"]:
                print(f"\n{backtest['ticker']} - {backtest['strategy']}:")
                print(f"  Parameters: {backtest['parameters']}")
                print(f"  Return: {backtest['return_pct']:.2f}%")
                print(f"  Sharpe Ratio: {backtest['sharpe_ratio']:.2f}")
                print(f"  Max Drawdown: {backtest['max_drawdown_pct']:.2f}%")
                print(f"  Number of Trades: {backtest['num_trades']}")
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
                    print(f"\n{forecast['ticker']}:")
                    print(f"  Current Price: ${forecast['current_price']:.2f}")
                    print(
                        f"  {forecast['forecast_length']}-Day Forecast: ${forecast['forecast_price']:.2f}"
                    )
                    print(
                        f"  Confidence Range: ${forecast['lower_bound']:.2f} - ${forecast['upper_bound']:.2f}"
                    )
                    print(f"  Best Model: {forecast['best_model']}")


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

    # Load configuration
    try:
        from pathlib import Path

        # Check if we're using a specific config file or looking for defaults
        if args.config:
            config = load_config(args.config)
            print(f"Using configuration from: {args.config}")
        else:
            # Check for default files
            default_files = [
                ".config.yaml",
                ".config.yml",
                "config.yaml",
                "config.yml",
                ".stockula.yaml",
                ".stockula.yml",
                "stockula.yaml",
                "stockula.yml",
            ]
            found_config = None
            for filename in default_files:
                if Path(filename).exists():
                    found_config = filename
                    break

            if found_config:
                config = load_config()
                print(f"Using configuration from: {found_config}")
            else:
                config = load_config()
                print("No configuration file found. Using default settings.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Using default configuration...")
        config = StockulaConfig()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Set up logging based on configuration
    setup_logging(config)

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

    # Create domain objects from configuration
    factory = DomainFactory()
    portfolio = factory.create_portfolio(config)

    logger.info("\nPortfolio Summary:")
    logger.info(f"  Name: {portfolio.name}")
    logger.info(f"  Initial Capital: ${portfolio.initial_capital:,.2f}")
    logger.info(f"  Total Assets: {len(portfolio.get_all_assets())}")
    logger.info(f"  Allocation Method: {portfolio.allocation_method}")

    # Get portfolio value at start of backtest period
    fetcher = DataFetcher()
    symbols = [asset.symbol for asset in portfolio.get_all_assets()]

    # Get prices at the start date if backtesting
    if args.mode in ["all", "backtest"] and config.data.start_date:
        logger.debug(f"\nFetching prices at start date ({config.data.start_date})...")
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
                logger.warning(f"Could not get start price for {symbol}: {e}")

        initial_portfolio_value = portfolio.get_portfolio_value(start_prices)
        logger.info(f"\nPortfolio Value at Start Date: ${initial_portfolio_value:,.2f}")
    else:
        logger.debug("\nFetching current prices...")
        current_prices = fetcher.get_current_prices(symbols)
        initial_portfolio_value = portfolio.get_portfolio_value(current_prices)
        logger.info(f"\nCurrent Portfolio Value: ${initial_portfolio_value:,.2f}")

    # Calculate returns (always needed, not just for logging)
    initial_return = initial_portfolio_value - portfolio.initial_capital
    initial_return_pct = (initial_return / portfolio.initial_capital) * 100

    logger.info(f"Initial Capital: ${portfolio.initial_capital:,.2f}")
    logger.info(
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
            logger.warning(
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
        logger.info("\nHold-only assets (excluded from backtesting):")
        for asset in hold_only_assets:
            logger.info(f"  {asset.symbol} ({asset.category})")

    # Get ticker symbols for processing
    ticker_symbols = [asset.symbol for asset in all_assets]

    for ticker in ticker_symbols:
        logger.debug(f"\nProcessing {ticker}...")

        # Get the asset to check its category
        asset = next((a for a in all_assets if a.symbol == ticker), None)
        is_hold_only = asset and asset.category in hold_only_categories

        if args.mode in ["all", "ta"]:
            if "technical_analysis" not in results:
                results["technical_analysis"] = []
            results["technical_analysis"].append(run_technical_analysis(ticker, config))

        if args.mode in ["all", "backtest"]:
            if is_hold_only:
                logger.debug(f"  Skipping backtest for {ticker} (hold-only asset)")
            else:
                if "backtesting" not in results:
                    results["backtesting"] = []
                results["backtesting"].extend(run_backtest(ticker, config))

        if args.mode in ["all", "forecast"]:
            if "forecasting" not in results:
                results["forecasting"] = []
            results["forecasting"].append(run_forecast(ticker, config))

    # Output results
    output_format = args.output or config.output.get("format", "console")
    print_results(results, output_format)

    # Show final portfolio summary after backtesting
    if args.mode in ["all", "backtest"]:
        print("\n" + "=" * 50)
        print("PORTFOLIO PERFORMANCE SUMMARY")
        print("=" * 50)

        # Re-fetch current prices to get the most up-to-date values
        logger.debug("\nFetching latest prices...")
        final_prices = fetcher.get_current_prices(symbols)
        final_value = portfolio.get_portfolio_value(final_prices)

        print(
            f"\nPortfolio Value at Start Date: ${results['initial_portfolio_value']:,.2f}"
        )
        print(f"Portfolio Value at End (Current): ${final_value:,.2f}")

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

                    print(f"  {category}:")
                    print(f"    Start Value: ${start_value:,.2f}")
                    print(f"    Current Value: ${final_value:,.2f}")
                    print(
                        f"    Return: ${category_return:,.2f} ({category_return_pct:+.2f}%)"
                    )
                    print(
                        f"    Assets: {', '.join(final_category_allocations[category]['assets'])}"
                    )
                else:
                    # New category added during the period
                    final_value = final_category_allocations[category]["value"]
                    print(f"  {category}:")
                    print(f"    Start Value: $0.00 (new category)")
                    print(f"    Current Value: ${final_value:,.2f}")
                    print(
                        f"    Assets: {', '.join(final_category_allocations[category]['assets'])}"
                    )

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

            print(f"  Hold-only Assets: ${hold_only_value:,.2f}")
            print(f"  Tradeable Assets: ${tradeable_value:,.2f}")
            print(f"  Total Portfolio: ${hold_only_value + tradeable_value:,.2f}")

        # Show return during period at the very end
        print(
            f"\nReturn During Period: ${period_return:,.2f} ({period_return_pct:+.2f}%)"
        )

    # Save results if configured
    if config.output.get("save_results", False):
        results_dir = Path(config.output.get("results_dir", "./results"))
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"stockula_results_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
