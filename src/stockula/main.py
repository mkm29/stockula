"""Stockula main entry point."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

from .data import DataFetcher
from .technical_analysis import TechnicalIndicators
from .backtesting import (
    BacktestRunner,
    SMACrossStrategy,
    RSIStrategy,
    MACDStrategy,
    DoubleEMACrossStrategy,
    TripleEMACrossStrategy,
)
from .forecasting import StockForecaster
from .config import load_config, StockulaConfig


def get_strategy_class(strategy_name: str):
    """Get strategy class by name."""
    strategies = {
        "smacross": SMACrossStrategy,
        "rsi": RSIStrategy,
        "macd": MACDStrategy,
        "doubleemacross": DoubleEMACrossStrategy,
        "tripleemacross": TripleEMACrossStrategy,
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
                    print(f"  Win Rate: N/A (no trades)")

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
            default_files = ["stockula.yaml", "stockula.yml"]
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

    # Override ticker if provided
    if args.ticker:
        config.data.tickers = [args.ticker]

    # Save configuration if requested
    if args.save_config:
        from .config.settings import save_config

        save_config(config, args.save_config)
        print(f"Configuration saved to {args.save_config}")
        return

    # Run operations
    results = {}

    for ticker in config.data.tickers:
        print(f"\nProcessing {ticker}...")

        if args.mode in ["all", "ta"]:
            if "technical_analysis" not in results:
                results["technical_analysis"] = []
            results["technical_analysis"].append(run_technical_analysis(ticker, config))

        if args.mode in ["all", "backtest"]:
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

    # Save results if configured
    if config.output.get("save_results", False):
        results_dir = Path(config.output.get("results_dir", "./results"))
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"stockula_results_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
