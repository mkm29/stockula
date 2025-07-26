"""Stockula main entry point."""

import argparse
from datetime import datetime, timedelta
from .data import DataFetcher
from .technical_analysis import TechnicalIndicators
from .backtesting import BacktestRunner, SMACrossStrategy, RSIStrategy
from .forecasting import StockForecaster


def demo_technical_analysis(symbol: str = "AAPL"):
    """Demonstrate technical analysis functionality."""
    print(f"\n=== Technical Analysis Demo for {symbol} ===")

    fetcher = DataFetcher()
    data = fetcher.get_stock_data(symbol)

    # Calculate indicators
    ta = TechnicalIndicators(data)

    # Get latest values
    print(f"\nLatest Technical Indicators:")
    print(f"SMA(20): {ta.sma(20).iloc[-1]:.2f}")
    print(f"EMA(20): {ta.ema(20).iloc[-1]:.2f}")
    print(f"RSI(14): {ta.rsi(14).iloc[-1]:.2f}")
    print(f"ATR(14): {ta.atr(14).iloc[-1]:.2f}")


def demo_backtesting(symbol: str = "AAPL"):
    """Demonstrate backtesting functionality."""
    print(f"\n=== Backtesting Demo for {symbol} ===")

    runner = BacktestRunner(cash=10000)

    # Test SMA Cross Strategy
    print("\nTesting SMA Cross Strategy...")
    results = runner.run_from_symbol(symbol, SMACrossStrategy)
    print(f"Return: {results['Return [%]']:.2f}%")
    print(f"Sharpe Ratio: {results['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown: {results['Max. Drawdown [%]']:.2f}%")
    print(f"Number of Trades: {results['# Trades']}")

    # Test RSI Strategy
    print("\nTesting RSI Strategy...")
    results = runner.run_from_symbol(symbol, RSIStrategy)
    print(f"Return: {results['Return [%]']:.2f}%")
    print(f"Sharpe Ratio: {results['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown: {results['Max. Drawdown [%]']:.2f}%")
    print(f"Number of Trades: {results['# Trades']}")


def demo_forecasting(symbol: str = "AAPL"):
    """Demonstrate forecasting functionality."""
    print(f"\n=== Forecasting Demo for {symbol} ===")

    forecaster = StockForecaster(forecast_length=30)

    print("Training forecast model (this may take a moment)...")
    predictions = forecaster.forecast_from_symbol(symbol, model_list="fast")

    print(f"\n30-Day Price Forecast:")
    print(f"Current Price: {predictions['forecast'].iloc[0]:.2f}")
    print(f"30-Day Forecast: {predictions['forecast'].iloc[-1]:.2f}")
    print(
        f"Expected Range: ${predictions['lower_bound'].iloc[-1]:.2f} - ${predictions['upper_bound'].iloc[-1]:.2f}"
    )

    # Get best model info
    model_info = forecaster.get_best_model()
    print(f"\nBest Model: {model_info['model_name']}")


def demo_data_fetching(symbol: str = "AAPL"):
    """Demonstrate data fetching functionality."""
    print(f"\n=== Data Fetching Demo for {symbol} ===")

    fetcher = DataFetcher()

    # Get stock info
    info = fetcher.get_info(symbol)
    print(f"\nStock Info:")
    print(f"Company: {info.get('longName', 'N/A')}")
    print(f"Sector: {info.get('sector', 'N/A')}")
    print(f"Market Cap: ${info.get('marketCap', 0):,.0f}")

    # Get current price
    current_price = fetcher.get_realtime_price(symbol)
    if current_price:
        print(f"Current Price: ${current_price:.2f}")

    # Get recent data
    data = fetcher.get_stock_data(
        symbol, start=(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    )
    print(f"\nLast 5 days of data:")
    print(data.tail())


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Stockula Trading Platform Demo")
    parser.add_argument("--symbol", default="AAPL", help="Stock symbol to analyze")
    parser.add_argument(
        "--demo",
        choices=["all", "data", "ta", "backtest", "forecast"],
        default="all",
        help="Demo to run",
    )

    args = parser.parse_args()

    if args.demo in ["all", "data"]:
        demo_data_fetching(args.symbol)

    if args.demo in ["all", "ta"]:
        demo_technical_analysis(args.symbol)

    if args.demo in ["all", "backtest"]:
        demo_backtesting(args.symbol)

    if args.demo in ["all", "forecast"]:
        demo_forecasting(args.symbol)


if __name__ == "__main__":
    main()
