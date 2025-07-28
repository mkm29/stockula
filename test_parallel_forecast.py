#!/usr/bin/env python3
"""Test parallel forecast to debug hanging issue."""

import sys
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Import stockula modules
from stockula.data import DataFetcher
from stockula.forecasting import StockForecaster


def test_parallel_forecast():
    """Test parallel forecasting for multiple tickers."""
    print("Starting parallel forecast test...")

    # Configure data fetcher
    data_fetcher = DataFetcher()

    # Test with just 2 tickers
    tickers = ["MSFT", "AAPL"]
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    def progress_callback(symbol, status, status_info=None):
        """Simple progress callback."""
        if status == "status_update" and status_info:
            print(f"Progress: {status_info}")
        elif status == "completed":
            print(f"✓ {symbol} completed")
        elif status == "error":
            print(f"✗ {symbol} error")

    print(f"\nForecasting {tickers}...")
    results = StockForecaster.forecast_multiple_parallel(
        symbols=tickers,
        start_date=start_date,
        end_date=end_date,
        forecast_length=7,
        model_list=["LastValueNaive"],  # Use just one simple model
        ensemble="simple",
        max_generations=1,
        max_workers=2,  # Can use multiple workers now with global lock
        data_fetcher=data_fetcher,
        progress_callback=progress_callback,
        status_update_interval=1,  # Update every second
    )

    print(f"\n\nResults received: {list(results.keys())}")
    for symbol, result in results.items():
        if "error" in result:
            print(f"{symbol}: ERROR - {result['error']}")
        else:
            print(
                f"{symbol}: {result['best_model']} - Forecast: ${result['forecast_price']:.2f}"
            )


if __name__ == "__main__":
    test_parallel_forecast()
