#!/usr/bin/env python3
"""Test sequential forecasting to verify it works without threading."""

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


def test_sequential_forecast():
    """Test sequential forecasting for multiple tickers."""
    print("Starting sequential forecast test...")

    # Configure data fetcher
    data_fetcher = DataFetcher()

    # Test with just 2 tickers
    tickers = ["MSFT", "AAPL"]
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    results = {}

    for ticker in tickers:
        print(f"\n\nForecasting {ticker}...")
        try:
            # Create a new forecaster for each ticker
            forecaster = StockForecaster(forecast_length=7, data_fetcher=data_fetcher)

            # Fetch data
            print(f"Fetching data for {ticker}...")
            data = data_fetcher.get_stock_data(ticker, start_date, end_date)
            print(f"Fetched {len(data)} data points")

            # Fit and predict
            print(f"Fitting model for {ticker}...")
            predictions = forecaster.fit_predict(
                data,
                target_column="Close",
                model_list=["LastValueNaive"],
                ensemble="simple",
                max_generations=1,
                show_progress=False,
            )

            print(f"Predictions generated for {ticker}")
            print(predictions.head())

            # Get best model info
            model_info = forecaster.get_best_model()

            results[ticker] = {
                "ticker": ticker,
                "best_model": model_info["model_name"],
                "forecast_price": predictions["forecast"].iloc[-1],
                "current_price": predictions["forecast"].iloc[0],
            }

            print(f"✓ {ticker} completed: {model_info['model_name']}")

        except Exception as e:
            print(f"✗ {ticker} error: {e}")
            results[ticker] = {"ticker": ticker, "error": str(e)}

    print(f"\n\nResults:")
    for symbol, result in results.items():
        if "error" in result:
            print(f"{symbol}: ERROR - {result['error']}")
        else:
            print(
                f"{symbol}: {result['best_model']} - Forecast: ${result['forecast_price']:.2f}"
            )


if __name__ == "__main__":
    test_sequential_forecast()
