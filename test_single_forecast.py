#!/usr/bin/env python3
"""Test single ticker forecast to debug hanging issue."""

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


def test_single_forecast():
    """Test forecasting for a single ticker."""
    print("Starting single ticker forecast test...")

    # Configure data fetcher
    data_fetcher = DataFetcher()

    # Create forecaster
    forecaster = StockForecaster(forecast_length=7, data_fetcher=data_fetcher)

    # Test data
    ticker = "MSFT"
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"\nFetching data for {ticker}...")
    data = data_fetcher.get_stock_data(ticker, start_date, end_date)
    print(f"Fetched {len(data)} data points")

    print(f"\nFitting model...")
    # Use ultra_fast models for quick test
    predictions = forecaster.fit_predict(
        data,
        target_column="Close",
        model_list="ultra_fast",
        ensemble="simple",
        max_generations=1,
        show_progress=True,
    )

    print(f"\nPredictions generated:")
    print(predictions.head())

    # Get best model info
    model_info = forecaster.get_best_model()
    print(f"\nBest model: {model_info['model_name']}")


if __name__ == "__main__":
    test_single_forecast()
