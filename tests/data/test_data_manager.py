"""Test data management for strategies, technical analysis, and forecasting modules.

This module provides utilities for fetching, saving, and loading real financial data
from yfinance for use in testing. Using real data ensures consistent test results
and better coverage of edge cases in calculations.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import warnings

try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


class TestDataManager:
    """Manages test data for financial strategies and analysis modules."""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the test data manager.

        Args:
            data_dir: Directory to store test data files. Defaults to tests/data/
        """
        if data_dir is None:
            data_dir = Path(__file__).parent

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def fetch_and_save_data(
        self,
        tickers: List[str],
        period: str = "2y",
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch real market data and save to pickle files.

        Args:
            tickers: List of ticker symbols to fetch
            period: Period to fetch (e.g., "1y", "2y", "5y")
            interval: Data interval (e.g., "1d", "1h")
            force_refresh: Whether to force refresh existing data

        Returns:
            Dictionary mapping ticker symbols to DataFrames
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance is required for fetching test data")

        data = {}

        for ticker in tickers:
            filepath = self.data_dir / f"{ticker}_{period}_{interval}.pkl"

            if not force_refresh and filepath.exists():
                print(f"Loading existing data for {ticker}")
                with open(filepath, "rb") as f:
                    data[ticker] = pickle.load(f)
                continue

            print(f"Fetching data for {ticker}")
            try:
                ticker_obj = yf.Ticker(ticker)
                df = ticker_obj.history(period=period, interval=interval)

                if df.empty:
                    warnings.warn(f"No data retrieved for {ticker}")
                    continue

                # Ensure we have all required columns
                required_cols = ["Open", "High", "Low", "Close", "Volume"]
                if not all(col in df.columns for col in required_cols):
                    warnings.warn(f"Missing required columns for {ticker}")
                    continue

                # Clean the data
                df = df.dropna()

                # Save to pickle
                with open(filepath, "wb") as f:
                    pickle.dump(df, f)

                data[ticker] = df
                print(f"Saved {len(df)} rows for {ticker}")

            except Exception as e:
                warnings.warn(f"Failed to fetch data for {ticker}: {e}")
                continue

        return data

    def load_data(
        self, ticker: str, period: str = "2y", interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Load test data from pickle file.

        Args:
            ticker: Ticker symbol
            period: Period identifier
            interval: Interval identifier

        Returns:
            DataFrame with OHLCV data or None if not found
        """
        filepath = self.data_dir / f"{ticker}_{period}_{interval}.pkl"

        if not filepath.exists():
            return None

        try:
            with open(filepath, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            warnings.warn(f"Failed to load data for {ticker}: {e}")
            return None

    def get_test_data_subset(
        self,
        ticker: str,
        days: int = 252,
        offset: int = 0,
        period: str = "2y",
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """Get a subset of test data for specific testing needs.

        Args:
            ticker: Ticker symbol
            days: Number of days to return
            offset: Number of days to skip from the start
            period: Period identifier
            interval: Interval identifier

        Returns:
            DataFrame subset or None if not available
        """
        df = self.load_data(ticker, period, interval)

        if df is None or len(df) < (days + offset):
            return None

        return df.iloc[offset : offset + days].copy()

    def create_synthetic_data(
        self,
        days: int = 252,
        start_price: float = 100.0,
        volatility: float = 0.02,
        trend: float = 0.001,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Create synthetic test data when real data is not available.

        Args:
            days: Number of trading days
            start_price: Starting price
            volatility: Daily volatility
            trend: Daily trend (return)
            seed: Random seed for reproducibility

        Returns:
            DataFrame with synthetic OHLCV data
        """
        np.random.seed(seed)

        # Generate dates
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days), periods=days, freq="D"
        )

        # Generate price series with trend and volatility
        returns = np.random.normal(trend, volatility, days)
        prices = [start_price]

        for r in returns[1:]:
            prices.append(prices[-1] * (1 + r))

        # Create OHLC data
        close_prices = np.array(prices)
        high_prices = close_prices * np.random.uniform(1.0, 1.05, days)
        low_prices = close_prices * np.random.uniform(0.95, 1.0, days)
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = start_price

        # Generate volumes
        volumes = np.random.lognormal(mean=10, sigma=0.5, size=days).astype(int) * 1000

        return pd.DataFrame(
            {
                "Open": open_prices,
                "High": high_prices,
                "Low": low_prices,
                "Close": close_prices,
                "Volume": volumes,
            },
            index=dates,
        )


# Global instance for easy access in tests
test_data_manager = TestDataManager()


def setup_test_data(force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
    """Set up test data for all modules.

    Args:
        force_refresh: Whether to force refresh all data

    Returns:
        Dictionary of loaded test data
    """
    # Common tickers used for testing different market conditions
    tickers = [
        "AAPL",  # Large cap tech stock
        "SPY",  # S&P 500 ETF
        "QQQ",  # NASDAQ ETF
        "GLD",  # Gold ETF (low volatility)
        "TSLA",  # High volatility stock
        "BRK-B",  # Value stock
        "VIX",  # Volatility index
        "TLT",  # Long-term treasury ETF
    ]

    return test_data_manager.fetch_and_save_data(
        tickers=tickers, period="2y", interval="1d", force_refresh=force_refresh
    )


def get_test_data_for_strategy(strategy_name: str, min_days: int = 100) -> pd.DataFrame:
    """Get appropriate test data for a specific strategy.

    Args:
        strategy_name: Name of the strategy being tested
        min_days: Minimum number of days required

    Returns:
        DataFrame with appropriate test data
    """
    # Choose ticker based on strategy characteristics
    ticker_mapping = {
        "SMA": "SPY",  # Trend-following strategies work well with SPY
        "EMA": "SPY",
        "RSI": "TSLA",  # RSI works well with volatile stocks
        "MACD": "AAPL",  # MACD good for tech stocks
        "VIDYA": "TSLA",  # Volatility-based strategies for volatile stocks
        "KAMA": "BRK-B",  # Adaptive strategies for steady stocks
        "FRAMA": "AAPL",  # Fractal strategies for trending stocks
        "TRIMA": "SPY",  # Triangular MA for smooth data
        "TEMA": "QQQ",  # Triple EMA for tech-heavy index
        "VAMA": "SPY",  # Volume-adjusted for high volume
        "Kaufman": "BRK-B",  # Efficiency-based for steady trends
    }

    # Find matching ticker
    ticker = "SPY"  # Default
    for key, value in ticker_mapping.items():
        if key.lower() in strategy_name.lower():
            ticker = value
            break

    # Try to get real data first
    data = test_data_manager.get_test_data_subset(
        ticker=ticker,
        days=min_days + 50,  # Add buffer
        offset=0,
    )

    # Fall back to synthetic data if real data not available
    if data is None:
        data = test_data_manager.create_synthetic_data(
            days=min_days + 50,
            seed=hash(strategy_name) % 1000,  # Consistent seed per strategy
        )

    return data
