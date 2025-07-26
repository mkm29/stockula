"""Data fetching utilities using yfinance."""

import yfinance as yf
import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta


class DataFetcher:
    """Fetch financial data using yfinance."""

    def __init__(self):
        """Initialize data fetcher."""
        pass

    def get_stock_data(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical stock data.

        Args:
            symbol: Stock ticker symbol
            start: Start date (YYYY-MM-DD), defaults to 1 year ago
            end: End date (YYYY-MM-DD), defaults to today
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            DataFrame with OHLCV data
        """
        if start is None:
            start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start, end=end, interval=interval)

        # Ensure consistent column naming for backtesting compatibility
        # The backtesting library expects capitalized column names
        column_mapping = {
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
            "Dividends": "Dividends",
            "Stock Splits": "Stock Splits",
        }

        # Rename columns to ensure consistency
        data = data.rename(columns=column_mapping)

        # Keep only the required columns if they exist
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        available_cols = [col for col in required_cols if col in data.columns]
        data = data[available_cols]

        return data

    def get_multiple_stocks(
        self,
        symbols: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple stocks.

        Args:
            symbols: List of stock ticker symbols
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            interval: Data interval

        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        data = {}
        for symbol in symbols:
            try:
                data[symbol] = self.get_stock_data(symbol, start, end, interval)
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")

        return data

    def get_info(self, symbol: str) -> Dict[str, Any]:
        """Get stock information.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with stock information
        """
        ticker = yf.Ticker(symbol)
        return ticker.info

    def get_realtime_price(self, symbol: str) -> float:
        """Get current price for a stock.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Current price
        """
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d", interval="1m")
        return data["Close"].iloc[-1] if not data.empty else None

    def get_options_chain(self, symbol: str) -> tuple:
        """Get options chain for a stock.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Tuple of (calls DataFrame, puts DataFrame)
        """
        ticker = yf.Ticker(symbol)
        options_dates = ticker.options

        if not options_dates:
            return pd.DataFrame(), pd.DataFrame()

        # Get options for the nearest expiration
        opt = ticker.option_chain(options_dates[0])
        return opt.calls, opt.puts

    def get_dividends(self, symbol: str) -> pd.Series:
        """Get dividend history.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Series with dividend history
        """
        ticker = yf.Ticker(symbol)
        return ticker.dividends

    def get_splits(self, symbol: str) -> pd.Series:
        """Get stock split history.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Series with split history
        """
        ticker = yf.Ticker(symbol)
        return ticker.splits
