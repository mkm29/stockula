"""Data fetching utilities using yfinance."""

import yfinance as yf
import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from ..database import DatabaseManager


class DataFetcher:
    """Fetch financial data using yfinance with SQLite caching."""

    def __init__(self, use_cache: bool = True, db_path: str = "stockula.db"):
        """Initialize data fetcher.

        Args:
            use_cache: Whether to use database caching
            db_path: Path to SQLite database file
        """
        self.use_cache = use_cache
        self.db = DatabaseManager(db_path) if use_cache else None

    def get_stock_data(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch historical stock data with database caching.

        Args:
            symbol: Stock ticker symbol
            start: Start date (YYYY-MM-DD), defaults to 1 year ago
            end: End date (YYYY-MM-DD), defaults to today
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            force_refresh: Force fetch from yfinance even if cached data exists

        Returns:
            DataFrame with OHLCV data
        """
        if start is None:
            start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        # Try to get data from cache first
        if self.use_cache and not force_refresh:
            try:
                cached_data = self.db.get_price_history(symbol, start, end, interval)
                if not cached_data.empty:
                    # Check if we have complete data for the requested range
                    if self.db.has_data(symbol, start, end):
                        return cached_data
            except Exception as e:
                # If database fails, fall back to yfinance
                print(f"Database error, falling back to yfinance: {e}")

        # Fetch from yfinance
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

        # Store in database if caching is enabled
        if self.use_cache and not data.empty:
            self.db.store_price_history(symbol, data, interval)

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

    def get_current_prices(self, symbols: List[str] | str) -> Dict[str, float]:
        """Get current prices for multiple symbols.

        Args:
            symbols: List of stock ticker symbols or single symbol string

        Returns:
            Dictionary mapping symbols to their current prices
        """
        # Handle single symbol case
        if isinstance(symbols, str):
            symbols = [symbols]

        prices = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                # Get the most recent price
                history = ticker.history(period="1d")
                if not history.empty:
                    prices[symbol] = history["Close"].iloc[-1]
                else:
                    # Fallback to info if history is not available
                    info = ticker.info
                    if "currentPrice" in info:
                        prices[symbol] = info["currentPrice"]
                    elif "regularMarketPrice" in info:
                        prices[symbol] = info["regularMarketPrice"]
                    else:
                        print(f"Warning: Could not get current price for {symbol}")
            except Exception as e:
                print(f"Error fetching price for {symbol}: {e}")

        return prices

    def get_info(self, symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Get stock information with database caching.

        Args:
            symbol: Stock ticker symbol
            force_refresh: Force fetch from yfinance even if cached data exists

        Returns:
            Dictionary with stock information
        """
        # Try to get from cache first
        if self.use_cache and not force_refresh:
            cached_info = self.db.get_stock_info(symbol)
            if cached_info:
                return cached_info

        # Fetch from yfinance
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Store in database if caching is enabled
        if self.use_cache and info:
            self.db.store_stock_info(symbol, info)

        return info

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

    def get_options_chain(
        self,
        symbol: str,
        expiration_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> tuple:
        """Get options chain for a stock with database caching.

        Args:
            symbol: Stock ticker symbol
            expiration_date: Specific expiration date (YYYY-MM-DD), uses nearest if None
            force_refresh: Force fetch from yfinance even if cached data exists

        Returns:
            Tuple of (calls DataFrame, puts DataFrame)
        """
        ticker = yf.Ticker(symbol)
        options_dates = ticker.options

        if not options_dates:
            return pd.DataFrame(), pd.DataFrame()

        # Use nearest expiration if not specified
        if expiration_date is None:
            expiration_date = options_dates[0]

        # Try to get from cache first
        if self.use_cache and not force_refresh:
            cached_calls, cached_puts = self.db.get_options_chain(
                symbol, expiration_date
            )
            if not cached_calls.empty or not cached_puts.empty:
                return cached_calls, cached_puts

        # Fetch from yfinance
        try:
            opt = ticker.option_chain(expiration_date)
            calls, puts = opt.calls, opt.puts

            # Store in database if caching is enabled
            if self.use_cache:
                self.db.store_options_chain(symbol, calls, puts, expiration_date)

            return calls, puts
        except Exception as e:
            print(f"Error fetching options chain for {symbol}: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def get_dividends(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        force_refresh: bool = False,
    ) -> pd.Series:
        """Get dividend history with database caching.

        Args:
            symbol: Stock ticker symbol
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            force_refresh: Force fetch from yfinance even if cached data exists

        Returns:
            Series with dividend history
        """
        # Try to get from cache first
        if self.use_cache and not force_refresh:
            cached_dividends = self.db.get_dividends(symbol, start, end)
            if not cached_dividends.empty:
                return cached_dividends

        # Fetch from yfinance
        ticker = yf.Ticker(symbol)
        dividends = ticker.dividends

        # Store in database if caching is enabled
        if self.use_cache and not dividends.empty:
            self.db.store_dividends(symbol, dividends)

        # Filter by date range if specified
        if start or end:
            if start:
                dividends = dividends[dividends.index >= start]
            if end:
                dividends = dividends[dividends.index <= end]

        return dividends

    def get_splits(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        force_refresh: bool = False,
    ) -> pd.Series:
        """Get stock split history with database caching.

        Args:
            symbol: Stock ticker symbol
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            force_refresh: Force fetch from yfinance even if cached data exists

        Returns:
            Series with split history
        """
        # Try to get from cache first
        if self.use_cache and not force_refresh:
            cached_splits = self.db.get_splits(symbol, start, end)
            if not cached_splits.empty:
                return cached_splits

        # Fetch from yfinance
        ticker = yf.Ticker(symbol)
        splits = ticker.splits

        # Store in database if caching is enabled
        if self.use_cache and not splits.empty:
            self.db.store_splits(symbol, splits)

        # Filter by date range if specified
        if start or end:
            if start:
                splits = splits[splits.index >= start]
            if end:
                splits = splits[splits.index <= end]

        return splits

    def fetch_and_store_all_data(
        self, symbol: str, start: Optional[str] = None, end: Optional[str] = None
    ) -> None:
        """Fetch and store all available data for a symbol.

        Args:
            symbol: Stock ticker symbol
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
        """
        if not self.use_cache:
            print("Warning: Caching is disabled, data will not be stored")
            return

        print(f"Fetching all data for {symbol}...")

        # Fetch and store price history
        try:
            self.get_stock_data(symbol, start, end, force_refresh=True)
            print(f"  ✓ Price history stored")
        except Exception as e:
            print(f"  ✗ Error fetching price history: {e}")

        # Fetch and store stock info
        try:
            self.get_info(symbol, force_refresh=True)
            print(f"  ✓ Stock info stored")
        except Exception as e:
            print(f"  ✗ Error fetching stock info: {e}")

        # Fetch and store dividends
        try:
            dividends = self.get_dividends(symbol, start, end, force_refresh=True)
            if not dividends.empty:
                print(f"  ✓ Dividends stored ({len(dividends)} records)")
            else:
                print(f"  ○ No dividends found")
        except Exception as e:
            print(f"  ✗ Error fetching dividends: {e}")

        # Fetch and store splits
        try:
            splits = self.get_splits(symbol, start, end, force_refresh=True)
            if not splits.empty:
                print(f"  ✓ Splits stored ({len(splits)} records)")
            else:
                print(f"  ○ No splits found")
        except Exception as e:
            print(f"  ✗ Error fetching splits: {e}")

        # Fetch and store options chain
        try:
            calls, puts = self.get_options_chain(symbol, force_refresh=True)
            if not calls.empty or not puts.empty:
                print(
                    f"  ✓ Options chain stored ({len(calls)} calls, {len(puts)} puts)"
                )
            else:
                print(f"  ○ No options chain found")
        except Exception as e:
            print(f"  ✗ Error fetching options chain: {e}")

    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics.

        Returns:
            Dictionary with table row counts
        """
        if not self.use_cache:
            return {}
        return self.db.get_database_stats()

    def cleanup_old_data(self, days_to_keep: int = 365) -> None:
        """Clean up old data to keep database size manageable.

        Args:
            days_to_keep: Number of days of data to keep
        """
        if not self.use_cache:
            print("Warning: Caching is disabled, no data to clean up")
            return
        self.db.cleanup_old_data(days_to_keep)
        print(f"Cleaned up data older than {days_to_keep} days")

    def get_cached_symbols(self) -> List[str]:
        """Get all symbols that have cached data.

        Returns:
            List of symbols with cached data
        """
        if not self.use_cache:
            return []
        return self.db.get_all_symbols()

    def disable_cache(self) -> None:
        """Disable database caching for this session."""
        self.use_cache = False
        self.db = None

    def enable_cache(self, db_path: str = "stockula.db") -> None:
        """Enable database caching for this session.

        Args:
            db_path: Path to SQLite database file
        """
        self.use_cache = True
        self.db = DatabaseManager(db_path)
