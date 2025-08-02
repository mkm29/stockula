"""Data fetching utilities using yfinance."""

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf
from dependency_injector.wiring import Provide, inject
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from ..database import DatabaseManager
from ..interfaces import ILoggingManager

console = Console()


class DataFetcher:
    """Fetch financial data using yfinance with SQLite caching."""

    # Treasury ETF tickers that track short-term rates
    TREASURY_TICKERS = {
        "3_month": "^IRX",  # 3-Month Treasury Bill
        "13_week": "^IRX",  # Same as 3-month
        "1_year": "^FVX",  # 5-Year Treasury (closest proxy)
        "tbill_etf": "BIL",  # SPDR Bloomberg Barclays 1-3 Month T-Bill ETF
        "sgov": "SGOV",  # iShares 0-3 Month Treasury Bond ETF
    }

    @inject
    def __init__(
        self,
        use_cache: bool = True,
        db_path: str = "stockula.db",
        database_manager: DatabaseManager | None = None,
        logging_manager: ILoggingManager = Provide["logging_manager"],
    ):
        """Initialize data fetcher.

        Args:
            use_cache: Whether to use database caching
            db_path: Path to SQLite database file
            database_manager: Injected database manager instance
            logging_manager: Injected logging manager
        """
        self.use_cache = use_cache
        self._owns_db = False  # Track if we created the DB manager
        self.logger = logging_manager

        # Use injected database manager if provided, otherwise create one
        if database_manager is not None:
            self.db = database_manager if use_cache else None
        else:
            self.db = DatabaseManager(db_path) if use_cache else None
            self._owns_db = use_cache  # We own it if we created it

    def close(self) -> None:
        """Close the database manager if we own it."""
        if self._owns_db and self.db is not None:
            self.db.close()
            self.db = None

    def __del__(self) -> None:
        """Ensure database connections are closed when object is destroyed."""
        try:
            self.close()
        except Exception:
            # Ignore errors during cleanup
            pass

    def get_stock_data(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
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
        if self.use_cache and not force_refresh and self.db is not None:
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
        if self.use_cache and not data.empty and self.db is not None:
            self.db.store_price_history(symbol, data, interval)

        return data

    def get_multiple_stocks(
        self,
        symbols: list[str],
        start: str | None = None,
        end: str | None = None,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
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

    def get_current_prices(self, symbols: list[str] | str, show_progress: bool = True) -> dict[str, float]:
        """Get current prices for multiple symbols.

        Args:
            symbols: List of stock ticker symbols or single symbol string
            show_progress: Whether to show progress bars for multiple symbols

        Returns:
            Dictionary mapping symbols to their current prices
        """
        # Handle single symbol case
        if isinstance(symbols, str):
            symbols = [symbols]

        prices = {}

        # Show progress bar only for multiple symbols
        if show_progress and len(symbols) > 1:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task(
                    f"[magenta]Fetching current prices for {len(symbols)} symbols...",
                    total=len(symbols),
                )

                for symbol in symbols:
                    progress.update(task, description=f"[magenta]Fetching price for {symbol}...")
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
                                console.print(f"[yellow]Warning: Could not get current price for {symbol}[/yellow]")
                    except Exception as e:
                        console.print(f"[red]Error fetching price for {symbol}: {e}[/red]")

                    progress.advance(task)
        else:
            # No progress bar for single symbol or when disabled
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
                            console.print(f"[yellow]Warning: Could not get current price for {symbol}[/yellow]")
                except Exception as e:
                    console.print(f"[red]Error fetching price for {symbol}: {e}[/red]")

        return prices

    def get_info(self, symbol: str, force_refresh: bool = False) -> dict[str, Any]:
        """Get stock information with database caching.

        Args:
            symbol: Stock ticker symbol
            force_refresh: Force fetch from yfinance even if cached data exists

        Returns:
            Dictionary with stock information
        """
        # Try to get from cache first
        if self.use_cache and not force_refresh and self.db is not None:
            cached_info = self.db.get_stock_info(symbol)
            if cached_info:
                return cached_info

        # Fetch from yfinance
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Store in database if caching is enabled
        if self.use_cache and info and self.db is not None:
            self.db.store_stock_info(symbol, info)

        return info

    def get_realtime_price(self, symbol: str) -> float | None:
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
        expiration_date: str | None = None,
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
        if self.use_cache and not force_refresh and self.db is not None:
            cached_calls, cached_puts = self.db.get_options_chain(symbol, expiration_date)
            if not cached_calls.empty or not cached_puts.empty:
                return cached_calls, cached_puts

        # Fetch from yfinance
        try:
            opt = ticker.option_chain(expiration_date)
            calls, puts = opt.calls, opt.puts

            # Store in database if caching is enabled
            if self.use_cache and self.db is not None:
                self.db.store_options_chain(symbol, calls, puts, expiration_date)

            return calls, puts
        except Exception as e:
            print(f"Error fetching options chain for {symbol}: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def get_dividends(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
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
        if self.use_cache and not force_refresh and self.db is not None:
            cached_dividends = self.db.get_dividends(symbol, start, end)
            if not cached_dividends.empty:
                return cached_dividends

        # Fetch from yfinance
        ticker = yf.Ticker(symbol)
        dividends = ticker.dividends

        # Store in database if caching is enabled
        if self.use_cache and not dividends.empty and self.db is not None:
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
        start: str | None = None,
        end: str | None = None,
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
        if self.use_cache and not force_refresh and self.db is not None:
            cached_splits = self.db.get_splits(symbol, start, end)
            if not cached_splits.empty:
                return cached_splits

        # Fetch from yfinance
        ticker = yf.Ticker(symbol)
        splits = ticker.splits

        # Store in database if caching is enabled
        if self.use_cache and not splits.empty and self.db is not None:
            self.db.store_splits(symbol, splits)

        # Filter by date range if specified
        if start or end:
            if start:
                splits = splits[splits.index >= start]
            if end:
                splits = splits[splits.index <= end]

        return splits

    def get_stock_data_batch(
        self,
        symbols: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """Fetch stock data for multiple symbols efficiently.

        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1d, 1wk, 1mo)

        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        results = {}

        # Check cache first for each symbol
        symbols_to_fetch = []
        for symbol in symbols:
            if self.use_cache:
                cached_data = self.get_stock_data(symbol, start_date, end_date, interval)
                if not cached_data.empty:
                    results[symbol] = cached_data
                    continue
            symbols_to_fetch.append(symbol)

        # Batch download remaining symbols
        if symbols_to_fetch:
            try:
                # yfinance supports space-separated symbols for batch download
                batch_str = " ".join(symbols_to_fetch)
                data = yf.download(
                    batch_str,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    group_by="ticker" if len(symbols_to_fetch) > 1 else None,
                    auto_adjust=True,
                    prepost=False,
                    threads=True,
                    progress=False,
                )

                # Handle single vs multiple symbols
                if len(symbols_to_fetch) == 1:
                    symbol = symbols_to_fetch[0]
                    if not data.empty:
                        results[symbol] = data
                        # Store in cache
                        if self.use_cache and self.db is not None:
                            self.db.store_price_history(symbol, data, interval)
                else:
                    # Multiple symbols returns multi-level columns
                    for symbol in symbols_to_fetch:
                        try:
                            symbol_data = data[symbol]
                            if not symbol_data.empty and not symbol_data.isna().all().all():
                                symbol_data = symbol_data.dropna(how="all")
                                results[symbol] = symbol_data
                                # Store in cache
                                if self.use_cache and self.db is not None:
                                    self.db.store_price_history(symbol, symbol_data, interval)
                        except KeyError:
                            self.logger.warning(f"No data returned for {symbol}")

            except Exception as e:
                self.logger.error(f"Error in batch download: {e}")
                # Fall back to individual downloads
                for symbol in symbols_to_fetch:
                    try:
                        data = self.get_stock_data(symbol, start_date, end_date, interval)
                        if not data.empty:
                            results[symbol] = data
                    except Exception as e:
                        self.logger.error(f"Error fetching {symbol}: {e}")

        return results

    def get_current_prices_batch(self, symbols: list[str]) -> dict[str, float]:
        """Get current prices for multiple symbols efficiently.

        Args:
            symbols: List of stock symbols

        Returns:
            Dictionary mapping symbols to their current prices
        """
        prices = {}

        # Use Tickers class for batch operations
        tickers = yf.Tickers(" ".join(symbols))

        for symbol in symbols:
            try:
                ticker = tickers.tickers[symbol]
                # Try fast info first
                fast_info = ticker.fast_info
                if hasattr(fast_info, "last_price") and fast_info.last_price:
                    prices[symbol] = float(fast_info.last_price)
                else:
                    # Fallback to regular info
                    info = ticker.info
                    price = info.get("regularMarketPrice") or info.get("previousClose")
                    if price:
                        prices[symbol] = float(price)
            except Exception as e:
                self.logger.error(f"Error getting price for {symbol}: {e}")
                # Try individual fetch as fallback
                try:
                    individual_prices = self.get_current_prices(symbol, show_progress=False)
                    if symbol in individual_prices:
                        prices[symbol] = individual_prices[symbol]
                except Exception:
                    pass

        return prices

    def fetch_and_store_all_data(self, symbol: str, start: str | None = None, end: str | None = None) -> None:
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
            print("  ✓ Price history stored")
        except Exception as e:
            print(f"  ✗ Error fetching price history: {e}")

        # Fetch and store stock info
        try:
            self.get_info(symbol, force_refresh=True)
            print("  ✓ Stock info stored")
        except Exception as e:
            print(f"  ✗ Error fetching stock info: {e}")

        # Fetch and store dividends
        try:
            dividends = self.get_dividends(symbol, start, end, force_refresh=True)
            if not dividends.empty:
                print(f"  ✓ Dividends stored ({len(dividends)} records)")
            else:
                print("  ○ No dividends found")
        except Exception as e:
            print(f"  ✗ Error fetching dividends: {e}")

        # Fetch and store splits
        try:
            splits = self.get_splits(symbol, start, end, force_refresh=True)
            if not splits.empty:
                print(f"  ✓ Splits stored ({len(splits)} records)")
            else:
                print("  ○ No splits found")
        except Exception as e:
            print(f"  ✗ Error fetching splits: {e}")

        # Fetch and store options chain
        try:
            calls, puts = self.get_options_chain(symbol, force_refresh=True)
            if not calls.empty or not puts.empty:
                print(f"  ✓ Options chain stored ({len(calls)} calls, {len(puts)} puts)")
            else:
                print("  ○ No options chain found")
        except Exception as e:
            print(f"  ✗ Error fetching options chain: {e}")

    def get_database_stats(self) -> dict[str, int]:
        """Get database statistics.

        Returns:
            Dictionary with table row counts
        """
        if not self.use_cache or self.db is None:
            return {}
        return self.db.get_database_stats()

    def cleanup_old_data(self, days_to_keep: int = 365) -> None:
        """Clean up old data to keep database size manageable.

        Args:
            days_to_keep: Number of days of data to keep
        """
        if not self.use_cache or self.db is None:
            print("Warning: Caching is disabled, no data to clean up")
            return
        self.db.cleanup_old_data(days_to_keep)
        print(f"Cleaned up data older than {days_to_keep} days")

    def get_cached_symbols(self) -> list[str]:
        """Get all symbols that have cached data.

        Returns:
            List of symbols with cached data
        """
        if not self.use_cache or self.db is None:
            return []
        return self.db.get_all_symbols()

    def disable_cache(self) -> None:
        """Disable database caching for this session."""
        self.use_cache = False
        if self._owns_db and self.db is not None:
            self.db.close()
        self.db = None
        self._owns_db = False

    def enable_cache(self, db_path: str = "stockula.db") -> None:
        """Enable database caching for this session.

        Args:
            db_path: Path to SQLite database file
        """
        # Close existing DB if we own it
        if self._owns_db and self.db is not None:
            self.db.close()

        self.use_cache = True
        self.db = DatabaseManager(db_path)
        self._owns_db = True

    def get_treasury_rate(
        self,
        date: str | datetime,
        duration: str = "3_month",
        as_decimal: bool = True,
        force_refresh: bool = False,
    ) -> float | None:
        """Get Treasury rate for a specific date.

        Args:
            date: Date to get rate for (YYYY-MM-DD or datetime)
            duration: Treasury duration ('3_month', '13_week', '1_year')
            as_decimal: Return as decimal (0.05) vs percentage (5.0)
            force_refresh: Force fetch from yfinance even if cached

        Returns:
            Treasury rate or None if not available
        """
        # Convert date to datetime if it's a string
        date_obj: datetime
        if isinstance(date, str):
            date_obj = pd.to_datetime(date)
        else:
            date_obj = date

        # Get ticker for requested duration
        ticker = self.TREASURY_TICKERS.get(duration, self.TREASURY_TICKERS["3_month"])

        # Try to get cached data first
        if self.use_cache and self.db and not force_refresh:
            cached_data = self._get_cached_rate(ticker, date_obj)
            if cached_data is not None:
                return cached_data if as_decimal else cached_data * 100

        # Fetch from yfinance
        rate = self._fetch_rate_from_yfinance(ticker, date_obj)

        if rate is not None:
            # Cache the result
            if self.use_cache and self.db:
                self._cache_rate(ticker, date_obj, rate)

            return rate if as_decimal else rate * 100

        return None

    def get_average_treasury_rate(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
        duration: str = "3_month",
        as_decimal: bool = True,
    ) -> float | None:
        """Get average Treasury rate for a date range.

        Args:
            start_date: Start date (YYYY-MM-DD or datetime)
            end_date: End date (YYYY-MM-DD or datetime)
            duration: Treasury duration ('3_month', '13_week', '1_year')
            as_decimal: Return as decimal (0.05) vs percentage (5.0)

        Returns:
            Average Treasury rate or None if not available
        """
        # Type narrowing is handled in get_treasury_rates
        rates = self.get_treasury_rates(start_date, end_date, duration, as_decimal)

        if rates.empty:
            return None

        return rates.mean()

    def get_treasury_rates(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
        duration: str = "3_month",
        as_decimal: bool = True,
        force_refresh: bool = False,
    ) -> pd.Series:
        """Get Treasury rates for a date range.

        Args:
            start_date: Start date (YYYY-MM-DD or datetime)
            end_date: End date (YYYY-MM-DD or datetime)
            duration: Treasury duration ('3_month', '13_week', '1_year')
            as_decimal: Return as decimal (0.05) vs percentage (5.0)
            force_refresh: Force fetch from yfinance even if cached

        Returns:
            Series of Treasury rates indexed by date
        """
        # Convert to datetime objects
        start_obj: datetime
        end_obj: datetime
        if isinstance(start_date, str):
            start_obj = pd.to_datetime(start_date)
        else:
            start_obj = start_date
        if isinstance(end_date, str):
            end_obj = pd.to_datetime(end_date)
        else:
            end_obj = end_date

        ticker = self.TREASURY_TICKERS.get(duration, self.TREASURY_TICKERS["3_month"])

        # Try cache first
        if self.use_cache and self.db and not force_refresh:
            cached_data = self._get_cached_rates(ticker, start_obj, end_obj)
            if not cached_data.empty:
                return cached_data if as_decimal else cached_data * 100

        # Fetch from yfinance with progress indication
        rates = self._fetch_rates_from_yfinance(ticker, start_obj, end_obj, show_progress=True)

        # Cache the results
        if self.use_cache and self.db and not rates.empty:
            self._cache_rates(ticker, rates)

        return rates if as_decimal else rates * 100

    def _fetch_rate_from_yfinance(self, ticker: str, date: datetime) -> float | None:
        """Fetch single Treasury rate from yfinance."""
        # Fetch a few days around the target date
        start_date = date - timedelta(days=5)
        end_date = date + timedelta(days=1)

        try:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(start=start_date, end=end_date)

            if data.empty:
                return None

            # Find closest date
            closest_idx = data.index.get_indexer([date], method="nearest")[0]
            if closest_idx >= 0:
                # Convert percentage to decimal (^IRX returns as percentage)
                rate = data.iloc[closest_idx]["Close"]
                if ticker == "^IRX":  # Treasury bill indices are in percentage
                    rate = rate / 100.0
                return rate

        except Exception as e:
            print(f"Error fetching Treasury rate for {ticker} on {date}: {e}")

        return None

    def _fetch_rates_from_yfinance(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        show_progress: bool = False,
    ) -> pd.Series:
        """Fetch Treasury rates from yfinance for date range."""
        try:
            if show_progress:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True,
                ) as progress:
                    task = progress.add_task(
                        f"[yellow]Fetching Treasury rates ({ticker}) from "
                        f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...",
                        total=None,
                    )

                    ticker_obj = yf.Ticker(ticker)
                    data = ticker_obj.history(start=start_date, end=end_date)

                    progress.update(task, description="[yellow]Processing Treasury rate data...")
            else:
                ticker_obj = yf.Ticker(ticker)
                data = ticker_obj.history(start=start_date, end=end_date)

            if data.empty:
                return pd.Series(dtype=float)

            # Use Close prices as rates
            rates = data["Close"].copy()

            # Convert percentage to decimal for Treasury indices
            if ticker == "^IRX":  # Treasury bill indices are in percentage
                rates = rates / 100.0

            return rates

        except Exception as e:
            console.print(f"[red]Error fetching Treasury rates for {ticker}: {e}[/red]")
            return pd.Series(dtype=float)

    def _get_cached_rate(self, ticker: str, date: datetime) -> float | None:
        """Get cached Treasury rate from database."""
        if not self.db:
            return None

        # Format date for query
        date_str = date.strftime("%Y-%m-%d")

        # Use the existing price history table for Treasury data
        if self.db is None:
            return None
        data = self.db.get_price_history(ticker, start_date=date_str, end_date=date_str)

        if not data.empty:
            return data.iloc[0]["Close"]

        return None

    def _get_cached_rates(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.Series:
        """Get cached Treasury rates from database."""
        if not self.db:
            return pd.Series(dtype=float)

        # Format dates for query
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # Use the existing price history table for Treasury data
        if self.db is None:
            return pd.Series(dtype=float)
        data = self.db.get_price_history(ticker, start_date=start_str, end_date=end_str)

        if not data.empty:
            return data["Close"]

        return pd.Series(dtype=float)

    def _cache_rate(self, ticker: str, date: datetime, rate: float) -> None:
        """Cache Treasury rate in database."""
        if not self.db:
            return

        # Create DataFrame for storing
        df = pd.DataFrame(
            {
                "Open": [rate],
                "High": [rate],
                "Low": [rate],
                "Close": [rate],
                "Volume": [0],
            },
            index=[date],
        )

        # Store as stock info (Treasury ticker)
        if self.db is not None:
            self.db.store_stock_info(ticker, {"longName": f"Treasury Rate {ticker}"})

            # Store rate data
            self.db.store_price_history(ticker, df)

    def _cache_rates(self, ticker: str, rates: pd.Series) -> None:
        """Cache Treasury rates in database."""
        if not self.db or rates.empty:
            return

        # Create DataFrame for storing
        df = pd.DataFrame(
            {
                "Open": rates,
                "High": rates,
                "Low": rates,
                "Close": rates,
                "Volume": 0,
            }
        )

        # Store as stock info (Treasury ticker)
        if self.db is not None:
            self.db.store_stock_info(ticker, {"longName": f"Treasury Rate {ticker}"})

            # Store rate data
            self.db.store_price_history(ticker, df)

    def get_current_treasury_rate(self, duration: str = "3_month") -> float | None:
        """Get the most recent Treasury rate.

        Args:
            duration: Treasury duration ('3_month', '13_week', '1_year')

        Returns:
            Most recent Treasury rate as decimal or None
        """
        today = datetime.now()
        return self.get_treasury_rate(today, duration=duration)
