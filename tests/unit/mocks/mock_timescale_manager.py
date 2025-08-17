"""Smart mock implementation for TimescaleDB database manager.

This module provides a realistic mock that simulates TimescaleDB features
without requiring an actual TimescaleDB instance. Designed for unit testing
with behavior validation and TimescaleDB-specific feature simulation.
"""

import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from stockula.config.models import TimescaleDBConfig
from stockula.interfaces import IDatabaseManager


class MockTimescaleDBManager(IDatabaseManager):
    """Smart mock that simulates TimescaleDB features for testing.

    This mock provides:
    - Realistic data storage and retrieval
    - TimescaleDB-specific feature simulation
    - Validation of method calls and parameters
    - In-memory data storage for test isolation
    - Performance characteristics simulation
    """

    def __init__(self, config: TimescaleDBConfig):
        """Initialize mock with realistic setup."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # In-memory storage for test data
        self._price_data: dict[str, pd.DataFrame] = {}
        self._stock_info: dict[str, dict[str, Any]] = {}
        self._dividends: dict[str, pd.Series] = {}
        self._splits: dict[str, pd.Series] = {}
        self._options_chains: dict[str, dict[str, dict[str, pd.DataFrame]]] = {}
        self._symbols: set[str] = set()

        # Track method calls for validation
        self.call_history: list[tuple[str, tuple, dict]] = []

        # Simulate database state
        self._connected = True
        self._session_count = 0

        # Mock session object
        self._mock_session = MagicMock()

    def _record_call(self, method_name: str, *args, **kwargs) -> None:
        """Record method call for testing validation."""
        self.call_history.append((method_name, args, kwargs))

    @property
    def backend_type(self) -> str:
        """Get the database backend type."""
        return "timescaledb_mock"

    @property
    def is_timescaledb(self) -> bool:
        """Check if using TimescaleDB backend."""
        return True

    # Core time-series operations
    def get_price_history(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Get price history with realistic filtering."""
        self._record_call("get_price_history", symbol, start_date, end_date, interval)

        if symbol not in self._price_data:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        data = self._price_data[symbol].copy()

        # Apply date filtering
        if start_date:
            data = data[data.index >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data.index <= pd.to_datetime(end_date)]

        return data

    def store_price_history(self, symbol: str, data: pd.DataFrame, interval: str = "1d") -> None:
        """Store price history with validation."""
        self._record_call("store_price_history", symbol, data, interval)

        if data.empty:
            return

        # Validate required columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")

        # Simulate TimescaleDB upsert behavior
        if symbol in self._price_data:
            # Merge with existing data, updating overlapping dates
            existing = self._price_data[symbol]
            combined = pd.concat([existing, data])
            self._price_data[symbol] = combined[~combined.index.duplicated(keep="last")]
        else:
            self._price_data[symbol] = data.copy()

        self._symbols.add(symbol)

    def has_data(self, symbol: str, start_date: str, end_date: str) -> bool:
        """Check data existence with realistic logic."""
        self._record_call("has_data", symbol, start_date, end_date)

        if symbol not in self._price_data:
            return False

        data = self._price_data[symbol]
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # Check if we have any data in the range
        mask = (data.index >= start) & (data.index <= end)
        return bool(mask.any())

    # Stock info operations
    def store_stock_info(self, symbol: str, info: dict[str, Any]) -> None:
        """Store stock information."""
        self._record_call("store_stock_info", symbol, info)
        self._stock_info[symbol] = info.copy()
        self._symbols.add(symbol)

    def get_stock_info(self, symbol: str) -> dict[str, Any] | None:
        """Retrieve stock information."""
        self._record_call("get_stock_info", symbol)
        return self._stock_info.get(symbol)

    def store_dividends(self, symbol: str, dividends: pd.Series) -> None:
        """Store dividend data."""
        self._record_call("store_dividends", symbol, dividends)
        if not dividends.empty:
            self._dividends[symbol] = dividends.copy()

    def store_splits(self, symbol: str, splits: pd.Series) -> None:
        """Store stock split data."""
        self._record_call("store_splits", symbol, splits)
        if not splits.empty:
            self._splits[symbol] = splits.copy()

    # Utility operations
    def get_all_symbols(self) -> list[str]:
        """Get all symbols."""
        self._record_call("get_all_symbols")
        return sorted(self._symbols)

    def get_latest_price(self, symbol: str) -> float | None:
        """Get latest price."""
        self._record_call("get_latest_price", symbol)

        if symbol not in self._price_data:
            return None

        data = self._price_data[symbol]
        if data.empty:
            return None

        return float(data["Close"].iloc[-1])

    def get_database_stats(self) -> dict[str, int]:
        """Get database statistics."""
        self._record_call("get_database_stats")

        total_price_rows = sum(len(df) for df in self._price_data.values())

        return {
            "stocks": len(self._symbols),
            "price_history": total_price_rows,
            "dividends": len(self._dividends),
            "splits": len(self._splits),
        }

    # TimescaleDB-specific features (simulated)
    def get_price_history_aggregated(
        self,
        symbol: str,
        time_bucket: str = "1 day",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Simulate TimescaleDB time_bucket aggregation."""
        self._record_call("get_price_history_aggregated", symbol, time_bucket, start_date, end_date)

        # Get base data
        data = self.get_price_history(symbol, start_date, end_date)
        if data.empty:
            return data

        # Simulate time_bucket behavior
        if time_bucket == "1 hour":
            freq = "H"
        elif time_bucket == "1 day":
            freq = "D"
        elif time_bucket == "1 week":
            freq = "W"
        else:
            freq = "D"  # Default

        # Resample data to simulate time_bucket
        aggregated = (
            data.resample(freq)
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
            .dropna()
        )

        return aggregated

    def get_recent_price_changes(self, symbols: list[str] | None = None, hours: int = 24) -> pd.DataFrame:
        """Simulate recent price change analysis."""
        self._record_call("get_recent_price_changes", symbols, hours)

        if symbols is None:
            symbols = list(self._symbols)

        results = []
        cutoff_time = datetime.now() - timedelta(hours=hours)  # Use timezone-naive datetime

        for symbol in symbols:
            if symbol in self._price_data:
                data = self._price_data[symbol]
                recent_data = data[data.index >= cutoff_time]

                if len(recent_data) >= 2:
                    price_change = recent_data["Close"].iloc[-1] - recent_data["Close"].iloc[0]
                    pct_change = (price_change / recent_data["Close"].iloc[0]) * 100

                    results.append(
                        {
                            "symbol": symbol,
                            "price_change": price_change,
                            "pct_change": pct_change,
                            "current_price": recent_data["Close"].iloc[-1],
                            "volume": recent_data["Volume"].sum(),
                        }
                    )

        return pd.DataFrame(results)

    def get_chunk_statistics(self) -> pd.DataFrame:
        """Simulate TimescaleDB chunk statistics."""
        self._record_call("get_chunk_statistics")

        # Return mock chunk statistics
        chunks = []
        for symbol in self._symbols:
            if symbol in self._price_data:
                data_size = len(self._price_data[symbol])
                chunks.append(
                    {
                        "table_name": "price_history",
                        "chunk_name": f"price_history_{symbol.lower()}",
                        "rows": data_size,
                        "compressed": data_size > 1000,
                        "compression_ratio": 0.7 if data_size > 1000 else 1.0,
                    }
                )

        return pd.DataFrame(chunks)

    # Session management
    @contextmanager
    def get_session(self):
        """Get database session context manager."""
        self._record_call("get_session")
        self._session_count += 1
        try:
            yield self._mock_session
        finally:
            self._session_count -= 1

    def close(self) -> None:
        """Close database connections."""
        self._record_call("close")
        self._connected = False

    def test_connection(self) -> dict[str, Any]:
        """Test database connection."""
        self._record_call("test_connection")
        return {
            "connected": self._connected,
            "backend": "timescaledb_mock",
            "version": "mock-1.0.0",
            "active_sessions": self._session_count,
            "timescale_available": True,
            "url": "postgresql://user:***@localhost:5432/stockula",
            "hypertables_enabled": True,
            "connection_pool": True,
            "pool_size": 10,
            "max_overflow": 20,
        }

    # Analytics methods (simplified implementations)
    def get_moving_averages(
        self,
        symbol: str,
        periods: list[int] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Calculate moving averages using SQL execution pattern for testing."""
        self._record_call("get_moving_averages", symbol, periods, start_date, end_date)

        if periods is None:
            periods = [20, 50, 200]

        # Simulate SQL execution like the real implementation
        from sqlalchemy import text

        # Build a mock query (similar to real implementation)
        ma_clauses = []
        for period in periods:
            ma_clauses.append(f"""
                AVG(close_price) OVER (
                    PARTITION BY symbol
                    ORDER BY timestamp
                    ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW
                ) AS ma_{period}
            """)

        query = f"""
        SELECT
            timestamp,
            symbol,
            close_price,
            {", ".join(ma_clauses)}
        FROM price_history
        WHERE symbol = :symbol
        """

        params = {"symbol": symbol}
        if start_date:
            query += " AND timestamp >= :start_date"
            params["start_date"] = start_date
        if end_date:
            query += " AND timestamp <= :end_date"
            params["end_date"] = end_date

        query += " ORDER BY timestamp"

        # Use the session.exec pattern to match real implementation
        with self.get_session() as session:
            result = session.exec(text(query), params).fetchall()

            if not result:
                return pd.DataFrame()

            # Simulate the result processing
            data = self.get_price_history(symbol, start_date, end_date)
            if data.empty:
                return pd.DataFrame()

            result_df = data[["Close"]].copy()
            for period in periods:
                result_df[f"ma_{period}"] = data["Close"].rolling(window=period).mean()

            result_df = result_df.rename(columns={"Close": "close_price"})
            return result_df

    def get_bollinger_bands(
        self,
        symbol: str,
        period: int = 20,
        std_dev: float = 2.0,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands using SQL execution pattern for testing."""
        self._record_call("get_bollinger_bands", symbol, period, std_dev, start_date, end_date)

        # Simulate SQL execution like the real implementation
        from sqlalchemy import text

        query = f"""
        SELECT timestamp, symbol, close_price, middle_band, upper_band, lower_band, width
        FROM (
            SELECT timestamp, symbol, close_price,
                   AVG(close_price) OVER (
                       ORDER BY timestamp ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW
                   ) AS middle_band,
                   STDDEV(close_price) OVER (
                       ORDER BY timestamp ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW
                   ) AS std_dev
            FROM price_history
            WHERE symbol = :symbol
        ) bb_calc
        """

        params = {"symbol": symbol}
        if start_date:
            query += " AND timestamp >= :start_date"
            params["start_date"] = start_date
        if end_date:
            query += " AND timestamp <= :end_date"
            params["end_date"] = end_date

        query += " ORDER BY timestamp"

        # Use the session.exec pattern to match real implementation
        with self.get_session() as session:
            result = session.exec(text(query), params).fetchall()

            if not result:
                return pd.DataFrame()

            # Simulate the result processing
            data = self.get_price_history(symbol, start_date, end_date)
            if data.empty:
                return pd.DataFrame()

            result_df = data[["Close"]].copy()
            rolling_mean = data["Close"].rolling(window=period).mean()
            rolling_std = data["Close"].rolling(window=period).std()

            result_df["middle_band"] = rolling_mean
            result_df["upper_band"] = rolling_mean + (rolling_std * std_dev)
            result_df["lower_band"] = rolling_mean - (rolling_std * std_dev)
            result_df["width"] = result_df["upper_band"] - result_df["lower_band"]
            result_df = result_df.rename(columns={"Close": "close_price"})

            return result_df

    def get_rsi(
        self,
        symbol: str,
        period: int = 14,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Calculate RSI using SQL execution pattern for testing."""
        self._record_call("get_rsi", symbol, period, start_date, end_date)

        # Simulate SQL execution like the real implementation
        from sqlalchemy import text

        query = f"""
        WITH price_changes AS (
            SELECT timestamp, symbol, close_price,
                   close_price - LAG(close_price) OVER (ORDER BY timestamp) AS price_change
            FROM price_history
            WHERE symbol = :symbol
        ),
        gains_losses AS (
            SELECT timestamp, symbol, close_price, price_change,
                   CASE WHEN price_change > 0 THEN price_change ELSE 0 END AS gain,
                   CASE WHEN price_change < 0 THEN ABS(price_change) ELSE 0 END AS loss
            FROM price_changes
        )
        SELECT timestamp, symbol, close_price, price_change, gain, loss,
               CASE WHEN avg_loss = 0 THEN 100
                    ELSE 100 - (100 / (1 + (avg_gain / avg_loss)))
               END AS rsi
        FROM (
            SELECT timestamp, symbol, close_price, price_change, gain, loss,
                   AVG(gain) OVER (ORDER BY timestamp ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW) AS avg_gain,
                   AVG(loss) OVER (ORDER BY timestamp ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW) AS avg_loss
            FROM gains_losses
        ) rsi_calc
        """

        params = {"symbol": symbol}
        if start_date:
            query += " AND timestamp >= :start_date"
            params["start_date"] = start_date
        if end_date:
            query += " AND timestamp <= :end_date"
            params["end_date"] = end_date

        query += " ORDER BY timestamp"

        # Use the session.exec pattern to match real implementation
        with self.get_session() as session:
            result = session.exec(text(query), params).fetchall()

            if not result:
                return pd.DataFrame()

            # Simulate the result processing
            data = self.get_price_history(symbol, start_date, end_date)
            if data.empty:
                return pd.DataFrame()

            delta = data["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            result_df = data[["Close"]].copy()
            result_df["price_change"] = delta
            result_df["gain"] = delta.where(delta > 0, 0)
            result_df["loss"] = -delta.where(delta < 0, 0)
            result_df["rsi"] = rsi
            result_df = result_df.rename(columns={"Close": "close_price"})

            return result_df

    # Placeholder implementations for remaining analytics methods
    def get_price_momentum(
        self,
        symbols: list[str] | None = None,
        lookback_days: int = 30,
        time_bucket: str = "1 day",
    ) -> pd.DataFrame:
        """Calculate price momentum."""
        self._record_call("get_price_momentum", symbols, lookback_days, time_bucket)
        return pd.DataFrame()  # Simplified for space

    def get_correlation_matrix(
        self,
        symbols: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        time_bucket: str = "1 day",
    ) -> pd.DataFrame:
        """Calculate correlation matrix."""
        self._record_call("get_correlation_matrix", symbols, start_date, end_date, time_bucket)
        return pd.DataFrame()  # Simplified for space

    def get_volatility_analysis(
        self,
        symbols: list[str] | None = None,
        lookback_days: int = 30,
        time_bucket: str = "1 day",
    ) -> pd.DataFrame:
        """Calculate volatility metrics."""
        self._record_call("get_volatility_analysis", symbols, lookback_days, time_bucket)
        return pd.DataFrame()  # Simplified for space

    def get_seasonal_patterns(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Analyze seasonal patterns."""
        self._record_call("get_seasonal_patterns", symbol, start_date, end_date)
        return pd.DataFrame()  # Simplified for space

    def get_top_performers(
        self,
        timeframe_days: int = 30,
        limit: int = 10,
    ) -> pd.DataFrame:
        """Get top performing stocks."""
        self._record_call("get_top_performers", timeframe_days, limit)
        return pd.DataFrame()  # Simplified for space

    # Options chain operations
    def store_options_chain(self, symbol: str, calls: pd.DataFrame, puts: pd.DataFrame, expiration_date: str) -> None:
        """Store options chain data."""
        self._record_call("store_options_chain", symbol, calls, puts, expiration_date)

        if symbol not in self._options_chains:
            self._options_chains[symbol] = {}

        self._options_chains[symbol][expiration_date] = {
            "calls": calls.copy() if not calls.empty else pd.DataFrame(),
            "puts": puts.copy() if not puts.empty else pd.DataFrame(),
        }

    def get_options_chain(self, symbol: str, expiration_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Retrieve options chain data."""
        self._record_call("get_options_chain", symbol, expiration_date)

        if (
            hasattr(self, "_options_chains")
            and symbol in self._options_chains
            and expiration_date in self._options_chains[symbol]
        ):
            data = self._options_chains[symbol][expiration_date]
            return data["calls"], data["puts"]

        # Return empty DataFrames with expected columns
        empty_calls = pd.DataFrame(
            columns=["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]
        )
        empty_puts = pd.DataFrame(
            columns=["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]
        )
        return empty_calls, empty_puts

    def get_latest_price_date(self, symbol: str) -> datetime | None:
        """Get the latest price date for a symbol."""
        self._record_call("get_latest_price_date", symbol)

        if symbol in self._price_data and not self._price_data[symbol].empty:
            return self._price_data[symbol].index[-1].to_pydatetime()

        return None

    def cleanup_old_data(self, days_to_keep: int = 365) -> int:
        """Clean up old data, returning number of rows deleted."""
        self._record_call("cleanup_old_data", days_to_keep)

        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        deleted_rows = 0

        # Simulate cleanup by removing old data
        for symbol in list(self._price_data.keys()):
            if symbol in self._price_data:
                original_len = len(self._price_data[symbol])
                self._price_data[symbol] = self._price_data[symbol][self._price_data[symbol].index >= cutoff_date]
                deleted_rows += original_len - len(self._price_data[symbol])

        return deleted_rows

    def get_dividends(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.Series:
        """Get dividend data for a symbol."""
        self._record_call("get_dividends", symbol, start_date, end_date)

        if symbol in self._dividends:
            return self._dividends[symbol]

        # Return empty Series with datetime index
        return pd.Series([], dtype="float64", name="dividends")

    def get_splits(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.Series:
        """Get split data for a symbol."""
        self._record_call("get_splits", symbol, start_date, end_date)

        if symbol in self._splits:
            return self._splits[symbol]

        # Return empty Series with datetime index
        return pd.Series([], dtype="float64", name="splits")

    # Testing utilities
    def get_call_count(self, method_name: str) -> int:
        """Get number of times a method was called."""
        return sum(1 for call in self.call_history if call[0] == method_name)

    def get_last_call(self, method_name: str) -> tuple[tuple, dict] | None:
        """Get the last call arguments for a method."""
        for call in reversed(self.call_history):
            if call[0] == method_name:
                return call[1], call[2]
        return None

    def reset_call_history(self) -> None:
        """Reset call history for testing."""
        self.call_history.clear()

    def populate_test_data(self, symbols: list[str], days: int = 100) -> None:
        """Populate mock with test data for efficient testing."""
        base_date = datetime.now() - timedelta(days=days)

        for symbol in symbols:
            # Generate realistic price data
            dates = pd.date_range(start=base_date, periods=days, freq="D")
            base_price = np.random.uniform(50, 500)

            # Random walk with trend
            returns = np.random.normal(0.001, 0.02, days)  # 0.1% daily return, 2% volatility
            prices = base_price * np.exp(np.cumsum(returns))

            data = pd.DataFrame(
                {
                    "Open": prices * np.random.uniform(0.995, 1.005, days),
                    "High": prices * np.random.uniform(1.00, 1.03, days),
                    "Low": prices * np.random.uniform(0.97, 1.00, days),
                    "Close": prices,
                    "Volume": np.random.randint(100000, 10000000, days),
                },
                index=dates,
            )

            # Ensure High >= Close >= Low
            data["High"] = data[["Open", "High", "Close"]].max(axis=1)
            data["Low"] = data[["Open", "Low", "Close"]].min(axis=1)

            self.store_price_history(symbol, data)
            self.store_stock_info(
                symbol,
                {
                    "longName": f"{symbol} Inc.",
                    "sector": "Technology",
                    "marketCap": np.random.randint(1000000000, 2000000000000),
                },
            )
