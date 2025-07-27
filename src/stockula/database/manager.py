"""Database manager for storing and retrieving yfinance data."""

import sqlite3
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
from pathlib import Path


class DatabaseManager:
    """Manages SQLite database for storing financial data."""

    def __init__(self, db_path: str = "stockula.db"):
        """Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database and create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            # Enable foreign key constraints
            conn.execute("PRAGMA foreign_keys = ON")

            # Create stocks table for basic metadata
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stocks (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    sector TEXT,
                    industry TEXT,
                    market_cap REAL,
                    exchange TEXT,
                    currency TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create price_history table for OHLCV data
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    open_price REAL,
                    high_price REAL,
                    low_price REAL,
                    close_price REAL,
                    volume INTEGER,
                    interval TEXT DEFAULT '1d',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (symbol) REFERENCES stocks(symbol),
                    UNIQUE(symbol, date, interval)
                )
            """)

            # Create dividends table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dividends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    amount REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (symbol) REFERENCES stocks(symbol),
                    UNIQUE(symbol, date)
                )
            """)

            # Create splits table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS splits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    ratio REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (symbol) REFERENCES stocks(symbol),
                    UNIQUE(symbol, date)
                )
            """)

            # Create options_calls table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS options_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    expiration_date DATE NOT NULL,
                    strike REAL NOT NULL,
                    last_price REAL,
                    bid REAL,
                    ask REAL,
                    volume INTEGER,
                    open_interest INTEGER,
                    implied_volatility REAL,
                    in_the_money BOOLEAN,
                    contract_symbol TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (symbol) REFERENCES stocks(symbol),
                    UNIQUE(symbol, expiration_date, strike, contract_symbol)
                )
            """)

            # Create options_puts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS options_puts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    expiration_date DATE NOT NULL,
                    strike REAL NOT NULL,
                    last_price REAL,
                    bid REAL,
                    ask REAL,
                    volume INTEGER,
                    open_interest INTEGER,
                    implied_volatility REAL,
                    in_the_money BOOLEAN,
                    contract_symbol TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (symbol) REFERENCES stocks(symbol),
                    UNIQUE(symbol, expiration_date, strike, contract_symbol)
                )
            """)

            # Create stock_info table for raw yfinance info
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stock_info (
                    symbol TEXT PRIMARY KEY,
                    info_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (symbol) REFERENCES stocks(symbol)
                )
            """)

            # Create indexes for better query performance
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_price_history_symbol_date ON price_history(symbol, date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_dividends_symbol_date ON dividends(symbol, date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_splits_symbol_date ON splits(symbol, date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_options_calls_symbol_exp ON options_calls(symbol, expiration_date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_options_puts_symbol_exp ON options_puts(symbol, expiration_date)"
            )

    def store_stock_info(self, symbol: str, info: Dict[str, Any]) -> None:
        """Store basic stock information.

        Args:
            symbol: Stock ticker symbol
            info: Stock information dictionary from yfinance
        """
        with sqlite3.connect(self.db_path) as conn:
            # Extract key fields for stocks table
            name = info.get("longName") or info.get("shortName", "")
            sector = info.get("sector", "")
            industry = info.get("industry", "")
            market_cap = info.get("marketCap")
            exchange = info.get("exchange", "")
            currency = info.get("currency", "")

            # Insert or update stocks table
            conn.execute(
                """
                INSERT OR REPLACE INTO stocks 
                (symbol, name, sector, industry, market_cap, exchange, currency, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (symbol, name, sector, industry, market_cap, exchange, currency),
            )

            # Store full info as JSON
            info_json = json.dumps(info, default=str)
            conn.execute(
                """
                INSERT OR REPLACE INTO stock_info 
                (symbol, info_json, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
                (symbol, info_json),
            )

    def store_price_history(
        self, symbol: str, data: pd.DataFrame, interval: str = "1d"
    ) -> None:
        """Store historical price data.

        Args:
            symbol: Stock ticker symbol
            data: DataFrame with OHLCV data
            interval: Data interval (1d, 1h, etc.)
        """
        if data.empty:
            return

        with sqlite3.connect(self.db_path) as conn:
            for date, row in data.iterrows():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO price_history 
                    (symbol, date, open_price, high_price, low_price, close_price, volume, interval)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        symbol,
                        date.strftime("%Y-%m-%d"),
                        row.get("Open"),
                        row.get("High"),
                        row.get("Low"),
                        row.get("Close"),
                        row.get("Volume"),
                        interval,
                    ),
                )

    def store_dividends(self, symbol: str, dividends: pd.Series) -> None:
        """Store dividend data.

        Args:
            symbol: Stock ticker symbol
            dividends: Series with dividend data
        """
        if dividends.empty:
            return

        with sqlite3.connect(self.db_path) as conn:
            for date, amount in dividends.items():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO dividends (symbol, date, amount)
                    VALUES (?, ?, ?)
                """,
                    (symbol, date.strftime("%Y-%m-%d"), float(amount)),
                )

    def store_splits(self, symbol: str, splits: pd.Series) -> None:
        """Store stock split data.

        Args:
            symbol: Stock ticker symbol
            splits: Series with split data
        """
        if splits.empty:
            return

        with sqlite3.connect(self.db_path) as conn:
            for date, ratio in splits.items():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO splits (symbol, date, ratio)
                    VALUES (?, ?, ?)
                """,
                    (symbol, date.strftime("%Y-%m-%d"), float(ratio)),
                )

    def store_options_chain(
        self, symbol: str, calls: pd.DataFrame, puts: pd.DataFrame, expiration_date: str
    ) -> None:
        """Store options chain data.

        Args:
            symbol: Stock ticker symbol
            calls: DataFrame with call options
            puts: DataFrame with put options
            expiration_date: Options expiration date
        """
        with sqlite3.connect(self.db_path) as conn:
            # Store calls
            if not calls.empty:
                for _, row in calls.iterrows():
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO options_calls 
                        (symbol, expiration_date, strike, last_price, bid, ask, volume, 
                         open_interest, implied_volatility, in_the_money, contract_symbol)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            symbol,
                            expiration_date,
                            row.get("strike"),
                            row.get("lastPrice"),
                            row.get("bid"),
                            row.get("ask"),
                            row.get("volume"),
                            row.get("openInterest"),
                            row.get("impliedVolatility"),
                            row.get("inTheMoney"),
                            row.get("contractSymbol"),
                        ),
                    )

            # Store puts
            if not puts.empty:
                for _, row in puts.iterrows():
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO options_puts 
                        (symbol, expiration_date, strike, last_price, bid, ask, volume, 
                         open_interest, implied_volatility, in_the_money, contract_symbol)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            symbol,
                            expiration_date,
                            row.get("strike"),
                            row.get("lastPrice"),
                            row.get("bid"),
                            row.get("ask"),
                            row.get("volume"),
                            row.get("openInterest"),
                            row.get("impliedVolatility"),
                            row.get("inTheMoney"),
                            row.get("contractSymbol"),
                        ),
                    )

    def get_price_history(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Retrieve historical price data.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval

        Returns:
            DataFrame with historical price data
        """
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT date, open_price as Open, high_price as High, low_price as Low, 
                       close_price as Close, volume as Volume
                FROM price_history 
                WHERE symbol = ? AND interval = ?
            """
            params = [symbol, interval]

            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)

            query += " ORDER BY date"

            df = pd.read_sql_query(query, conn, params=params)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            return df

    def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieve stock information.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Stock information dictionary or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT info_json FROM stock_info WHERE symbol = ?", (symbol,)
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return None

    def get_dividends(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.Series:
        """Retrieve dividend data.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Series with dividend data
        """
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT date, amount FROM dividends WHERE symbol = ?"
            params = [symbol]

            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)

            query += " ORDER BY date"

            df = pd.read_sql_query(query, conn, params=params)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                return df.set_index("date")["amount"]
            return pd.Series(dtype=float)

    def get_splits(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.Series:
        """Retrieve stock split data.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Series with split data
        """
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT date, ratio FROM splits WHERE symbol = ?"
            params = [symbol]

            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)

            query += " ORDER BY date"

            df = pd.read_sql_query(query, conn, params=params)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                return df.set_index("date")["ratio"]
            return pd.Series(dtype=float)

    def get_options_chain(
        self, symbol: str, expiration_date: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Retrieve options chain data.

        Args:
            symbol: Stock ticker symbol
            expiration_date: Options expiration date

        Returns:
            Tuple of (calls DataFrame, puts DataFrame)
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get calls
            calls_df = pd.read_sql_query(
                """
                SELECT strike, last_price as lastPrice, bid, ask, volume, 
                       open_interest as openInterest, implied_volatility as impliedVolatility,
                       in_the_money as inTheMoney, contract_symbol as contractSymbol
                FROM options_calls 
                WHERE symbol = ? AND expiration_date = ?
                ORDER BY strike
            """,
                conn,
                params=(symbol, expiration_date),
            )

            # Get puts
            puts_df = pd.read_sql_query(
                """
                SELECT strike, last_price as lastPrice, bid, ask, volume, 
                       open_interest as openInterest, implied_volatility as impliedVolatility,
                       in_the_money as inTheMoney, contract_symbol as contractSymbol
                FROM options_puts 
                WHERE symbol = ? AND expiration_date = ?
                ORDER BY strike
            """,
                conn,
                params=(symbol, expiration_date),
            )

            return calls_df, puts_df

    def get_all_symbols(self) -> List[str]:
        """Get all symbols in the database.

        Returns:
            List of all ticker symbols
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT symbol FROM stocks ORDER BY symbol")
            return [row[0] for row in cursor.fetchall()]

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Latest close price or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT close_price FROM price_history 
                WHERE symbol = ? 
                ORDER BY date DESC 
                LIMIT 1
            """,
                (symbol,),
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def has_data(self, symbol: str, start_date: str, end_date: str) -> bool:
        """Check if we have data for a symbol in the given date range.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            True if we have data in the date range
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM price_history 
                WHERE symbol = ? AND date >= ? AND date <= ?
            """,
                (symbol, start_date, end_date),
            )
            count = cursor.fetchone()[0]
            return count > 0

    def cleanup_old_data(self, days_to_keep: int = 365) -> None:
        """Clean up old data to keep database size manageable.

        Args:
            days_to_keep: Number of days of data to keep
        """
        cutoff_date = (datetime.now() - pd.Timedelta(days=days_to_keep)).strftime(
            "%Y-%m-%d"
        )

        with sqlite3.connect(self.db_path) as conn:
            # Clean up old price history
            conn.execute("DELETE FROM price_history WHERE date < ?", (cutoff_date,))

            # Clean up old dividends
            conn.execute("DELETE FROM dividends WHERE date < ?", (cutoff_date,))

            # Clean up old splits
            conn.execute("DELETE FROM splits WHERE date < ?", (cutoff_date,))

            # Clean up old options (they expire anyway)
            conn.execute(
                "DELETE FROM options_calls WHERE expiration_date < ?", (cutoff_date,)
            )
            conn.execute(
                "DELETE FROM options_puts WHERE expiration_date < ?", (cutoff_date,)
            )

            # Vacuum to reclaim space
            conn.execute("VACUUM")

    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics.

        Returns:
            Dictionary with table row counts
        """
        stats = {}
        tables = [
            "stocks",
            "price_history",
            "dividends",
            "splits",
            "options_calls",
            "options_puts",
            "stock_info",
        ]

        with sqlite3.connect(self.db_path) as conn:
            for table in tables:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]

        return stats
