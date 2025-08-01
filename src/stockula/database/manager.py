"""Database manager using SQLModel for type-safe database operations."""

import os
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import event
from sqlmodel import Session, SQLModel, create_engine, select

from alembic import command  # type: ignore[attr-defined]
from alembic.config import Config

from .models import (
    Dividend,
    OptionsCall,
    OptionsPut,
    PriceHistory,
    Split,
    Stock,
    StockInfo,
)


class DatabaseManager:
    """Manages SQLite database using SQLModel for type-safe operations."""

    def __init__(self, db_path: str = "stockula.db"):
        """Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_url = f"sqlite:///{self.db_path}"

        # Create engine with foreign key support
        self.engine = create_engine(self.db_url, connect_args={"check_same_thread": False}, echo=False)

        # Enable foreign keys for SQLite
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        self._run_migrations()
        # Create tables if they don't exist (for development)
        SQLModel.metadata.create_all(self.engine)

    def _run_migrations(self) -> None:
        """Run Alembic migrations to ensure database schema is up to date."""
        import os

        # Skip migrations in test environment
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return

        # Find alembic.ini file relative to the project root
        project_root = Path(__file__).parents[3]
        alembic_ini_path = project_root / "alembic.ini"

        if not alembic_ini_path.exists():
            # Try common locations
            possible_paths = [
                Path.cwd() / "alembic.ini",
                Path(__file__).parent.parent.parent / "alembic.ini",
            ]
            for path in possible_paths:
                if path.exists():
                    alembic_ini_path = path
                    break
            else:
                # Skip migrations if alembic.ini not found
                return

        # Configure Alembic
        alembic_cfg = Config(str(alembic_ini_path))
        alembic_cfg.set_main_option("sqlalchemy.url", self.db_url)

        # Run migrations
        try:
            # Suppress Alembic info messages
            import logging

            logging.getLogger("alembic.runtime.migration").setLevel(logging.WARNING)
            command.upgrade(alembic_cfg, "head")
        except Exception as e:
            print(f"Warning: Could not run migrations: {e}")

    def close(self) -> None:
        """Close the database engine and dispose of all connections."""
        if hasattr(self, "engine") and self.engine:
            self.engine.dispose()

    def __del__(self) -> None:
        """Ensure database connections are closed when object is destroyed."""
        try:
            self.close()
        except Exception:
            # Ignore errors during cleanup
            pass

    @contextmanager
    def get_session(self) -> Session:
        """Get a database session as context manager."""
        with Session(self.engine) as session:
            yield session

    def store_stock_info(self, symbol: str, info: dict[str, Any]) -> None:
        """Store basic stock information.

        Args:
            symbol: Stock ticker symbol
            info: Stock information dictionary from yfinance
        """
        with self.get_session() as session:
            # Create or update stock
            stock = session.get(Stock, symbol)
            if not stock:
                stock = Stock(symbol=symbol)

            # Update fields
            stock.name = info.get("longName") or info.get("shortName", "")
            stock.sector = info.get("sector", "")
            stock.industry = info.get("industry", "")
            stock.market_cap = info.get("marketCap")
            stock.exchange = info.get("exchange", "")
            stock.currency = info.get("currency", "")
            stock.updated_at = datetime.now(UTC)

            session.add(stock)

            # Store full info as JSON
            stock_info = session.get(StockInfo, symbol)
            if not stock_info:
                stock_info = StockInfo(symbol=symbol)

            stock_info.set_info(info)
            stock_info.updated_at = datetime.now(UTC)

            session.add(stock_info)
            session.commit()

    def store_price_history(self, symbol: str, data: pd.DataFrame, interval: str = "1d") -> None:
        """Store historical price data.

        Args:
            symbol: Stock ticker symbol
            data: DataFrame with OHLCV data
            interval: Data interval (1d, 1h, etc.)
        """
        if data.empty:
            return

        with self.get_session() as session:
            # Ensure stock exists
            stock = session.get(Stock, symbol)
            if not stock:
                stock = Stock(symbol=symbol)
                session.add(stock)

            for date, row in data.iterrows():
                # Check if record exists
                stmt = select(PriceHistory).where(
                    PriceHistory.symbol == symbol,
                    PriceHistory.date == date.date(),
                    PriceHistory.interval == interval,
                )
                price_history = session.exec(stmt).first()

                if not price_history:
                    price_history = PriceHistory(symbol=symbol, date=date.date(), interval=interval)

                # Update values
                price_history.open_price = row.get("Open")
                price_history.high_price = row.get("High")
                price_history.low_price = row.get("Low")
                price_history.close_price = row.get("Close")
                price_history.volume = row.get("Volume")

                session.add(price_history)

            session.commit()

    def store_dividends(self, symbol: str, dividends: pd.Series) -> None:
        """Store dividend data.

        Args:
            symbol: Stock ticker symbol
            dividends: Series with dividend data
        """
        if dividends.empty:
            return

        with self.get_session() as session:
            # Ensure stock exists
            stock = session.get(Stock, symbol)
            if not stock:
                stock = Stock(symbol=symbol)
                session.add(stock)

            for date, amount in dividends.items():
                # Check if record exists
                stmt = select(Dividend).where(Dividend.symbol == symbol, Dividend.date == date.date())
                dividend = session.exec(stmt).first()

                if not dividend:
                    dividend = Dividend(symbol=symbol, date=date.date(), amount=float(amount))
                else:
                    dividend.amount = float(amount)

                session.add(dividend)

            session.commit()

    def store_splits(self, symbol: str, splits: pd.Series) -> None:
        """Store stock split data.

        Args:
            symbol: Stock ticker symbol
            splits: Series with split data
        """
        if splits.empty:
            return

        with self.get_session() as session:
            # Ensure stock exists
            stock = session.get(Stock, symbol)
            if not stock:
                stock = Stock(symbol=symbol)
                session.add(stock)

            for date, ratio in splits.items():
                # Check if record exists
                stmt = select(Split).where(Split.symbol == symbol, Split.date == date.date())
                split = session.exec(stmt).first()

                if not split:
                    split = Split(symbol=symbol, date=date.date(), ratio=float(ratio))
                else:
                    split.ratio = float(ratio)

                session.add(split)

            session.commit()

    def store_options_chain(self, symbol: str, calls: pd.DataFrame, puts: pd.DataFrame, expiration_date: str) -> None:
        """Store options chain data.

        Args:
            symbol: Stock ticker symbol
            calls: DataFrame with call options
            puts: DataFrame with put options
            expiration_date: Options expiration date
        """
        expiry_date = datetime.strptime(expiration_date, "%Y-%m-%d").date()

        with self.get_session() as session:
            # Ensure stock exists
            stock = session.get(Stock, symbol)
            if not stock:
                stock = Stock(symbol=symbol)
                session.add(stock)

            # Store calls
            if not calls.empty:
                for _, row in calls.iterrows():
                    # Check if record exists
                    stmt = select(OptionsCall).where(
                        OptionsCall.symbol == symbol,
                        OptionsCall.expiration_date == expiry_date,
                        OptionsCall.strike == row.get("strike"),
                        OptionsCall.contract_symbol == row.get("contractSymbol"),
                    )
                    option = session.exec(stmt).first()

                    if not option:
                        option = OptionsCall(
                            symbol=symbol,
                            expiration_date=expiry_date,
                            strike=row.get("strike"),
                        )

                    # Update values
                    option.last_price = row.get("lastPrice")
                    option.bid = row.get("bid")
                    option.ask = row.get("ask")
                    option.volume = row.get("volume")
                    option.open_interest = row.get("openInterest")
                    option.implied_volatility = row.get("impliedVolatility")
                    option.in_the_money = row.get("inTheMoney")
                    option.contract_symbol = row.get("contractSymbol")

                    session.add(option)

            # Store puts
            if not puts.empty:
                for _, row in puts.iterrows():
                    # Check if record exists
                    stmt = select(OptionsPut).where(
                        OptionsPut.symbol == symbol,
                        OptionsPut.expiration_date == expiry_date,
                        OptionsPut.strike == row.get("strike"),
                        OptionsPut.contract_symbol == row.get("contractSymbol"),
                    )
                    option = session.exec(stmt).first()

                    if not option:
                        option = OptionsPut(
                            symbol=symbol,
                            expiration_date=expiry_date,
                            strike=row.get("strike"),
                        )

                    # Update values
                    option.last_price = row.get("lastPrice")
                    option.bid = row.get("bid")
                    option.ask = row.get("ask")
                    option.volume = row.get("volume")
                    option.open_interest = row.get("openInterest")
                    option.implied_volatility = row.get("impliedVolatility")
                    option.in_the_money = row.get("inTheMoney")
                    option.contract_symbol = row.get("contractSymbol")

                    session.add(option)

            session.commit()

    def get_price_history(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
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
        with self.get_session() as session:
            stmt = select(PriceHistory).where(PriceHistory.symbol == symbol, PriceHistory.interval == interval)

            if start_date:
                start = datetime.strptime(start_date, "%Y-%m-%d").date()
                stmt = stmt.where(PriceHistory.date >= start)
            if end_date:
                end = datetime.strptime(end_date, "%Y-%m-%d").date()
                stmt = stmt.where(PriceHistory.date <= end)

            stmt = stmt.order_by(PriceHistory.date)

            results = session.exec(stmt).all()

            if not results:
                return pd.DataFrame()

            # Convert to DataFrame
            data = []
            for row in results:
                data.append(
                    {
                        "date": row.date,
                        "Open": row.open_price,
                        "High": row.high_price,
                        "Low": row.low_price,
                        "Close": row.close_price,
                        "Volume": row.volume,
                    }
                )

            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            return df

    def get_stock_info(self, symbol: str) -> dict[str, Any] | None:
        """Retrieve stock information.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Stock information dictionary or None if not found
        """
        with self.get_session() as session:
            stock_info = session.get(StockInfo, symbol)
            if stock_info:
                return stock_info.info_dict
            return None

    def get_dividends(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.Series:
        """Retrieve dividend data.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Series with dividend data
        """
        with self.get_session() as session:
            stmt = select(Dividend).where(Dividend.symbol == symbol)

            if start_date:
                start = datetime.strptime(start_date, "%Y-%m-%d").date()
                stmt = stmt.where(Dividend.date >= start)
            if end_date:
                end = datetime.strptime(end_date, "%Y-%m-%d").date()
                stmt = stmt.where(Dividend.date <= end)

            stmt = stmt.order_by(Dividend.date)

            results = session.exec(stmt).all()

            if not results:
                return pd.Series(dtype=float)

            # Convert to Series
            data = {pd.to_datetime(row.date): row.amount for row in results}
            return pd.Series(data)

    def get_splits(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.Series:
        """Retrieve stock split data.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Series with split data
        """
        with self.get_session() as session:
            stmt = select(Split).where(Split.symbol == symbol)

            if start_date:
                start = datetime.strptime(start_date, "%Y-%m-%d").date()
                stmt = stmt.where(Split.date >= start)
            if end_date:
                end = datetime.strptime(end_date, "%Y-%m-%d").date()
                stmt = stmt.where(Split.date <= end)

            stmt = stmt.order_by(Split.date)

            results = session.exec(stmt).all()

            if not results:
                return pd.Series(dtype=float)

            # Convert to Series
            data = {pd.to_datetime(row.date): row.ratio for row in results}
            return pd.Series(data)

    def get_options_chain(self, symbol: str, expiration_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Retrieve options chain data.

        Args:
            symbol: Stock ticker symbol
            expiration_date: Options expiration date

        Returns:
            Tuple of (calls DataFrame, puts DataFrame)
        """
        expiry_date = datetime.strptime(expiration_date, "%Y-%m-%d").date()

        with self.get_session() as session:
            # Get calls
            stmt = (
                select(OptionsCall)
                .where(
                    OptionsCall.symbol == symbol,
                    OptionsCall.expiration_date == expiry_date,
                )
                .order_by(OptionsCall.strike)
            )

            calls = session.exec(stmt).all()

            # Get puts
            stmt = (
                select(OptionsPut)
                .where(
                    OptionsPut.symbol == symbol,
                    OptionsPut.expiration_date == expiry_date,
                )
                .order_by(OptionsPut.strike)
            )

            puts = session.exec(stmt).all()

            # Convert to DataFrames
            calls_data = []
            for row in calls:
                calls_data.append(
                    {
                        "strike": row.strike,
                        "lastPrice": row.last_price,
                        "bid": row.bid,
                        "ask": row.ask,
                        "volume": row.volume,
                        "openInterest": row.open_interest,
                        "impliedVolatility": row.implied_volatility,
                        "inTheMoney": row.in_the_money,
                        "contractSymbol": row.contract_symbol,
                    }
                )

            puts_data = []
            for row in puts:
                puts_data.append(
                    {
                        "strike": row.strike,
                        "lastPrice": row.last_price,
                        "bid": row.bid,
                        "ask": row.ask,
                        "volume": row.volume,
                        "openInterest": row.open_interest,
                        "impliedVolatility": row.implied_volatility,
                        "inTheMoney": row.in_the_money,
                        "contractSymbol": row.contract_symbol,
                    }
                )

            return pd.DataFrame(calls_data), pd.DataFrame(puts_data)

    def get_all_symbols(self) -> list[str]:
        """Get all symbols in the database.

        Returns:
            List of all ticker symbols
        """
        with self.get_session() as session:
            stmt = select(Stock.symbol).order_by(Stock.symbol)
            results = session.exec(stmt).all()
            return list(results)

    def get_latest_price(self, symbol: str) -> float | None:
        """Get the latest price for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Latest close price or None if not found
        """
        with self.get_session() as session:
            from sqlalchemy import desc

            stmt = select(PriceHistory).where(PriceHistory.symbol == symbol).order_by(desc(PriceHistory.date)).limit(1)

            result = session.exec(stmt).first()
            if result:
                return result.close_price
            return None

    def has_data(self, symbol: str, start_date: str, end_date: str) -> bool:
        """Check if we have data for a symbol in the given date range.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            True if we have data in the date range
        """
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        with self.get_session() as session:
            stmt = (
                select(PriceHistory)
                .where(
                    PriceHistory.symbol == symbol,
                    PriceHistory.date >= start,
                    PriceHistory.date <= end,
                )
                .limit(1)
            )

            result = session.exec(stmt).first()
            return result is not None

    def get_database_stats(self) -> dict[str, int]:
        """Get database statistics.

        Returns:
            Dictionary with table row counts
        """
        with self.get_session() as session:
            stats = {
                "stocks": session.exec(select(Stock)).all().__len__(),
                "price_history": session.exec(select(PriceHistory)).all().__len__(),
                "dividends": session.exec(select(Dividend)).all().__len__(),
                "splits": session.exec(select(Split)).all().__len__(),
                "options_calls": session.exec(select(OptionsCall)).all().__len__(),
                "options_puts": session.exec(select(OptionsPut)).all().__len__(),
                "stock_info": session.exec(select(StockInfo)).all().__len__(),
            }
        return stats

    def get_latest_price_date(self, symbol: str) -> datetime | None:
        """Get latest price date for a symbol."""
        with self.get_session() as session:
            from sqlalchemy import desc

            stmt = (
                select(PriceHistory.date)
                .where(PriceHistory.symbol == symbol)
                .order_by(desc(PriceHistory.date))
                .limit(1)
            )

            result = session.exec(stmt).first()
            if result:
                return datetime.combine(result, datetime.min.time())
            return None

    def cleanup_old_data(self, days_to_keep: int = 365) -> int:
        """Clean up old data from database."""
        from datetime import timedelta

        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).date()

        with self.get_session() as session:
            # Delete old price history
            stmt = select(PriceHistory).where(PriceHistory.date < cutoff_date)
            old_prices = session.exec(stmt).all()
            deleted_count = len(old_prices)

            for price in old_prices:
                session.delete(price)

            # Delete old options calls
            stmt = select(OptionsCall).where(OptionsCall.expiration_date < cutoff_date)
            old_calls = session.exec(stmt).all()
            for call in old_calls:
                session.delete(call)

            # Delete old options puts
            stmt = select(OptionsPut).where(OptionsPut.expiration_date < cutoff_date)
            old_puts = session.exec(stmt).all()
            for put in old_puts:
                session.delete(put)

            session.commit()

            # VACUUM the database (SQLite specific)
            # Skip VACUUM for in-memory databases or during testing
            if self.db_path != ":memory:" and not os.environ.get("PYTEST_CURRENT_TEST"):
                from sqlalchemy import text

                try:
                    # Need to close all connections first for VACUUM to work
                    self.engine.dispose()
                    with self.engine.connect() as conn:
                        conn.execute(text("VACUUM"))
                        conn.commit()
                except Exception as e:
                    # VACUUM can fail if there are concurrent connections
                    # This is not critical, so we just log and continue
                    print(f"Warning: VACUUM failed: {e}")

            return deleted_count

    # Backward compatibility methods
    def add_stock(self, symbol: str, name: str, sector: str = "", market_cap: float | None = None):
        """Add stock to database (backward compatibility)."""
        self.store_stock_info(symbol, {"longName": name, "sector": sector, "marketCap": market_cap})

    def add_price_data(
        self,
        symbol: str,
        date,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: int,
        interval: str = "1d",
    ):
        """Add single price data point (backward compatibility)."""
        df = pd.DataFrame(
            [
                {
                    "Open": open_price,
                    "High": high,
                    "Low": low,
                    "Close": close,
                    "Volume": volume,
                }
            ],
            index=[pd.to_datetime(date)],
        )
        self.store_price_history(symbol, df, interval)

    def bulk_add_price_data(self, price_data_list):
        """Bulk add price data (backward compatibility)."""
        # Group by symbol for efficiency
        from collections import defaultdict

        grouped_data = defaultdict(list)

        for row in price_data_list:
            symbol, date, open_p, high, low, close, volume, interval = row
            grouped_data[(symbol, interval)].append(
                {
                    "date": pd.to_datetime(date),
                    "Open": open_p,
                    "High": high,
                    "Low": low,
                    "Close": close,
                    "Volume": volume,
                }
            )

        # Store each group
        for (symbol, interval), data_list in grouped_data.items():
            df = pd.DataFrame(data_list)
            df = df.set_index("date")
            self.store_price_history(symbol, df, interval)

    # Context manager support for backward compatibility
    @contextmanager
    def get_connection(self):
        """Get a database connection (returns raw SQLite connection for compatibility)."""
        pool_conn = self.engine.raw_connection()
        try:
            # Get the underlying SQLite connection
            sqlite_conn = pool_conn.driver_connection
            yield sqlite_conn
        finally:
            pool_conn.close()

    @property
    def conn(self):
        """Get database connection (deprecated)."""
        import warnings

        warnings.warn(
            "The 'conn' property is deprecated. Use 'get_session()' context manager instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Return the raw SQLite connection
        return self.engine.raw_connection().driver_connection

    def add_stock_info(self, symbol: str, info: dict):
        """Add stock info to database (backward compatibility)."""
        self.store_stock_info(symbol, info)

    def add_dividends(self, symbol: str, dividends_data: pd.Series):
        """Add dividends data (backward compatibility)."""
        self.store_dividends(symbol, dividends_data)

    def add_splits(self, symbol: str, splits_data: pd.Series):
        """Add splits data (backward compatibility)."""
        self.store_splits(symbol, splits_data)

    def add_options_data(
        self,
        symbol: str,
        expiration: str,
        calls_df: pd.DataFrame,
        puts_df: pd.DataFrame,
    ):
        """Add options data (backward compatibility)."""
        self.store_options_chain(symbol, expiration, calls_df, puts_df)
