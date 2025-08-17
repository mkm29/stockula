"""Pure TimescaleDB database manager with optimized interfaces.

This module provides a consolidated database manager focused exclusively on TimescaleDB
with advanced time-series capabilities, connection pooling, and performance optimizations.

Key Features:
- Pure TimescaleDB implementation
- Enhanced interface compliance
- Connection pooling and async operations
- Advanced time-series queries and analytics
- Comprehensive error handling and monitoring
- Optimized for high-performance time-series data
"""

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd
from sqlalchemy import create_engine, desc, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import QueuePool
from sqlmodel import Session, SQLModel, select

from alembic import command  # type: ignore[attr-defined]
from alembic.config import Config

from ..config.models import TimescaleDBConfig
from ..interfaces import IDatabaseManager
from .models import Dividend, OptionsCall, OptionsPut, PriceHistory, Split, Stock, StockInfo, get_timescale_setup_sql

# Configure module logger
logger = logging.getLogger(__name__)


class DatabaseManager(IDatabaseManager):
    """Pure TimescaleDB database manager with advanced time-series capabilities.

    Implements IDatabaseManager interface with:
    - TimescaleDB hypertables and compression
    - Connection pooling and async operations
    - Advanced time-series analytics
    - Comprehensive error handling and monitoring
    - Optimized for high-performance financial data
    """

    # Class-level tracking of setup completion
    _migrations_run: dict[str, bool] = {}
    _timescale_setup_run: dict[str, bool] = {}

    # Instance attributes for type checking
    async_engine: AsyncEngine | None
    async_session_factory: async_sessionmaker[AsyncSession] | None

    def __init__(self, config: TimescaleDBConfig | str, enable_async: bool = True):
        """Initialize TimescaleDB database manager.

        Args:
            config: TimescaleDB configuration or legacy string path (for test compatibility)
            enable_async: Enable async operations

        Raises:
            ValueError: If config is None
            ConnectionError: If TimescaleDB is not available
        """
        if config is None:
            raise ValueError("TimescaleDB configuration is required")

        # Handle legacy string path for test fixtures
        if isinstance(config, str):
            # Create a SQLite-like config for backward compatibility with tests
            from sqlalchemy import create_engine

            self._legacy_path: str = config  # Store the legacy path for tests
            self._legacy_mode = True
            self.config: TimescaleDBConfig | str = config

            # Create simple SQLite engine for tests
            self.engine = create_engine(f"sqlite:///{config}", echo=False)
            self.async_engine = None
            self.session_maker = None
            self.async_session_maker = None
            self.enable_async = False
            self.db_url = f"sqlite:///{config}"
            self.async_session_factory = None
            return
        else:
            self.config = config
            self._legacy_mode = False
        self.enable_async = enable_async

        # Setup engines and connections
        self._setup_engines()

        # Initialize database
        self._run_migrations()
        self._create_tables()
        self._setup_timescale_features()

    @property
    def backend_type(self) -> str:
        """Get the database backend type."""
        if hasattr(self, "_legacy_mode") and self._legacy_mode:
            return "sqlite"
        return "timescaledb"

    @property
    def is_timescaledb(self) -> bool:
        """Check if using TimescaleDB backend."""
        if hasattr(self, "_legacy_mode") and self._legacy_mode:
            return False
        return True

    def _test_timescale_connection(self) -> None:
        """Test TimescaleDB connection and extension availability.

        Raises:
            ConnectionError: If TimescaleDB is not available or connection fails
        """
        # Skip connection test in legacy mode
        if self._legacy_mode:
            return

        if not isinstance(self.config, TimescaleDBConfig):
            raise ValueError("TimescaleDB configuration required for connection test")

        try:
            # Test basic PostgreSQL connection
            test_url = self.config.get_connection_url()
            test_engine = create_engine(test_url, poolclass=QueuePool, pool_pre_ping=True)

            with test_engine.connect() as conn:
                # Test basic connectivity
                conn.execute(text("SELECT 1"))

                # Check if TimescaleDB extension is available
                result = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'")).fetchone()

                if result:
                    logger.info("TimescaleDB extension detected and available")
                else:
                    error_msg = "PostgreSQL connected but TimescaleDB extension not found"
                    logger.error(error_msg)
                    raise ConnectionError(error_msg)

            test_engine.dispose()

        except Exception as e:
            error_msg = f"TimescaleDB connection failed: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e

    def _setup_engines(self) -> None:
        """Setup TimescaleDB engines with connection pooling and validation."""
        # Skip engine setup in legacy mode (already handled in __init__)
        if self._legacy_mode:
            return

        if not isinstance(self.config, TimescaleDBConfig):
            raise ValueError("TimescaleDB configuration required for engine setup")

        # Test TimescaleDB connection and extension availability first
        self._test_timescale_connection()
        # Synchronous engine
        self.db_url = self.config.get_connection_url()
        self.engine = create_engine(
            self.db_url,
            poolclass=QueuePool,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle,
            pool_pre_ping=True,
            echo=False,
        )

        # Asynchronous engine if enabled
        if self.enable_async:
            self.async_db_url = self.config.get_connection_url(async_driver=True)
            self.async_engine = create_async_engine(
                self.async_db_url,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=True,
                echo=False,
            )
            self.async_session_factory = async_sessionmaker(
                self.async_engine, class_=AsyncSession, expire_on_commit=False
            )
        else:
            self.async_engine = None
            self.async_session_factory = None

    def _run_migrations(self) -> None:
        """Run Alembic migrations to ensure database schema is up to date."""
        # Skip migrations in test environment
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return

        # Check if migrations have already been run for this database URL
        if self.db_url in self._migrations_run:
            return

        try:
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
                    logger.warning("alembic.ini not found, skipping migrations")
                    return

            # Configure Alembic
            alembic_cfg = Config(str(alembic_ini_path))
            alembic_cfg.set_main_option("sqlalchemy.url", self.db_url)

            # Run migrations
            try:
                alembic_logger = logging.getLogger("alembic")
                original_level = alembic_logger.level
                alembic_logger.setLevel(logging.WARNING)
                command.upgrade(alembic_cfg, "head")
                alembic_logger.setLevel(original_level)
            except Exception as e:
                if "already exists" not in str(e):
                    logger.warning(f"TimescaleDB migration failed: {e}")

            # Mark migrations as run
            self._migrations_run[self.db_url] = True
            logger.info("Migrations completed for TimescaleDB backend")

        except Exception as e:
            if "already exists" not in str(e):
                logger.warning(f"Migration error: {e}")

    def _create_tables(self) -> None:
        """Create database tables."""
        try:
            SQLModel.metadata.create_all(self.engine, checkfirst=True)
            logger.info("Tables created/verified for TimescaleDB backend")
        except Exception as e:
            logger.error(f"Table creation failed: {e}")
            raise

    def _setup_timescale_features(self) -> None:
        """Setup TimescaleDB-specific features like hypertables and policies."""
        # Skip TimescaleDB features in legacy mode
        if self._legacy_mode:
            return

        if not isinstance(self.config, TimescaleDBConfig):
            logger.warning("TimescaleDB configuration required for hypertable setup")
            return

        if not self.config.enable_hypertables:
            logger.info("Hypertables disabled in configuration")
            return

        # Check if setup already completed
        if self.db_url in self._timescale_setup_run:
            return

        try:
            with self.get_session() as session:
                # Verify TimescaleDB extension
                result = session.exec(text("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'")).fetchone()

                if not result:
                    logger.warning("TimescaleDB extension not found, skipping hypertable setup")
                    return

                # Execute TimescaleDB setup commands
                setup_commands = get_timescale_setup_sql()

                for sql_command in setup_commands:
                    try:
                        session.exec(text(sql_command))
                        session.commit()
                    except Exception as e:
                        if "already exists" not in str(e).lower():
                            logger.warning(f"TimescaleDB setup command failed: {e}")
                        session.rollback()

                self._timescale_setup_run[self.db_url] = True
                logger.info("TimescaleDB features setup completed")

        except Exception as e:
            logger.error(f"TimescaleDB feature setup failed: {e}")

    def close(self) -> None:
        """Close database engines and dispose of all connections."""
        try:
            if hasattr(self, "engine") and self.engine:
                self.engine.dispose()

            if hasattr(self, "async_engine") and self.async_engine:
                # Schedule async engine disposal
                try:
                    if asyncio.get_event_loop().is_running():
                        asyncio.create_task(self.async_engine.adispose())
                    else:
                        asyncio.run(self.async_engine.adispose())
                except Exception as e:
                    logger.warning(f"Async engine disposal failed: {e}")

            logger.info("TimescaleDB database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")

    def __del__(self) -> None:
        """Ensure database connections are closed when object is destroyed."""
        try:
            self.close()
        except Exception:
            # Ignore errors during cleanup
            pass

    @contextmanager
    def get_session(self) -> Iterator[Session]:
        """Get a synchronous database session as context manager."""
        with Session(self.engine) as session:
            yield session

    @asynccontextmanager
    async def get_async_session(self) -> AsyncIterator[AsyncSession]:
        """Get an asynchronous database session as context manager.

        Raises:
            RuntimeError: If async operations not enabled
        """
        if not self.async_session_factory:
            raise RuntimeError("Async operations not enabled. Initialize with enable_async=True.")

        async with self.async_session_factory() as session:
            yield session

    # ========================================
    # IDatabaseManager Interface Implementation
    # ========================================

    def store_stock_info(self, symbol: str, info: dict[str, Any]) -> None:
        """Store comprehensive stock information.

        Args:
            symbol: Stock ticker symbol
            info: Stock information dictionary from yfinance
        """
        try:
            with self.get_session() as session:
                # Create or update stock
                stock = session.get(Stock, symbol)
                if not stock:
                    stock = Stock(symbol=symbol)

                # Update stock metadata
                stock.name = info.get("longName") or info.get("shortName", "")
                stock.sector = info.get("sector", "")
                stock.industry = info.get("industry", "")
                stock.market_cap = info.get("marketCap")
                stock.exchange = info.get("exchange", "")
                stock.currency = info.get("currency", "USD")
                stock.updated_at = datetime.now(UTC)

                session.add(stock)

                # Store comprehensive info as JSONB
                stock_info = session.get(StockInfo, symbol)
                if not stock_info:
                    stock_info = StockInfo(symbol=symbol)

                stock_info.set_info(info)
                stock_info.updated_at = datetime.now(UTC)

                session.add(stock_info)
                session.commit()

                logger.debug(f"Stored stock info for {symbol} using TimescaleDB")

        except Exception as e:
            logger.error(f"Failed to store stock info for {symbol}: {e}")
            raise

    def store_price_history(self, symbol: str, data: pd.DataFrame, interval: str = "1d") -> None:
        """Store historical price data with TimescaleDB optimizations.

        Args:
            symbol: Stock ticker symbol
            data: DataFrame with OHLCV data
            interval: Data interval (1d, 1h, etc.)
        """
        if data.empty:
            logger.warning(f"Empty DataFrame provided for {symbol}")
            return

        try:
            with self.get_session() as session:
                # Ensure stock exists
                stock = session.get(Stock, symbol)
                if not stock:
                    stock = Stock(symbol=symbol)
                    session.add(stock)

                stored_count = 0
                updated_count = 0

                for timestamp, row in data.iterrows():
                    # Convert to timezone-aware datetime for TimescaleDB
                    if hasattr(timestamp, "tz_localize"):
                        if timestamp.tz is None:
                            timestamp = timestamp.tz_localize(UTC)
                        else:
                            timestamp = timestamp.tz_convert(UTC)
                    elif isinstance(timestamp, datetime) and timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=UTC)

                    # Use timestamp for TimescaleDB hypertable
                    stmt = select(PriceHistory).where(
                        PriceHistory.symbol == symbol,
                        PriceHistory.timestamp == timestamp,
                        PriceHistory.interval == interval,
                    )
                    price_history = session.exec(stmt).first()

                    if not price_history:
                        price_history = PriceHistory(symbol=symbol, timestamp=timestamp, interval=interval)
                        stored_count += 1
                    else:
                        updated_count += 1

                    # Set computed date for backward compatibility
                    price_history.date = timestamp.date()

                    # Update OHLCV values
                    price_history.open_price = row.get("Open")
                    price_history.high_price = row.get("High")
                    price_history.low_price = row.get("Low")
                    price_history.close_price = row.get("Close")
                    price_history.volume = row.get("Volume")

                    # Calculate additional fields for TimescaleDB
                    if all([price_history.high_price, price_history.low_price, price_history.close_price]):
                        price_history.typical_price = (
                            price_history.high_price + price_history.low_price + price_history.close_price
                        ) / 3

                    # Set adjusted close if available
                    price_history.adjusted_close = row.get("Adj Close")

                    session.add(price_history)

                session.commit()

                logger.info(
                    f"Stored price history for {symbol}: {stored_count} new, "
                    f"{updated_count} updated records using TimescaleDB"
                )

        except Exception as e:
            logger.error(f"Failed to store price history for {symbol}: {e}")
            raise

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
        """Store options chain data with TimescaleDB optimizations.

        Args:
            symbol: Stock ticker symbol
            calls: DataFrame with call options
            puts: DataFrame with put options
            expiration_date: Options expiration date
        """
        try:
            # Convert to timezone-aware datetime for TimescaleDB
            expiry_datetime = datetime.strptime(expiration_date, "%Y-%m-%d").replace(tzinfo=UTC)
            data_timestamp = datetime.now(UTC)  # When the data was recorded

            with self.get_session() as session:
                # Ensure stock exists
                stock = session.get(Stock, symbol)
                if not stock:
                    stock = Stock(symbol=symbol)
                    session.add(stock)

                calls_stored = 0
                puts_stored = 0

                # Store calls
                if not calls.empty:
                    for _, row in calls.iterrows():
                        # TimescaleDB uses timestamp-based unique constraints
                        stmt = select(OptionsCall).where(
                            OptionsCall.symbol == symbol,
                            OptionsCall.expiration_timestamp == expiry_datetime,
                            OptionsCall.strike == row.get("strike"),
                            OptionsCall.contract_symbol == row.get("contractSymbol"),
                            OptionsCall.data_timestamp == data_timestamp,
                        )
                        option = session.exec(stmt).first()

                        if not option:
                            option = OptionsCall(
                                symbol=symbol,
                                expiration_timestamp=expiry_datetime,
                                data_timestamp=data_timestamp,
                                strike=row.get("strike"),
                            )
                            option.expiration_date = expiry_datetime.date()  # Backward compatibility

                        # Update common values
                        option.last_price = row.get("lastPrice")
                        option.bid = row.get("bid")
                        option.ask = row.get("ask")
                        option.volume = row.get("volume")
                        option.open_interest = row.get("openInterest")
                        option.implied_volatility = row.get("impliedVolatility")
                        option.in_the_money = row.get("inTheMoney")
                        option.contract_symbol = row.get("contractSymbol")

                        # Store Greeks (TimescaleDB specific fields)
                        option.delta = row.get("delta")
                        option.gamma = row.get("gamma")
                        option.theta = row.get("theta")
                        option.vega = row.get("vega")
                        option.intrinsic_value = row.get("intrinsicValue")
                        option.time_value = row.get("timeValue")

                        session.add(option)
                        calls_stored += 1

                # Store puts
                if not puts.empty:
                    for _, row in puts.iterrows():
                        # TimescaleDB uses timestamp-based unique constraints
                        stmt_put = select(OptionsPut).where(
                            OptionsPut.symbol == symbol,
                            OptionsPut.expiration_timestamp == expiry_datetime,
                            OptionsPut.strike == row.get("strike"),
                            OptionsPut.contract_symbol == row.get("contractSymbol"),
                            OptionsPut.data_timestamp == data_timestamp,
                        )
                        option_put = session.exec(stmt_put).first()

                        if not option_put:
                            option_put = OptionsPut(
                                symbol=symbol,
                                expiration_timestamp=expiry_datetime,
                                data_timestamp=data_timestamp,
                                strike=row.get("strike"),
                            )
                            option_put.expiration_date = expiry_datetime.date()  # Backward compatibility

                        # Update common values
                        option_put.last_price = row.get("lastPrice")
                        option_put.bid = row.get("bid")
                        option_put.ask = row.get("ask")
                        option_put.volume = row.get("volume")
                        option_put.open_interest = row.get("openInterest")
                        option_put.implied_volatility = row.get("impliedVolatility")
                        option_put.in_the_money = row.get("inTheMoney")
                        option_put.contract_symbol = row.get("contractSymbol")

                        # Store Greeks (TimescaleDB specific fields)
                        option_put.delta = row.get("delta")
                        option_put.gamma = row.get("gamma")
                        option_put.theta = row.get("theta")
                        option_put.vega = row.get("vega")
                        option_put.intrinsic_value = row.get("intrinsicValue")
                        option_put.time_value = row.get("timeValue")

                        session.add(option_put)
                        puts_stored += 1

                session.commit()

                logger.info(
                    f"Stored options chain for {symbol} expiring {expiration_date}: "
                    f"{calls_stored} calls, {puts_stored} puts using TimescaleDB"
                )

        except Exception as e:
            logger.error(f"Failed to store options chain for {symbol}: {e}")
            raise

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
                start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
                stmt = stmt.where(PriceHistory.timestamp >= start)
            if end_date:
                end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)
                stmt = stmt.where(PriceHistory.timestamp <= end)

            stmt = stmt.order_by(PriceHistory.timestamp)  # type: ignore[arg-type]

            results = session.exec(stmt).all()

            if not results:
                return pd.DataFrame()

            # Convert to DataFrame
            data = []
            for row in results:
                data.append(
                    {
                        "timestamp": row.timestamp,
                        "Open": row.open_price,
                        "High": row.high_price,
                        "Low": row.low_price,
                        "Close": row.close_price,
                        "Volume": row.volume,
                    }
                )

            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
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
                return cast(dict[str, Any] | None, stock_info.info_dict)
            return None

    def get_dividends(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.Series:
        """Retrieve dividend data with TimescaleDB optimizations.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Series with dividend data
        """
        try:
            with self.get_session() as session:
                stmt = select(Dividend).where(Dividend.symbol == symbol)

                # Use timestamp-based queries for TimescaleDB
                if start_date:
                    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
                    stmt = stmt.where(Dividend.timestamp >= start)
                if end_date:
                    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)
                    stmt = stmt.where(Dividend.timestamp <= end)

                stmt = stmt.order_by(Dividend.__table__.c.date)
                results = session.exec(stmt).all()

                if not results:
                    return pd.Series(dtype=float)

                # Convert to Series using timestamp
                data = {row.timestamp: row.amount for row in results}

                logger.debug(f"Retrieved {len(data)} dividend records for {symbol}")
                return pd.Series(data)

        except Exception as e:
            logger.error(f"Failed to retrieve dividends for {symbol}: {e}")
            return pd.Series(dtype=float)

    def get_splits(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.Series:
        """Retrieve stock split data with TimescaleDB optimizations.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Series with split data
        """
        try:
            with self.get_session() as session:
                stmt = select(Split).where(Split.symbol == symbol)

                # Use timestamp-based queries for TimescaleDB
                if start_date:
                    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
                    stmt = stmt.where(Split.timestamp >= start)
                if end_date:
                    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)
                    stmt = stmt.where(Split.timestamp <= end)

                stmt = stmt.order_by(Split.__table__.c.date)
                results = session.exec(stmt).all()

                if not results:
                    return pd.Series(dtype=float)

                # Convert to Series using timestamp
                data = {row.timestamp: row.ratio for row in results}

                logger.debug(f"Retrieved {len(data)} split records for {symbol}")
                return pd.Series(data)

        except Exception as e:
            logger.error(f"Failed to retrieve splits for {symbol}: {e}")
            return pd.Series(dtype=float)

    def get_options_chain(self, symbol: str, expiration_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Retrieve options chain data with TimescaleDB optimizations.

        Args:
            symbol: Stock ticker symbol
            expiration_date: Options expiration date

        Returns:
            Tuple of (calls DataFrame, puts DataFrame)
        """
        try:
            # Convert to timezone-aware datetime for TimescaleDB
            expiry_datetime = datetime.strptime(expiration_date, "%Y-%m-%d").replace(tzinfo=UTC)

            with self.get_session() as session:
                # Get calls
                stmt = (
                    select(OptionsCall)
                    .where(
                        OptionsCall.symbol == symbol,
                        OptionsCall.expiration_timestamp == expiry_datetime,
                    )
                    .order_by(OptionsCall.__table__.c.strike)
                )

                calls = session.exec(stmt).all()

                # Get puts
                stmt_puts = (
                    select(OptionsPut)
                    .where(
                        OptionsPut.symbol == symbol,
                        OptionsPut.expiration_timestamp == expiry_datetime,
                    )
                    .order_by(OptionsPut.__table__.c.strike)
                )

                puts = session.exec(stmt_puts).all()

                # Convert to DataFrames with enhanced data for TimescaleDB
                calls_data = []
                for row in calls:
                    record = {
                        "strike": row.strike,
                        "lastPrice": row.last_price,
                        "bid": row.bid,
                        "ask": row.ask,
                        "volume": row.volume,
                        "openInterest": row.open_interest,
                        "impliedVolatility": row.implied_volatility,
                        "inTheMoney": row.in_the_money,
                        "contractSymbol": row.contract_symbol,
                        # Add Greeks and advanced fields for TimescaleDB
                        "delta": row.delta,
                        "gamma": row.gamma,
                        "theta": row.theta,
                        "vega": row.vega,
                        "intrinsicValue": row.intrinsic_value,
                        "timeValue": row.time_value,
                    }
                    calls_data.append(record)

                puts_data = []
                for put_row in puts:
                    record = {
                        "strike": put_row.strike,
                        "lastPrice": put_row.last_price,
                        "bid": put_row.bid,
                        "ask": put_row.ask,
                        "volume": put_row.volume,
                        "openInterest": put_row.open_interest,
                        "impliedVolatility": put_row.implied_volatility,
                        "inTheMoney": put_row.in_the_money,
                        "contractSymbol": put_row.contract_symbol,
                        # Add Greeks and advanced fields for TimescaleDB
                        "delta": put_row.delta,
                        "gamma": put_row.gamma,
                        "theta": put_row.theta,
                        "vega": put_row.vega,
                        "intrinsicValue": put_row.intrinsic_value,
                        "timeValue": put_row.time_value,
                    }
                    puts_data.append(record)

                logger.debug(
                    f"Retrieved options chain for {symbol} expiring {expiration_date}: "
                    f"{len(calls_data)} calls, {len(puts_data)} puts"
                )
                return pd.DataFrame(calls_data), pd.DataFrame(puts_data)

        except Exception as e:
            logger.error(f"Failed to retrieve options chain for {symbol}: {e}")
            return pd.DataFrame(), pd.DataFrame()

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
            stmt = (
                select(PriceHistory)
                .where(PriceHistory.symbol == symbol)
                .order_by(desc(PriceHistory.__table__.c.timestamp))
                .limit(1)
            )

            result = session.exec(stmt).first()
            if result:
                return cast(float | None, result.close_price)
            return None

    def has_data(self, symbol: str, start_date: str, end_date: str) -> bool:
        """Check if data exists for symbol in date range.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            True if data exists in the date range
        """
        try:
            # Use timestamp-based query for TimescaleDB
            start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
            end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)

            with self.get_session() as session:
                stmt = (
                    select(PriceHistory)
                    .where(
                        PriceHistory.symbol == symbol,
                        PriceHistory.timestamp >= start,
                        PriceHistory.timestamp <= end,
                    )
                    .limit(1)
                )
                result = session.exec(stmt).first()
                return result is not None

        except Exception as e:
            logger.error(f"Error checking data for {symbol}: {e}")
            return False

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
        try:
            with self.get_session() as session:
                stmt = (
                    select(PriceHistory.timestamp)
                    .where(PriceHistory.symbol == symbol)
                    .order_by(desc(PriceHistory.__table__.c.timestamp))
                    .limit(1)
                )
                result = session.exec(stmt).first()
                return result if result is None else cast(datetime, result)
        except Exception as e:
            logger.error(f"Failed to get latest price date for {symbol}: {e}")
            return None

    def cleanup_old_data(self, days_to_keep: int = 365) -> int:
        """Clean up old data from database with TimescaleDB optimizations."""
        try:
            cutoff_datetime = datetime.now(UTC) - timedelta(days=days_to_keep)

            with self.get_session() as session:
                deleted_count = 0

                # Use TimescaleDB-optimized cleanup with timestamp
                # Delete old price history
                stmt = select(PriceHistory).where(PriceHistory.timestamp < cutoff_datetime)
                old_prices = session.exec(stmt).all()
                deleted_count += len(old_prices)

                for price in old_prices:
                    session.delete(price)

                # Delete old options with expiration_timestamp
                stmt_calls = select(OptionsCall).where(OptionsCall.expiration_timestamp < cutoff_datetime)
                old_calls = session.exec(stmt_calls).all()
                for call in old_calls:
                    session.delete(call)

                stmt_puts = select(OptionsPut).where(OptionsPut.expiration_timestamp < cutoff_datetime)
                old_puts = session.exec(stmt_puts).all()
                for put in old_puts:
                    session.delete(put)

                deleted_count += len(old_calls) + len(old_puts)

                session.commit()

                logger.info(f"Cleaned up {deleted_count} old records using TimescaleDB")
                return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return 0

    def test_connection(self) -> dict[str, Any]:
        """Test TimescaleDB connection and return health information.

        Returns:
            Dictionary with connection status and health information
        """
        # Handle legacy mode
        if self._legacy_mode:
            return {
                "backend": "sqlite",
                "url": f"sqlite:///{self._legacy_path}",
                "timescale_available": False,
                "hypertables_enabled": False,
                "connection_pool": False,
            }

        if not isinstance(self.config, TimescaleDBConfig):
            return {
                "backend": "unknown",
                "error": "Invalid configuration type",
            }

        info = {
            "backend": "timescaledb",
            "url": self.config.get_connection_url().replace(self.config.password or "", "***"),
            "timescale_available": True,
            "hypertables_enabled": self.config.enable_hypertables,
            "connection_pool": True,
            "pool_size": self.config.pool_size,
            "max_overflow": self.config.max_overflow,
        }

        try:
            # Test actual connectivity
            with self.get_session() as session:
                result = session.exec(text("SELECT version()")).fetchone()
                if result:
                    info["database_version"] = str(result[0])

                # Check TimescaleDB version
                try:
                    ts_result = session.exec(
                        text("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
                    ).fetchone()
                    if ts_result:
                        info["timescaledb_version"] = str(ts_result[0])
                except Exception:
                    pass

        except Exception as e:
            info["error"] = str(e)

        return info

    # ========================================
    # TimescaleDB Analytics Methods
    # ========================================

    def get_moving_averages(
        self,
        symbol: str,
        periods: list[int] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Calculate moving averages using TimescaleDB window functions.

        Args:
            symbol: Stock ticker symbol
            periods: List of periods for moving averages
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with moving averages
        """
        if periods is None:
            periods = [20, 50, 200]
        # Build window function clauses for each period
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

        params: dict[str, Any] = {"symbol": symbol}

        if start_date:
            query += " AND timestamp >= :start_date"
            params["start_date"] = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
            query += " AND timestamp <= :end_date"
            params["end_date"] = datetime.strptime(end_date, "%Y-%m-%d")

        query += " ORDER BY timestamp"

        with self.get_session() as session:
            result = session.execute(text(query), params).fetchall()

            if not result:
                return pd.DataFrame()

            columns = ["timestamp", "symbol", "close_price"] + [f"ma_{p}" for p in periods]
            df = pd.DataFrame(result, columns=columns)
            df = df.set_index("timestamp")
            return df

    def get_bollinger_bands(
        self,
        symbol: str,
        period: int = 20,
        std_dev: float = 2.0,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands using TimescaleDB window functions.

        Args:
            symbol: Stock ticker symbol
            period: Period for moving average and standard deviation
            std_dev: Standard deviation multiplier
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with Bollinger Bands
        """
        query = f"""
        SELECT
            timestamp,
            symbol,
            close_price,
            AVG(close_price) OVER (
                PARTITION BY symbol
                ORDER BY timestamp
                ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW
            ) AS bb_middle,
            AVG(close_price) OVER (
                PARTITION BY symbol
                ORDER BY timestamp
                ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW
            ) + ({std_dev} * STDDEV(close_price) OVER (
                PARTITION BY symbol
                ORDER BY timestamp
                ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW
            )) AS bb_upper,
            AVG(close_price) OVER (
                PARTITION BY symbol
                ORDER BY timestamp
                ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW
            ) - ({std_dev} * STDDEV(close_price) OVER (
                PARTITION BY symbol
                ORDER BY timestamp
                ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW
            )) AS bb_lower,
            STDDEV(close_price) OVER (
                PARTITION BY symbol
                ORDER BY timestamp
                ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW
            ) AS price_volatility
        FROM price_history
        WHERE symbol = :symbol
        """

        params: dict[str, Any] = {"symbol": symbol}

        if start_date:
            query += " AND timestamp >= :start_date"
            params["start_date"] = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
            query += " AND timestamp <= :end_date"
            params["end_date"] = datetime.strptime(end_date, "%Y-%m-%d")

        query += " ORDER BY timestamp"

        with self.get_session() as session:
            result = session.execute(text(query), params).fetchall()

            if not result:
                return pd.DataFrame()

            df = pd.DataFrame(
                result,
                columns=["timestamp", "symbol", "close_price", "bb_middle", "bb_upper", "bb_lower", "price_volatility"],
            )
            df = df.set_index("timestamp")
            return df

    def get_rsi(
        self,
        symbol: str,
        period: int = 14,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Calculate RSI using TimescaleDB window functions.

        Args:
            symbol: Stock ticker symbol
            period: RSI period
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with RSI values
        """
        query = f"""
        WITH price_changes AS (
            SELECT
                timestamp,
                symbol,
                close_price,
                close_price - LAG(close_price) OVER (
                    PARTITION BY symbol
                    ORDER BY timestamp
                ) AS price_change
            FROM price_history
            WHERE symbol = :symbol
        ),
        gains_losses AS (
            SELECT
                timestamp,
                symbol,
                close_price,
                price_change,
                CASE WHEN price_change > 0 THEN price_change ELSE 0 END AS gain,
                CASE WHEN price_change < 0 THEN ABS(price_change) ELSE 0 END AS loss
            FROM price_changes
            WHERE price_change IS NOT NULL
        ),
        rsi_calc AS (
            SELECT
                timestamp,
                symbol,
                close_price,
                price_change,
                gain,
                loss,
                AVG(gain) OVER (
                    PARTITION BY symbol
                    ORDER BY timestamp
                    ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW
                ) AS avg_gain,
                AVG(loss) OVER (
                    PARTITION BY symbol
                    ORDER BY timestamp
                    ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW
                ) AS avg_loss
            FROM gains_losses
        )
        SELECT
            timestamp,
            symbol,
            close_price,
            price_change,
            avg_gain,
            avg_loss,
            CASE
                WHEN avg_loss = 0 THEN 100
                ELSE 100 - (100 / (1 + (avg_gain / avg_loss)))
            END AS rsi
        FROM rsi_calc
        """

        params: dict[str, Any] = {"symbol": symbol}

        if start_date:
            query = query.replace("WHERE symbol = :symbol", "WHERE symbol = :symbol AND timestamp >= :start_date")
            params["start_date"] = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
            # Add to the innermost WHERE clause
            if start_date:
                query = query.replace(
                    "AND timestamp >= :start_date", "AND timestamp >= :start_date AND timestamp <= :end_date"
                )
            else:
                query = query.replace("WHERE symbol = :symbol", "WHERE symbol = :symbol AND timestamp <= :end_date")
            params["end_date"] = datetime.strptime(end_date, "%Y-%m-%d")

        query += " ORDER BY timestamp"

        with self.get_session() as session:
            result = session.execute(text(query), params).fetchall()

            if not result:
                return pd.DataFrame()

            df = pd.DataFrame(
                result, columns=["timestamp", "symbol", "close_price", "price_change", "avg_gain", "avg_loss", "rsi"]
            )
            df = df.set_index("timestamp")
            return df

    def get_price_momentum(
        self,
        symbols: list[str] | None = None,
        lookback_days: int = 30,
        time_bucket: str = "1 day",
    ) -> pd.DataFrame:
        """Calculate price momentum across multiple timeframes.

        Args:
            symbols: List of symbols to analyze (None for all)
            lookback_days: Number of days to look back
            time_bucket: Time aggregation bucket

        Returns:
            DataFrame with momentum analysis
        """
        cutoff_date = datetime.now() - timedelta(days=lookback_days)

        query = f"""
        WITH bucketed_prices AS (
            SELECT
                symbol,
                time_bucket('{time_bucket}', timestamp) AS bucket,
                first(close_price, timestamp) AS open_price,
                last(close_price, timestamp) AS close_price,
                max(high_price) AS high_price,
                min(low_price) AS low_price,
                sum(volume) AS volume
            FROM price_history
            WHERE timestamp >= :cutoff_date
        """

        params: dict[str, Any] = {"cutoff_date": cutoff_date}

        if symbols:
            placeholders = ",".join([f":symbol_{i}" for i in range(len(symbols))])
            query += f" AND symbol IN ({placeholders})"
            for i, symbol in enumerate(symbols):
                params[f"symbol_{i}"] = symbol

        query += """
            GROUP BY symbol, bucket
            ORDER BY symbol, bucket
        ),
        momentum_calc AS (
            SELECT
                symbol,
                bucket,
                open_price,
                close_price,
                high_price,
                low_price,
                volume,
                LAG(close_price, 1) OVER (PARTITION BY symbol ORDER BY bucket) AS prev_close_1d,
                LAG(close_price, 7) OVER (PARTITION BY symbol ORDER BY bucket) AS prev_close_7d,
                LAG(close_price, 30) OVER (PARTITION BY symbol ORDER BY bucket) AS prev_close_30d
            FROM bucketed_prices
        )
        SELECT
            symbol,
            bucket,
            close_price,
            volume,
            CASE
                WHEN prev_close_1d IS NOT NULL
                THEN ((close_price - prev_close_1d) / prev_close_1d) * 100
                ELSE NULL
            END AS momentum_1d,
            CASE
                WHEN prev_close_7d IS NOT NULL
                THEN ((close_price - prev_close_7d) / prev_close_7d) * 100
                ELSE NULL
            END AS momentum_7d,
            CASE
                WHEN prev_close_30d IS NOT NULL
                THEN ((close_price - prev_close_30d) / prev_close_30d) * 100
                ELSE NULL
            END AS momentum_30d,
            (high_price - low_price) / open_price * 100 AS daily_range_pct
        FROM momentum_calc
        WHERE bucket >= :cutoff_date
        ORDER BY symbol, bucket
        """

        with self.get_session() as session:
            result = session.execute(text(query), params).fetchall()

            if not result:
                return pd.DataFrame()

            df = pd.DataFrame(
                result,
                columns=[
                    "symbol",
                    "timestamp",
                    "close_price",
                    "volume",
                    "momentum_1d",
                    "momentum_7d",
                    "momentum_30d",
                    "daily_range_pct",
                ],
            )
            return df

    def get_correlation_matrix(
        self,
        symbols: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        time_bucket: str = "1 day",
    ) -> pd.DataFrame:
        """Calculate correlation matrix between symbols using TimescaleDB.

        Args:
            symbols: List of symbols to analyze
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            time_bucket: Time aggregation bucket

        Returns:
            DataFrame with correlation matrix
        """
        if len(symbols) < 2:
            raise ValueError("At least 2 symbols required for correlation analysis")

        # Create symbol placeholders
        symbol_placeholders = ",".join([f":symbol_{i}" for i in range(len(symbols))])
        params: dict[str, Any] = {f"symbol_{i}": symbol for i, symbol in enumerate(symbols)}

        query = f"""
        WITH bucketed_prices AS (
            SELECT
                symbol,
                time_bucket('{time_bucket}', timestamp) AS bucket,
                last(close_price, timestamp) AS close_price
            FROM price_history
            WHERE symbol IN ({symbol_placeholders})
        """

        if start_date:
            query += " AND timestamp >= :start_date"
            params["start_date"] = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
            query += " AND timestamp <= :end_date"
            params["end_date"] = datetime.strptime(end_date, "%Y-%m-%d")

        query += """
            GROUP BY symbol, bucket
            HAVING last(close_price, timestamp) IS NOT NULL
        ),
        price_returns AS (
            SELECT
                symbol,
                bucket,
                close_price,
                (close_price - LAG(close_price) OVER (
                    PARTITION BY symbol ORDER BY bucket
                )) / LAG(close_price) OVER (
                    PARTITION BY symbol ORDER BY bucket
                ) AS daily_return
            FROM bucketed_prices
        )
        SELECT
            s1.symbol AS symbol1,
            s2.symbol AS symbol2,
            CORR(s1.daily_return, s2.daily_return) AS correlation
        FROM price_returns s1
        JOIN price_returns s2 ON s1.bucket = s2.bucket
        WHERE s1.daily_return IS NOT NULL
            AND s2.daily_return IS NOT NULL
        GROUP BY s1.symbol, s2.symbol
        ORDER BY s1.symbol, s2.symbol
        """

        with self.get_session() as session:
            result = session.execute(text(query), params).fetchall()

            if not result:
                return pd.DataFrame()

            # Convert to correlation matrix format
            correlation_data = []
            for row in result:
                correlation_data.append({"symbol1": row[0], "symbol2": row[1], "correlation": row[2]})

            df = pd.DataFrame(correlation_data)

            # Pivot to create correlation matrix
            correlation_matrix = df.pivot(index="symbol1", columns="symbol2", values="correlation")

            return correlation_matrix

    def get_volatility_analysis(
        self,
        symbols: list[str] | None = None,
        lookback_days: int = 30,
        time_bucket: str = "1 day",
    ) -> pd.DataFrame:
        """Calculate volatility metrics using TimescaleDB statistical functions.

        Args:
            symbols: List of symbols to analyze (None for all)
            lookback_days: Number of days to analyze
            time_bucket: Time aggregation bucket

        Returns:
            DataFrame with volatility analysis
        """
        cutoff_date = datetime.now() - timedelta(days=lookback_days)

        query = f"""
        WITH bucketed_prices AS (
            SELECT
                symbol,
                time_bucket('{time_bucket}', timestamp) AS bucket,
                first(close_price, timestamp) AS open_price,
                last(close_price, timestamp) AS close_price,
                max(high_price) AS high_price,
                min(low_price) AS low_price
            FROM price_history
            WHERE timestamp >= :cutoff_date
        """

        params: dict[str, Any] = {"cutoff_date": cutoff_date}

        if symbols:
            placeholders = ",".join([f":symbol_{i}" for i in range(len(symbols))])
            query += f" AND symbol IN ({placeholders})"
            for i, symbol in enumerate(symbols):
                params[f"symbol_{i}"] = symbol

        query += """
            GROUP BY symbol, bucket
        ),
        daily_returns AS (
            SELECT
                symbol,
                bucket,
                close_price,
                (close_price - LAG(close_price) OVER (
                    PARTITION BY symbol ORDER BY bucket
                )) / LAG(close_price) OVER (
                    PARTITION BY symbol ORDER BY bucket
                ) AS daily_return,
                (high_price - low_price) / open_price AS daily_range
            FROM bucketed_prices
        )
        SELECT
            symbol,
            COUNT(*) AS data_points,
            AVG(daily_return) * 100 AS avg_daily_return_pct,
            STDDEV(daily_return) * 100 AS daily_volatility_pct,
            STDDEV(daily_return) * SQRT(252) * 100 AS annualized_volatility_pct,
            MIN(daily_return) * 100 AS min_daily_return_pct,
            MAX(daily_return) * 100 AS max_daily_return_pct,
            AVG(daily_range) * 100 AS avg_daily_range_pct,
            STDDEV(daily_range) * 100 AS range_volatility_pct
        FROM daily_returns
        WHERE daily_return IS NOT NULL
        GROUP BY symbol
        ORDER BY annualized_volatility_pct DESC
        """

        with self.get_session() as session:
            result = session.execute(text(query), params).fetchall()

            if not result:
                return pd.DataFrame()

            df = pd.DataFrame(
                result,
                columns=[
                    "symbol",
                    "data_points",
                    "avg_daily_return_pct",
                    "daily_volatility_pct",
                    "annualized_volatility_pct",
                    "min_daily_return_pct",
                    "max_daily_return_pct",
                    "avg_daily_range_pct",
                    "range_volatility_pct",
                ],
            )
            return df

    def get_seasonal_patterns(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Analyze seasonal patterns in stock price movements.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with seasonal pattern analysis
        """
        query = """
        WITH daily_returns AS (
            SELECT
                timestamp,
                symbol,
                close_price,
                (close_price - LAG(close_price) OVER (
                    PARTITION BY symbol ORDER BY timestamp
                )) / LAG(close_price) OVER (
                    PARTITION BY symbol ORDER BY timestamp
                ) AS daily_return,
                EXTRACT(DOW FROM timestamp) AS day_of_week,
                EXTRACT(MONTH FROM timestamp) AS month,
                EXTRACT(QUARTER FROM timestamp) AS quarter,
                EXTRACT(DAY FROM timestamp) AS day_of_month
            FROM price_history
            WHERE symbol = :symbol
        """

        params: dict[str, Any] = {"symbol": symbol}

        if start_date:
            query += " AND timestamp >= :start_date"
            params["start_date"] = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
            query += " AND timestamp <= :end_date"
            params["end_date"] = datetime.strptime(end_date, "%Y-%m-%d")

        query += """
        )
        SELECT
            'day_of_week' AS pattern_type,
            day_of_week::text AS pattern_value,
            COUNT(*) AS occurrences,
            AVG(daily_return) * 100 AS avg_return_pct,
            STDDEV(daily_return) * 100 AS volatility_pct,
            MIN(daily_return) * 100 AS min_return_pct,
            MAX(daily_return) * 100 AS max_return_pct
        FROM daily_returns
        WHERE daily_return IS NOT NULL
        GROUP BY day_of_week

        UNION ALL

        SELECT
            'month' AS pattern_type,
            month::text AS pattern_value,
            COUNT(*) AS occurrences,
            AVG(daily_return) * 100 AS avg_return_pct,
            STDDEV(daily_return) * 100 AS volatility_pct,
            MIN(daily_return) * 100 AS min_return_pct,
            MAX(daily_return) * 100 AS max_return_pct
        FROM daily_returns
        WHERE daily_return IS NOT NULL
        GROUP BY month

        UNION ALL

        SELECT
            'quarter' AS pattern_type,
            quarter::text AS pattern_value,
            COUNT(*) AS occurrences,
            AVG(daily_return) * 100 AS avg_return_pct,
            STDDEV(daily_return) * 100 AS volatility_pct,
            MIN(daily_return) * 100 AS min_return_pct,
            MAX(daily_return) * 100 AS max_return_pct
        FROM daily_returns
        WHERE daily_return IS NOT NULL
        GROUP BY quarter

        ORDER BY pattern_type, pattern_value::integer
        """

        with self.get_session() as session:
            result = session.execute(text(query), params).fetchall()

            if not result:
                return pd.DataFrame()

            df = pd.DataFrame(
                result,
                columns=[
                    "pattern_type",
                    "pattern_value",
                    "occurrences",
                    "avg_return_pct",
                    "volatility_pct",
                    "min_return_pct",
                    "max_return_pct",
                ],
            )
            return df

    def get_top_performers(
        self,
        timeframe_days: int = 30,
        limit: int = 10,
    ) -> pd.DataFrame:
        """Get top performing stocks over a specified timeframe.

        Args:
            timeframe_days: Number of days to analyze
            limit: Number of top performers to return

        Returns:
            DataFrame with top performers
        """
        cutoff_date = datetime.now() - timedelta(days=timeframe_days)

        query = """
        WITH period_performance AS (
            SELECT
                symbol,
                first(close_price, timestamp) AS start_price,
                last(close_price, timestamp) AS end_price,
                max(high_price) AS period_high,
                min(low_price) AS period_low,
                sum(volume) AS total_volume,
                count(*) AS trading_days
            FROM price_history
            WHERE timestamp >= :cutoff_date
            GROUP BY symbol
            HAVING count(*) >= :min_trading_days
        )
        SELECT
            symbol,
            start_price,
            end_price,
            ((end_price - start_price) / start_price) * 100 AS total_return_pct,
            ((period_high - start_price) / start_price) * 100 AS max_gain_pct,
            ((period_low - start_price) / start_price) * 100 AS max_loss_pct,
            total_volume,
            trading_days
        FROM period_performance
        WHERE start_price > 0 AND end_price > 0
        ORDER BY total_return_pct DESC
        LIMIT :limit
        """

        params: dict[str, Any] = {
            "cutoff_date": cutoff_date,
            "min_trading_days": max(1, timeframe_days * 0.7),  # Require at least 70% trading days
            "limit": limit,
        }

        with self.get_session() as session:
            result = session.execute(text(query), params).fetchall()

            if not result:
                return pd.DataFrame()

            df = pd.DataFrame(
                result,
                columns=[
                    "symbol",
                    "start_price",
                    "end_price",
                    "total_return_pct",
                    "max_gain_pct",
                    "max_loss_pct",
                    "total_volume",
                    "trading_days",
                ],
            )
            return df

    # ========================================
    # Context Manager Support
    # ========================================

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Parameters are required by context manager protocol
        _ = exc_type, exc_val, exc_tb  # Mark as used
        self.close()
        return None


# ========================================
# Factory Functions
# ========================================
