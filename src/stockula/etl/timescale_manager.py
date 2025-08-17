"""TimescaleDB manager for time-series data operations."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text
from sqlmodel import Session

from ..config.exceptions import DatabaseException
from ..interfaces import ILoggingManager


class TimescaleDBManager:
    """Manages TimescaleDB operations for time-series stock data."""

    def __init__(
        self,
        connection_string: str,
        logging_manager: ILoggingManager,
        pool_size: int = 20,
        max_overflow: int = 30,
    ):
        """Initialize TimescaleDB manager.

        Args:
            connection_string: PostgreSQL connection string
            logging_manager: Logging manager instance
            pool_size: Connection pool size
            max_overflow: Maximum pool overflow
        """
        self.connection_string = connection_string
        self.logger = logging_manager

        # Set SQL files directory
        self.sql_dir = Path(__file__).parent / "sql"

        # Create SQLAlchemy engine with connection pooling
        self.engine = create_engine(
            connection_string,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            echo=False,
        )

        # Initialize schema
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Initialize TimescaleDB schema and hypertables."""
        try:
            with self.engine.connect() as conn:
                # Create TimescaleDB extension if not exists
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))

                # Create tables with proper schema
                self._create_tables(conn)

                # Create hypertables
                self._create_hypertables(conn)

                # Create indexes
                self._create_indexes(conn)

                # Set up compression and retention policies
                self._setup_policies(conn)

                conn.commit()

        except Exception as e:
            self.logger.error(f"Failed to initialize TimescaleDB schema: {e}")
            raise DatabaseException(f"Schema initialization failed: {e}") from e

    def _load_sql_file(self, filename: str) -> str:
        """Load SQL content from file.

        Args:
            filename: Name of the SQL file (without .sql extension)

        Returns:
            SQL content as string

        Raises:
            DatabaseException: If file cannot be read
        """
        sql_file = self.sql_dir / f"{filename}.sql"
        try:
            return sql_file.read_text(encoding="utf-8")
        except FileNotFoundError as e:
            raise DatabaseException(f"SQL file not found: {sql_file}") from e
        except Exception as e:
            raise DatabaseException(f"Failed to load SQL file {sql_file}: {e}") from e

    def _create_tables(self, conn) -> None:
        """Create database tables."""
        tables_sql = self._load_sql_file("tables")
        conn.execute(text(tables_sql))

    def _create_hypertables(self, conn) -> None:
        """Create TimescaleDB hypertables."""
        hypertables_sql = self._load_sql_file("hypertables")
        conn.execute(text(hypertables_sql))

    def _create_indexes(self, conn) -> None:
        """Create optimized indexes for time-series queries."""
        indexes_sql = self._load_sql_file("indexes")
        conn.execute(text(indexes_sql))

    def _setup_policies(self, conn) -> None:
        """Set up compression and retention policies."""
        policies_sql = self._load_sql_file("policies")

        try:
            conn.execute(text(policies_sql))
        except Exception as e:
            # Policies might already exist, log warning but don't fail
            self.logger.warning(f"Could not set up all policies: {e}")

    @contextmanager
    def get_session(self) -> Iterator[Session]:
        """Get a database session as context manager."""
        with Session(self.engine) as session:
            yield session

    def bulk_insert_price_history(self, data: list[dict[str, Any]], batch_size: int = 10000) -> int:
        """Bulk insert price history data using COPY for maximum performance.

        Args:
            data: List of price history records
            batch_size: Number of records per batch

        Returns:
            Number of records inserted
        """
        if not data:
            return 0

        total_inserted = 0

        try:
            # Use raw psycopg2 connection for COPY performance
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cursor:
                    # Process data in batches
                    for i in range(0, len(data), batch_size):
                        batch = data[i : i + batch_size]

                        # Prepare data for COPY
                        values = [
                            (
                                record["symbol"],
                                record["timestamp"],
                                record.get("interval_type", "1d"),
                                record.get("open_price"),
                                record.get("high_price"),
                                record.get("low_price"),
                                record.get("close_price"),
                                record.get("volume"),
                            )
                            for record in batch
                        ]

                        # Use ON CONFLICT for upsert behavior
                        insert_sql = """
                        INSERT INTO price_history
                        (symbol, timestamp, interval_type, open_price, high_price,
                         low_price, close_price, volume)
                        VALUES %s
                        ON CONFLICT (symbol, timestamp, interval_type)
                        DO UPDATE SET
                            open_price = EXCLUDED.open_price,
                            high_price = EXCLUDED.high_price,
                            low_price = EXCLUDED.low_price,
                            close_price = EXCLUDED.close_price,
                            volume = EXCLUDED.volume,
                            updated_at = NOW()
                        """

                        execute_values(cursor, insert_sql, values, page_size=1000)
                        total_inserted += len(batch)

                    conn.commit()

        except Exception as e:
            self.logger.error(f"Bulk insert failed: {e}")
            raise DatabaseException(f"Bulk insert failed: {e}") from e

        self.logger.info(f"Successfully inserted {total_inserted} price history records")
        return total_inserted

    def upsert_stock_info(self, symbol: str, info: dict[str, Any]) -> None:
        """Upsert stock information.

        Args:
            symbol: Stock symbol
            info: Stock information dictionary
        """
        try:
            with self.get_session() as session:
                # First ensure stock exists
                stock_sql = """
                INSERT INTO stocks (symbol, name, sector, industry, market_cap, exchange, currency)
                VALUES (:symbol, :name, :sector, :industry, :market_cap, :exchange, :currency)
                ON CONFLICT (symbol)
                DO UPDATE SET
                    name = EXCLUDED.name,
                    sector = EXCLUDED.sector,
                    industry = EXCLUDED.industry,
                    market_cap = EXCLUDED.market_cap,
                    exchange = EXCLUDED.exchange,
                    currency = EXCLUDED.currency,
                    updated_at = NOW()
                """

                session.exec(
                    text(stock_sql),
                    {
                        "symbol": symbol,
                        "name": info.get("longName") or info.get("shortName"),
                        "sector": info.get("sector"),
                        "industry": info.get("industry"),
                        "market_cap": info.get("marketCap"),
                        "exchange": info.get("exchange"),
                        "currency": info.get("currency"),
                    },
                )

                # Upsert stock info JSON
                info_sql = """
                INSERT INTO stock_info (symbol, info_json)
                VALUES (:symbol, :info_json)
                ON CONFLICT (symbol)
                DO UPDATE SET
                    info_json = EXCLUDED.info_json,
                    updated_at = NOW()
                """

                session.exec(
                    text(info_sql),
                    {
                        "symbol": symbol,
                        "info_json": info,
                    },
                )

                session.commit()

        except Exception as e:
            self.logger.error(f"Failed to upsert stock info for {symbol}: {e}")
            raise DatabaseException(f"Stock info upsert failed: {e}") from e

    def get_price_history(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Retrieve price history data.

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval

        Returns:
            DataFrame with price history
        """
        query = """
        SELECT timestamp, open_price, high_price, low_price, close_price, volume
        FROM price_history
        WHERE symbol = :symbol AND interval_type = :interval
        """

        params = {"symbol": symbol, "interval": interval}

        if start_date:
            query += " AND timestamp >= :start_date"
            params["start_date"] = start_date

        if end_date:
            query += " AND timestamp <= :end_date"
            params["end_date"] = end_date

        query += " ORDER BY timestamp"

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql_query(
                    text(query), conn, params=params, parse_dates=["timestamp"], index_col="timestamp"
                )

                # Rename columns to match expected format
                df.columns = ["Open", "High", "Low", "Close", "Volume"]
                return df

        except Exception as e:
            self.logger.error(f"Failed to retrieve price history for {symbol}: {e}")
            raise DatabaseException(f"Price history retrieval failed: {e}") from e

    def get_data_quality_metrics(self) -> dict[str, Any]:
        """Get data quality metrics for monitoring.

        Returns:
            Dictionary with quality metrics
        """
        metrics: dict[str, Any] = {}

        try:
            with self.engine.connect() as conn:
                # Data completeness metrics
                completeness_sql = """
                SELECT
                    'price_history' as table_name,
                    COUNT(*) as total_records,
                    COUNT(open_price) as open_price_count,
                    COUNT(high_price) as high_price_count,
                    COUNT(low_price) as low_price_count,
                    COUNT(close_price) as close_price_count,
                    COUNT(volume) as volume_count
                FROM price_history
                WHERE timestamp >= NOW() - INTERVAL '7 days'
                """

                result = conn.execute(text(completeness_sql)).fetchone()
                if result:
                    total = result.total_records
                    metrics["completeness"] = {
                        "open_price": result.open_price_count / total if total > 0 else 0,
                        "high_price": result.high_price_count / total if total > 0 else 0,
                        "low_price": result.low_price_count / total if total > 0 else 0,
                        "close_price": result.close_price_count / total if total > 0 else 0,
                        "volume": result.volume_count / total if total > 0 else 0,
                    }

                # Data freshness metrics
                freshness_sql = """
                SELECT
                    symbol,
                    MAX(timestamp) as latest_timestamp,
                    EXTRACT(EPOCH FROM (NOW() - MAX(timestamp))) / 3600 as hours_since_update
                FROM price_history
                GROUP BY symbol
                HAVING MAX(timestamp) < NOW() - INTERVAL '24 hours'
                """

                stale_symbols = conn.execute(text(freshness_sql)).fetchall()
                metrics["freshness"] = {
                    "stale_symbols_count": len(stale_symbols),
                    "stale_symbols": [row.symbol for row in stale_symbols[:10]],  # Limit to 10
                }

                # Data consistency metrics
                consistency_sql = """
                SELECT COUNT(*) as invalid_price_records
                FROM price_history
                WHERE high_price < low_price
                   OR open_price < 0
                   OR close_price < 0
                   OR volume < 0
                   AND timestamp >= NOW() - INTERVAL '7 days'
                """

                result = conn.execute(text(consistency_sql)).fetchone()
                metrics["consistency"] = {"invalid_price_records": result.invalid_price_records if result else 0}

        except Exception as e:
            self.logger.error(f"Failed to get data quality metrics: {e}")
            metrics["error"] = str(e)

        return metrics

    def create_continuous_aggregates(self) -> None:
        """Create continuous aggregates for common queries."""
        aggregates_sql = self._load_sql_file("aggregates")

        try:
            with self.engine.connect() as conn:
                conn.execute(text(aggregates_sql))
                conn.commit()
                self.logger.info("Successfully created continuous aggregates")

        except Exception as e:
            self.logger.warning(f"Could not create all continuous aggregates: {e}")

    def get_database_stats(self) -> dict[str, Any]:
        """Get comprehensive database statistics.

        Returns:
            Dictionary with database statistics
        """
        stats = {}

        try:
            with self.engine.connect() as conn:
                # Table row counts
                tables = [
                    "stocks",
                    "price_history",
                    "dividends",
                    "splits",
                    "options_calls",
                    "options_puts",
                    "stock_info",
                ]

                for table in tables:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()
                    stats[f"{table}_count"] = result[0] if result else 0

                # Hypertable statistics
                hypertable_stats = """
                SELECT
                    hypertable_name,
                    num_chunks,
                    table_size,
                    index_size,
                    total_size
                FROM timescaledb_information.hypertables h
                LEFT JOIN timescaledb_information.hypertable_detailed_size(h.hypertable_name) s
                ON true
                """

                hypertable_results = conn.execute(text(hypertable_stats)).fetchall()
                stats["hypertables"] = [
                    {
                        "name": row.hypertable_name,
                        "chunks": row.num_chunks,
                        "table_size": row.table_size,
                        "index_size": row.index_size,
                        "total_size": row.total_size,
                    }
                    for row in hypertable_results
                ]

                # Data range statistics
                range_stats = """
                SELECT
                    MIN(timestamp) as earliest_data,
                    MAX(timestamp) as latest_data,
                    COUNT(DISTINCT symbol) as unique_symbols
                FROM price_history
                """

                result = conn.execute(text(range_stats)).fetchone()
                if result:
                    stats["data_range"] = {
                        "earliest_data": result.earliest_data,
                        "latest_data": result.latest_data,
                        "unique_symbols": result.unique_symbols,
                    }

        except Exception as e:
            self.logger.error(f"Failed to get database stats: {e}")
            stats["error"] = str(e)

        return stats

    def close(self) -> None:
        """Close database connections."""
        if hasattr(self, "engine") and self.engine:
            self.engine.dispose()
