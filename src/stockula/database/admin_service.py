"""
Database administration service following SRP.
Handles only database administration tasks like migrations, cleanup, and maintenance.
"""

import logging
from datetime import date, timedelta
from typing import Any, Dict

from sqlalchemy import text

from .connection_manager import DatabaseConnectionManager

logger = logging.getLogger(__name__)


class DatabaseAdminService:
    """Service for database administration - Single Responsibility: Database Administration."""

    def __init__(self, connection_manager: DatabaseConnectionManager):
        """Initialize admin service with connection manager.

        Args:
            connection_manager: Database connection manager
        """
        self.connection_manager = connection_manager
        self._migrations_run = False
        self._timescale_setup_run = False

    def run_migrations(self) -> None:
        """Run database migrations."""
        if self._migrations_run:
            return

        try:
            # Create tables first
            self.connection_manager.create_tables()

            # Run TimescaleDB-specific setup
            self._setup_timescale_features()

            self._migrations_run = True
            logger.info("Database migrations completed successfully")

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise

    def _setup_timescale_features(self) -> None:
        """Setup TimescaleDB-specific features."""
        if self._timescale_setup_run:
            return

        if not self.connection_manager.test_timescale_extension():
            logger.warning("TimescaleDB extension not available, skipping TimescaleDB setup")
            return

        with self.connection_manager.get_session() as session:
            try:
                # Create hypertable for price_history if not exists
                session.execute(
                    text("""
                    SELECT create_hypertable('price_history', 'timestamp',
                                           chunk_time_interval => INTERVAL '1 week',
                                           if_not_exists => TRUE);
                """)
                )

                # Add compression policy
                session.execute(
                    text("""
                    SELECT add_compression_policy('price_history', INTERVAL '30 days', if_not_exists => TRUE);
                """)
                )

                # Add retention policy (optional - keep 2 years of data)
                session.execute(
                    text("""
                    SELECT add_retention_policy('price_history', INTERVAL '2 years', if_not_exists => TRUE);
                """)
                )

                logger.info("TimescaleDB features setup completed")
                self._timescale_setup_run = True

            except Exception as e:
                logger.error(f"TimescaleDB setup failed: {e}")
                # Don't raise - allow system to work without TimescaleDB features

    def cleanup_old_data(self, symbol: str, days_to_keep: int = 365) -> int:
        """Clean up old data for a symbol.

        Args:
            symbol: Stock symbol
            days_to_keep: Number of days of data to keep

        Returns:
            Number of records deleted
        """
        cutoff_date = date.today() - timedelta(days=days_to_keep)

        with self.connection_manager.get_session() as session:
            # Delete old price history
            result = session.execute(
                text("""
                DELETE FROM price_history
                WHERE symbol = :symbol AND timestamp < :cutoff_date
            """),
                {"symbol": symbol, "cutoff_date": cutoff_date},
            )

            deleted_count = result.rowcount

            # Delete old dividends
            result2 = session.execute(
                text("""
                DELETE FROM dividends
                WHERE symbol = :symbol AND ex_date < :cutoff_date
            """),
                {"symbol": symbol, "cutoff_date": cutoff_date},
            )
            deleted_count += result2.rowcount

            # Delete old splits
            result3 = session.execute(
                text("""
                DELETE FROM splits
                WHERE symbol = :symbol AND split_date < :cutoff_date
            """),
                {"symbol": symbol, "cutoff_date": cutoff_date},
            )
            deleted_count += result3.rowcount

            # Delete old options
            result4 = session.execute(
                text("""
                DELETE FROM options
                WHERE symbol = :symbol AND expiration_date < :cutoff_date
            """),
                {"symbol": symbol, "cutoff_date": cutoff_date},
            )
            deleted_count += result4.rowcount

            logger.info(f"Cleaned up {deleted_count} old records for {symbol}")
            return int(deleted_count)

    def vacuum_analyze(self) -> None:
        """Run VACUUM ANALYZE on all tables for performance."""
        tables = ["price_history", "stock_info", "dividends", "splits", "options"]

        with self.connection_manager.get_session() as session:
            for table in tables:
                try:
                    session.execute(text(f"VACUUM ANALYZE {table}"))
                    logger.debug(f"VACUUM ANALYZE completed for {table}")
                except Exception as e:
                    logger.warning(f"VACUUM ANALYZE failed for {table}: {e}")

    def get_table_sizes(self) -> Dict[str, Dict[str, Any]]:
        """Get size information for all tables.

        Returns:
            Dictionary with table size information
        """
        with self.connection_manager.get_session() as session:
            result = session.execute(
                text("""
                SELECT
                    schemaname,
                    tablename,
                    attname as column_name,
                    n_distinct,
                    correlation
                FROM pg_stats
                WHERE schemaname = 'public'
                AND tablename IN ('price_history', 'stock_info', 'dividends', 'splits', 'options')
                ORDER BY tablename, attname
            """)
            ).fetchall()

            table_stats: Dict[str, Dict[str, Any]] = {}
            for row in result:
                table_name = row.tablename
                if table_name not in table_stats:
                    table_stats[table_name] = {
                        "columns": [],
                        "row_count": 0,
                        "size_bytes": 0,
                    }

                table_stats[table_name]["columns"].append(
                    {
                        "name": row.column_name,
                        "n_distinct": row.n_distinct,
                        "correlation": row.correlation,
                    }
                )

            # Get row counts and sizes
            for table_name in table_stats:
                try:
                    # Get row count
                    count_result = session.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
                    table_stats[table_name]["row_count"] = count_result

                    # Get table size
                    size_result = session.execute(text(f"SELECT pg_total_relation_size('{table_name}')")).scalar()
                    table_stats[table_name]["size_bytes"] = size_result

                except Exception as e:
                    logger.warning(f"Failed to get stats for {table_name}: {e}")

            return table_stats

    def optimize_tables(self) -> None:
        """Optimize tables for better performance."""
        with self.connection_manager.get_session() as session:
            try:
                # Update table statistics
                session.execute(text("ANALYZE"))

                # Reindex if needed (for PostgreSQL)
                session.execute(text("REINDEX DATABASE CONCURRENTLY"))

                logger.info("Table optimization completed")

            except Exception as e:
                logger.warning(f"Table optimization failed: {e}")

    def get_chunk_statistics(self) -> Dict[str, Any]:
        """Get TimescaleDB chunk statistics (if available).

        Returns:
            Dictionary with chunk statistics
        """
        if not self.connection_manager.test_timescale_extension():
            return {}

        with self.connection_manager.get_session() as session:
            try:
                result = session.execute(
                    text("""
                    SELECT
                        chunk_schema,
                        chunk_name,
                        table_name,
                        range_start,
                        range_end,
                        is_compressed,
                        chunk_tablespace
                    FROM timescaledb_information.chunks
                    WHERE hypertable_name = 'price_history'
                    ORDER BY range_start DESC
                    LIMIT 20
                """)
                ).fetchall()

                chunks = []
                for row in result:
                    chunks.append(
                        {
                            "schema": row.chunk_schema,
                            "name": row.chunk_name,
                            "table": row.table_name,
                            "range_start": row.range_start,
                            "range_end": row.range_end,
                            "is_compressed": row.is_compressed,
                            "tablespace": row.chunk_tablespace,
                        }
                    )

                return {
                    "chunk_count": len(chunks),
                    "chunks": chunks,
                }

            except Exception as e:
                logger.error(f"Failed to get chunk statistics: {e}")
                return {}

    def compress_chunks(self, older_than_days: int = 30) -> int:
        """Compress old chunks for space savings.

        Args:
            older_than_days: Compress chunks older than this many days

        Returns:
            Number of chunks compressed
        """
        if not self.connection_manager.test_timescale_extension():
            return 0

        with self.connection_manager.get_session() as session:
            try:
                cutoff_date = date.today() - timedelta(days=older_than_days)

                result = session.execute(
                    text("""
                    SELECT compress_chunk(show_chunks('price_history', older_than => :cutoff_date))
                """),
                    {"cutoff_date": cutoff_date},
                ).fetchall()

                compressed_count = len(result)
                logger.info(f"Compressed {compressed_count} chunks")
                return compressed_count

            except Exception as e:
                logger.error(f"Chunk compression failed: {e}")
                return 0
