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

import logging

# Configure module logger
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Simplified DatabaseManager following SRP - delegates to specialized services.
    Single Responsibility: Coordinating database operations through focused services.
    """

    def __init__(self, config=None, enable_async: bool = True):
        """Initialize database manager with configuration.

        Args:
            config: TimescaleDB configuration or legacy string path
            enable_async: Whether to enable async database operations

        Raises:
            ValueError: If configuration is invalid
            ConnectionError: If TimescaleDB is not available
        """
        # Import here to avoid circular imports
        from .admin_service import DatabaseAdminService
        from .analytics_service import DatabaseAnalyticsService
        from .connection_manager import DatabaseConnectionManager
        from .data_repository import DatabaseDataRepository

        # Initialize connection manager
        self.connection_manager = DatabaseConnectionManager(config, enable_async)

        # Initialize specialized services
        self.data_repository = DatabaseDataRepository(self.connection_manager)
        self.analytics_service = DatabaseAnalyticsService(self.connection_manager)
        self.admin_service = DatabaseAdminService(self.connection_manager)

        # Run initial setup
        self.admin_service.run_migrations()

    @property
    def backend_type(self) -> str:
        """Get the backend type."""
        return "timescaledb"

    @property
    def is_timescaledb(self) -> bool:
        """Check if using TimescaleDB."""
        return True

    # Connection management delegation
    def get_session(self):
        """Get a database session."""
        return self.connection_manager.get_session()

    async def get_async_session(self):
        """Get an async database session."""
        return self.connection_manager.get_async_session()

    def test_connection(self) -> bool:
        """Test database connection."""
        return self.connection_manager.test_connection()

    def close(self) -> None:
        """Close database connections."""
        self.connection_manager.close()

    # Data repository delegation
    def store_stock_info(self, symbol: str, info_data: dict) -> None:
        """Store stock information."""
        self.data_repository.store_stock_info(symbol, info_data)

    def get_stock_info(self, symbol: str):
        """Get stock information."""
        return self.data_repository.get_stock_info(symbol)

    def store_price_history(self, symbol: str, data) -> None:
        """Store price history data."""
        self.data_repository.store_price_history(symbol, data)

    def get_price_history(self, symbol: str, start_date=None, end_date=None):
        """Get price history data."""
        return self.data_repository.get_price_history(symbol, start_date, end_date)

    def store_dividends(self, symbol: str, dividends_data) -> None:
        """Store dividend data."""
        self.data_repository.store_dividends(symbol, dividends_data)

    def get_dividends(self, symbol: str, start_date=None, end_date=None):
        """Get dividend data."""
        return self.data_repository.get_dividends(symbol, start_date, end_date)

    def store_splits(self, symbol: str, splits_data) -> None:
        """Store stock split data."""
        self.data_repository.store_splits(symbol, splits_data)

    def get_splits(self, symbol: str, start_date=None, end_date=None):
        """Get stock split data."""
        return self.data_repository.get_splits(symbol, start_date, end_date)

    def store_options_chain(self, symbol: str, options_data) -> None:
        """Store options chain data."""
        self.data_repository.store_options_chain(symbol, options_data)

    def get_options_chain(self, symbol: str, expiration_date=None, option_type=None):
        """Get options chain data."""
        return self.data_repository.get_options_chain(symbol, expiration_date, option_type)

    def get_all_symbols(self):
        """Get all unique symbols."""
        return self.data_repository.get_all_symbols()

    def get_latest_price(self, symbol: str):
        """Get latest price for a symbol."""
        return self.data_repository.get_latest_price(symbol)

    def get_latest_price_date(self, symbol: str):
        """Get latest price date for a symbol."""
        return self.data_repository.get_latest_price_date(symbol)

    def has_data(self, symbol: str, start_date=None) -> bool:
        """Check if data exists for a symbol."""
        return self.data_repository.has_data(symbol, start_date)

    def get_database_stats(self):
        """Get database statistics."""
        return self.data_repository.get_database_stats()

    # Analytics service delegation
    def get_moving_averages(self, symbol: str, windows, start_date=None, end_date=None):
        """Calculate moving averages."""
        return self.analytics_service.get_moving_averages(symbol, windows, start_date, end_date)

    def get_bollinger_bands(self, symbol: str, period=20, std_dev=2.0, start_date=None, end_date=None):
        """Calculate Bollinger Bands."""
        return self.analytics_service.get_bollinger_bands(symbol, period, std_dev, start_date, end_date)

    def get_rsi(self, symbol: str, period=14, start_date=None, end_date=None):
        """Calculate RSI."""
        return self.analytics_service.get_rsi(symbol, period, start_date, end_date)

    def get_price_momentum(self, symbol: str, periods, start_date=None, end_date=None):
        """Calculate price momentum."""
        return self.analytics_service.get_price_momentum(symbol, periods, start_date, end_date)

    def get_correlation_matrix(self, symbols, start_date=None, end_date=None):
        """Calculate correlation matrix."""
        return self.analytics_service.get_correlation_matrix(symbols, start_date, end_date)

    def get_volatility_analysis(self, symbol: str, windows, start_date=None, end_date=None):
        """Calculate volatility analysis."""
        return self.analytics_service.get_volatility_analysis(symbol, windows, start_date, end_date)

    def get_seasonal_patterns(self, symbol: str, start_date=None, end_date=None):
        """Analyze seasonal patterns."""
        return self.analytics_service.get_seasonal_patterns(symbol, start_date, end_date)

    def get_top_performers(self, period_days=30, limit=10):
        """Get top performing symbols."""
        return self.analytics_service.get_top_performers(period_days, limit)

    # Admin service delegation
    def cleanup_old_data(self, symbol: str, days_to_keep: int = 365) -> int:
        """Clean up old data."""
        return self.admin_service.cleanup_old_data(symbol, days_to_keep)

    def get_chunk_statistics(self):
        """Get TimescaleDB chunk statistics."""
        return self.admin_service.get_chunk_statistics()

    def vacuum_analyze(self) -> None:
        """Run VACUUM ANALYZE on tables."""
        self.admin_service.vacuum_analyze()

    def get_table_sizes(self):
        """Get table size information."""
        return self.admin_service.get_table_sizes()

    def optimize_tables(self) -> None:
        """Optimize tables for performance."""
        self.admin_service.optimize_tables()

    def compress_chunks(self, older_than_days: int = 30) -> int:
        """Compress old chunks."""
        return self.admin_service.compress_chunks(older_than_days)

    # Context management
    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor."""
        try:
            self.close()
        except Exception:
            pass


# ========================================
# Factory Functions
# ========================================
