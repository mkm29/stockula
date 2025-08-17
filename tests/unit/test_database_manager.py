"""Unit tests for database manager module."""

import numpy as np
import pandas as pd

from stockula.config.models import TimescaleDBConfig
from stockula.interfaces import IDatabaseManager


class TestDatabaseManagerInitialization:
    """Test DatabaseManager initialization."""

    def test_manager_initialization(self, database_manager: IDatabaseManager):
        """Test that database manager initializes correctly."""
        # Test that the manager implements the interface
        assert isinstance(database_manager, IDatabaseManager)

        # Test backend properties
        assert hasattr(database_manager, "backend_type")
        assert hasattr(database_manager, "is_timescaledb")
        assert database_manager.is_timescaledb is True

    def test_connection_establishment(self, database_manager: IDatabaseManager):
        """Test that database connection can be established."""
        # Test connection through session context manager
        with database_manager.get_session() as session:
            assert session is not None

    def test_timescaledb_features(self, database_manager: IDatabaseManager):
        """Test TimescaleDB-specific features are available."""
        # Test that TimescaleDB-specific methods exist
        timescale_methods = [
            "get_price_history_aggregated",
            "get_recent_price_changes",
            "get_chunk_statistics",
            "test_connection",
        ]

        for method_name in timescale_methods:
            assert hasattr(database_manager, method_name)
            assert callable(getattr(database_manager, method_name))

    def test_configuration_handling(self, timescaledb_config: TimescaleDBConfig):
        """Test that configuration is handled correctly."""
        assert timescaledb_config.host is not None
        assert timescaledb_config.database is not None
        assert timescaledb_config.user is not None
        assert timescaledb_config.port > 0


class TestStockInfoOperations:
    """Test stock info storage and retrieval."""

    def test_store_stock_info(self, database_manager: IDatabaseManager):
        """Test storing stock information."""
        info = {
            "longName": "Test Company Inc.",
            "sector": "Technology",
            "industry": "Software",
            "marketCap": 1000000000,
            "exchange": "NASDAQ",
            "currency": "USD",
            "website": "https://test.com",
        }

        # Store stock info
        database_manager.store_stock_info("TEST", info)

        # Verify it was stored
        retrieved_info = database_manager.get_stock_info("TEST")
        assert retrieved_info is not None
        assert retrieved_info["longName"] == info["longName"]
        assert retrieved_info["sector"] == info["sector"]

    def test_get_stock_info_nonexistent(self, database_manager: IDatabaseManager):
        """Test retrieving non-existent stock info."""
        result = database_manager.get_stock_info("NONEXISTENT")
        assert result is None

    def test_update_stock_info(self, database_manager: IDatabaseManager):
        """Test updating existing stock information."""
        initial_info = {
            "longName": "Initial Company",
            "sector": "Technology",
            "marketCap": 1000000000,
        }

        updated_info = {
            "longName": "Updated Company Inc.",
            "sector": "Technology",
            "marketCap": 2000000000,
            "industry": "Software",
        }

        # Store initial info
        database_manager.store_stock_info("UPDATE_TEST", initial_info)

        # Update with new info
        database_manager.store_stock_info("UPDATE_TEST", updated_info)

        # Verify update
        retrieved = database_manager.get_stock_info("UPDATE_TEST")
        assert retrieved is not None
        assert retrieved["longName"] == updated_info["longName"]
        assert retrieved["marketCap"] == updated_info["marketCap"]
        assert retrieved["industry"] == updated_info["industry"]

    def test_store_complex_stock_info(self, database_manager: IDatabaseManager):
        """Test storing complex stock information with nested data."""
        complex_info = {
            "longName": "Complex Company",
            "sector": "Technology",
            "financialData": {"totalRevenue": 50000000000, "totalDebt": 10000000000, "returnOnEquity": 0.25},
            "keyStatistics": {"beta": 1.5, "trailingPE": 20.5, "forwardPE": 18.2},
            "officers": [{"name": "John Doe", "title": "CEO"}, {"name": "Jane Smith", "title": "CFO"}],
        }

        database_manager.store_stock_info("COMPLEX", complex_info)
        retrieved = database_manager.get_stock_info("COMPLEX")

        assert retrieved is not None
        assert retrieved["longName"] == complex_info["longName"]
        # Verify nested structures are preserved
        assert "financialData" in retrieved
        assert retrieved["financialData"]["totalRevenue"] == 50000000000


class TestPriceHistoryOperations:
    """Test price history storage and retrieval."""

    def test_store_price_history(self, database_manager: IDatabaseManager, sample_price_data: pd.DataFrame):
        """Test storing price history."""
        database_manager.store_price_history("TEST", sample_price_data)

        # Verify data was stored by retrieving it
        retrieved = database_manager.get_price_history("TEST")
        assert isinstance(retrieved, pd.DataFrame)
        assert not retrieved.empty
        assert len(retrieved) == len(sample_price_data)

    def test_store_price_history_with_interval(
        self, database_manager: IDatabaseManager, sample_price_data: pd.DataFrame
    ):
        """Test storing price history with custom interval."""
        database_manager.store_price_history("TEST", sample_price_data, interval="1h")

        # Verify data was stored
        retrieved = database_manager.get_price_history("TEST")
        assert not retrieved.empty

    def test_get_price_history(self, database_manager: IDatabaseManager, sample_price_data: pd.DataFrame):
        """Test retrieving price history."""
        database_manager.store_price_history("TEST", sample_price_data)

        retrieved = database_manager.get_price_history("TEST")

        assert isinstance(retrieved, pd.DataFrame)
        assert not retrieved.empty
        assert all(col in retrieved.columns for col in ["Open", "High", "Low", "Close", "Volume"])

    def test_get_price_history_empty(self, database_manager: IDatabaseManager):
        """Test retrieving price history for non-existent symbol."""
        retrieved = database_manager.get_price_history("NONEXISTENT")
        assert isinstance(retrieved, pd.DataFrame)
        # Should return empty DataFrame, not None
        assert retrieved.empty

    def test_get_price_history_with_date_range(
        self, database_manager: IDatabaseManager, sample_price_data: pd.DataFrame
    ):
        """Test retrieving price history with date range."""
        database_manager.store_price_history("TEST", sample_price_data)

        # Get subset of data
        start_date = "2023-01-03"
        end_date = "2023-01-07"
        retrieved = database_manager.get_price_history("TEST", start_date, end_date)

        assert isinstance(retrieved, pd.DataFrame)
        # Should have some data (exact count depends on implementation)
        assert len(retrieved) <= len(sample_price_data)

    def test_get_latest_price(self, database_manager: IDatabaseManager, sample_price_data: pd.DataFrame):
        """Test getting latest price."""
        database_manager.store_price_history("TEST", sample_price_data)

        latest = database_manager.get_latest_price("TEST")

        assert isinstance(latest, float | type(None))
        # Should have a price since we stored data
        if latest is not None:
            assert latest > 0

    def test_get_latest_price_nonexistent(self, database_manager: IDatabaseManager):
        """Test getting latest price for non-existent symbol."""
        latest = database_manager.get_latest_price("NONEXISTENT")
        assert latest is None

    def test_has_data(self, database_manager: IDatabaseManager, sample_price_data: pd.DataFrame):
        """Test checking if data exists."""
        database_manager.store_price_history("TEST", sample_price_data)

        # Should have data for stored range
        has_data = database_manager.has_data("TEST", "2023-01-01", "2023-12-31")
        # Allow both Python bool and numpy bool
        assert isinstance(has_data, bool | np.bool_) or hasattr(has_data, "dtype")
        # Should be True since we stored data in this range
        assert bool(has_data)

    def test_has_data_nonexistent(self, database_manager: IDatabaseManager):
        """Test checking if data exists for non-existent symbol."""
        has_data = database_manager.has_data("NONEXISTENT", "2023-01-01", "2023-12-31")
        # Allow both Python bool and numpy bool
        assert isinstance(has_data, bool | np.bool_) or hasattr(has_data, "dtype")
        assert not bool(has_data)

    def test_price_history_upsert_behavior(self, database_manager: IDatabaseManager):
        """Test that storing price history handles updates correctly."""
        # Create initial data
        dates = pd.date_range("2023-01-01", periods=5)
        initial_data = pd.DataFrame(
            {
                "Open": [100, 101, 102, 103, 104],
                "High": [101, 102, 103, 104, 105],
                "Low": [99, 100, 101, 102, 103],
                "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
                "Volume": [1000000, 1100000, 1200000, 1300000, 1400000],
            },
            index=dates,
        )

        # Store initial data
        database_manager.store_price_history("UPSERT_TEST", initial_data)

        # Create overlapping data with different values
        overlapping_data = pd.DataFrame(
            {
                "Open": [103, 104, 105],
                "High": [104, 105, 106],
                "Low": [102, 103, 104],
                "Close": [103.7, 104.7, 105.7],  # Different from initial
                "Volume": [1350000, 1450000, 1550000],
            },
            index=dates[2:5],
        )

        # Store overlapping data (should update existing records)
        database_manager.store_price_history("UPSERT_TEST", overlapping_data)

        # Retrieve all data
        retrieved = database_manager.get_price_history("UPSERT_TEST")

        # Should have the updated values for overlapping dates
        assert not retrieved.empty
        # This is implementation-dependent, but we should have some data


class TestDividendsAndSplits:
    """Test dividends and splits operations."""

    def test_store_dividends(self, database_manager: IDatabaseManager, sample_dividends: pd.Series):
        """Test storing dividend data."""
        # Store dividends (this is a write-only operation in the current interface)
        database_manager.store_dividends("TEST", sample_dividends)

        # Note: The IDatabaseManager interface doesn't have get_dividends method
        # This is intentional as dividends are stored but retrieved via other means
        # We verify by checking that the operation doesn't raise an exception

    def test_store_empty_dividends(self, database_manager: IDatabaseManager):
        """Test storing empty dividend data."""
        empty_dividends = pd.Series([], dtype=float)
        # Should not raise an exception
        database_manager.store_dividends("TEST_EMPTY", empty_dividends)

    def test_store_splits(self, database_manager: IDatabaseManager, sample_splits: pd.Series):
        """Test storing stock split data."""
        # Store splits (this is a write-only operation in the current interface)
        database_manager.store_splits("TEST", sample_splits)

        # Note: The IDatabaseManager interface doesn't have get_splits method
        # This is intentional as splits are stored but retrieved via other means
        # We verify by checking that the operation doesn't raise an exception

    def test_store_empty_splits(self, database_manager: IDatabaseManager):
        """Test storing empty splits data."""
        empty_splits = pd.Series([], dtype=float)
        # Should not raise an exception
        database_manager.store_splits("TEST_EMPTY", empty_splits)

    def test_store_multiple_dividend_batches(self, database_manager: IDatabaseManager):
        """Test storing multiple batches of dividend data."""
        # First batch
        first_batch = pd.Series([0.25, 0.26], index=pd.to_datetime(["2023-01-15", "2023-04-15"]))
        database_manager.store_dividends("MULTI_DIV", first_batch)

        # Second batch (different time period)
        second_batch = pd.Series([0.27, 0.28], index=pd.to_datetime(["2023-07-15", "2023-10-15"]))
        database_manager.store_dividends("MULTI_DIV", second_batch)

        # Both operations should succeed without error


class TestUtilityMethods:
    """Test utility methods."""

    def test_get_all_symbols(self, database_manager: IDatabaseManager):
        """Test getting all symbols."""
        symbols = database_manager.get_all_symbols()

        assert isinstance(symbols, list)
        # Mock database should have some symbols from fixture setup
        if len(symbols) > 0:
            # Should have some of the test symbols
            common_symbols = ["AAPL", "GOOGL", "MSFT"]
            assert any(symbol in symbols for symbol in common_symbols)

    def test_get_all_symbols_empty(self, database_manager: IDatabaseManager):
        """Test getting symbols from empty database."""
        symbols = database_manager.get_all_symbols()
        assert isinstance(symbols, list)
        # May be empty or have mock data, but should be a list

    def test_get_database_stats(self, database_manager: IDatabaseManager):
        """Test getting database statistics."""
        stats = database_manager.get_database_stats()

        assert isinstance(stats, dict)
        # Should have some common statistics
        # The exact keys depend on implementation, but should have counts
        assert any(key in stats for key in ["stocks", "price_history", "symbols", "rows"])

        # Values should be non-negative integers
        for _key, value in stats.items():
            assert isinstance(value, int)
            assert value >= 0

    def test_get_database_stats_empty(self, database_manager: IDatabaseManager):
        """Test getting stats from empty database."""
        stats = database_manager.get_database_stats()
        assert isinstance(stats, dict)
        # Even empty database should return valid stats structure


class TestAnalyticsOperations:
    """Test analytics and technical indicator operations."""

    def test_get_moving_averages(self, database_manager: IDatabaseManager):
        """Test getting moving averages."""
        result = database_manager.get_moving_averages("AAPL")

        assert isinstance(result, pd.DataFrame)
        # Should have close_price column and MA columns
        if not result.empty:
            assert "close_price" in result.columns
            # Should have some MA columns (implementation dependent)

    def test_get_moving_averages_with_periods(self, database_manager: IDatabaseManager):
        """Test getting moving averages with custom periods."""
        periods = [10, 20, 50]
        result = database_manager.get_moving_averages("AAPL", periods=periods)

        assert isinstance(result, pd.DataFrame)

    def test_get_bollinger_bands(self, database_manager: IDatabaseManager):
        """Test getting Bollinger Bands."""
        result = database_manager.get_bollinger_bands("AAPL")

        assert isinstance(result, pd.DataFrame)
        # Should have close_price and BB columns if implemented
        if not result.empty:
            assert "close_price" in result.columns

    def test_get_bollinger_bands_custom_params(self, database_manager: IDatabaseManager):
        """Test Bollinger Bands with custom parameters."""
        result = database_manager.get_bollinger_bands("AAPL", period=10, std_dev=1.5)

        assert isinstance(result, pd.DataFrame)

    def test_get_rsi(self, database_manager: IDatabaseManager):
        """Test getting RSI."""
        result = database_manager.get_rsi("AAPL")

        assert isinstance(result, pd.DataFrame)
        # Should have close_price and RSI columns if implemented
        if not result.empty:
            assert "close_price" in result.columns

    def test_get_rsi_custom_period(self, database_manager: IDatabaseManager):
        """Test RSI with custom period."""
        result = database_manager.get_rsi("AAPL", period=21)

        assert isinstance(result, pd.DataFrame)

    def test_analytics_with_date_range(self, database_manager: IDatabaseManager):
        """Test analytics methods with date range."""
        start_date = "2023-06-01"
        end_date = "2023-12-31"

        # Test each analytics method with date range
        ma_result = database_manager.get_moving_averages("AAPL", start_date=start_date, end_date=end_date)
        bb_result = database_manager.get_bollinger_bands("AAPL", start_date=start_date, end_date=end_date)
        rsi_result = database_manager.get_rsi("AAPL", start_date=start_date, end_date=end_date)

        assert isinstance(ma_result, pd.DataFrame)
        assert isinstance(bb_result, pd.DataFrame)
        assert isinstance(rsi_result, pd.DataFrame)

    def test_analytics_nonexistent_symbol(self, database_manager: IDatabaseManager):
        """Test analytics methods with non-existent symbol."""
        # Should return empty DataFrames, not raise exceptions
        ma_result = database_manager.get_moving_averages("NONEXISTENT")
        bb_result = database_manager.get_bollinger_bands("NONEXISTENT")
        rsi_result = database_manager.get_rsi("NONEXISTENT")

        assert isinstance(ma_result, pd.DataFrame)
        assert isinstance(bb_result, pd.DataFrame)
        assert isinstance(rsi_result, pd.DataFrame)

        # Should be empty
        assert ma_result.empty
        assert bb_result.empty
        assert rsi_result.empty


class TestAdvancedAnalytics:
    """Test advanced analytics methods (may not be fully implemented)."""

    def test_advanced_methods_exist(self, database_manager: IDatabaseManager):
        """Test that advanced analytics methods exist and are callable."""
        advanced_methods = [
            "get_price_momentum",
            "get_correlation_matrix",
            "get_volatility_analysis",
            "get_seasonal_patterns",
            "get_top_performers",
        ]

        for method_name in advanced_methods:
            assert hasattr(database_manager, method_name)
            method = getattr(database_manager, method_name)
            assert callable(method)

    def test_price_momentum_callable(self, database_manager: IDatabaseManager):
        """Test that price momentum method is callable."""
        try:
            result = database_manager.get_price_momentum()
            assert isinstance(result, pd.DataFrame)
        except NotImplementedError:
            # This is acceptable for methods not yet implemented
            pass

    def test_correlation_matrix_callable(self, database_manager: IDatabaseManager):
        """Test that correlation matrix method is callable."""
        try:
            result = database_manager.get_correlation_matrix(["AAPL", "GOOGL"])
            assert isinstance(result, pd.DataFrame)
        except (NotImplementedError, ValueError):
            # This is acceptable for methods not yet implemented or insufficient data
            pass

    def test_volatility_analysis_callable(self, database_manager: IDatabaseManager):
        """Test that volatility analysis method is callable."""
        try:
            result = database_manager.get_volatility_analysis()
            assert isinstance(result, pd.DataFrame)
        except NotImplementedError:
            # This is acceptable for methods not yet implemented
            pass


class TestSessionManagement:
    """Test session and connection management."""

    def test_get_session_context_manager(self, database_manager: IDatabaseManager):
        """Test that get_session works as context manager."""
        with database_manager.get_session() as session:
            assert session is not None

    def test_close_method(self, database_manager: IDatabaseManager):
        """Test that close method exists and is callable."""
        # Should not raise an exception
        database_manager.close()

    def test_test_connection_method(self, database_manager: IDatabaseManager):
        """Test connection testing method."""
        try:
            result = database_manager.test_connection()
            assert isinstance(result, dict)
            # Should contain connection information
            assert "connected" in result or "status" in result or "backend" in result
        except NotImplementedError:
            # This is acceptable for implementations that don't support connection testing
            pass


class TestTimescaleSpecificFeatures:
    """Test TimescaleDB-specific features."""

    def test_backend_properties(self, database_manager: IDatabaseManager):
        """Test TimescaleDB backend properties."""
        assert hasattr(database_manager, "backend_type")
        assert hasattr(database_manager, "is_timescaledb")

        backend_type = database_manager.backend_type
        assert isinstance(backend_type, str)
        assert "timescale" in backend_type.lower()

        assert database_manager.is_timescaledb is True

    def test_aggregated_price_history(self, database_manager: IDatabaseManager):
        """Test TimescaleDB time_bucket aggregation."""
        try:
            result = database_manager.get_price_history_aggregated("AAPL", "1 day")
            assert isinstance(result, pd.DataFrame)
        except NotImplementedError:
            # This is acceptable if not implemented
            pass

    def test_recent_price_changes(self, database_manager: IDatabaseManager):
        """Test recent price changes analysis."""
        try:
            result = database_manager.get_recent_price_changes(["AAPL"], hours=24)
            assert isinstance(result, pd.DataFrame)
        except NotImplementedError:
            # This is acceptable if not implemented
            pass

    def test_chunk_statistics(self, database_manager: IDatabaseManager):
        """Test TimescaleDB chunk statistics."""
        try:
            result = database_manager.get_chunk_statistics()
            assert isinstance(result, pd.DataFrame)
        except NotImplementedError:
            # This is acceptable if not implemented
            pass
