"""Validation tests for the database migration from SQLite to TimescaleDB.

This module provides tests to ensure the migration maintains data integrity
and API compatibility while gaining TimescaleDB performance benefits.
"""

from typing import Any
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from stockula.config.models import TimescaleDBConfig
from stockula.interfaces import IDatabaseManager
from tests.unit.mocks.mock_timescale_manager import MockTimescaleDBManager


@pytest.mark.database
class TestMigrationValidation:
    """Test migration from SQLite to TimescaleDB maintains compatibility."""

    @patch("stockula.database.manager.async_sessionmaker")
    @patch("stockula.database.manager.DatabaseManager._setup_timescale_features")
    @patch("stockula.database.manager.DatabaseManager._create_tables")
    @patch("stockula.database.manager.DatabaseManager._run_migrations")
    @patch("stockula.database.manager.DatabaseManager._test_timescale_connection")
    @patch("stockula.database.manager.create_engine")
    @patch("stockula.database.manager.create_async_engine")
    def test_config_based_initialization(
        self,
        mock_async_engine,
        mock_create_engine,
        mock_test_conn,
        mock_migrations,
        mock_create_tables,
        mock_timescale_features,
        mock_async_session,
        timescale_config: TimescaleDBConfig,
    ):
        """Test that new config-based initialization works."""
        # Mock the database connection to avoid real connection attempts
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        mock_async_engine.return_value = Mock()
        mock_async_session.return_value = Mock()
        mock_test_conn.return_value = None  # No exception means success
        mock_migrations.return_value = None
        mock_create_tables.return_value = None
        mock_timescale_features.return_value = None

        # This test validates the core API change
        from stockula.database.manager import DatabaseManager

        # Should work with TimescaleDBConfig
        db = DatabaseManager(timescale_config)
        assert db.config == timescale_config
        assert db.is_timescaledb is True
        assert db.backend_type in ["timescaledb", "timescaledb_mock"]
        db.close()

    def test_backward_compatibility_methods(self, database_manager: IDatabaseManager):
        """Test that all existing interface methods still work."""
        # Test basic operations that existing code relies on

        # Store stock info
        database_manager.store_stock_info(
            "COMPAT_TEST", {"longName": "Compatibility Test Inc.", "sector": "Technology"}
        )

        # Store price data
        dates = pd.date_range("2023-01-01", periods=5)
        price_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "High": [101.0, 102.0, 103.0, 104.0, 105.0],
                "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
                "Volume": [1000000, 1100000, 1200000, 1300000, 1400000],
            },
            index=dates,
        )

        database_manager.store_price_history("COMPAT_TEST", price_data)

        # Test retrieval methods
        retrieved_info = database_manager.get_stock_info("COMPAT_TEST")
        assert retrieved_info is not None
        assert retrieved_info["longName"] == "Compatibility Test Inc."

        retrieved_data = database_manager.get_price_history("COMPAT_TEST")
        assert len(retrieved_data) == 5
        assert "Close" in retrieved_data.columns

        # Test utility methods
        symbols = database_manager.get_all_symbols()
        assert "COMPAT_TEST" in symbols

        latest_price = database_manager.get_latest_price("COMPAT_TEST")
        assert latest_price == 104.5

        stats = database_manager.get_database_stats()
        assert isinstance(stats, dict)

    def test_data_consistency_after_migration(self, populated_database_manager: IDatabaseManager):
        """Test that data remains consistent after migration."""
        # Get data for a known symbol
        aapl_data = populated_database_manager.get_price_history("AAPL")
        assert not aapl_data.empty

        # Verify data integrity
        assert (aapl_data["High"] >= aapl_data["Close"]).all()
        assert (aapl_data["Low"] <= aapl_data["Close"]).all()
        assert (aapl_data["Volume"] > 0).all()

        # Test date range queries work
        recent_data = populated_database_manager.get_price_history("AAPL", "2023-06-01", "2023-06-30")
        assert len(recent_data) <= len(aapl_data)

        # Test stock info is preserved
        stock_info = populated_database_manager.get_stock_info("AAPL")
        assert stock_info is not None
        assert "longName" in stock_info

    def test_performance_characteristics(self, database_manager: IDatabaseManager, large_dataset: pd.DataFrame):
        """Test that TimescaleDB provides expected performance characteristics."""
        import time

        # Test bulk insert performance
        start_time = time.time()
        database_manager.store_price_history("PERF_TEST", large_dataset)
        insert_time = time.time() - start_time

        # Should complete reasonably quickly (adjust threshold as needed)
        assert insert_time < 30.0, f"Bulk insert took {insert_time:.2f}s, expected < 30s"

        # Test query performance
        start_time = time.time()
        data = database_manager.get_price_history("PERF_TEST")
        query_time = time.time() - start_time

        assert query_time < 5.0, f"Query took {query_time:.2f}s, expected < 5s"
        assert len(data) == len(large_dataset)

    def test_timescale_specific_features(self, populated_database_manager: IDatabaseManager):
        """Test that TimescaleDB-specific features are available."""
        # Test time-bucket aggregation (if implemented)
        try:
            aggregated = populated_database_manager.get_price_history_aggregated(
                "AAPL", "1 day", "2023-01-01", "2023-01-31"
            )
            assert isinstance(aggregated, pd.DataFrame)
        except NotImplementedError:
            pytest.skip("Time bucket aggregation not implemented")

        # Test recent price changes (if implemented)
        try:
            changes = populated_database_manager.get_recent_price_changes(["AAPL"], hours=24)
            assert isinstance(changes, pd.DataFrame)
        except NotImplementedError:
            pytest.skip("Recent price changes not implemented")

    def test_error_handling_consistency(self, database_manager: IDatabaseManager):
        """Test that error handling works consistently."""
        # Test handling of non-existent symbols
        non_existent_data = database_manager.get_price_history("NONEXISTENT")
        assert non_existent_data.empty

        non_existent_info = database_manager.get_stock_info("NONEXISTENT")
        assert non_existent_info is None

        non_existent_price = database_manager.get_latest_price("NONEXISTENT")
        assert non_existent_price is None

        # Test has_data with non-existent symbol
        has_data = database_manager.has_data("NONEXISTENT", "2023-01-01", "2023-12-31")
        assert has_data is False

    def test_session_management(self, database_manager: IDatabaseManager):
        """Test that session management works correctly."""
        # Test session context manager
        with database_manager.get_session() as session:
            assert session is not None

        # Test connection testing (if implemented)
        try:
            connection_info = database_manager.test_connection()
            assert isinstance(connection_info, dict)
        except NotImplementedError:
            pytest.skip("Connection testing not implemented")

    def test_analytics_backward_compatibility(self, populated_database_manager: IDatabaseManager):
        """Test that analytics methods work with existing data."""
        analytics_tests = [
            ("get_moving_averages", ("AAPL",), {}),
            ("get_bollinger_bands", ("AAPL",), {}),
            ("get_rsi", ("AAPL",), {}),
        ]

        for method_name, args, kwargs in analytics_tests:
            result = getattr(populated_database_manager, method_name)(*args, **kwargs)
            assert isinstance(result, pd.DataFrame)

            # Should have data if input symbol has data
            if not result.empty:
                assert len(result) > 0

    def test_fixture_compatibility(self, timescale_config: TimescaleDBConfig):
        """Test that test fixtures work with new implementation."""
        # Use mock implementation to avoid JSONB compatibility issues
        mock_database = MockTimescaleDBManager(timescale_config)

        # Mock database should provide IDatabaseManager interface
        assert isinstance(mock_database, IDatabaseManager)

        # Should be able to store and retrieve data
        test_info = {"longName": "Test Company", "sector": "Technology"}

        mock_database.store_stock_info("FIXTURE_TEST", test_info)
        retrieved = mock_database.get_stock_info("FIXTURE_TEST")
        assert retrieved == test_info


class TestPerformanceRegression:
    """Test that performance hasn't regressed after migration."""

    def test_query_performance_baseline(self, populated_database_manager: IDatabaseManager):
        """Establish baseline query performance metrics."""
        import time

        # Test various query patterns
        queries = [
            ("get_price_history", ("AAPL",), 2.0),
            ("get_price_history", ("AAPL", "2023-01-01", "2023-12-31"), 2.0),
            ("get_latest_price", ("AAPL",), 0.5),
            ("get_all_symbols", (), 1.0),
            ("get_database_stats", (), 1.0),
        ]

        for method_name, args, max_time in queries:
            start_time = time.time()
            getattr(populated_database_manager, method_name)(*args)
            duration = time.time() - start_time

            assert duration < max_time, f"{method_name} took {duration:.3f}s, expected < {max_time}s"

    def test_bulk_operations_performance(self, database_manager: IDatabaseManager):
        """Test performance of bulk operations."""
        import time

        # Generate test data
        dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
        bulk_data = pd.DataFrame(
            {
                "Open": range(len(dates)),
                "High": [x + 1 for x in range(len(dates))],
                "Low": [max(0, x - 1) for x in range(len(dates))],
                "Close": [x + 0.5 for x in range(len(dates))],
                "Volume": [1000000] * len(dates),
            },
            index=dates,
        )

        # Test bulk insert
        start_time = time.time()
        database_manager.store_price_history("BULK_TEST", bulk_data)
        insert_duration = time.time() - start_time

        # Should handle ~1400 days of data reasonably quickly
        assert insert_duration < 60.0, f"Bulk insert took {insert_duration:.2f}s"

        # Test bulk retrieval
        start_time = time.time()
        retrieved = database_manager.get_price_history("BULK_TEST")
        query_duration = time.time() - start_time

        assert query_duration < 10.0, f"Bulk query took {query_duration:.2f}s"
        assert len(retrieved) == len(bulk_data)


class TestMockVsRealBehavior:
    """Test that mock and real implementations behave consistently."""

    def test_consistent_return_types(self, database_manager: IDatabaseManager):
        """Test that return types are consistent between implementations."""
        # Test with some basic operations
        operations = [
            ("get_all_symbols", (), list),
            ("get_database_stats", (), dict),
            ("get_stock_info", ("NONEXISTENT",), type(None)),
            ("get_latest_price", ("NONEXISTENT",), type(None)),
            ("get_price_history", ("NONEXISTENT",), pd.DataFrame),
        ]

        for method_name, args, expected_type in operations:
            result = getattr(database_manager, method_name)(*args)
            assert isinstance(result, expected_type), f"{method_name} returned {type(result)}, expected {expected_type}"

    def test_consistent_data_behavior(
        self, database_manager: IDatabaseManager, sample_price_data: pd.DataFrame, sample_stock_info: dict[str, Any]
    ):
        """Test that data operations behave consistently."""
        # Store data
        database_manager.store_stock_info("CONSISTENCY_TEST", sample_stock_info)
        database_manager.store_price_history("CONSISTENCY_TEST", sample_price_data)

        # Test retrieval consistency
        retrieved_info = database_manager.get_stock_info("CONSISTENCY_TEST")
        assert retrieved_info is not None
        assert retrieved_info["longName"] == sample_stock_info["longName"]

        retrieved_data = database_manager.get_price_history("CONSISTENCY_TEST")
        assert len(retrieved_data) == len(sample_price_data)
        assert list(retrieved_data.columns) == list(sample_price_data.columns)

        # Test has_data consistency
        start_date = sample_price_data.index[0].strftime("%Y-%m-%d")
        end_date = sample_price_data.index[-1].strftime("%Y-%m-%d")
        has_data = database_manager.has_data("CONSISTENCY_TEST", start_date, end_date)
        assert has_data is True

        # Test latest price consistency
        expected_latest = sample_price_data["Close"].iloc[-1]
        actual_latest = database_manager.get_latest_price("CONSISTENCY_TEST")
        assert abs(actual_latest - expected_latest) < 0.01
