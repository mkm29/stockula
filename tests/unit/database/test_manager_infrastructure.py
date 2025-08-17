"""Infrastructure behavior testing for DatabaseManager.

This module tests the infrastructure behavior patterns of DatabaseManager
using the proven mock-based approaches for reliable unit testing. Tests focus on:
- Backend type and property behavior validation
- Connection testing behavior
- Session management behavior
- Error handling patterns
- Resource cleanup behavior

All tests use the working mock patterns to avoid real database connections.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from stockula.config.models import TimescaleDBConfig
from stockula.database.manager import DatabaseManager
from stockula.interfaces import IDatabaseManager


class TestDatabaseManagerProperties:
    """Test DatabaseManager property behavior using working mock patterns."""

    def test_backend_type_detection(self, database_manager: IDatabaseManager):
        """Test that backend_type property works correctly."""
        backend_type = database_manager.backend_type
        # Mock returns "timescaledb_mock", real TimescaleDB returns "timescaledb", SQLite returns "sqlite"
        assert backend_type in ["timescaledb", "timescaledb_mock", "sqlite"]
        assert isinstance(backend_type, str)

    def test_timescaledb_property_behavior(self, database_manager: IDatabaseManager):
        """Test is_timescaledb property behavior."""
        is_timescale = database_manager.is_timescaledb
        assert isinstance(is_timescale, bool)

        # Mock and real TimescaleDB both return True
        backend_type = database_manager.backend_type
        if "timescale" in backend_type.lower():
            assert is_timescale is True
        else:
            assert is_timescale is False

    def test_connection_testing_behavior(self, database_manager: IDatabaseManager):
        """Test connection testing functionality."""
        # Should be callable without errors
        connection_result = database_manager.test_connection()
        # Should return something indicating connection status
        assert connection_result is not None

    def test_session_management_behavior(self, database_manager: IDatabaseManager):
        """Test session management behavior."""
        # Should be able to get a session without errors
        session = database_manager.get_session()
        assert session is not None

    def test_cleanup_behavior(self, database_manager: IDatabaseManager):
        """Test cleanup and resource management behavior."""
        # Should be callable without errors
        database_manager.close()
        # Multiple calls should be safe
        database_manager.close()


class TestLegacyDatabaseManager:
    """Test legacy SQLite DatabaseManager behavior."""

    def test_legacy_initialization_behavior(self):
        """Test legacy SQLite initialization behavior."""
        legacy_path = "test_stockula.db"

        # This creates a legacy SQLite manager
        db = DatabaseManager(legacy_path)

        # Verify legacy mode properties
        assert hasattr(db, "_legacy_mode") and db._legacy_mode is True
        assert hasattr(db, "_legacy_path") and db._legacy_path == legacy_path
        assert db.config == legacy_path
        assert db.enable_async is False
        assert db.backend_type == "sqlite"
        assert db.is_timescaledb is False

    def test_none_config_error_behavior(self):
        """Test error behavior with None configuration."""
        with pytest.raises(ValueError, match="TimescaleDB configuration is required"):
            DatabaseManager(None)


class TestConnectionBehavior:
    """Test connection behavior patterns with proper mocking."""

    @patch.object(DatabaseManager, "_setup_engines")
    @patch.object(DatabaseManager, "_run_migrations")
    @patch.object(DatabaseManager, "_create_tables")
    @patch.object(DatabaseManager, "_setup_timescale_features")
    def test_initialization_call_sequence(
        self,
        mock_setup_features,
        mock_create_tables,
        mock_run_migrations,
        mock_setup_engines,
        timescale_config: TimescaleDBConfig,
    ):
        """Test that initialization follows expected call sequence."""
        # Should complete initialization without error
        db = DatabaseManager(timescale_config, enable_async=True)

        # Verify config is stored
        assert db.config == timescale_config
        assert db.enable_async is True

        # Verify methods are called
        mock_setup_engines.assert_called_once()
        mock_run_migrations.assert_called_once()
        mock_create_tables.assert_called_once()
        mock_setup_features.assert_called_once()

    @patch.object(DatabaseManager, "_setup_engines")
    @patch.object(DatabaseManager, "_run_migrations")
    @patch.object(DatabaseManager, "_create_tables")
    @patch.object(DatabaseManager, "_setup_timescale_features")
    def test_async_disabled_behavior(
        self,
        mock_setup_features,
        mock_create_tables,
        mock_run_migrations,
        mock_setup_engines,
        timescale_config: TimescaleDBConfig,
    ):
        """Test behavior when async is disabled."""
        db = DatabaseManager(timescale_config, enable_async=False)
        assert db.enable_async is False


class TestErrorHandlingBehavior:
    """Test error handling behavior patterns."""

    def test_graceful_error_handling_in_operations(self, error_prone_database_manager):
        """Test graceful error handling in database operations."""
        # Should handle errors appropriately
        with pytest.raises(ValueError, match="Simulated database error"):
            error_prone_database_manager.get_price_history("ERROR")

        # Should continue working for valid operations
        result = error_prone_database_manager.get_price_history("AAPL")
        assert result is not None

    @patch.object(DatabaseManager, "_setup_engines")
    @patch.object(DatabaseManager, "_run_migrations")
    @patch.object(DatabaseManager, "_create_tables")
    @patch.object(DatabaseManager, "_setup_timescale_features")
    def test_initialization_error_propagation(
        self,
        mock_setup_features,
        mock_create_tables,
        mock_run_migrations,
        mock_setup_engines,
        timescale_config: TimescaleDBConfig,
    ):
        """Test that initialization errors are properly propagated."""
        # Setup mock to raise error
        mock_setup_engines.side_effect = Exception("Setup failed")

        # Should propagate the error appropriately
        with pytest.raises(Exception, match="Setup failed"):
            DatabaseManager(timescale_config)


class TestMockDatabaseManagerBehavior:
    """Test specific mock database manager behavior."""

    def test_mock_call_tracking(self, mock_database_manager):
        """Test that mock tracks method calls correctly."""
        # Make some method calls
        mock_database_manager.get_price_history("AAPL")
        mock_database_manager.store_stock_info("AAPL", {"name": "Apple Inc."})

        # Verify calls are tracked
        assert hasattr(mock_database_manager, "call_history")
        assert len(mock_database_manager.call_history) >= 2

    def test_mock_data_persistence(self, mock_database_manager):
        """Test that mock properly persists data across calls."""
        # Store some data
        test_info = {"longName": "Apple Inc.", "sector": "Technology"}
        mock_database_manager.store_stock_info("AAPL", test_info)

        # Retrieve and verify
        retrieved_info = mock_database_manager.get_stock_info("AAPL")
        assert retrieved_info == test_info

    def test_mock_realistic_behavior(self, mock_database_manager):
        """Test that mock provides realistic behavior."""
        # Test with non-existent symbol
        result = mock_database_manager.get_price_history("NONEXISTENT")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

        # Test has_data functionality
        has_data = mock_database_manager.has_data("NONEXISTENT", "2023-01-01", "2023-12-31")
        assert has_data is False


class TestResourceManagement:
    """Test resource management behavior."""

    def test_explicit_close_behavior(self, mock_database_manager):
        """Test explicit close behavior."""
        # Should be callable multiple times safely
        mock_database_manager.close()
        mock_database_manager.close()

        # Should not raise errors
        assert True  # Test passes if no exceptions

    def test_cleanup_safety_patterns(self, mock_database_manager):
        """Test that cleanup operations are safe."""
        # Should handle missing resources gracefully
        mock_database_manager.close()

        # Should still be able to call methods safely after close
        backend_type = mock_database_manager.backend_type
        assert isinstance(backend_type, str)


class TestConfigurationBehavior:
    """Test configuration behavior patterns."""

    def test_timescale_config_storage(self, timescale_config: TimescaleDBConfig, database_manager: IDatabaseManager):
        """Test that TimescaleDB configuration is properly stored."""
        # The mock should store the config appropriately
        assert hasattr(database_manager, "config")

        # For mock, config should be the TimescaleDBConfig object
        if hasattr(database_manager.config, "host"):
            assert database_manager.config.host == timescale_config.host
            assert database_manager.config.database == timescale_config.database

    def test_backend_consistency(self, database_manager: IDatabaseManager):
        """Test that backend type and TimescaleDB property are consistent."""
        backend_type = database_manager.backend_type
        is_timescale = database_manager.is_timescaledb

        # Should be consistent
        if "timescale" in backend_type.lower():
            assert is_timescale is True
        else:
            assert is_timescale is False
