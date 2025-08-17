"""
Comprehensive tests for DatabaseManager basic analytics methods.

This module tests the analytics methods in DatabaseManager (lines 1087-1316):
- get_moving_averages (lines 1087-1130)
- get_bollinger_bands (lines 1152-1211)
- get_rsi (lines 1231-1316)

Focus on method execution and parameter validation. Tests exercise actual code paths
by mocking the session context manager appropriately.
"""

from contextlib import contextmanager
from unittest.mock import Mock, patch

import pandas as pd
import pytest


class TestMovingAveragesExecution:
    """Test get_moving_averages method execution and SQL generation."""

    def test_moving_averages_with_default_parameters(self, database_manager):
        """Test moving averages executes successfully with default parameters."""
        # Mock the get_session context manager
        mock_session = Mock()
        mock_session.exec.return_value.fetchall.return_value = [
            ("2023-01-01", "AAPL", 150.0, 148.5, 147.2, 146.8),
            ("2023-01-02", "AAPL", 151.0, 149.1, 147.8, 147.2),
        ]

        @contextmanager
        def mock_get_session():
            yield mock_session

        with patch.object(database_manager, "get_session", mock_get_session):
            result = database_manager.get_moving_averages("AAPL")

            # Verify method executed and returned DataFrame
            assert isinstance(result, pd.DataFrame)
            assert mock_session.exec.called

            # Verify SQL query was constructed
            call_args = mock_session.exec.call_args
            query_text = str(call_args[0][0])

            # Validate SQL structure for moving averages
            assert "SELECT" in query_text
            assert "price_history" in query_text
            assert "symbol = :symbol" in query_text
            assert "AVG(" in query_text
            assert "OVER" in query_text

    def test_moving_averages_with_custom_parameters(self, database_manager):
        """Test moving averages with custom periods and date range."""
        mock_session = Mock()
        mock_session.exec.return_value.fetchall.return_value = []

        @contextmanager
        def mock_get_session():
            yield mock_session

        with patch.object(database_manager, "get_session", mock_get_session):
            # Test with custom parameters
            result = database_manager.get_moving_averages(
                symbol="AAPL", periods=[5, 10, 20], start_date="2023-01-01", end_date="2023-12-31"
            )

            # Verify method executed
            assert isinstance(result, pd.DataFrame)
            assert mock_session.exec.called

            # Verify parameters were passed
            call_args = mock_session.exec.call_args
            # Parameters are passed as second positional argument: session.exec(text(query), params)
            params = call_args[0][1] if len(call_args[0]) > 1 else {}

            # Should include symbol and date parameters
            assert params.get("symbol") == "AAPL"

    def test_moving_averages_empty_results(self, database_manager):
        """Test moving averages handles empty results gracefully."""
        mock_session = Mock()
        mock_session.exec.return_value.fetchall.return_value = []

        @contextmanager
        def mock_get_session():
            yield mock_session

        with patch.object(database_manager, "get_session", mock_get_session):
            result = database_manager.get_moving_averages("NONEXISTENT")

            # Should return empty DataFrame
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0

    def test_moving_averages_parameter_validation(self, database_manager):
        """Test moving averages parameter handling."""
        mock_session = Mock()
        mock_session.exec.return_value.fetchall.return_value = []

        @contextmanager
        def mock_get_session():
            yield mock_session

        with patch.object(database_manager, "get_session", mock_get_session):
            # Test various parameter combinations
            database_manager.get_moving_averages("AAPL", periods=[10, 20])
            database_manager.get_moving_averages("AAPL", start_date="2023-01-01")
            database_manager.get_moving_averages("AAPL", end_date="2023-12-31")

            # All calls should execute without error
            assert mock_session.exec.call_count == 3


class TestBollingerBandsExecution:
    """Test get_bollinger_bands method execution."""

    def test_bollinger_bands_with_default_parameters(self, database_manager):
        """Test Bollinger Bands executes successfully with default parameters."""
        mock_session = Mock()
        mock_session.exec.return_value.fetchall.return_value = [
            ("2023-01-01", "AAPL", 150.0, 148.5, 152.1, 144.9, 1.8),
        ]

        @contextmanager
        def mock_get_session():
            yield mock_session

        with patch.object(database_manager, "get_session", mock_get_session):
            result = database_manager.get_bollinger_bands("AAPL")

            # Verify method executed
            assert isinstance(result, pd.DataFrame)
            assert mock_session.exec.called

            # Verify SQL query for Bollinger Bands
            call_args = mock_session.exec.call_args
            query_text = str(call_args[0][0])

            assert "SELECT" in query_text
            assert "price_history" in query_text
            assert "AVG(" in query_text
            assert "STDDEV(" in query_text or "STD(" in query_text

    def test_bollinger_bands_with_custom_parameters(self, database_manager):
        """Test Bollinger Bands with custom period and standard deviation."""
        mock_session = Mock()
        mock_session.exec.return_value.fetchall.return_value = []

        @contextmanager
        def mock_get_session():
            yield mock_session

        with patch.object(database_manager, "get_session", mock_get_session):
            result = database_manager.get_bollinger_bands(
                symbol="AAPL", period=10, std_dev=1.5, start_date="2023-06-01"
            )

            # Verify method executed
            assert isinstance(result, pd.DataFrame)
            assert mock_session.exec.called

    def test_bollinger_bands_date_filtering(self, database_manager):
        """Test Bollinger Bands with date range filtering."""
        mock_session = Mock()
        mock_session.exec.return_value.fetchall.return_value = []

        @contextmanager
        def mock_get_session():
            yield mock_session

        with patch.object(database_manager, "get_session", mock_get_session):
            result = database_manager.get_bollinger_bands("AAPL", start_date="2023-01-01", end_date="2023-12-31")

            assert isinstance(result, pd.DataFrame)
            assert mock_session.exec.called

    def test_bollinger_bands_empty_results(self, database_manager):
        """Test Bollinger Bands handles empty results."""
        mock_session = Mock()
        mock_session.exec.return_value.fetchall.return_value = []

        @contextmanager
        def mock_get_session():
            yield mock_session

        with patch.object(database_manager, "get_session", mock_get_session):
            result = database_manager.get_bollinger_bands("INVALID")

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0


class TestRSIExecution:
    """Test get_rsi method execution."""

    def test_rsi_with_default_parameters(self, database_manager):
        """Test RSI executes successfully with default parameters."""
        mock_session = Mock()
        mock_session.exec.return_value.fetchall.return_value = [
            ("2023-01-01", "AAPL", 150.0, 0.5, 0.3, 0.2, 65.2),
        ]

        @contextmanager
        def mock_get_session():
            yield mock_session

        with patch.object(database_manager, "get_session", mock_get_session):
            result = database_manager.get_rsi("AAPL")

            # Verify method executed
            assert isinstance(result, pd.DataFrame)
            assert mock_session.exec.called

            # Verify SQL query structure
            call_args = mock_session.exec.call_args
            query_text = str(call_args[0][0])

            # RSI queries are complex, just verify basic structure
            assert "SELECT" in query_text
            assert "price_history" in query_text

    def test_rsi_with_custom_period(self, database_manager):
        """Test RSI with custom period."""
        mock_session = Mock()
        mock_session.exec.return_value.fetchall.return_value = []

        @contextmanager
        def mock_get_session():
            yield mock_session

        with patch.object(database_manager, "get_session", mock_get_session):
            result = database_manager.get_rsi("AAPL", period=21)

            assert isinstance(result, pd.DataFrame)
            assert mock_session.exec.called

    def test_rsi_with_date_range(self, database_manager):
        """Test RSI with date range parameters."""
        mock_session = Mock()
        mock_session.exec.return_value.fetchall.return_value = []

        @contextmanager
        def mock_get_session():
            yield mock_session

        with patch.object(database_manager, "get_session", mock_get_session):
            result = database_manager.get_rsi("AAPL", start_date="2023-03-01", end_date="2023-09-30")

            assert isinstance(result, pd.DataFrame)
            assert mock_session.exec.called

    def test_rsi_empty_results(self, database_manager):
        """Test RSI handles empty results."""
        mock_session = Mock()
        mock_session.exec.return_value.fetchall.return_value = []

        @contextmanager
        def mock_get_session():
            yield mock_session

        with patch.object(database_manager, "get_session", mock_get_session):
            result = database_manager.get_rsi("INVALID")

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0

    def test_rsi_complex_sql_execution(self, database_manager):
        """Test RSI executes complex SQL with CTEs."""
        mock_session = Mock()
        # Mock a more realistic RSI result
        mock_session.exec.return_value.fetchall.return_value = [
            ("2023-01-01", "AAPL", 150.0, 1.0, 1.0, 0.0, None),  # First day
            ("2023-01-02", "AAPL", 152.0, 2.0, 2.0, 0.0, None),  # Gain day
            ("2023-01-03", "AAPL", 149.0, -3.0, 0.0, 3.0, 40.0),  # Loss day with RSI
        ]

        @contextmanager
        def mock_get_session():
            yield mock_session

        with patch.object(database_manager, "get_session", mock_get_session):
            result = database_manager.get_rsi("AAPL", period=14)

            assert isinstance(result, pd.DataFrame)
            assert mock_session.exec.called


class TestAnalyticsErrorHandling:
    """Test error handling across analytics methods."""

    def test_analytics_database_error_handling(self, database_manager):
        """Test analytics methods handle database errors gracefully."""
        mock_session = Mock()
        mock_session.exec.side_effect = Exception("Database connection failed")

        @contextmanager
        def mock_get_session():
            yield mock_session

        with patch.object(database_manager, "get_session", mock_get_session):
            # Methods should propagate database errors
            with pytest.raises((Exception, RuntimeError, ConnectionError)):
                database_manager.get_moving_averages("AAPL")

            with pytest.raises((Exception, RuntimeError, ConnectionError)):
                database_manager.get_bollinger_bands("AAPL")

            with pytest.raises((Exception, RuntimeError, ConnectionError)):
                database_manager.get_rsi("AAPL")

    def test_analytics_invalid_parameters(self, database_manager):
        """Test analytics methods parameter validation."""
        mock_session = Mock()
        mock_session.exec.return_value.fetchall.return_value = []

        @contextmanager
        def mock_get_session():
            yield mock_session

        with patch.object(database_manager, "get_session", mock_get_session):
            # Test empty symbol
            database_manager.get_moving_averages("")
            database_manager.get_bollinger_bands("")
            database_manager.get_rsi("")

            # Methods should handle empty symbols gracefully
            assert mock_session.exec.call_count == 3

    def test_analytics_date_parameter_handling(self, database_manager):
        """Test analytics methods handle various date formats."""
        mock_session = Mock()
        mock_session.exec.return_value.fetchall.return_value = []

        @contextmanager
        def mock_get_session():
            yield mock_session

        with patch.object(database_manager, "get_session", mock_get_session):
            # Test various date formats and edge cases
            database_manager.get_moving_averages("AAPL", start_date="2023-01-01")
            database_manager.get_bollinger_bands("AAPL", end_date="2023-12-31")
            database_manager.get_rsi("AAPL", start_date="2023-06-01", end_date="2023-06-30")

            # All should execute without error
            assert mock_session.exec.call_count == 3


class TestAnalyticsIntegration:
    """Integration tests for analytics methods working together."""

    def test_analytics_method_chain_execution(self, database_manager):
        """Test multiple analytics methods can be called in sequence."""
        mock_session = Mock()
        mock_session.exec.return_value.fetchall.return_value = [
            ("2023-01-01", "AAPL", 150.0, 148.5, 147.2),
        ]

        @contextmanager
        def mock_get_session():
            yield mock_session

        with patch.object(database_manager, "get_session", mock_get_session):
            # Call all three analytics methods in sequence
            ma_result = database_manager.get_moving_averages("AAPL")
            bb_result = database_manager.get_bollinger_bands("AAPL")
            rsi_result = database_manager.get_rsi("AAPL")

            # All should return DataFrames
            assert isinstance(ma_result, pd.DataFrame)
            assert isinstance(bb_result, pd.DataFrame)
            assert isinstance(rsi_result, pd.DataFrame)

            # Verify all methods were called
            assert mock_session.exec.call_count == 3

    def test_analytics_consistent_parameter_handling(self, database_manager):
        """Test that all analytics methods handle parameters consistently."""
        mock_session = Mock()
        mock_session.exec.return_value.fetchall.return_value = []

        @contextmanager
        def mock_get_session():
            yield mock_session

        with patch.object(database_manager, "get_session", mock_get_session):
            # Test consistent date parameter handling across methods
            start_date = "2023-01-01"
            end_date = "2023-12-31"
            symbol = "AAPL"

            database_manager.get_moving_averages(symbol, start_date=start_date, end_date=end_date)
            database_manager.get_bollinger_bands(symbol, start_date=start_date, end_date=end_date)
            database_manager.get_rsi(symbol, start_date=start_date, end_date=end_date)

            # All should execute successfully
            assert mock_session.exec.call_count == 3
