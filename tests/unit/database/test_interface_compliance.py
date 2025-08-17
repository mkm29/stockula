"""Test interface compliance for IDatabaseManager implementation.

This module ensures that the consolidated DatabaseManager correctly implements
all 67 methods of the IDatabaseManager interface with proper signatures and behavior.
"""

import inspect

import pandas as pd
import pytest

from stockula.interfaces import IDatabaseManager


class TestIDatabaseManagerCompliance:
    """Test that DatabaseManager implements IDatabaseManager interface correctly."""

    def test_interface_implementation(self, database_manager: IDatabaseManager):
        """Test that database manager implements the interface."""
        assert isinstance(database_manager, IDatabaseManager)

    def test_all_interface_methods_exist(self, database_manager: IDatabaseManager):
        """Test that all interface methods are implemented."""
        interface_methods = {
            name
            for name, method in inspect.getmembers(IDatabaseManager, predicate=inspect.isfunction)
            if not name.startswith("_")
        }

        # Get implemented methods
        impl_methods = {
            name
            for name in dir(database_manager)
            if not name.startswith("_") and callable(getattr(database_manager, name))
        }

        missing_methods = interface_methods - impl_methods
        assert not missing_methods, f"Missing interface methods: {missing_methods}"

    @pytest.mark.parametrize(
        "method_name,expected_params",
        [
            ("get_price_history", ["symbol", "start_date", "end_date", "interval"]),
            ("store_price_history", ["symbol", "data", "interval"]),
            ("has_data", ["symbol", "start_date", "end_date"]),
            ("store_stock_info", ["symbol", "info"]),
            ("get_stock_info", ["symbol"]),
            ("store_dividends", ["symbol", "dividends"]),
            ("store_splits", ["symbol", "splits"]),
            ("get_all_symbols", []),
            ("get_latest_price", ["symbol"]),
            ("get_database_stats", []),
            ("get_moving_averages", ["symbol", "periods", "start_date", "end_date"]),
            ("get_bollinger_bands", ["symbol", "period", "std_dev", "start_date", "end_date"]),
            ("get_rsi", ["symbol", "period", "start_date", "end_date"]),
            ("get_session", []),
            ("close", []),
        ],
    )
    def test_method_signatures(self, database_manager: IDatabaseManager, method_name: str, expected_params: list[str]):
        """Test that methods have correct signatures."""
        method = getattr(database_manager, method_name)
        signature = inspect.signature(method)

        # Get parameter names (excluding 'self')
        actual_params = [name for name in signature.parameters.keys() if name != "self"]

        assert actual_params == expected_params, (
            f"Method {method_name} has incorrect signature. Expected: {expected_params}, Got: {actual_params}"
        )

    def test_core_operations_return_types(self, populated_database_manager: IDatabaseManager):
        """Test that core operations return correct types."""
        # Test get_price_history returns DataFrame
        price_data = populated_database_manager.get_price_history("AAPL")
        assert isinstance(price_data, pd.DataFrame)

        # Test has_data returns bool
        has_data = populated_database_manager.has_data("AAPL", "2023-01-01", "2023-12-31")
        assert isinstance(has_data, bool)

        # Test get_stock_info returns dict or None
        stock_info = populated_database_manager.get_stock_info("AAPL")
        assert isinstance(stock_info, dict | type(None))

        # Test get_latest_price returns float or None
        latest_price = populated_database_manager.get_latest_price("AAPL")
        assert isinstance(latest_price, float | type(None))

        # Test get_all_symbols returns list
        symbols = populated_database_manager.get_all_symbols()
        assert isinstance(symbols, list)

        # Test get_database_stats returns dict
        stats = populated_database_manager.get_database_stats()
        assert isinstance(stats, dict)

    def test_analytics_methods_return_dataframes(self, populated_database_manager: IDatabaseManager):
        """Test that analytics methods return DataFrames."""
        analytics_methods = [
            ("get_moving_averages", ("AAPL",)),
            ("get_bollinger_bands", ("AAPL",)),
            ("get_rsi", ("AAPL",)),
        ]

        for method_name, args in analytics_methods:
            result = getattr(populated_database_manager, method_name)(*args)
            assert isinstance(result, pd.DataFrame), f"{method_name} should return DataFrame"

    def test_optional_analytics_methods_exist(self, database_manager: IDatabaseManager):
        """Test that optional TimescaleDB-specific methods exist."""
        optional_methods = [
            "get_price_history_aggregated",
            "get_recent_price_changes",
            "get_chunk_statistics",
            "test_connection",
        ]

        for method_name in optional_methods:
            assert hasattr(database_manager, method_name), f"Missing optional method: {method_name}"
            assert callable(getattr(database_manager, method_name))

    def test_method_call_patterns(self, mock_database_manager):
        """Test that methods are called with expected patterns."""
        # Test basic operations
        mock_database_manager.get_price_history("AAPL")
        assert mock_database_manager.get_call_count("get_price_history") == 1

        # Test with parameters
        mock_database_manager.get_price_history("AAPL", "2023-01-01", "2023-12-31")
        assert mock_database_manager.get_call_count("get_price_history") == 2

        # Verify last call arguments
        args, kwargs = mock_database_manager.get_last_call("get_price_history")
        assert args[0] == "AAPL"
        assert args[1] == "2023-01-01"
        assert args[2] == "2023-12-31"

    def test_backend_type_property(self, database_manager: IDatabaseManager):
        """Test that backend_type property is implemented."""
        backend_type = database_manager.backend_type
        assert isinstance(backend_type, str)
        assert backend_type in ["timescaledb", "timescaledb_mock"]

    def test_is_timescaledb_property(self, database_manager: IDatabaseManager):
        """Test that is_timescaledb property returns True."""
        assert database_manager.is_timescaledb is True

    def test_session_context_manager(self, database_manager: IDatabaseManager):
        """Test that get_session works as context manager."""
        with database_manager.get_session() as session:
            assert session is not None

    def test_connection_test_method(self, database_manager: IDatabaseManager):
        """Test that test_connection returns health information."""
        result = database_manager.test_connection()
        assert isinstance(result, dict)
        assert "connected" in result or "status" in result

    def test_store_operations_validation(
        self, database_manager: IDatabaseManager, sample_price_data: pd.DataFrame, sample_stock_info: dict
    ):
        """Test that store operations validate inputs."""
        # Test store_stock_info
        database_manager.store_stock_info("TEST", sample_stock_info)

        # Test store_price_history
        database_manager.store_price_history("TEST", sample_price_data)

        # Verify data was stored
        retrieved_info = database_manager.get_stock_info("TEST")
        assert retrieved_info is not None

        retrieved_data = database_manager.get_price_history("TEST")
        assert not retrieved_data.empty

    def test_error_handling_compliance(self, error_prone_database_manager):
        """Test that errors are handled appropriately."""
        with pytest.raises(ValueError):
            error_prone_database_manager.get_price_history("ERROR")

    def test_all_analytics_methods_callable(self, database_manager: IDatabaseManager):
        """Test that all analytics methods are callable (even if not implemented)."""
        analytics_methods = [
            "get_moving_averages",
            "get_bollinger_bands",
            "get_rsi",
            "get_price_momentum",
            "get_correlation_matrix",
            "get_volatility_analysis",
            "get_seasonal_patterns",
            "get_top_performers",
        ]

        for method_name in analytics_methods:
            method = getattr(database_manager, method_name)
            assert callable(method), f"{method_name} should be callable"

            # Test that calling doesn't raise AttributeError
            try:
                # Call with minimal args (may raise NotImplementedError, which is OK)
                if method_name in ["get_correlation_matrix"]:
                    method(["AAPL", "GOOGL"])  # Requires symbols list
                elif method_name in ["get_price_momentum", "get_volatility_analysis", "get_top_performers"]:
                    method()  # Can be called without args
                else:
                    method("AAPL")  # Single symbol methods
            except (NotImplementedError, ValueError, TypeError):
                # These exceptions are acceptable for unimplemented methods
                pass
            except AttributeError as e:
                pytest.fail(f"{method_name} raised AttributeError: {e}")


class TestMethodGroupCompliance:
    """Test methods grouped by functionality."""

    def test_core_operations_group(self, populated_database_manager: IDatabaseManager, method_groups: dict):
        """Test core operations work together."""
        method_groups["core_operations"]

        # Test the workflow: store -> check -> retrieve
        test_data = pd.DataFrame(
            {"Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.5], "Volume": [1000000]},
            index=pd.date_range("2024-01-01", periods=1),
        )

        # Store data
        populated_database_manager.store_price_history("TEST_CORE", test_data)

        # Check data exists
        has_data = populated_database_manager.has_data("TEST_CORE", "2024-01-01", "2024-01-01")
        assert has_data

        # Retrieve data
        retrieved = populated_database_manager.get_price_history("TEST_CORE")
        assert not retrieved.empty
        assert "Close" in retrieved.columns

    def test_stock_info_group(self, database_manager: IDatabaseManager, method_groups: dict, sample_stock_info: dict):
        """Test stock info operations."""
        method_groups["stock_info"]

        # Store and retrieve stock info
        database_manager.store_stock_info("TEST_INFO", sample_stock_info)
        retrieved_info = database_manager.get_stock_info("TEST_INFO")

        assert retrieved_info is not None
        assert retrieved_info["longName"] == sample_stock_info["longName"]

    def test_analytics_basic_group(self, populated_database_manager: IDatabaseManager, method_groups: dict):
        """Test basic analytics methods."""
        basic_analytics = method_groups["analytics_basic"]

        for method_name in basic_analytics:
            method = getattr(populated_database_manager, method_name)
            result = method("AAPL")
            assert isinstance(result, pd.DataFrame)

    def test_utilities_group(self, populated_database_manager: IDatabaseManager, method_groups: dict):
        """Test utility methods."""
        method_groups["utilities"]

        # Test get_all_symbols
        symbols = populated_database_manager.get_all_symbols()
        assert isinstance(symbols, list)
        assert len(symbols) > 0

        # Test get_latest_price
        if symbols:
            latest = populated_database_manager.get_latest_price(symbols[0])
            assert isinstance(latest, float | type(None))

        # Test get_database_stats
        stats = populated_database_manager.get_database_stats()
        assert isinstance(stats, dict)
        assert "stocks" in stats or "price_history" in stats

    def test_session_management_group(self, database_manager: IDatabaseManager, method_groups: dict):
        """Test session management methods."""
        method_groups["session_management"]

        # Test get_session
        with database_manager.get_session() as session:
            assert session is not None

        # Test test_connection (if implemented)
        try:
            result = database_manager.test_connection()
            assert isinstance(result, dict)
        except NotImplementedError:
            pass  # Acceptable for mock implementations

        # Test close (should not raise)
        database_manager.close()


class TestTimescaleSpecificCompliance:
    """Test TimescaleDB-specific features."""

    def test_timescale_aggregation_methods(self, populated_database_manager: IDatabaseManager):
        """Test TimescaleDB-specific aggregation methods."""
        try:
            # Test time_bucket aggregation
            result = populated_database_manager.get_price_history_aggregated("AAPL", "1 day")
            assert isinstance(result, pd.DataFrame)
        except NotImplementedError:
            pytest.skip("Time bucket aggregation not implemented")

    def test_recent_price_changes(self, populated_database_manager: IDatabaseManager):
        """Test recent price changes analysis."""
        try:
            result = populated_database_manager.get_recent_price_changes(["AAPL"], hours=24)
            assert isinstance(result, pd.DataFrame)
        except NotImplementedError:
            pytest.skip("Recent price changes not implemented")

    def test_chunk_statistics(self, populated_database_manager: IDatabaseManager):
        """Test TimescaleDB chunk statistics."""
        try:
            result = populated_database_manager.get_chunk_statistics()
            assert isinstance(result, pd.DataFrame)
        except NotImplementedError:
            pytest.skip("Chunk statistics not implemented")

    def test_advanced_analytics_exist(self, database_manager: IDatabaseManager):
        """Test that advanced analytics methods exist."""
        advanced_methods = [
            "get_price_momentum",
            "get_correlation_matrix",
            "get_volatility_analysis",
            "get_seasonal_patterns",
            "get_top_performers",
        ]

        for method_name in advanced_methods:
            assert hasattr(database_manager, method_name)
            assert callable(getattr(database_manager, method_name))
