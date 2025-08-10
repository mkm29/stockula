"""Tests for custom exceptions."""

import pytest

from stockula.config.exceptions import (
    APIException,
    ConfigurationException,
    DatabaseException,
    DataFetchException,
    NetworkException,
    StockulaException,
    ValidationException,
)


class TestStockulaException:
    """Test the base StockulaException."""

    def test_base_exception(self):
        """Test creating a base exception."""
        exc = StockulaException("Test error")
        assert str(exc) == "Test error"
        assert isinstance(exc, Exception)

    def test_base_exception_inheritance(self):
        """Test that base exception inherits from Exception."""
        exc = StockulaException()
        assert isinstance(exc, Exception)


class TestDataFetchException:
    """Test DataFetchException."""

    def test_with_symbol_and_message(self):
        """Test exception with symbol and custom message."""
        exc = DataFetchException(symbol="AAPL", message="Custom error message")
        assert str(exc) == "Custom error message"
        assert exc.symbol == "AAPL"
        assert exc.cause is None

    def test_with_symbol_and_cause(self):
        """Test exception with symbol and underlying cause."""
        cause = ValueError("Underlying error")
        exc = DataFetchException(symbol="AAPL", cause=cause)
        assert str(exc) == "Failed to fetch data for AAPL: Underlying error"
        assert exc.symbol == "AAPL"
        assert exc.cause == cause

    def test_with_symbol_only(self):
        """Test exception with only symbol."""
        exc = DataFetchException(symbol="AAPL")
        assert str(exc) == "Failed to fetch data for AAPL"
        assert exc.symbol == "AAPL"
        assert exc.cause is None

    def test_with_cause_only(self):
        """Test exception with only cause."""
        cause = RuntimeError("Network timeout")
        exc = DataFetchException(cause=cause)
        assert str(exc) == "Data fetch failed: Network timeout"
        assert exc.symbol is None
        assert exc.cause == cause

    def test_with_no_arguments(self):
        """Test exception with no arguments."""
        exc = DataFetchException()
        assert str(exc) == "Data fetch failed"
        assert exc.symbol is None
        assert exc.cause is None

    def test_with_all_arguments(self):
        """Test exception with all arguments."""
        cause = Exception("Root cause")
        exc = DataFetchException(symbol="AAPL", message="Custom message", cause=cause)
        assert str(exc) == "Custom message"
        assert exc.symbol == "AAPL"
        assert exc.cause == cause

    def test_inheritance(self):
        """Test that DataFetchException inherits from StockulaException."""
        exc = DataFetchException()
        assert isinstance(exc, StockulaException)
        assert isinstance(exc, Exception)


class TestNetworkException:
    """Test NetworkException."""

    def test_with_symbol_and_message(self):
        """Test exception with symbol and custom message."""
        exc = NetworkException(symbol="AAPL", message="Connection refused")
        assert str(exc) == "Connection refused"
        assert exc.symbol == "AAPL"

    def test_with_symbol_only(self):
        """Test exception with only symbol."""
        exc = NetworkException(symbol="AAPL")
        assert str(exc) == "Network error while fetching AAPL"
        assert exc.symbol == "AAPL"

    def test_with_no_arguments(self):
        """Test exception with no arguments."""
        exc = NetworkException()
        assert str(exc) == "Network error occurred"
        assert exc.symbol is None

    def test_with_cause(self):
        """Test exception with underlying cause."""
        cause = ConnectionError("Connection timeout")
        exc = NetworkException(symbol="AAPL", cause=cause)
        assert str(exc) == "Network error while fetching AAPL"
        assert exc.cause == cause

    def test_inheritance(self):
        """Test that NetworkException inherits from DataFetchException."""
        exc = NetworkException()
        assert isinstance(exc, DataFetchException)
        assert isinstance(exc, StockulaException)
        assert isinstance(exc, Exception)


class TestAPIException:
    """Test APIException."""

    def test_with_symbol_and_api_name(self):
        """Test exception with symbol and API name."""
        exc = APIException(symbol="AAPL", api_name="yahoo")
        assert str(exc) == "yahoo API failed for AAPL"
        assert exc.symbol == "AAPL"
        assert exc.api_name == "yahoo"

    def test_with_api_name_only(self):
        """Test exception with only API name."""
        exc = APIException(api_name="alpha_vantage")
        assert str(exc) == "alpha_vantage API call failed"
        assert exc.api_name == "alpha_vantage"
        assert exc.symbol is None

    def test_with_custom_message(self):
        """Test exception with custom message."""
        exc = APIException(symbol="AAPL", api_name="yahoo", message="Rate limit exceeded")
        assert str(exc) == "Rate limit exceeded"
        assert exc.symbol == "AAPL"
        assert exc.api_name == "yahoo"

    def test_default_api_name(self):
        """Test exception with default API name."""
        exc = APIException(symbol="AAPL")
        assert str(exc) == "yfinance API failed for AAPL"
        assert exc.api_name == "yfinance"

    def test_with_cause(self):
        """Test exception with underlying cause."""
        cause = ValueError("Invalid response")
        exc = APIException(symbol="AAPL", cause=cause)
        assert str(exc) == "yfinance API failed for AAPL"
        assert exc.cause == cause

    def test_inheritance(self):
        """Test that APIException inherits from DataFetchException."""
        exc = APIException()
        assert isinstance(exc, DataFetchException)
        assert isinstance(exc, StockulaException)
        assert isinstance(exc, Exception)


class TestDatabaseException:
    """Test DatabaseException."""

    def test_with_operation_and_message(self):
        """Test exception with operation and custom message."""
        exc = DatabaseException(operation="insert", message="Unique constraint violated")
        assert str(exc) == "Unique constraint violated"
        assert exc.operation == "insert"
        assert exc.cause is None

    def test_with_operation_and_cause(self):
        """Test exception with operation and cause."""
        cause = RuntimeError("Connection lost")
        exc = DatabaseException(operation="query", cause=cause)
        assert str(exc) == "Database query failed: Connection lost"
        assert exc.operation == "query"
        assert exc.cause == cause

    def test_with_operation_only(self):
        """Test exception with only operation."""
        exc = DatabaseException(operation="update")
        assert str(exc) == "Database update failed"
        assert exc.operation == "update"

    def test_with_cause_only(self):
        """Test exception with only cause."""
        cause = ConnectionError("Database unreachable")
        exc = DatabaseException(cause=cause)
        assert str(exc) == "Database error: Database unreachable"
        assert exc.operation is None
        assert exc.cause == cause

    def test_with_no_arguments(self):
        """Test exception with no arguments."""
        exc = DatabaseException()
        assert str(exc) == "Database error occurred"
        assert exc.operation is None
        assert exc.cause is None

    def test_with_all_arguments(self):
        """Test exception with all arguments."""
        cause = Exception("Root cause")
        exc = DatabaseException(operation="delete", message="Custom message", cause=cause)
        assert str(exc) == "Custom message"
        assert exc.operation == "delete"
        assert exc.cause == cause

    def test_inheritance(self):
        """Test that DatabaseException inherits from StockulaException."""
        exc = DatabaseException()
        assert isinstance(exc, StockulaException)
        assert isinstance(exc, Exception)


class TestConfigurationException:
    """Test ConfigurationException."""

    def test_with_field_and_message(self):
        """Test exception with field and custom message."""
        exc = ConfigurationException(field="api_key", message="API key is required")
        assert str(exc) == "API key is required"
        assert exc.field == "api_key"

    def test_with_field_only(self):
        """Test exception with only field."""
        exc = ConfigurationException(field="database_url")
        assert str(exc) == "Invalid configuration for field: database_url"
        assert exc.field == "database_url"

    def test_with_message_only(self):
        """Test exception with only message."""
        exc = ConfigurationException(message="Configuration file not found")
        assert str(exc) == "Configuration file not found"
        assert exc.field is None

    def test_with_no_arguments(self):
        """Test exception with no arguments."""
        exc = ConfigurationException()
        assert str(exc) == "Invalid configuration"
        assert exc.field is None

    def test_inheritance(self):
        """Test that ConfigurationException inherits from StockulaException."""
        exc = ConfigurationException()
        assert isinstance(exc, StockulaException)
        assert isinstance(exc, Exception)


class TestValidationException:
    """Test ValidationException."""

    def test_with_field_value_and_message(self):
        """Test exception with field, value, and custom message."""
        exc = ValidationException(field="period", value=-1, message="Period must be positive")
        assert str(exc) == "Period must be positive"
        assert exc.field == "period"
        assert exc.value == -1

    def test_with_field_and_value(self):
        """Test exception with field and value."""
        exc = ValidationException(field="symbol", value="INVALID!")
        assert str(exc) == "Validation failed for symbol: invalid value INVALID!"
        assert exc.field == "symbol"
        assert exc.value == "INVALID!"

    def test_with_field_only(self):
        """Test exception with only field."""
        exc = ValidationException(field="end_date")
        assert str(exc) == "Validation failed for end_date"
        assert exc.field == "end_date"
        assert exc.value is None

    def test_with_message_only(self):
        """Test exception with only message."""
        exc = ValidationException(message="Date range is invalid")
        assert str(exc) == "Date range is invalid"
        assert exc.field is None
        assert exc.value is None

    def test_with_no_arguments(self):
        """Test exception with no arguments."""
        exc = ValidationException()
        assert str(exc) == "Validation failed"
        assert exc.field is None
        assert exc.value is None

    def test_with_none_value(self):
        """Test exception with None as value."""
        exc = ValidationException(field="data", value=None)
        assert str(exc) == "Validation failed for data"
        assert exc.field == "data"
        assert exc.value is None

    def test_with_zero_value(self):
        """Test exception with zero as value (should be included in message)."""
        exc = ValidationException(field="count", value=0)
        assert str(exc) == "Validation failed for count: invalid value 0"
        assert exc.field == "count"
        assert exc.value == 0

    def test_with_false_value(self):
        """Test exception with False as value (should be included in message)."""
        exc = ValidationException(field="flag", value=False)
        assert str(exc) == "Validation failed for flag: invalid value False"
        assert exc.field == "flag"
        assert exc.value is False

    def test_with_empty_string_value(self):
        """Test exception with empty string as value."""
        exc = ValidationException(field="name", value="")
        assert str(exc) == "Validation failed for name: invalid value "
        assert exc.field == "name"
        assert exc.value == ""

    def test_inheritance(self):
        """Test that ValidationException inherits from StockulaException."""
        exc = ValidationException()
        assert isinstance(exc, StockulaException)
        assert isinstance(exc, Exception)


class TestExceptionHierarchy:
    """Test the exception hierarchy and relationships."""

    def test_all_inherit_from_stockula_exception(self):
        """Test that all custom exceptions inherit from StockulaException."""
        exceptions = [
            DataFetchException(),
            NetworkException(),
            APIException(),
            DatabaseException(),
            ConfigurationException(),
            ValidationException(),
        ]

        for exc in exceptions:
            assert isinstance(exc, StockulaException)
            assert isinstance(exc, Exception)

    def test_network_and_api_inherit_from_data_fetch(self):
        """Test that NetworkException and APIException inherit from DataFetchException."""
        network_exc = NetworkException()
        api_exc = APIException()

        assert isinstance(network_exc, DataFetchException)
        assert isinstance(api_exc, DataFetchException)

    def test_exception_attributes(self):
        """Test that exceptions have the expected attributes."""
        # DataFetchException attributes
        data_exc = DataFetchException(symbol="TEST")
        assert hasattr(data_exc, "symbol")
        assert hasattr(data_exc, "cause")

        # APIException specific attributes
        api_exc = APIException(api_name="test_api")
        assert hasattr(api_exc, "api_name")
        assert hasattr(api_exc, "symbol")
        assert hasattr(api_exc, "cause")

        # DatabaseException attributes
        db_exc = DatabaseException(operation="test_op")
        assert hasattr(db_exc, "operation")
        assert hasattr(db_exc, "cause")

        # ConfigurationException attributes
        config_exc = ConfigurationException(field="test_field")
        assert hasattr(config_exc, "field")

        # ValidationException attributes
        val_exc = ValidationException(field="test", value=123)
        assert hasattr(val_exc, "field")
        assert hasattr(val_exc, "value")

    def test_exception_can_be_raised_and_caught(self):
        """Test that exceptions can be raised and caught properly."""
        # Test raising and catching StockulaException
        with pytest.raises(StockulaException) as exc_info:
            raise StockulaException("Test error")
        assert str(exc_info.value) == "Test error"

        # Test raising and catching DataFetchException
        with pytest.raises(DataFetchException) as exc_info:
            raise DataFetchException(symbol="AAPL")
        assert "AAPL" in str(exc_info.value)

        # Test catching NetworkException as DataFetchException
        with pytest.raises(DataFetchException):
            raise NetworkException(symbol="AAPL")

        # Test catching all as StockulaException
        with pytest.raises(StockulaException):
            raise ValidationException(field="test")
