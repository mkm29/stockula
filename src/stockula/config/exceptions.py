"""Custom exceptions for Stockula."""

from typing import Any


class StockulaException(Exception):
    """Base exception for all Stockula-specific errors."""

    pass


class DataFetchException(StockulaException):
    """Exception raised when data fetching fails."""

    def __init__(self, symbol: str | None = None, message: str | None = None, cause: Exception | None = None):
        """Initialize DataFetchException.

        Args:
            symbol: The stock symbol that failed to fetch
            message: Custom error message
            cause: The underlying exception that caused this error
        """
        self.symbol = symbol
        self.cause = cause

        if message:
            super().__init__(message)
        elif symbol and cause:
            super().__init__(f"Failed to fetch data for {symbol}: {cause}")
        elif symbol:
            super().__init__(f"Failed to fetch data for {symbol}")
        elif cause:
            super().__init__(f"Data fetch failed: {cause}")
        else:
            super().__init__("Data fetch failed")


class NetworkException(DataFetchException):
    """Exception raised when network operations fail."""

    def __init__(self, symbol: str | None = None, message: str | None = None, cause: Exception | None = None):
        """Initialize NetworkException."""
        if not message and symbol:
            message = f"Network error while fetching {symbol}"
        elif not message:
            message = "Network error occurred"
        super().__init__(symbol=symbol, message=message, cause=cause)


class APIException(DataFetchException):
    """Exception raised when API calls fail."""

    def __init__(
        self,
        symbol: str | None = None,
        api_name: str = "yfinance",
        message: str | None = None,
        cause: Exception | None = None,
    ):
        """Initialize APIException.

        Args:
            symbol: The stock symbol that failed
            api_name: Name of the API that failed
            message: Custom error message
            cause: The underlying exception
        """
        self.api_name = api_name
        if not message and symbol:
            message = f"{api_name} API failed for {symbol}"
        elif not message:
            message = f"{api_name} API call failed"
        super().__init__(symbol=symbol, message=message, cause=cause)


class DatabaseException(StockulaException):
    """Exception raised when database operations fail."""

    def __init__(self, operation: str | None = None, message: str | None = None, cause: Exception | None = None):
        """Initialize DatabaseException.

        Args:
            operation: The database operation that failed
            message: Custom error message
            cause: The underlying exception
        """
        self.operation = operation
        self.cause = cause

        if message:
            super().__init__(message)
        elif operation and cause:
            super().__init__(f"Database {operation} failed: {cause}")
        elif operation:
            super().__init__(f"Database {operation} failed")
        elif cause:
            super().__init__(f"Database error: {cause}")
        else:
            super().__init__("Database error occurred")


class ConfigurationException(StockulaException):
    """Exception raised when configuration is invalid."""

    def __init__(self, field: str | None = None, message: str | None = None):
        """Initialize ConfigurationException.

        Args:
            field: The configuration field that is invalid
            message: Custom error message
        """
        self.field = field

        if message:
            super().__init__(message)
        elif field:
            super().__init__(f"Invalid configuration for field: {field}")
        else:
            super().__init__("Invalid configuration")


class ValidationException(StockulaException):
    """Exception raised when data validation fails."""

    def __init__(self, field: str | None = None, value: Any = None, message: str | None = None):
        """Initialize ValidationException.

        Args:
            field: The field that failed validation
            value: The value that was invalid
            message: Custom error message
        """
        self.field = field
        self.value = value

        if message:
            super().__init__(message)
        elif field and value is not None:
            super().__init__(f"Validation failed for {field}: invalid value {value}")
        elif field:
            super().__init__(f"Validation failed for {field}")
        else:
            super().__init__("Validation failed")
