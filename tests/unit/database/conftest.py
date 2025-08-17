"""Database-specific pytest fixtures for comprehensive testing.

This module provides fixtures for testing the consolidated TimescaleDB database layer
with both real and mock implementations, optimized for the 67-method interface.
"""

import os
from typing import Any

import pandas as pd
import pytest

from stockula.config.models import TimescaleDBConfig
from stockula.database.manager import DatabaseManager
from stockula.interfaces import IDatabaseManager
from tests.unit.mocks.mock_timescale_manager import MockTimescaleDBManager


# Environment detection for adaptive testing
def is_timescaledb_available() -> bool:
    """Check if TimescaleDB is available for testing."""
    # Check environment variable first
    if os.getenv("TIMESCALEDB_AVAILABLE", "").lower() in ("true", "1", "yes"):
        return True

    # Try to connect if config is available
    try:
        config = get_test_timescale_config()
        if config:
            # Quick connection test
            from sqlalchemy import create_engine

            engine = create_engine(config.get_connection_url())
            with engine.connect() as conn:
                result = conn.execute("SELECT 1")
                return result.fetchone() is not None
    except Exception:
        pass

    return False


def get_test_timescale_config() -> TimescaleDBConfig | None:
    """Get TimescaleDB configuration for testing."""
    # Try environment variables
    host = os.getenv("TIMESCALEDB_HOST", "localhost")
    port = int(os.getenv("TIMESCALEDB_PORT", "5432"))
    database = os.getenv("TIMESCALEDB_DATABASE", "stockula_test")
    username = os.getenv("TIMESCALEDB_USERNAME", "stockula")
    password = os.getenv("TIMESCALEDB_PASSWORD", "password")

    # Only create config if minimum requirements are met
    if all([host, database, username]):
        return TimescaleDBConfig(
            host=host, port=port, database=database, user=username, password=password, pool_size=5, max_overflow=10
        )

    return None


# Fixture for TimescaleDB configuration
@pytest.fixture(scope="session")
def timescale_config() -> TimescaleDBConfig:
    """Create TimescaleDB configuration for testing."""
    config = get_test_timescale_config()
    if config is None:
        # Fallback to minimal config for mock testing
        config = TimescaleDBConfig(
            host="localhost", port=5432, database="test_stockula", user="test_user", password="test_password"
        )
    return config


# Smart database manager fixture with adaptive behavior
@pytest.fixture(scope="function")
def database_manager(timescale_config: TimescaleDBConfig) -> IDatabaseManager:
    """Provide database manager - real TimescaleDB if available, mock otherwise.

    This fixture automatically adapts based on environment:
    - Uses real TimescaleDB if TIMESCALEDB_AVAILABLE=true or connection succeeds
    - Falls back to mock implementation for fast unit testing
    """
    if is_timescaledb_available():
        # Use real TimescaleDB
        db = DatabaseManager(timescale_config)
        try:
            yield db
        finally:
            db.close()
    else:
        # Use mock implementation
        mock_db = MockTimescaleDBManager(timescale_config)
        # Populate with test data for realistic testing
        mock_db.populate_test_data(["AAPL", "GOOGL", "MSFT", "TSLA", "SPY"])
        yield mock_db


# Dedicated mock fixture for pure unit testing
@pytest.fixture(scope="function")
def mock_database_manager(timescale_config: TimescaleDBConfig) -> MockTimescaleDBManager:
    """Provide mock database manager for isolated unit testing."""
    mock_db = MockTimescaleDBManager(timescale_config)
    # Reset call history for clean testing
    mock_db.reset_call_history()
    return mock_db


# Real TimescaleDB fixture (skip if not available)
@pytest.fixture(scope="function")
def real_database_manager(timescale_config: TimescaleDBConfig) -> IDatabaseManager:
    """Provide real TimescaleDB manager (skips test if not available)."""
    if not is_timescaledb_available():
        pytest.skip("TimescaleDB not available for integration testing")

    db = DatabaseManager(timescale_config)
    try:
        yield db
    finally:
        db.close()


# Populated database fixture
@pytest.fixture(scope="function")
def populated_database_manager(database_manager: IDatabaseManager) -> IDatabaseManager:
    """Provide database manager pre-populated with test data."""
    # Add comprehensive test data
    test_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY", "QQQ", "NVDA"]

    # Add stock info
    for symbol in test_symbols:
        stock_info = {
            "longName": f"{symbol} Inc.",
            "sector": "Technology" if symbol != "SPY" else "ETF",
            "industry": "Software",
            "marketCap": 1000000000000,
            "exchange": "NASDAQ",
            "currency": "USD",
        }
        database_manager.store_stock_info(symbol, stock_info)

    # Add price history
    for symbol in test_symbols:
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        # Create realistic price data
        import numpy as np

        base_price = np.random.uniform(100, 500)
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))

        price_data = pd.DataFrame(
            {
                "Open": prices * np.random.uniform(0.995, 1.005, len(dates)),
                "High": prices * np.random.uniform(1.005, 1.025, len(dates)),
                "Low": prices * np.random.uniform(0.975, 0.995, len(dates)),
                "Close": prices,
                "Volume": np.random.randint(1000000, 50000000, len(dates)),
            },
            index=dates,
        )

        # Ensure price relationships
        price_data["High"] = price_data[["Open", "High", "Close"]].max(axis=1)
        price_data["Low"] = price_data[["Open", "Low", "Close"]].min(axis=1)

        database_manager.store_price_history(symbol, price_data)

    return database_manager


# Sample data fixtures
@pytest.fixture(scope="session")
def sample_stock_info() -> dict[str, Any]:
    """Sample stock information for testing."""
    return {
        "longName": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "marketCap": 3000000000000,
        "exchange": "NASDAQ",
        "currency": "USD",
        "website": "https://www.apple.com",
        "fullTimeEmployees": 150000,
        "previousClose": 180.0,
        "beta": 1.2,
    }


@pytest.fixture(scope="session")
def sample_price_data() -> pd.DataFrame:
    """Sample price data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    import numpy as np

    # Create realistic price movement
    base_price = 150.0
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))

    data = pd.DataFrame(
        {
            "Open": prices * np.random.uniform(0.995, 1.005, len(dates)),
            "High": prices * np.random.uniform(1.005, 1.025, len(dates)),
            "Low": prices * np.random.uniform(0.975, 0.995, len(dates)),
            "Close": prices,
            "Volume": np.random.randint(1000000, 50000000, len(dates)),
        },
        index=dates,
    )

    # Ensure price relationships
    data["High"] = data[["Open", "High", "Close"]].max(axis=1)
    data["Low"] = data[["Open", "Low", "Close"]].min(axis=1)

    return data


@pytest.fixture(scope="session")
def sample_dividends() -> pd.Series:
    """Sample dividend data for testing."""
    dividend_dates = pd.to_datetime(["2023-03-15", "2023-06-15", "2023-09-15", "2023-12-15"])
    return pd.Series([0.22, 0.23, 0.24, 0.25], index=dividend_dates)


@pytest.fixture(scope="session")
def sample_splits() -> pd.Series:
    """Sample stock split data for testing."""
    split_dates = pd.to_datetime(["2023-06-01"])
    return pd.Series([4.0], index=split_dates)  # 4:1 split


# Performance testing fixtures
@pytest.fixture(scope="function")
def large_dataset() -> pd.DataFrame:
    """Large dataset for performance testing."""
    dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
    import numpy as np

    base_price = 100.0
    returns = np.random.normal(0.0005, 0.015, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))

    return pd.DataFrame(
        {
            "Open": prices * np.random.uniform(0.995, 1.005, len(dates)),
            "High": prices * np.random.uniform(1.005, 1.025, len(dates)),
            "Low": prices * np.random.uniform(0.975, 0.995, len(dates)),
            "Close": prices,
            "Volume": np.random.randint(1000000, 100000000, len(dates)),
        },
        index=dates,
    )


# Interface validation fixtures
@pytest.fixture(scope="session")
def interface_method_signatures() -> dict[str, dict]:
    """Expected method signatures for interface compliance testing."""
    return {
        # Core operations
        "get_price_history": {"symbol": str, "start_date": str | None, "end_date": str | None, "interval": str},
        "store_price_history": {"symbol": str, "data": pd.DataFrame, "interval": str},
        "has_data": {"symbol": str, "start_date": str, "end_date": str},
        # Stock info
        "store_stock_info": {"symbol": str, "info": dict},
        "get_stock_info": {"symbol": str},
        # Dividends and splits
        "store_dividends": {"symbol": str, "dividends": pd.Series},
        "store_splits": {"symbol": str, "splits": pd.Series},
        # Utilities
        "get_all_symbols": {},
        "get_latest_price": {"symbol": str},
        "get_database_stats": {},
        # Analytics (subset)
        "get_moving_averages": {
            "symbol": str,
            "periods": list | None,
            "start_date": str | None,
            "end_date": str | None,
        },
        "get_bollinger_bands": {
            "symbol": str,
            "period": int,
            "std_dev": float,
            "start_date": str | None,
            "end_date": str | None,
        },
        "get_rsi": {"symbol": str, "period": int, "start_date": str | None, "end_date": str | None},
        # Session management
        "get_session": {},
        "close": {},
        "test_connection": {},
    }


# Method grouping for organized testing
@pytest.fixture(scope="session")
def method_groups() -> dict[str, list[str]]:
    """Group methods by functionality for organized testing."""
    return {
        "core_operations": ["get_price_history", "store_price_history", "has_data"],
        "stock_info": ["store_stock_info", "get_stock_info"],
        "dividends_splits": ["store_dividends", "store_splits"],
        "utilities": ["get_all_symbols", "get_latest_price", "get_database_stats"],
        "analytics_basic": ["get_moving_averages", "get_bollinger_bands", "get_rsi"],
        "analytics_advanced": [
            "get_price_momentum",
            "get_correlation_matrix",
            "get_volatility_analysis",
            "get_seasonal_patterns",
            "get_top_performers",
        ],
        "timescale_specific": ["get_price_history_aggregated", "get_recent_price_changes", "get_chunk_statistics"],
        "session_management": ["get_session", "close", "test_connection"],
    }


# Markers for conditional test execution
pytestmark = [
    pytest.mark.database,  # Mark all tests in this module as database tests
]


# Environment-specific fixtures
@pytest.fixture(scope="session")
def test_environment() -> dict[str, Any]:
    """Information about the test environment."""
    return {
        "timescaledb_available": is_timescaledb_available(),
        "config_available": get_test_timescale_config() is not None,
        "test_mode": "integration" if is_timescaledb_available() else "unit",
    }


# Error simulation fixtures
@pytest.fixture(scope="function")
def error_prone_database_manager(mock_database_manager: MockTimescaleDBManager) -> MockTimescaleDBManager:
    """Database manager that simulates various error conditions."""
    # Monkey patch methods to simulate errors
    original_get_price_history = mock_database_manager.get_price_history

    def error_on_invalid_symbol(symbol: str, *args, **kwargs):
        if symbol == "ERROR":
            raise ValueError("Simulated database error")
        return original_get_price_history(symbol, *args, **kwargs)

    mock_database_manager.get_price_history = error_on_invalid_symbol
    return mock_database_manager
