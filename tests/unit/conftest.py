"""Shared test fixtures and configuration for unit tests."""

import os
import sys
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from stockula.config.models import TimescaleDBConfig
from stockula.interfaces import IDatabaseManager

# Import the mock implementation
from tests.unit.mocks.mock_timescale_manager import MockTimescaleDBManager


@pytest.fixture(scope="session", autouse=True)
def mock_chronos_if_unavailable():
    """Mock chronos module if not available to prevent import errors."""
    try:
        import importlib

        importlib.import_module("chronos")
        chronos_available = True
    except ImportError:
        chronos_available = False
        # Create a mock chronos module to prevent import errors
        if "chronos" not in sys.modules:
            chronos_mock = MagicMock()
            chronos_mock.__spec__ = MagicMock()
            sys.modules["chronos"] = chronos_mock

    return chronos_available


@pytest.fixture(scope="session")
def chronos_available(mock_chronos_if_unavailable):
    """Check if chronos is available."""
    return mock_chronos_if_unavailable


def is_chronos_available():
    """Helper function to check chronos availability for skipif decorators."""
    try:
        import importlib

        importlib.import_module("chronos")
        return True
    except ImportError:
        return False


def is_timescaledb_available() -> bool:
    """Check if TimescaleDB is available for integration tests."""
    try:
        from sqlalchemy import create_engine, text

        # Check for TimescaleDB connection environment variables
        db_host = os.getenv("TIMESCALEDB_HOST", "localhost")
        db_port = os.getenv("TIMESCALEDB_PORT", "5432")
        db_name = os.getenv("TIMESCALEDB_DATABASE", "stockula_test")
        db_user = os.getenv("TIMESCALEDB_USER", "postgres")
        db_password = os.getenv("TIMESCALEDB_PASSWORD", "postgres")

        # Try to connect to TimescaleDB
        connection_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        engine = create_engine(connection_url, connect_args={"connect_timeout": 5})

        with engine.connect() as conn:
            # Test TimescaleDB extension
            result = conn.execute(text("SELECT extname FROM pg_extension WHERE extname = 'timescaledb';"))
            has_timescale = result.fetchone() is not None

        return has_timescale
    except Exception:
        return False


@pytest.fixture(scope="session")
def timescaledb_available() -> bool:
    """Check if TimescaleDB is available for tests."""
    return is_timescaledb_available()


@pytest.fixture
def timescaledb_config() -> TimescaleDBConfig:
    """Create TimescaleDB configuration for testing."""
    return TimescaleDBConfig(
        host=os.getenv("TIMESCALEDB_HOST", "localhost"),
        port=int(os.getenv("TIMESCALEDB_PORT", "5432")),
        database=os.getenv("TIMESCALEDB_DATABASE", "stockula_test"),
        user=os.getenv("TIMESCALEDB_USER", "postgres"),
        password=os.getenv("TIMESCALEDB_PASSWORD", "postgres"),
        schema="public",
        pool_size=2,  # Smaller for tests
        max_overflow=5,
        pool_timeout=10,
        pool_recycle=1800,
    )


@pytest.fixture
def mock_database_manager(timescaledb_config: TimescaleDBConfig) -> MockTimescaleDBManager:
    """Create a mock database manager for unit tests."""
    mock_manager = MockTimescaleDBManager(timescaledb_config)

    # Populate with some basic test data
    test_symbols = ["AAPL", "GOOGL", "MSFT"]
    mock_manager.populate_test_data(test_symbols, days=50)

    yield mock_manager

    # Cleanup
    mock_manager.close()


@pytest.fixture
def database_manager(
    timescaledb_config: TimescaleDBConfig, timescaledb_available: bool
) -> Generator[IDatabaseManager, None, None]:
    """Create appropriate database manager based on availability.

    Returns real TimescaleDB manager if available, otherwise mock.
    Automatically handles cleanup.
    """
    if timescaledb_available:
        # Use real TimescaleDB manager for integration tests
        from stockula.database.manager import DatabaseManager

        manager = DatabaseManager(timescaledb_config)
        try:
            yield manager
        finally:
            manager.close()
    else:
        # Use mock for unit tests when TimescaleDB not available
        mock_manager = MockTimescaleDBManager(timescaledb_config)
        test_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        mock_manager.populate_test_data(test_symbols, days=100)

        try:
            yield mock_manager
        finally:
            mock_manager.close()


@pytest.fixture
def sample_price_data() -> pd.DataFrame:
    """Create sample price data for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

    # Generate realistic OHLCV data
    base_price = 150.0
    price_data = []

    for i, _date in enumerate(dates):
        # Add some randomness and trend
        trend = i * 0.01  # Small upward trend
        volatility = 2.0  # 2% daily volatility

        daily_change = (np.random.random() - 0.5) * volatility / 100
        close = base_price * (1 + trend / 100) * (1 + daily_change)

        # Generate OHLC from close
        high = close * (1 + abs(np.random.random() * 0.02))
        low = close * (1 - abs(np.random.random() * 0.02))
        open_price = close * (1 + (np.random.random() - 0.5) * 0.01)

        # Ensure logical relationships
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        volume = int(np.random.uniform(1000000, 10000000))

        price_data.append({"Open": open_price, "High": high, "Low": low, "Close": close, "Volume": volume})

    return pd.DataFrame(price_data, index=dates)


@pytest.fixture
def sample_stock_info() -> dict[str, Any]:
    """Create sample stock info for testing."""
    return {
        "symbol": "AAPL",
        "longName": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "marketCap": 3000000000000,
        "enterpriseValue": 2900000000000,
        "trailingPE": 28.5,
        "forwardPE": 25.2,
        "pegRatio": 2.1,
        "priceToBook": 45.2,
        "priceToSales": 7.8,
        "dividendYield": 0.005,
        "beta": 1.2,
        "52WeekLow": 124.17,
        "52WeekHigh": 198.23,
        "currency": "USD",
        "exchange": "NASDAQ",
        "country": "US",
    }


@pytest.fixture
def sample_dividends() -> pd.Series:
    """Create sample dividend data for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="Q")
    dividends = pd.Series([0.23, 0.24, 0.24, 0.25], index=dates)
    dividends.name = "Dividends"
    return dividends


@pytest.fixture
def sample_splits() -> pd.Series:
    """Create sample stock split data for testing."""
    # Most stocks don't split frequently, so create minimal data
    dates = pd.DatetimeIndex(["2023-06-15"])
    splits = pd.Series([2.0], index=dates)  # 2-for-1 split
    splits.name = "Stock Splits"
    return splits


@pytest.fixture(scope="function")
def isolated_mock_manager(timescaledb_config: TimescaleDBConfig) -> MockTimescaleDBManager:
    """Create a fresh mock manager for each test (function scope)."""
    manager = MockTimescaleDBManager(timescaledb_config)
    manager.reset_call_history()
    return manager
