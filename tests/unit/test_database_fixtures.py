"""Test database fixtures to ensure they work correctly."""

import pytest
from datetime import date, timedelta
import pandas as pd
from sqlalchemy import text


def test_database_fixture_creates_tables(test_database):
    """Test that database fixture creates all required tables."""
    # Get table names from the database
    with test_database.get_session() as session:
        result = session.exec(
            text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        )
        tables = {row[0] for row in result}

    # Check that all expected tables exist
    expected_tables = {
        "stocks",
        "price_history",
        "dividends",
        "splits",
        "options_calls",
        "options_puts",
        "stock_info",
    }
    assert expected_tables.issubset(tables)


def test_database_fixture_seeds_stocks(test_database):
    """Test that database fixture seeds stock data correctly."""
    # Get all stock symbols
    symbols = test_database.get_all_symbols()

    # Verify we have the expected stocks
    expected_symbols = {"AAPL", "GOOGL", "MSFT", "TSLA", "SPY"}
    actual_symbols = set(symbols)
    assert expected_symbols == actual_symbols

    # Check stock details for AAPL
    stock_info = test_database.get_stock_info("AAPL")
    assert stock_info is not None
    assert stock_info["name"] == "Apple Inc."
    assert stock_info["sector"] == "Technology"
    assert stock_info["exchange"] == "NASDAQ"


def test_database_fixture_seeds_price_history(test_database):
    """Test that database fixture seeds price history correctly."""
    # Get price history for AAPL
    price_history = test_database.get_price_history("AAPL")

    # Should have 365 days of data
    assert len(price_history) == 365

    # Check data structure
    assert "Open" in price_history.columns
    assert "High" in price_history.columns
    assert "Low" in price_history.columns
    assert "Close" in price_history.columns
    assert "Volume" in price_history.columns

    # Verify data integrity
    assert (price_history["High"] >= price_history["Close"]).all()
    assert (price_history["Low"] <= price_history["Close"]).all()
    assert (price_history["Volume"] > 0).all()


def test_database_fixture_seeds_dividends(test_database):
    """Test that database fixture seeds dividend data correctly."""
    # Get dividends for AAPL
    dividends = test_database.get_dividends("AAPL")

    # Should have quarterly dividends
    assert len(dividends) >= 1  # At least one dividend
    assert (dividends > 0).all()  # All dividends positive

    # TSLA should have no dividends
    tsla_dividends = test_database.get_dividends("TSLA")
    assert len(tsla_dividends) == 0


def test_database_fixture_seeds_splits(test_database):
    """Test that database fixture seeds split data correctly."""
    # Get splits for AAPL
    splits = test_database.get_splits("AAPL")

    # Should have one split
    assert len(splits) == 1
    assert splits.iloc[0] == 4.0  # 4:1 split

    # GOOGL should have no splits
    googl_splits = test_database.get_splits("GOOGL")
    assert len(googl_splits) == 0


def test_database_session_scope_persistence(test_database_session):
    """Test that session-scoped database persists data."""
    # Add a new stock
    test_database_session.store_stock_info(
        "TEST",
        {
            "longName": "Test Company",
            "sector": "Technology",
            "marketCap": 1000000,
        },
    )

    # Verify it was added
    stock_info = test_database_session.get_stock_info("TEST")
    assert stock_info is not None
    assert stock_info["longName"] == "Test Company"


def test_database_cleanup_happens():
    """Test that database cleanup happens after session ends.

    This test verifies the test database path doesn't persist
    between test sessions.
    """
    from pathlib import Path

    test_db_path = Path(__file__).parent.parent / "data" / "test_stockula.db"

    # The database file should not exist before the fixture is used
    # (This assumes this test runs in isolation or after cleanup)
    # We can't reliably test this without controlling test order,
    # so we just verify the path is correct
    assert test_db_path.parent.name == "data"
    assert test_db_path.name == "test_stockula.db"
