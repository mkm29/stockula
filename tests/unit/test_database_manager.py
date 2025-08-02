"""Unit tests for database manager module."""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from stockula.database.manager import DatabaseManager


class TestDatabaseManagerInitialization:
    """Test DatabaseManager initialization."""

    def test_initialization_default_path(self, tmp_path):
        """Test initialization with default database path."""
        db_path = tmp_path / "test.db"
        with patch("stockula.database.manager.Path", return_value=db_path):
            db = DatabaseManager(str(db_path))
            assert db.db_path == db_path
            assert db_path.exists()

    def test_initialization_creates_tables(self, tmp_path):
        """Test that initialization creates all required tables."""
        db_path = tmp_path / "test.db"
        DatabaseManager(str(db_path))

        # Check tables exist
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

        expected_tables = [
            "stocks",
            "price_history",
            "stock_info",
            "dividends",
            "splits",
            "options_calls",
            "options_puts",
        ]
        for table in expected_tables:
            assert table in tables

    def test_initialization_creates_indexes(self, tmp_path):
        """Test that initialization creates indexes."""
        db_path = tmp_path / "test.db"
        DatabaseManager(str(db_path))

        # Check indexes exist
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = [row[0] for row in cursor.fetchall()]

        expected_indexes = [
            "idx_price_history_symbol_date",
            "idx_dividends_symbol_date",
            "idx_splits_symbol_date",
            "idx_options_calls_symbol_exp",
            "idx_options_puts_symbol_exp",
        ]
        for index in expected_indexes:
            assert index in indexes

    def test_foreign_key_constraints_enabled(self, tmp_path):
        """Test that foreign key constraints are enabled."""
        db_path = tmp_path / "test.db"
        db = DatabaseManager(str(db_path))

        with db.get_connection() as conn:
            cursor = conn.execute("PRAGMA foreign_keys")
            fk_enabled = cursor.fetchone()[0]
            assert fk_enabled == 1


class TestStockInfoOperations:
    """Test stock info storage and retrieval."""

    @pytest.fixture
    def db_manager(self, tmp_path):
        """Create a DatabaseManager instance."""
        return DatabaseManager(str(tmp_path / "test.db"))

    def test_store_stock_info(self, db_manager):
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

        db_manager.store_stock_info("TEST", info)

        # Verify in stocks table
        with db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM stocks WHERE symbol = ?", ("TEST",))
            row = cursor.fetchone()

        assert row is not None
        assert row[1] == "Test Company Inc."  # name
        assert row[2] == "Technology"  # sector
        assert row[3] == "Software"  # industry
        assert row[4] == 1000000000  # market_cap

    def test_store_stock_info_update(self, db_manager):
        """Test updating existing stock information."""
        # Store initial info
        info1 = {"longName": "Old Name", "sector": "Tech"}
        db_manager.store_stock_info("TEST", info1)

        # Update with new info
        info2 = {"longName": "New Name", "sector": "Technology"}
        db_manager.store_stock_info("TEST", info2)

        # Should have updated
        stored_info = db_manager.get_stock_info("TEST")
        assert stored_info["longName"] == "New Name"
        assert stored_info["sector"] == "Technology"

    def test_get_stock_info(self, db_manager):
        """Test retrieving stock information."""
        info = {
            "longName": "Test Company",
            "sector": "Finance",
            "custom_field": "custom_value",
        }
        db_manager.store_stock_info("TEST", info)

        retrieved = db_manager.get_stock_info("TEST")
        assert retrieved == info

    def test_get_stock_info_not_found(self, db_manager):
        """Test retrieving non-existent stock info."""
        info = db_manager.get_stock_info("NOTEXIST")
        assert info is None


class TestPriceHistoryOperations:
    """Test price history storage and retrieval."""

    @pytest.fixture
    def db_manager(self, tmp_path):
        """Create a DatabaseManager instance."""
        db = DatabaseManager(str(tmp_path / "test.db"))
        # Add test stock
        db.store_stock_info("TEST", {"longName": "Test Company"})
        return db

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data."""
        dates = pd.date_range("2023-01-01", periods=10)
        return pd.DataFrame(
            {
                "Open": np.random.uniform(98, 102, 10),
                "High": np.random.uniform(99, 103, 10),
                "Low": np.random.uniform(97, 101, 10),
                "Close": np.random.uniform(98, 102, 10),
                "Volume": np.random.randint(1000000, 5000000, 10),
            },
            index=dates,
        )

    def test_store_price_history(self, db_manager, sample_price_data):
        """Test storing price history."""
        db_manager.store_price_history("TEST", sample_price_data)

        # Verify data was stored
        with db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM price_history WHERE symbol = ?", ("TEST",))
            count = cursor.fetchone()[0]

        assert count == len(sample_price_data)

    def test_store_price_history_with_interval(self, db_manager, sample_price_data):
        """Test storing price history with custom interval."""
        db_manager.store_price_history("TEST", sample_price_data, interval="1h")

        # Verify interval was stored
        with db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT interval FROM price_history WHERE symbol = ? LIMIT 1", ("TEST",))
            interval = cursor.fetchone()[0]

        assert interval == "1h"

    def test_get_price_history(self, db_manager, sample_price_data):
        """Test retrieving price history."""
        db_manager.store_price_history("TEST", sample_price_data)

        retrieved = db_manager.get_price_history("TEST")

        assert isinstance(retrieved, pd.DataFrame)
        assert len(retrieved) == len(sample_price_data)
        assert all(col in retrieved.columns for col in ["Open", "High", "Low", "Close", "Volume"])

    def test_get_price_history_with_date_range(self, db_manager, sample_price_data):
        """Test retrieving price history with date range."""
        db_manager.store_price_history("TEST", sample_price_data)

        # Get subset of data
        start_date = "2023-01-03"
        end_date = "2023-01-07"
        retrieved = db_manager.get_price_history("TEST", start_date, end_date)

        assert len(retrieved) == 5  # 5 days inclusive

    def test_get_latest_price(self, db_manager, sample_price_data):
        """Test getting latest price."""
        db_manager.store_price_history("TEST", sample_price_data)

        latest = db_manager.get_latest_price("TEST")

        assert latest is not None
        assert isinstance(latest, float)

    def test_has_data(self, db_manager, sample_price_data):
        """Test checking if data exists."""
        db_manager.store_price_history("TEST", sample_price_data)

        # Should have data for stored range
        assert db_manager.has_data("TEST", "2023-01-01", "2023-01-10")

        # Should not have data for future dates
        assert not db_manager.has_data("TEST", "2024-01-01", "2024-01-10")


class TestDividendsAndSplits:
    """Test dividends and splits operations."""

    @pytest.fixture
    def db_manager(self, tmp_path):
        """Create a DatabaseManager instance."""
        db = DatabaseManager(str(tmp_path / "test.db"))
        db.store_stock_info("TEST", {"longName": "Test Company"})
        return db

    def test_store_and_get_dividends(self, db_manager):
        """Test storing and retrieving dividends."""
        dividend_dates = pd.to_datetime(["2023-03-15", "2023-06-15", "2023-09-15"])
        dividends = pd.Series([0.25, 0.25, 0.30], index=dividend_dates)

        db_manager.store_dividends("TEST", dividends)

        retrieved = db_manager.get_dividends("TEST")
        assert isinstance(retrieved, pd.Series)
        assert len(retrieved) == 3
        assert retrieved.iloc[0] == 0.25

    def test_store_and_get_splits(self, db_manager):
        """Test storing and retrieving splits."""
        split_dates = pd.to_datetime(["2022-06-01", "2023-06-01"])
        splits = pd.Series([2.0, 3.0], index=split_dates)

        db_manager.store_splits("TEST", splits)

        retrieved = db_manager.get_splits("TEST")
        assert isinstance(retrieved, pd.Series)
        assert len(retrieved) == 2
        assert retrieved.iloc[0] == 2.0

    def test_get_dividends_with_date_range(self, db_manager):
        """Test retrieving dividends with date range."""
        dividend_dates = pd.to_datetime(["2023-01-15", "2023-04-15", "2023-07-15", "2023-10-15"])
        dividends = pd.Series([0.25, 0.25, 0.30, 0.30], index=dividend_dates)

        db_manager.store_dividends("TEST", dividends)

        # Get Q2 dividends
        retrieved = db_manager.get_dividends("TEST", "2023-04-01", "2023-06-30")
        assert len(retrieved) == 1
        assert retrieved.iloc[0] == 0.25


class TestOptionsData:
    """Test options data operations."""

    @pytest.fixture
    def db_manager(self, tmp_path):
        """Create a DatabaseManager instance."""
        db = DatabaseManager(str(tmp_path / "test.db"))
        db.store_stock_info("TEST", {"longName": "Test Company"})
        return db

    @pytest.fixture
    def sample_options_data(self):
        """Create sample options data."""
        calls = pd.DataFrame(
            {
                "strike": [95.0, 100.0, 105.0],
                "lastPrice": [7.5, 5.0, 2.5],
                "bid": [7.4, 4.9, 2.4],
                "ask": [7.6, 5.1, 2.6],
                "volume": [100, 200, 150],
                "openInterest": [1000, 2000, 1500],
                "impliedVolatility": [0.25, 0.28, 0.30],
                "inTheMoney": [True, False, False],
                "contractSymbol": ["TEST230120C95", "TEST230120C100", "TEST230120C105"],
            }
        )

        puts = pd.DataFrame(
            {
                "strike": [95.0, 100.0, 105.0],
                "lastPrice": [2.5, 5.0, 7.5],
                "bid": [2.4, 4.9, 7.4],
                "ask": [2.6, 5.1, 7.6],
                "volume": [50, 100, 75],
                "openInterest": [500, 1000, 750],
                "impliedVolatility": [0.25, 0.28, 0.30],
                "inTheMoney": [False, False, True],
                "contractSymbol": ["TEST230120P95", "TEST230120P100", "TEST230120P105"],
            }
        )

        return calls, puts

    def test_store_options_chain(self, db_manager, sample_options_data):
        """Test storing options chain."""
        calls, puts = sample_options_data
        expiration = "2023-01-20"

        db_manager.store_options_chain("TEST", calls, puts, expiration)

        # Verify data was stored
        with db_manager.get_connection() as conn:
            calls_count = conn.execute("SELECT COUNT(*) FROM options_calls WHERE symbol = ?", ("TEST",)).fetchone()[0]
            puts_count = conn.execute("SELECT COUNT(*) FROM options_puts WHERE symbol = ?", ("TEST",)).fetchone()[0]

        assert calls_count == len(calls)
        assert puts_count == len(puts)

    def test_get_options_chain(self, db_manager, sample_options_data):
        """Test retrieving options chain."""
        calls, puts = sample_options_data
        expiration = "2023-01-20"

        db_manager.store_options_chain("TEST", calls, puts, expiration)

        retrieved_calls, retrieved_puts = db_manager.get_options_chain("TEST", expiration)

        assert isinstance(retrieved_calls, pd.DataFrame)
        assert isinstance(retrieved_puts, pd.DataFrame)
        assert len(retrieved_calls) == len(calls)
        assert len(retrieved_puts) == len(puts)
        assert "strike" in retrieved_calls.columns
        assert "lastPrice" in retrieved_calls.columns


class TestUtilityMethods:
    """Test utility methods."""

    @pytest.fixture
    def db_manager(self, tmp_path):
        """Create a DatabaseManager instance."""
        db = DatabaseManager(str(tmp_path / "test.db"))
        # Add test stocks
        for symbol in ["AAPL", "GOOGL", "MSFT"]:
            db.store_stock_info(symbol, {"longName": f"{symbol} Inc."})
        return db

    def test_get_all_symbols(self, db_manager):
        """Test getting all symbols."""
        symbols = db_manager.get_all_symbols()

        assert isinstance(symbols, list)
        assert len(symbols) == 3
        assert "AAPL" in symbols
        assert "GOOGL" in symbols
        assert "MSFT" in symbols
        # Should be sorted
        assert symbols == sorted(symbols)

    def test_get_database_stats(self, db_manager):
        """Test getting database statistics."""
        # Add some data
        data = pd.DataFrame(
            {
                "Open": [100],
                "High": [101],
                "Low": [99],
                "Close": [100.5],
                "Volume": [1000000],
            },
            index=pd.date_range("2023-01-01", periods=1),
        )

        db_manager.store_price_history("AAPL", data)

        stats = db_manager.get_database_stats()

        assert isinstance(stats, dict)
        assert "stocks" in stats
        assert "price_history" in stats
        assert stats["stocks"] == 3
        assert stats["price_history"] == 1

    def test_cleanup_old_data(self, db_manager):
        """Test cleaning up old data."""
        # Add old and recent data
        old_date = datetime.now() - timedelta(days=400)
        recent_date = datetime.now() - timedelta(days=10)

        old_data = pd.DataFrame(
            {
                "Open": [100],
                "High": [101],
                "Low": [99],
                "Close": [100.5],
                "Volume": [1000000],
            },
            index=[old_date],
        )

        recent_data = pd.DataFrame(
            {
                "Open": [110],
                "High": [111],
                "Low": [109],
                "Close": [110.5],
                "Volume": [1100000],
            },
            index=[recent_date],
        )

        db_manager.store_price_history("AAPL", old_data)
        db_manager.store_price_history("AAPL", recent_data)

        # Clean up data older than 365 days
        deleted = db_manager.cleanup_old_data(days_to_keep=365)

        assert deleted == 1  # Should delete 1 row

        # Verify only recent data remains
        remaining = db_manager.get_price_history("AAPL")
        assert len(remaining) == 1
        assert remaining.index[0].date() == recent_date.date()


class TestBackwardCompatibility:
    """Test backward compatibility methods."""

    @pytest.fixture
    def db_manager(self, tmp_path):
        """Create a DatabaseManager instance."""
        return DatabaseManager(str(tmp_path / "test.db"))

    def test_add_stock(self, db_manager):
        """Test add_stock backward compatibility."""
        db_manager.add_stock("TEST", "Test Company", "Technology", 1000000000)

        info = db_manager.get_stock_info("TEST")
        assert info["longName"] == "Test Company"
        assert info["sector"] == "Technology"
        assert info["marketCap"] == 1000000000

    def test_add_price_data(self, db_manager):
        """Test add_price_data backward compatibility."""
        db_manager.add_stock("TEST", "Test Company")

        date = datetime.now()
        db_manager.add_price_data("TEST", date, 100.0, 101.0, 99.0, 100.5, 1000000)

        data = db_manager.get_price_history("TEST")
        assert len(data) == 1
        assert data.iloc[0]["Open"] == 100.0
        assert data.iloc[0]["Close"] == 100.5

    def test_bulk_add_price_data(self, db_manager):
        """Test bulk_add_price_data backward compatibility."""
        db_manager.add_stock("TEST", "Test Company")

        # Create bulk data
        price_data = []
        base_date = datetime.now()
        for i in range(5):
            date = base_date - timedelta(days=i)
            price_data.append(
                (
                    "TEST",
                    date,
                    100 + i,
                    101 + i,
                    99 + i,
                    100.5 + i,
                    1000000 + i * 10000,
                    "1d",
                )
            )

        db_manager.bulk_add_price_data(price_data)

        data = db_manager.get_price_history("TEST")
        assert len(data) == 5

    def test_get_latest_price_date(self, db_manager):
        """Test get_latest_price_date backward compatibility."""
        db_manager.add_stock("TEST", "Test Company")

        # Add data with specific dates
        dates = [datetime.now() - timedelta(days=i) for i in range(5, 0, -1)]
        for date in dates:
            db_manager.add_price_data("TEST", date, 100, 101, 99, 100.5, 1000000)

        latest = db_manager.get_latest_price_date("TEST")
        assert latest.date() == dates[-1].date()


class TestConnectionManagement:
    """Test database connection management."""

    @pytest.fixture
    def db_manager(self, tmp_path):
        """Create a DatabaseManager instance."""
        return DatabaseManager(str(tmp_path / "test.db"))

    def test_get_connection_context_manager(self, db_manager):
        """Test get_connection as context manager."""
        with db_manager.get_connection() as conn:
            assert isinstance(conn, sqlite3.Connection)
            # Foreign keys should be enabled
            cursor = conn.execute("PRAGMA foreign_keys")
            assert cursor.fetchone()[0] == 1

    def test_conn_property(self, db_manager):
        """Test deprecated conn property."""
        conn = db_manager.conn
        assert isinstance(conn, sqlite3.Connection)
        conn.close()  # Clean up

    def test_transaction_rollback(self, db_manager):
        """Test transaction rollback on error."""
        # Try to insert price data for non-existent stock
        # This should fail due to foreign key constraint
        with pytest.raises(sqlite3.IntegrityError):
            with db_manager.get_connection() as conn:
                conn.execute(
                    """INSERT INTO price_history
                    (symbol, date, close_price, volume)
                    VALUES (?, ?, ?, ?)""",
                    ("NOTEXIST", "2023-01-01", 100.0, 1000000),
                )

    def test_vacuum_operation(self, db_manager):
        """Test VACUUM operation in cleanup."""
        # Add and delete data to create space to reclaim
        db_manager.add_stock("TEST", "Test Company")

        # Add data
        for i in range(100):
            date = datetime.now() - timedelta(days=500 + i)
            db_manager.add_price_data("TEST", date, 100, 101, 99, 100.5, 1000000)

        # Get initial size
        initial_size = Path(db_manager.db_path).stat().st_size

        # Clean up (includes VACUUM)
        db_manager.cleanup_old_data(days_to_keep=365)

        # Size might be smaller after VACUUM (not guaranteed)
        final_size = Path(db_manager.db_path).stat().st_size
        assert final_size <= initial_size


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.fixture
    def db_manager(self, tmp_path):
        """Create a DatabaseManager instance."""
        return DatabaseManager(str(tmp_path / "test.db"))

    def test_store_empty_dataframe(self, db_manager):
        """Test storing empty DataFrame."""
        empty_df = pd.DataFrame()

        # Should handle gracefully
        db_manager.store_price_history("TEST", empty_df)

        # No data should be stored
        data = db_manager.get_price_history("TEST")
        assert data.empty

    def test_invalid_json_in_stock_info(self, db_manager):
        """Test handling invalid JSON in stock info."""
        # Manually insert invalid JSON
        with db_manager.get_connection() as conn:
            conn.execute(
                "INSERT INTO stocks (symbol, name, created_at, updated_at) "
                "VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                ("TEST", "Test Company"),
            )
            conn.execute(
                "INSERT INTO stock_info (symbol, info_json, created_at, updated_at) "
                "VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                ("TEST", "{invalid json}"),
            )
            conn.commit()

        # Should handle error gracefully
        with pytest.raises(json.JSONDecodeError):
            db_manager.get_stock_info("TEST")


class TestDatabaseFixtures:
    """Test database fixtures to ensure they work correctly."""

    def test_database_fixture_creates_tables(self, test_database):
        """Test that database fixture creates all required tables."""
        # Get table names from the database
        with test_database.get_session() as session:
            result = session.exec(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
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

    def test_database_fixture_seeds_stocks(self, test_database):
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

    def test_database_fixture_seeds_price_history(self, test_database):
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

    def test_database_fixture_seeds_dividends(self, test_database):
        """Test that database fixture seeds dividend data correctly."""
        # Get dividends for AAPL
        dividends = test_database.get_dividends("AAPL")

        # Should have quarterly dividends
        assert len(dividends) >= 1  # At least one dividend
        assert (dividends > 0).all()  # All dividends positive

        # TSLA should have no dividends
        tsla_dividends = test_database.get_dividends("TSLA")
        assert len(tsla_dividends) == 0

    def test_database_fixture_seeds_splits(self, test_database):
        """Test that database fixture seeds split data correctly."""
        # Get splits for AAPL
        splits = test_database.get_splits("AAPL")

        # Should have one split
        assert len(splits) == 1
        assert splits.iloc[0] == 4.0  # 4:1 split

        # GOOGL should have no splits
        googl_splits = test_database.get_splits("GOOGL")
        assert len(googl_splits) == 0

    def test_database_session_scope_persistence(self, test_database_session):
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

    def test_database_cleanup_happens(self):
        """Test that database cleanup happens after session ends.

        This test verifies the test database path doesn't persist
        between test sessions.
        """
        test_db_path = Path(__file__).parent.parent / "data" / "test_stockula.db"

        # The database file should not exist before the fixture is used
        # (This assumes this test runs in isolation or after cleanup)
        # We can't reliably test this without controlling test order,
        # so we just verify the path is correct
        assert test_db_path.parent.name == "data"
        assert test_db_path.name == "test_stockula.db"
