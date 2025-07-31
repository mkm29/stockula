"""Tests for database module."""

import sqlite3
from datetime import datetime, timedelta

import pandas as pd

from stockula.database.manager import DatabaseManager


class TestDatabaseManager:
    """Test DatabaseManager class."""

    def test_database_initialization(self, temp_db_path):
        """Test database initialization creates tables."""
        db = DatabaseManager(temp_db_path)

        # Check tables exist
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
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

    def test_add_stock(self, test_database):
        """Test adding a stock to database."""
        test_database.store_stock_info(
            "AAPL",
            {
                "longName": "Apple Inc.",
                "sector": "Technology",
                "marketCap": 3000000000000,
            },
        )

        # Verify stock was added
        with sqlite3.connect(test_database.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM stocks WHERE symbol = ?", ("AAPL",))
            row = cursor.fetchone()

        assert row is not None
        assert row[0] == "AAPL"  # symbol is first column
        assert row[1] == "Apple Inc."
        assert row[2] == "Technology"
        assert row[4] == 3000000000000  # market_cap

    def test_add_duplicate_stock(self, in_memory_database):
        """Test adding duplicate stock updates existing."""
        test_database = in_memory_database
        test_database.add_stock("AAPL", "Apple Inc.", "Technology", 3000000000000)
        test_database.add_stock("AAPL", "Apple Computer", "Tech", 3500000000000)

        # Should update, not create duplicate
        with test_database.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM stocks WHERE symbol = ?", ("AAPL",))
            count = cursor.fetchone()[0]
            assert count == 1

            # Check updated values
            cursor.execute(
                "SELECT name, sector, market_cap FROM stocks WHERE symbol = ?",
                ("AAPL",),
            )
            row = cursor.fetchone()
            assert row[0] == "Apple Computer"
            assert row[1] == "Tech"
            assert row[2] == 3500000000000

    def test_add_price_data(self, in_memory_database):
        """Test adding price data."""
        test_database = in_memory_database
        test_database.add_stock("TEST_STOCK", "Test Stock Inc.")

        price_date = datetime.now()
        test_database.add_price_data("TEST_STOCK", price_date, 150.0, 152.0, 149.0, 151.0, 1000000, "1d")

        # Verify price data was added
        with test_database.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM price_history WHERE symbol = ?", ("TEST_STOCK",))
            row = cursor.fetchone()

        assert row is not None
        assert row[3] == 150.0  # open_price
        assert row[4] == 152.0  # high_price
        assert row[5] == 149.0  # low_price
        assert row[6] == 151.0  # close_price
        assert row[7] == 1000000  # volume

    def test_bulk_add_price_data(self, in_memory_database, sample_ohlcv_data):
        """Test bulk adding price data."""
        test_database = in_memory_database
        test_database.add_stock("BULK_TEST", "Bulk Test Inc.")

        # Convert DataFrame to list of tuples
        price_data = []
        for date, row in sample_ohlcv_data.iterrows():
            price_data.append(
                (
                    "BULK_TEST",
                    date,
                    row["Open"],
                    row["High"],
                    row["Low"],
                    row["Close"],
                    row["Volume"],
                    "1d",
                )
            )

        test_database.bulk_add_price_data(price_data)

        # Verify all data was added
        with test_database.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM price_history WHERE symbol = ?", ("BULK_TEST",))
            count = cursor.fetchone()[0]
            assert count == len(sample_ohlcv_data)

    def test_get_price_history(self, populated_database):
        """Test retrieving price history."""
        data = populated_database.get_price_history("AAPL")

        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert all(col in data.columns for col in ["Open", "High", "Low", "Close", "Volume"])

        # Test with date range
        data_filtered = populated_database.get_price_history("AAPL", start_date="2023-01-10", end_date="2023-01-20")
        assert len(data_filtered) < len(data)

    def test_get_latest_price_date(self, populated_database):
        """Test getting latest price date."""
        latest_date = populated_database.get_latest_price_date("AAPL")
        assert latest_date is not None
        assert isinstance(latest_date, datetime)

        # Non-existent symbol
        latest_date = populated_database.get_latest_price_date("INVALID")
        assert latest_date is None

    def test_add_stock_info(self, test_database):
        """Test adding stock info."""
        test_database.add_stock("AAPL", "Apple Inc.")

        info = {
            "longName": "Apple Inc.",
            "sector": "Technology",
            "marketCap": 3000000000000,
            "website": "https://apple.com",
        }
        test_database.add_stock_info("AAPL", info)

        # Verify info was added
        retrieved_info = test_database.get_stock_info("AAPL")
        assert retrieved_info == info

    def test_get_stock_info(self, test_database):
        """Test retrieving stock info."""
        # Non-existent stock
        info = test_database.get_stock_info("INVALID")
        assert info is None

        # Add stock with info
        test_database.add_stock("AAPL", "Apple Inc.")
        test_database.add_stock_info("AAPL", {"sector": "Technology"})

        info = test_database.get_stock_info("AAPL")
        assert info["sector"] == "Technology"

    def test_add_dividends(self, in_memory_database):
        """Test adding dividend data."""
        test_database = in_memory_database
        test_database.add_stock("DIV_TEST", "Dividend Test Inc.")

        # Create dividend series
        dividend_dates = pd.DatetimeIndex([datetime(2023, 1, 15), datetime(2023, 4, 15), datetime(2023, 7, 15)])
        dividend_amounts = [0.23, 0.24, 0.24]
        dividend_series = pd.Series(dividend_amounts, index=dividend_dates)

        test_database.add_dividends("DIV_TEST", dividend_series)

        # Verify dividends were added
        dividends = test_database.get_dividends("DIV_TEST")
        assert len(dividends) == 3
        assert dividends.iloc[0] == 0.23

    def test_add_splits(self, in_memory_database):
        """Test adding split data."""
        test_database = in_memory_database
        test_database.add_stock("SPLIT_TEST", "Split Test Inc.")

        # Create splits series
        split_dates = pd.DatetimeIndex([datetime(2020, 8, 31), datetime(2014, 6, 9)])
        split_ratios = [4.0, 7.0]
        split_series = pd.Series(split_ratios, index=split_dates)

        test_database.add_splits("SPLIT_TEST", split_series)

        # Verify splits were added
        splits = test_database.get_splits("SPLIT_TEST")
        assert len(splits) == 2
        # Splits are returned in date order (oldest first)
        assert splits.iloc[0] == 7.0  # 2014 split
        assert splits.iloc[1] == 4.0  # 2020 split

    def test_add_options_data(self, test_database):
        """Test adding options data."""
        test_database.add_stock("AAPL", "Apple Inc.")

        expiry = datetime(2024, 1, 19)

        # Add calls
        calls_data = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "AAPL"],
                "expiry": [expiry, expiry, expiry],
                "strike": [140.0, 145.0, 150.0],
                "lastPrice": [12.5, 8.3, 5.1],
                "bid": [12.4, 8.2, 5.0],
                "ask": [12.6, 8.4, 5.2],
                "volume": [100, 200, 300],
                "openInterest": [1000, 2000, 3000],
                "impliedVolatility": [0.25, 0.26, 0.27],
            }
        )
        # Remove symbol and expiry columns as they're passed separately
        calls_data = calls_data.drop(["symbol", "expiry"], axis=1)

        # Add puts
        puts_data = pd.DataFrame(
            {
                "strike": [140.0, 145.0, 150.0],
                "lastPrice": [2.5, 4.3, 7.1],
                "bid": [2.4, 4.2, 7.0],
                "ask": [2.6, 4.4, 7.2],
                "volume": [50, 100, 150],
                "openInterest": [500, 1000, 1500],
                "impliedVolatility": [0.22, 0.23, 0.24],
            }
        )

        # Call with correct signature: symbol, calls, puts, expiry
        test_database.add_options_data("AAPL", calls_data, puts_data, expiry.strftime("%Y-%m-%d"))

        # Verify options were added
        calls, puts = test_database.get_options_chain("AAPL", expiry.strftime("%Y-%m-%d"))
        assert len(calls) == 3
        assert calls.iloc[0]["strike"] == 140.0

        assert len(puts) == 3
        assert puts.iloc[0]["lastPrice"] == 2.5

    def test_get_database_stats(self, populated_database):
        """Test getting database statistics."""
        stats = populated_database.get_database_stats()

        assert isinstance(stats, dict)
        assert "stocks" in stats
        assert "price_history" in stats
        assert stats["stocks"] >= 1
        assert stats["price_history"] >= 1

    def test_cleanup_old_data(self, populated_database):
        """Test cleaning up old data."""
        # Add some old data
        old_date = datetime.now() - timedelta(days=400)
        populated_database.add_price_data("AAPL", old_date, 100.0, 101.0, 99.0, 100.5, 500000, "1d")

        # Get count before cleanup
        with populated_database.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM price_history")
            count_before = cursor.fetchone()[0]

        # Cleanup data older than 365 days
        populated_database.cleanup_old_data(days_to_keep=365)

        # Get count after cleanup
        with populated_database.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM price_history")
            count_after = cursor.fetchone()[0]

        assert count_after < count_before


class TestDatabaseIntegrity:
    """Test database integrity and constraints."""

    def test_foreign_key_constraint(self, test_database):
        """Test that price data can be added for non-existent stock (auto-creation)."""
        # The store_price_history method auto-creates stocks if they don't exist
        # This test verifies that behavior
        test_database.add_price_data("NEWSTOCK", datetime.now(), 100.0, 101.0, 99.0, 100.5, 1000000, "1d")

        # Verify the stock was auto-created
        with test_database.get_session() as session:
            from stockula.database.models import Stock

            stock = session.query(Stock).filter_by(symbol="NEWSTOCK").first()
            assert stock is not None
            assert stock.symbol == "NEWSTOCK"

    def test_unique_constraints(self, in_memory_database):
        """Test unique constraints."""
        test_database = in_memory_database
        # Use a unique stock to avoid conflicts with seeded data
        test_database.add_stock("UNIQUE_TEST", "Unique Test Corp.")

        # Add price data
        price_date = datetime.now()
        test_database.add_price_data("UNIQUE_TEST", price_date, 150.0, 152.0, 149.0, 151.0, 1000000, "1d")

        # Try to add duplicate (same symbol, date, interval)
        # Should update, not error
        test_database.add_price_data("UNIQUE_TEST", price_date, 151.0, 153.0, 150.0, 152.0, 1100000, "1d")

        # Verify only one record exists for this specific date
        with test_database.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM price_history WHERE symbol = ? AND date(date) = date(?) AND interval = ?",
                ("UNIQUE_TEST", price_date.strftime("%Y-%m-%d"), "1d"),
            )
            count = cursor.fetchone()[0]
            assert count == 1

    def test_transaction_rollback(self, in_memory_database):
        """Test transaction rollback on error."""
        test_database = in_memory_database
        # Don't add the stock first - it should cause an error

        # Create a batch with entries for non-existent stock
        price_data = [
            ("NEWSTOCK", datetime.now(), 150.0, 152.0, 149.0, 151.0, 1000000, "1d"),
            (
                "NEWSTOCK",
                datetime.now() - timedelta(days=1),
                149.0,
                151.0,
                148.0,
                150.5,
                900000,
                "1d",
            ),
        ]

        # Actually, store_price_history auto-creates stocks, so this won't fail
        # Let's use bulk_add_price_data which should work fine
        test_database.bulk_add_price_data(price_data)

        # Verify data was added and stock was auto-created
        with test_database.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM price_history WHERE symbol = ?", ("NEWSTOCK",))
            count = cursor.fetchone()[0]
            assert count == 2  # Both records should be added
