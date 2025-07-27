"""Unit tests for data fetcher module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, call
import yfinance as yf

from stockula.data.fetcher import DataFetcher


class TestDataFetcherInitialization:
    """Test DataFetcher initialization."""

    def test_initialization_with_defaults(self):
        """Test DataFetcher initialization with default parameters."""
        with patch("stockula.data.fetcher.DatabaseManager"):
            fetcher = DataFetcher()
            assert fetcher.use_cache is True
            assert fetcher.db is not None

    def test_initialization_without_cache(self):
        """Test DataFetcher initialization without cache."""
        fetcher = DataFetcher(use_cache=False)
        assert fetcher.use_cache is False
        assert fetcher.db is None

    def test_initialization_with_custom_db_path(self):
        """Test DataFetcher initialization with custom database path."""
        with patch("stockula.data.fetcher.DatabaseManager") as mock_db:
            fetcher = DataFetcher(use_cache=True, db_path="custom.db")
            mock_db.assert_called_once_with("custom.db")
            assert fetcher.db is not None


class TestDataFetcherStockData:
    """Test stock data fetching functionality."""

    @pytest.fixture
    def mock_ticker(self):
        """Create a mock yfinance Ticker."""
        ticker = Mock(spec=yf.Ticker)
        # Mock history method
        ticker.history = Mock(
            return_value=pd.DataFrame(
                {
                    "Open": [100, 101, 102],
                    "High": [101, 102, 103],
                    "Low": [99, 100, 101],
                    "Close": [100.5, 101.5, 102.5],
                    "Volume": [1000000, 1100000, 1200000],
                },
                index=pd.date_range("2023-01-01", periods=3),
            )
        )
        # Mock info property
        ticker.info = {
            "longName": "Test Company",
            "sector": "Technology",
            "marketCap": 1000000000,
        }
        return ticker

    def test_get_stock_data_no_cache(self, mock_ticker):
        """Test fetching stock data without cache."""
        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False)
            data = fetcher.get_stock_data("TEST", start="2023-01-01", end="2023-01-03")

            assert isinstance(data, pd.DataFrame)
            assert len(data) == 3
            assert all(
                col in data.columns
                for col in ["Open", "High", "Low", "Close", "Volume"]
            )
            mock_ticker.history.assert_called_once()

    def test_get_stock_data_with_cache_miss(self, mock_ticker):
        """Test fetching stock data with cache miss."""
        mock_db = Mock()
        mock_db.get_price_history.return_value = pd.DataFrame()
        mock_db.has_data.return_value = False

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            with patch("yfinance.Ticker", return_value=mock_ticker):
                fetcher = DataFetcher(use_cache=True)
                data = fetcher.get_stock_data(
                    "TEST", start="2023-01-01", end="2023-01-03"
                )

                # Should try to get from cache first
                mock_db.get_price_history.assert_called_once()
                # Should fetch from yfinance
                mock_ticker.history.assert_called_once()
                # Should store in cache
                mock_db.store_price_history.assert_called_once()

    def test_get_stock_data_with_cache_hit(self, mock_ticker):
        """Test fetching stock data with cache hit."""
        cached_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [101, 102, 103],
                "Low": [99, 100, 101],
                "Close": [100.5, 101.5, 102.5],
                "Volume": [1000000, 1100000, 1200000],
            },
            index=pd.date_range("2023-01-01", periods=3),
        )

        mock_db = Mock()
        mock_db.has_data.return_value = True
        mock_db.get_price_history.return_value = cached_data

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            with patch("yfinance.Ticker", return_value=mock_ticker):
                fetcher = DataFetcher(use_cache=True)
                data = fetcher.get_stock_data(
                    "TEST", start="2023-01-01", end="2023-01-03"
                )

                # Should not call yfinance
                mock_ticker.history.assert_not_called()
                # Should return cached data
                assert data.equals(cached_data)

    def test_get_stock_data_force_refresh(self, mock_ticker):
        """Test forcing refresh bypasses cache."""
        mock_db = Mock()
        mock_db.has_data.return_value = True

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            with patch("yfinance.Ticker", return_value=mock_ticker):
                fetcher = DataFetcher(use_cache=True)
                data = fetcher.get_stock_data(
                    "TEST", start="2023-01-01", end="2023-01-03", force_refresh=True
                )

                # Should not check cache
                mock_db.has_data.assert_not_called()
                # Should fetch from yfinance
                mock_ticker.history.assert_called_once()

    def test_get_stock_data_empty_response(self):
        """Test handling empty response from yfinance."""
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.history.return_value = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False)
            data = fetcher.get_stock_data("INVALID")

            assert isinstance(data, pd.DataFrame)
            assert data.empty

    def test_get_stock_data_exception_handling(self):
        """Test exception handling in stock data fetching."""
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.history.side_effect = Exception("Network error")

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False)
            # Exception should propagate since there's no explicit handling
            with pytest.raises(Exception, match="Network error"):
                fetcher.get_stock_data("TEST")


class TestDataFetcherRealtimePrice:
    """Test real-time price fetching."""

    def test_get_realtime_price_success(self):
        """Test successful real-time price fetch."""
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.history.return_value = pd.DataFrame(
            {"Close": [150.50]}, index=pd.date_range("2023-01-01", periods=1)
        )

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False)
            price = fetcher.get_realtime_price("TEST")

            assert price == 150.50

    def test_get_realtime_price_no_data(self):
        """Test real-time price when no data available."""
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.history.return_value = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False)
            price = fetcher.get_realtime_price("TEST")

            assert price is None

    def test_get_realtime_price_exception(self):
        """Test exception handling in real-time price fetch."""
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.history.side_effect = Exception("API error")

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False)
            # Exception should propagate
            with pytest.raises(Exception, match="API error"):
                fetcher.get_realtime_price("TEST")


class TestDataFetcherInfo:
    """Test stock info fetching."""

    def test_get_info_no_cache(self):
        """Test fetching stock info without cache."""
        expected_info = {
            "longName": "Test Company",
            "sector": "Technology",
            "marketCap": 1000000000,
        }
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.info = expected_info

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False)
            info = fetcher.get_info("TEST")

            assert info == expected_info

    def test_get_info_with_cache_miss(self):
        """Test fetching stock info with cache miss."""
        expected_info = {
            "longName": "Test Company",
            "sector": "Technology",
            "marketCap": 1000000000,
        }
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.info = expected_info

        mock_db = Mock()
        mock_db.get_stock_info.return_value = None

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            with patch("yfinance.Ticker", return_value=mock_ticker):
                fetcher = DataFetcher(use_cache=True)
                info = fetcher.get_info("TEST")

                # Should check cache
                mock_db.get_stock_info.assert_called_once_with("TEST")
                # Should store in cache
                mock_db.store_stock_info.assert_called_once_with("TEST", expected_info)
                assert info == expected_info

    def test_get_info_with_cache_hit(self):
        """Test fetching stock info with cache hit."""
        cached_info = {
            "longName": "Cached Company",
            "sector": "Finance",
            "marketCap": 500000000,
        }

        mock_db = Mock()
        mock_db.get_stock_info.return_value = cached_info

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            fetcher = DataFetcher(use_cache=True)
            info = fetcher.get_info("TEST")

            assert info == cached_info

    def test_get_info_force_refresh(self):
        """Test forcing refresh of stock info."""
        fresh_info = {
            "longName": "Fresh Company",
            "sector": "Healthcare",
            "marketCap": 2000000000,
        }
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.info = fresh_info

        mock_db = Mock()
        mock_db.get_stock_info.return_value = {"old": "info"}

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            with patch("yfinance.Ticker", return_value=mock_ticker):
                fetcher = DataFetcher(use_cache=True)
                info = fetcher.get_info("TEST", force_refresh=True)

                # Should not check cache
                mock_db.get_stock_info.assert_not_called()
                # Should fetch fresh
                assert info == fresh_info


class TestDataFetcherDividendsAndSplits:
    """Test dividends and splits fetching."""

    def test_get_dividends(self):
        """Test fetching dividends."""
        dividend_series = pd.Series(
            [0.25, 0.30], index=pd.to_datetime(["2023-01-15", "2023-04-15"])
        )
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.dividends = dividend_series

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False)
            dividends = fetcher.get_dividends("TEST")

            assert isinstance(dividends, pd.Series)
            assert len(dividends) == 2
            assert dividends.iloc[0] == 0.25

    def test_get_splits(self):
        """Test fetching stock splits."""
        split_series = pd.Series(
            [2.0, 3.0], index=pd.to_datetime(["2022-06-01", "2023-06-01"])
        )
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.splits = split_series

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False)
            splits = fetcher.get_splits("TEST")

            assert isinstance(splits, pd.Series)
            assert len(splits) == 2
            assert splits.iloc[0] == 2.0


class TestDataFetcherOptions:
    """Test options chain fetching."""

    def test_get_options_chain(self):
        """Test fetching options chain."""
        # Create mock options data
        calls_df = pd.DataFrame(
            {
                "strike": [100, 105, 110],
                "lastPrice": [5.0, 3.0, 1.5],
                "volume": [100, 200, 150],
            }
        )
        puts_df = pd.DataFrame(
            {
                "strike": [90, 95, 100],
                "lastPrice": [1.0, 2.0, 4.0],
                "volume": [50, 75, 100],
            }
        )

        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.options = ("2024-01-19", "2024-02-16")
        mock_ticker.option_chain.return_value = Mock(calls=calls_df, puts=puts_df)

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False)
            calls, puts = fetcher.get_options_chain("TEST", "2024-01-19")

            assert isinstance(calls, pd.DataFrame)
            assert isinstance(puts, pd.DataFrame)
            assert len(calls) == 3
            assert len(puts) == 3

    def test_get_options_chain_no_data(self):
        """Test fetching options chain with no data."""
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.options = ()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False)
            calls, puts = fetcher.get_options_chain("TEST")

            assert calls.empty
            assert puts.empty


class TestDataFetcherBulkOperations:
    """Test bulk data fetching operations."""

    def test_fetch_and_store_all_data(self):
        """Test fetching and storing all data types."""
        # Create mock ticker with all data types
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.history.return_value = pd.DataFrame(
            {
                "Open": [100],
                "High": [101],
                "Low": [99],
                "Close": [100.5],
                "Volume": [1000000],
            },
            index=pd.date_range("2023-01-01", periods=1),
        )
        mock_ticker.info = {"longName": "Test Company"}
        mock_ticker.dividends = pd.Series([0.25], index=pd.to_datetime(["2023-01-15"]))
        mock_ticker.splits = pd.Series([2.0], index=pd.to_datetime(["2023-06-01"]))
        mock_ticker.options = ("2024-01-19",)
        mock_ticker.option_chain.return_value = Mock(
            calls=pd.DataFrame({"strike": [100]}), puts=pd.DataFrame({"strike": [100]})
        )

        mock_db = Mock()

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            with patch("yfinance.Ticker", return_value=mock_ticker):
                fetcher = DataFetcher(use_cache=True)
                fetcher.fetch_and_store_all_data("TEST", start="2023-01-01")

                # Should store all data types
                mock_db.store_stock_info.assert_called_once()
                mock_db.store_price_history.assert_called_once()
                mock_db.store_dividends.assert_called_once()
                mock_db.store_splits.assert_called_once()
                mock_db.store_options_chain.assert_called_once()

    def test_get_multiple_stocks(self):
        """Test fetching data for multiple stocks."""
        symbols = ["AAPL", "GOOGL", "MSFT"]

        # Create different data for each stock
        mock_data = {
            "AAPL": pd.DataFrame(
                {"Close": [150, 151, 152]}, index=pd.date_range("2023-01-01", periods=3)
            ),
            "GOOGL": pd.DataFrame(
                {"Close": [100, 101, 102]}, index=pd.date_range("2023-01-01", periods=3)
            ),
            "MSFT": pd.DataFrame(
                {"Close": [300, 301, 302]}, index=pd.date_range("2023-01-01", periods=3)
            ),
        }

        with patch("yfinance.Ticker") as mock_ticker:
            # Mock different data for each symbol call
            def side_effect(symbol):
                mock_instance = Mock()
                mock_instance.history.return_value = mock_data[symbol]
                return mock_instance

            mock_ticker.side_effect = side_effect

            fetcher = DataFetcher(use_cache=False)
            results = fetcher.get_multiple_stocks(symbols)

            assert isinstance(results, dict)
            assert len(results) == 3
            assert all(symbol in results for symbol in symbols)
            assert len(results["AAPL"]) == 3


class TestDataFetcherCacheManagement:
    """Test cache management functionality."""

    def test_cache_partial_data_fetch(self):
        """Test fetching when cache has partial data."""
        # Cache has data for Jan 1-15
        cached_data = pd.DataFrame(
            {
                "Open": [100] * 15,
                "High": [101] * 15,
                "Low": [99] * 15,
                "Close": [100.5] * 15,
                "Volume": [1000000] * 15,
            },
            index=pd.date_range("2023-01-01", periods=15),
        )

        # Fresh data for Jan 16-31
        fresh_data = pd.DataFrame(
            {
                "Open": [102] * 16,
                "High": [103] * 16,
                "Low": [101] * 16,
                "Close": [102.5] * 16,
                "Volume": [1100000] * 16,
            },
            index=pd.date_range("2023-01-16", periods=16),
        )

        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.history.return_value = fresh_data

        mock_db = Mock()
        # First call returns partial data, second call returns nothing (for the gap)
        mock_db.get_price_history.side_effect = [cached_data, pd.DataFrame()]
        mock_db.has_data.return_value = False

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            with patch("yfinance.Ticker", return_value=mock_ticker):
                fetcher = DataFetcher(use_cache=True)
                data = fetcher.get_stock_data(
                    "TEST", start="2023-01-01", end="2023-01-31"
                )

                # Should return combined data
                assert len(data) >= len(cached_data)  # At least cached data


class TestDataFetcherHelpers:
    """Test helper methods and edge cases."""

    def test_data_fetcher_attributes(self):
        """Test DataFetcher has expected attributes."""
        with patch("stockula.data.fetcher.DatabaseManager"):
            fetcher = DataFetcher(use_cache=True, db_path="test.db")
            assert hasattr(fetcher, "use_cache")
            assert hasattr(fetcher, "db")
            assert fetcher.use_cache is True

    def test_invalid_date_formats(self):
        """Test handling of invalid date formats."""
        fetcher = DataFetcher(use_cache=False)

        # Should handle various date formats
        with patch("yfinance.Ticker") as mock_yf:
            mock_ticker = Mock()
            mock_ticker.history.return_value = pd.DataFrame()
            mock_yf.return_value = mock_ticker

            # Test different date formats
            fetcher.get_stock_data("TEST", start="2023/01/01", end="2023/12/31")
            fetcher.get_stock_data("TEST", start="01-01-2023", end="31-12-2023")
            fetcher.get_stock_data(
                "TEST", start=datetime(2023, 1, 1), end=datetime(2023, 12, 31)
            )

            # All should work
            assert mock_ticker.history.call_count >= 3

    def test_concurrent_requests(self):
        """Test handling multiple requests for same symbol."""
        with patch("stockula.data.fetcher.DatabaseManager") as mock_db_class:
            mock_db = Mock()
            mock_db.get_price_history.return_value = pd.DataFrame()
            mock_db.has_data.return_value = False
            mock_db_class.return_value = mock_db

            fetcher = DataFetcher(use_cache=True)

            with patch("yfinance.Ticker") as mock_yf:
                mock_ticker = Mock()
                mock_ticker.history.return_value = pd.DataFrame(
                    {"Close": [100, 101, 102]},
                    index=pd.date_range("2023-01-01", periods=3),
                )
                mock_yf.return_value = mock_ticker

                # Multiple calls for same symbol
                data1 = fetcher.get_stock_data("TEST")
                data2 = fetcher.get_stock_data("TEST")

                assert len(data1) == len(data2)
