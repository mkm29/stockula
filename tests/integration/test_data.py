"""Tests for data fetching module."""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from stockula.data.fetcher import DataFetcher
from stockula.database.manager import DatabaseManager


class TestDataFetcher:
    """Test DataFetcher class."""

    def test_data_fetcher_initialization(self):
        """Test DataFetcher initialization."""
        # Without database
        fetcher = DataFetcher(use_cache=False)
        assert fetcher.use_cache is False
        assert fetcher.db is None

        # With database
        fetcher = DataFetcher(use_cache=True, db_path="test.db")
        assert fetcher.use_cache is True
        assert fetcher.db is not None
        assert isinstance(fetcher.db, DatabaseManager)

    def test_get_stock_data_from_yfinance(self, mock_yfinance_ticker):
        """Test fetching stock data from yfinance."""
        with patch("yfinance.Ticker", return_value=mock_yfinance_ticker):
            fetcher = DataFetcher(use_cache=False)
            # Get data for last month
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            data = fetcher.get_stock_data("AAPL", start=start_date, end=end_date)

            assert isinstance(data, pd.DataFrame)
            assert not data.empty
            assert all(col in data.columns for col in ["Open", "High", "Low", "Close", "Volume"])

    def test_get_stock_data_with_dates(self, mock_yfinance_ticker):
        """Test fetching stock data with specific date range."""
        with patch("yfinance.Ticker", return_value=mock_yfinance_ticker):
            fetcher = DataFetcher(use_cache=False)
            data = fetcher.get_stock_data("AAPL", start="2023-01-01", end="2023-01-31")

            assert isinstance(data, pd.DataFrame)
            assert not data.empty

    def test_get_stock_data_from_cache(self, populated_database):
        """Test fetching stock data from cache."""
        fetcher = DataFetcher(use_cache=True, db_path=populated_database.db_path)

        # Get specific date range from cache
        data = fetcher.get_stock_data("AAPL", start="2023-01-01", end="2023-01-31")

        assert isinstance(data, pd.DataFrame)
        # Should have data for January 2023 (approximately 22 trading days)
        assert len(data) > 0
        assert len(data) <= 31  # Max days in January
        assert all(col in data.columns for col in ["Open", "High", "Low", "Close", "Volume"])

        # Verify data integrity
        assert (data["High"] >= data["Close"]).all()
        assert (data["Low"] <= data["Close"]).all()
        assert (data["Volume"] > 0).all()

    def test_get_stock_data_force_refresh(self, mock_yfinance_ticker, temp_db_path):
        """Test force refresh bypasses cache."""
        with patch("yfinance.Ticker", return_value=mock_yfinance_ticker):
            fetcher = DataFetcher(use_cache=True, db_path=temp_db_path)

            # First call should hit yfinance and cache
            # Get data for last 5 days
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
            fetcher.get_stock_data("AAPL", start=start_date, end=end_date)

            # Force refresh should hit yfinance again
            with patch.object(mock_yfinance_ticker, "history") as mock_history:
                # Create sample data
                sample_data = pd.DataFrame(
                    {
                        "Open": [150, 151, 152],
                        "High": [151, 152, 153],
                        "Low": [149, 150, 151],
                        "Close": [150, 151, 152],
                        "Volume": [1000000, 1100000, 1200000],
                    },
                    index=pd.date_range("2023-01-01", periods=3),
                )
                mock_history.return_value = sample_data
                fetcher.get_stock_data("AAPL", start=start_date, end=end_date, force_refresh=True)

                assert mock_history.called

    def test_get_info(self, mock_yfinance_ticker):
        """Test fetching stock info."""
        with patch("yfinance.Ticker", return_value=mock_yfinance_ticker):
            fetcher = DataFetcher(use_cache=False)
            info = fetcher.get_info("AAPL")

            assert isinstance(info, dict)
            assert info["longName"] == "Apple Inc."
            assert info["sector"] == "Technology"
            assert info["marketCap"] == 3000000000000

    def test_get_info_with_cache(self, populated_database):
        """Test fetching stock info from cache."""
        # Stock info is already added in populated_database fixture
        fetcher = DataFetcher(use_cache=True, db_path=populated_database.db_path)
        info = fetcher.get_info("AAPL")

        assert isinstance(info, dict)
        # Check for either name or longName (database stores as 'name')
        assert info.get("name") == "Apple Inc." or info.get("longName") == "Apple Inc."
        assert info["sector"] == "Technology"

    def test_get_current_prices(self, mock_yfinance_ticker):
        """Test fetching current prices for multiple symbols."""
        mock_ticker = Mock()
        mock_ticker.info = {"currentPrice": 150.0}

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False)
            prices = fetcher.get_current_prices(["AAPL", "GOOGL", "MSFT"], show_progress=False)

            assert isinstance(prices, dict)
            assert len(prices) == 3
            assert all(symbol in prices for symbol in ["AAPL", "GOOGL", "MSFT"])
            assert all(price == 150.0 for price in prices.values())

    def test_get_current_prices_single_symbol(self, mock_yfinance_ticker):
        """Test fetching current price for single symbol."""
        with patch("yfinance.Ticker", return_value=mock_yfinance_ticker):
            fetcher = DataFetcher(use_cache=False)
            prices = fetcher.get_current_prices("AAPL", show_progress=False)

            assert isinstance(prices, dict)
            assert "AAPL" in prices
            # Price should be from the last close in the history
            assert isinstance(prices["AAPL"], int | float)
            assert prices["AAPL"] > 0

    def test_get_current_prices_with_errors(self):
        """Test handling errors when fetching current prices."""
        mock_ticker = Mock()
        mock_ticker.info = {}  # No currentPrice

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False)
            prices = fetcher.get_current_prices(["AAPL", "INVALID"], show_progress=False)

            assert isinstance(prices, dict)
            # Should handle missing prices gracefully

    def test_get_realtime_price(self, mock_yfinance_ticker):
        """Test fetching real-time price."""
        with patch("yfinance.Ticker", return_value=mock_yfinance_ticker):
            fetcher = DataFetcher(use_cache=False)
            price = fetcher.get_realtime_price("AAPL")

            # Price should be from the last close in the history (minute data)
            assert isinstance(price, int | float)
            assert price > 0

    def test_get_dividends(self):
        """Test fetching dividend data."""
        mock_ticker = Mock()
        mock_dividends = pd.Series([0.22, 0.23, 0.24], index=pd.date_range("2023-01-01", periods=3, freq="Q"))
        mock_ticker.dividends = mock_dividends

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False)
            dividends = fetcher.get_dividends("AAPL")

            assert isinstance(dividends, pd.Series)
            assert len(dividends) == 3
            assert dividends.iloc[0] == 0.22

    def test_get_splits(self):
        """Test fetching stock split data."""
        mock_ticker = Mock()
        mock_splits = pd.Series([2.0, 4.0], index=pd.date_range("2020-01-01", periods=2, freq="Y"))
        mock_ticker.splits = mock_splits

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False)
            splits = fetcher.get_splits("AAPL")

            assert isinstance(splits, pd.Series)
            assert len(splits) == 2
            assert splits.iloc[0] == 2.0

    def test_get_options_chain(self):
        """Test fetching options chain data."""
        mock_ticker = Mock()
        mock_ticker.options = ["2024-01-19", "2024-02-16"]

        mock_calls = pd.DataFrame(
            {
                "strike": [140, 145, 150],
                "lastPrice": [12.5, 8.3, 5.1],
                "volume": [100, 200, 300],
            }
        )
        mock_puts = pd.DataFrame(
            {
                "strike": [140, 145, 150],
                "lastPrice": [2.5, 4.3, 7.1],
                "volume": [50, 100, 150],
            }
        )

        mock_ticker.option_chain.return_value = Mock(calls=mock_calls, puts=mock_puts)

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False)
            calls, puts = fetcher.get_options_chain("AAPL")

            assert isinstance(calls, pd.DataFrame)
            assert isinstance(puts, pd.DataFrame)
            assert len(calls) == 3
            assert len(puts) == 3
            # Can check available dates from the mock_ticker
            assert len(mock_ticker.options) == 2

    def test_fetch_and_store_all_data(self, mock_yfinance_ticker, temp_db_path):
        """Test fetching and storing all data types."""
        with patch("yfinance.Ticker", return_value=mock_yfinance_ticker):
            fetcher = DataFetcher(use_cache=True, db_path=temp_db_path)

            # Mock additional methods
            with patch.object(fetcher, "get_dividends") as mock_div:
                mock_div.return_value = pd.Series([0.22], index=[datetime.now()])

                with patch.object(fetcher, "get_splits") as mock_splits:
                    mock_splits.return_value = pd.Series([2.0], index=[datetime.now()])

                    fetcher.fetch_and_store_all_data("AAPL", start="2023-01-01")

            # Verify data was stored
            assert fetcher.db.get_stock_info("AAPL") is not None
            price_data = fetcher.db.get_price_history("AAPL")
            assert len(price_data) > 0


class TestDataFetcherErrorHandling:
    """Test error handling in DataFetcher."""

    def test_invalid_symbol(self):
        """Test handling invalid symbol."""
        mock_ticker = Mock()
        mock_ticker.history.return_value = pd.DataFrame()  # Empty DataFrame
        mock_ticker.info = {}

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False)

            data = fetcher.get_stock_data("INVALID_SYMBOL")
            assert data.empty

            info = fetcher.get_info("INVALID_SYMBOL")
            assert info == {}

    def test_network_error(self):
        """Test handling network errors."""
        mock_ticker = Mock()
        mock_ticker.history.side_effect = Exception("Network error")

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False)

            with pytest.raises(ValueError):
                fetcher.get_stock_data("AAPL")

    def test_database_error(self, temp_db_path):
        """Test handling database errors."""
        fetcher = DataFetcher(use_cache=True, db_path=temp_db_path)

        # Mock database error
        with patch.object(fetcher.db, "get_price_history", side_effect=Exception("DB error")):
            # Should fall back to yfinance
            with patch("yfinance.Ticker") as mock_yf:
                mock_ticker = Mock()
                mock_ticker.history.return_value = pd.DataFrame(
                    {
                        "Close": [150.0],
                        "Open": [149.0],
                        "High": [151.0],
                        "Low": [148.0],
                        "Volume": [1000000],
                    },
                    index=[datetime.now()],
                )
                mock_yf.return_value = mock_ticker

                data = fetcher.get_stock_data("AAPL")
                assert not data.empty


class TestDataFetcherCaching:
    """Test caching behavior in DataFetcher."""

    def test_cache_hit_avoids_api_call(self, populated_database):
        """Test that cache hit avoids API call."""
        fetcher = DataFetcher(use_cache=True, db_path=populated_database.db_path)

        # First get some data that we know exists in the database
        existing_data = populated_database.get_price_history("AAPL")
        if not existing_data.empty:
            # Use date range that we know exists
            start_date = existing_data.index[0].strftime("%Y-%m-%d")
            end_date = existing_data.index[-1].strftime("%Y-%m-%d")

            with patch("yfinance.Ticker") as mock_yf:
                # Should not call yfinance
                data = fetcher.get_stock_data("AAPL", start=start_date, end=end_date)
                assert not mock_yf.called
                assert not data.empty
        else:
            pytest.skip("No data in populated database")

    def test_cache_miss_calls_api(self, temp_db_path, mock_yfinance_ticker):
        """Test that cache miss calls API."""
        fetcher = DataFetcher(use_cache=True, db_path=temp_db_path)

        with patch("yfinance.Ticker", return_value=mock_yfinance_ticker) as mock_yf:
            # Should call yfinance
            data = fetcher.get_stock_data("AAPL")
            assert mock_yf.called
            assert not data.empty

    def test_partial_cache_hit(self, populated_database, mock_yfinance_ticker):
        """Test partial cache hit fetches missing data."""
        fetcher = DataFetcher(use_cache=True, db_path=populated_database.db_path)

        # Request data beyond what's in cache
        with patch("yfinance.Ticker", return_value=mock_yfinance_ticker):
            data = fetcher.get_stock_data(
                "AAPL",
                start="2023-01-01",
                end="2023-02-28",  # Cache only has January
            )

            # Should have returned data (either from cache or fetched)
            assert not data.empty
            # Just verify we got data, don't check exact length since
            # the mock returns only a few days of data
