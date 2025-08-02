"""Unit tests for data fetcher module."""

from datetime import datetime
from unittest.mock import Mock, PropertyMock, patch

import pandas as pd
import pytest
import yfinance as yf

from stockula.data.fetcher import DataFetcher


@pytest.fixture
def mock_logging_manager():
    """Create a mock logging manager."""
    logger = Mock()
    logger.debug = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    return logger


class TestDataFetcherInitialization:
    """Test DataFetcher initialization."""

    def test_initialization_with_defaults(self, mock_logging_manager):
        """Test DataFetcher initialization with default parameters."""
        with patch("stockula.data.fetcher.DatabaseManager"):
            fetcher = DataFetcher(logging_manager=mock_logging_manager)
            assert fetcher.use_cache is True
            assert fetcher.db is not None

    def test_initialization_without_cache(self, mock_logging_manager):
        """Test DataFetcher initialization without cache."""
        fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
        assert fetcher.use_cache is False
        assert fetcher.db is None

    def test_initialization_with_custom_db_path(self, mock_logging_manager):
        """Test DataFetcher initialization with custom database path."""
        with patch("stockula.data.fetcher.DatabaseManager") as mock_db:
            fetcher = DataFetcher(use_cache=True, db_path="custom.db", logging_manager=mock_logging_manager)
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

    def test_get_stock_data_no_cache(self, mock_logging_manager, mock_ticker):
        """Test fetching stock data without cache."""
        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            data = fetcher.get_stock_data("TEST", start="2023-01-01", end="2023-01-03")

            assert isinstance(data, pd.DataFrame)
            assert len(data) == 3
            assert all(col in data.columns for col in ["Open", "High", "Low", "Close", "Volume"])
            mock_ticker.history.assert_called_once()

    def test_get_stock_data_with_cache_miss(self, mock_logging_manager, mock_ticker):
        """Test fetching stock data with cache miss."""
        mock_db = Mock()
        mock_db.get_price_history.return_value = pd.DataFrame()
        mock_db.has_data.return_value = False

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            with patch("yfinance.Ticker", return_value=mock_ticker):
                fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
                fetcher.get_stock_data("TEST", start="2023-01-01", end="2023-01-03")

                # Should try to get from cache first
                mock_db.get_price_history.assert_called_once()
                # Should fetch from yfinance
                mock_ticker.history.assert_called_once()
                # Should store in cache
                mock_db.store_price_history.assert_called_once()

    def test_get_stock_data_with_cache_hit(self, mock_logging_manager, mock_ticker):
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
                fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
                data = fetcher.get_stock_data("TEST", start="2023-01-01", end="2023-01-03")

                # Should not call yfinance
                mock_ticker.history.assert_not_called()
                # Should return cached data
                assert data.equals(cached_data)

    def test_get_stock_data_force_refresh(self, mock_logging_manager, mock_ticker):
        """Test forcing refresh bypasses cache."""
        mock_db = Mock()
        mock_db.has_data.return_value = True

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            with patch("yfinance.Ticker", return_value=mock_ticker):
                fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
                fetcher.get_stock_data("TEST", start="2023-01-01", end="2023-01-03", force_refresh=True)

                # Should not check cache
                mock_db.has_data.assert_not_called()
                # Should fetch from yfinance
                mock_ticker.history.assert_called_once()

    def test_get_stock_data_empty_response(self, mock_logging_manager):
        """Test handling empty response from yfinance."""
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.history.return_value = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            data = fetcher.get_stock_data("INVALID")

            assert isinstance(data, pd.DataFrame)
            assert data.empty

    def test_get_stock_data_exception_handling(self, mock_logging_manager):
        """Test exception handling in stock data fetching."""
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.history.side_effect = Exception("Network error")

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            # Exception should propagate since there's no explicit handling
            with pytest.raises(Exception, match="Network error"):
                fetcher.get_stock_data("TEST")


class TestDataFetcherRealtimePrice:
    """Test real-time price fetching."""

    def test_get_realtime_price_success(self, mock_logging_manager):
        """Test successful real-time price fetch."""
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.history.return_value = pd.DataFrame(
            {"Close": [150.50]}, index=pd.date_range("2023-01-01", periods=1)
        )

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            price = fetcher.get_realtime_price("TEST")

            assert price == 150.50

    def test_get_realtime_price_no_data(self, mock_logging_manager):
        """Test real-time price when no data available."""
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.history.return_value = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            price = fetcher.get_realtime_price("TEST")

            assert price is None

    def test_get_realtime_price_exception(self, mock_logging_manager):
        """Test exception handling in real-time price fetch."""
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.history.side_effect = Exception("API error")

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            # Exception should propagate
            with pytest.raises(Exception, match="API error"):
                fetcher.get_realtime_price("TEST")


class TestDataFetcherInfo:
    """Test stock info fetching."""

    def test_get_info_no_cache(self, mock_logging_manager):
        """Test fetching stock info without cache."""
        expected_info = {
            "longName": "Test Company",
            "sector": "Technology",
            "marketCap": 1000000000,
        }
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.info = expected_info

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            info = fetcher.get_info("TEST")

            assert info == expected_info

    def test_get_info_with_cache_miss(self, mock_logging_manager):
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
                fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
                info = fetcher.get_info("TEST")

                # Should check cache
                mock_db.get_stock_info.assert_called_once_with("TEST")
                # Should store in cache
                mock_db.store_stock_info.assert_called_once_with("TEST", expected_info)
                assert info == expected_info

    def test_get_info_with_cache_hit(self, mock_logging_manager):
        """Test fetching stock info with cache hit."""
        cached_info = {
            "longName": "Cached Company",
            "sector": "Finance",
            "marketCap": 500000000,
        }

        mock_db = Mock()
        mock_db.get_stock_info.return_value = cached_info

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
            info = fetcher.get_info("TEST")

            assert info == cached_info

    def test_get_info_force_refresh(self, mock_logging_manager):
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
                fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
                info = fetcher.get_info("TEST", force_refresh=True)

                # Should not check cache
                mock_db.get_stock_info.assert_not_called()
                # Should fetch fresh
                assert info == fresh_info


class TestDataFetcherDividendsAndSplits:
    """Test dividends and splits fetching."""

    def test_get_dividends(self, mock_logging_manager):
        """Test fetching dividends."""
        dividend_series = pd.Series([0.25, 0.30], index=pd.to_datetime(["2023-01-15", "2023-04-15"]))
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.dividends = dividend_series

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            dividends = fetcher.get_dividends("TEST")

            assert isinstance(dividends, pd.Series)
            assert len(dividends) == 2
            assert dividends.iloc[0] == 0.25

    def test_get_splits(self, mock_logging_manager):
        """Test fetching stock splits."""
        split_series = pd.Series([2.0, 3.0], index=pd.to_datetime(["2022-06-01", "2023-06-01"]))
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.splits = split_series

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            splits = fetcher.get_splits("TEST")

            assert isinstance(splits, pd.Series)
            assert len(splits) == 2
            assert splits.iloc[0] == 2.0


class TestDataFetcherOptions:
    """Test options chain fetching."""

    def test_get_options_chain(self, mock_logging_manager):
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
            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            calls, puts = fetcher.get_options_chain("TEST", "2024-01-19")

            assert isinstance(calls, pd.DataFrame)
            assert isinstance(puts, pd.DataFrame)
            assert len(calls) == 3
            assert len(puts) == 3

    def test_get_options_chain_no_data(self, mock_logging_manager):
        """Test fetching options chain with no data."""
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.options = ()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            calls, puts = fetcher.get_options_chain("TEST")

            assert calls.empty
            assert puts.empty


class TestDataFetcherBulkOperations:
    """Test bulk data fetching operations."""

    def test_fetch_and_store_all_data(self, mock_logging_manager):
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
                fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
                fetcher.fetch_and_store_all_data("TEST", start="2023-01-01")

                # Should store all data types
                mock_db.store_stock_info.assert_called_once()
                mock_db.store_price_history.assert_called_once()
                mock_db.store_dividends.assert_called_once()
                mock_db.store_splits.assert_called_once()
                mock_db.store_options_chain.assert_called_once()

    def test_get_multiple_stocks(self, mock_logging_manager):
        """Test fetching data for multiple stocks."""
        symbols = ["AAPL", "GOOGL", "MSFT"]

        # Create different data for each stock
        mock_data = {
            "AAPL": pd.DataFrame({"Close": [150, 151, 152]}, index=pd.date_range("2023-01-01", periods=3)),
            "GOOGL": pd.DataFrame({"Close": [100, 101, 102]}, index=pd.date_range("2023-01-01", periods=3)),
            "MSFT": pd.DataFrame({"Close": [300, 301, 302]}, index=pd.date_range("2023-01-01", periods=3)),
        }

        with patch("yfinance.Ticker") as mock_ticker:
            # Mock different data for each symbol call
            def side_effect(symbol):
                mock_instance = Mock()
                mock_instance.history.return_value = mock_data[symbol]
                return mock_instance

            mock_ticker.side_effect = side_effect

            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            results = fetcher.get_multiple_stocks(symbols)

            assert isinstance(results, dict)
            assert len(results) == 3
            assert all(symbol in results for symbol in symbols)
            assert len(results["AAPL"]) == 3


class TestDataFetcherCacheManagement:
    """Test cache management functionality."""

    def test_cache_partial_data_fetch(self, mock_logging_manager):
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
                fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
                data = fetcher.get_stock_data("TEST", start="2023-01-01", end="2023-01-31")

                # Should return combined data
                assert len(data) >= len(cached_data)  # At least cached data


class TestDataFetcherHelpers:
    """Test helper methods and edge cases."""

    def test_data_fetcher_attributes(self, mock_logging_manager):
        """Test DataFetcher has expected attributes."""
        with patch("stockula.data.fetcher.DatabaseManager"):
            fetcher = DataFetcher(use_cache=True, db_path="test.db", logging_manager=mock_logging_manager)
            assert hasattr(fetcher, "use_cache")
            assert hasattr(fetcher, "db")
            assert fetcher.use_cache is True

    def test_invalid_date_formats(self, mock_logging_manager):
        """Test handling of invalid date formats."""
        fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)

        # Should handle various date formats
        with patch("yfinance.Ticker") as mock_yf:
            mock_ticker = Mock()
            mock_ticker.history.return_value = pd.DataFrame()
            mock_yf.return_value = mock_ticker

            # Test different date formats
            fetcher.get_stock_data("TEST", start="2023/01/01", end="2023/12/31")
            fetcher.get_stock_data("TEST", start="01-01-2023", end="31-12-2023")
            fetcher.get_stock_data("TEST", start=datetime(2023, 1, 1), end=datetime(2023, 12, 31))

            # All should work
            assert mock_ticker.history.call_count >= 3

    def test_concurrent_requests(self, mock_logging_manager):
        """Test handling multiple requests for same symbol."""
        with patch("stockula.data.fetcher.DatabaseManager") as mock_db_class:
            mock_db = Mock()
            mock_db.get_price_history.return_value = pd.DataFrame()
            mock_db.has_data.return_value = False
            mock_db_class.return_value = mock_db

            fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)

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


class TestDataFetcherErrorHandling:
    """Test comprehensive error handling scenarios."""

    def test_database_error_fallback(self, mock_logging_manager):
        """Test fallback to yfinance when database fails - covers lines 67-69."""
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

        mock_db = Mock()
        mock_db.get_price_history.side_effect = Exception("Database connection failed")
        mock_db.has_data.side_effect = Exception("Database error")

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            with patch("yfinance.Ticker", return_value=mock_ticker):
                with patch("builtins.print") as mock_print:
                    fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
                    data = fetcher.get_stock_data("TEST")

                    # Should print database error message
                    mock_print.assert_called_with(
                        "Database error, falling back to yfinance: Database connection failed"
                    )
                    # Should still return data from yfinance
                    assert not data.empty
                    assert len(data) == 1

    def test_get_multiple_stocks_with_errors(self, mock_logging_manager):
        """Test get_multiple_stocks with individual stock errors - covers lines 123-124."""
        with patch("yfinance.Ticker") as mock_yf:

            def side_effect(symbol):
                mock_ticker = Mock()
                if symbol == "INVALID":
                    mock_ticker.history.side_effect = Exception(f"Error fetching data for {symbol}")
                else:
                    mock_ticker.history.return_value = pd.DataFrame(
                        {"Close": [100]}, index=pd.date_range("2023-01-01", periods=1)
                    )
                return mock_ticker

            mock_yf.side_effect = side_effect

            with patch("builtins.print") as mock_print:
                fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
                results = fetcher.get_multiple_stocks(["AAPL", "INVALID", "GOOGL"])

                # Should print error for invalid symbol
                mock_print.assert_called_with("Error fetching data for INVALID: Error fetching data for INVALID")
                # Should return data for valid symbols only
                assert len(results) == 2
                assert "AAPL" in results
                assert "GOOGL" in results
                assert "INVALID" not in results


class TestDataFetcherCurrentPrices:
    """Test current price fetching with comprehensive scenarios."""

    def test_get_current_prices_with_info_fallback(self, mock_logging_manager):
        """Test current price fallback to info when history is empty - covers lines 150-157."""
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.history.return_value = pd.DataFrame()  # Empty history
        mock_ticker.info = {"currentPrice": 155.75}

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            prices = fetcher.get_current_prices(["TEST"])

            assert prices == {"TEST": 155.75}

    def test_get_current_prices_with_regular_market_price_fallback(self, mock_logging_manager):
        """Test fallback to regularMarketPrice - covers lines 154-155."""
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.history.return_value = pd.DataFrame()  # Empty history
        mock_ticker.info = {"regularMarketPrice": 160.25}  # No currentPrice, but has regularMarketPrice

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            prices = fetcher.get_current_prices(["TEST"])

            assert prices == {"TEST": 160.25}

    def test_get_current_prices_no_price_available(self, mock_logging_manager):
        """Test warning when no price is available - covers lines 156-157."""
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.history.return_value = pd.DataFrame()  # Empty history
        mock_ticker.info = {"longName": "Test Company"}  # No price fields

        with patch("yfinance.Ticker", return_value=mock_ticker):
            with patch("stockula.data.fetcher.console.print") as mock_console_print:
                fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
                prices = fetcher.get_current_prices(["TEST"], show_progress=False)

                mock_console_print.assert_called_with("[yellow]Warning: Could not get current price for TEST[/yellow]")
                assert prices == {}

    def test_get_current_prices_with_exception(self, mock_logging_manager):
        """Test exception handling in current price fetching - covers lines 158-159."""
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.history.side_effect = Exception("API error")

        with patch("yfinance.Ticker", return_value=mock_ticker):
            with patch("stockula.data.fetcher.console.print") as mock_console_print:
                fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
                prices = fetcher.get_current_prices(["TEST"], show_progress=False)

                mock_console_print.assert_called_with("[red]Error fetching price for TEST: API error[/red]")
                assert prices == {}

    def test_get_current_prices_single_string_input(self, mock_logging_manager):
        """Test single string input conversion - covers lines 138-139."""
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.history.return_value = pd.DataFrame(
            {"Close": [150.50]}, index=pd.date_range("2023-01-01", periods=1)
        )

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            prices = fetcher.get_current_prices("TEST", show_progress=False)  # Single string instead of list

            assert prices == {"TEST": 150.50}


class TestDataFetcherOptionsChainAdvanced:
    """Test advanced options chain scenarios."""

    def test_get_options_chain_with_cache_hit(self, mock_logging_manager):
        """Test options chain cache hit - covers lines 230-234."""
        cached_calls = pd.DataFrame({"strike": [100, 105], "lastPrice": [5.0, 3.0]})
        cached_puts = pd.DataFrame({"strike": [95, 100], "lastPrice": [2.0, 4.0]})

        mock_db = Mock()
        # Mock database to return cached data (non-empty DataFrames)
        mock_db.get_options_chain.return_value = (cached_calls, cached_puts)

        # Mock ticker with options available
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.options = ("2024-01-19",)

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            with patch("yfinance.Ticker", return_value=mock_ticker):
                fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
                calls, puts = fetcher.get_options_chain("TEST", "2024-01-19")

                # Should return cached data
                pd.testing.assert_frame_equal(calls, cached_calls)
                pd.testing.assert_frame_equal(puts, cached_puts)
                # Should not call option_chain method since cache hit
                mock_ticker.option_chain.assert_not_called() if hasattr(mock_ticker, "option_chain") else None

    def test_get_options_chain_with_exception(self, mock_logging_manager):
        """Test options chain exception handling - covers lines 246-248."""
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.options = ("2024-01-19",)
        mock_ticker.option_chain.side_effect = Exception("Options API error")

        with patch("yfinance.Ticker", return_value=mock_ticker):
            with patch("builtins.print") as mock_print:
                fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
                calls, puts = fetcher.get_options_chain("TEST", "2024-01-19")

                mock_print.assert_called_with("Error fetching options chain for TEST: Options API error")
                assert calls.empty
                assert puts.empty

    def test_get_options_chain_default_expiration(self, mock_logging_manager):
        """Test options chain with default expiration selection - covers lines 225-226."""
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.options = ("2024-01-19", "2024-02-16", "2024-03-15")
        mock_ticker.option_chain.return_value = Mock(
            calls=pd.DataFrame({"strike": [100]}), puts=pd.DataFrame({"strike": [100]})
        )

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            _, _ = fetcher.get_options_chain("TEST")  # No expiration specified

            # Should use first available expiration
            mock_ticker.option_chain.assert_called_once_with("2024-01-19")


class TestDataFetcherDividendsAndSplitsAdvanced:
    """Test advanced dividends and splits scenarios."""

    def test_get_dividends_with_cache_hit(self, mock_logging_manager):
        """Test dividends cache hit - covers lines 270-272."""
        cached_dividends = pd.Series([0.25, 0.30], index=pd.to_datetime(["2023-01-15", "2023-04-15"]))

        mock_db = Mock()
        mock_db.get_dividends.return_value = cached_dividends

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
            dividends = fetcher.get_dividends("TEST")

            assert dividends.equals(cached_dividends)
            mock_db.get_dividends.assert_called_once_with("TEST", None, None)

    def test_get_dividends_with_date_filtering(self, mock_logging_manager):
        """Test dividends with date range filtering - covers lines 284-287."""
        dividend_series = pd.Series(
            [0.20, 0.25, 0.30, 0.35],
            index=pd.to_datetime(["2022-12-15", "2023-01-15", "2023-04-15", "2023-07-15"]),
        )
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.dividends = dividend_series

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            dividends = fetcher.get_dividends("TEST", start="2023-01-01", end="2023-06-30")

            # Should filter to only dividends within date range
            assert len(dividends) == 2  # Should include 2023-01-15 and 2023-04-15
            assert dividends.iloc[0] == 0.25
            assert dividends.iloc[1] == 0.30

    def test_get_splits_with_cache_hit(self, mock_logging_manager):
        """Test splits cache hit - covers lines 311-313."""
        cached_splits = pd.Series([2.0, 3.0], index=pd.to_datetime(["2022-06-01", "2023-06-01"]))

        mock_db = Mock()
        mock_db.get_splits.return_value = cached_splits

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
            splits = fetcher.get_splits("TEST")

            assert splits.equals(cached_splits)
            mock_db.get_splits.assert_called_once_with("TEST", None, None)

    def test_get_splits_with_date_filtering(self, mock_logging_manager):
        """Test splits with date range filtering - covers lines 325-328."""
        split_series = pd.Series(
            [1.5, 2.0, 3.0, 2.5],
            index=pd.to_datetime(["2021-06-01", "2022-06-01", "2023-06-01", "2024-06-01"]),
        )
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.splits = split_series

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            splits = fetcher.get_splits("TEST", start="2022-01-01", end="2023-12-31")

            # Should filter to only splits within date range
            assert len(splits) == 2  # Should include 2022-06-01 and 2023-06-01
            assert splits.iloc[0] == 2.0
            assert splits.iloc[1] == 3.0


class TestDataFetcherBulkOperationsAdvanced:
    """Test advanced bulk operations scenarios."""

    def test_fetch_and_store_all_data_cache_disabled(self, mock_logging_manager):
        """Test bulk fetch with cache disabled - covers lines 343-344."""
        with patch("builtins.print") as mock_print:
            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            fetcher.fetch_and_store_all_data("TEST")

            mock_print.assert_called_with("Warning: Caching is disabled, data will not be stored")

    def test_fetch_and_store_all_data_with_errors(self, mock_logging_manager):
        """Test bulk fetch with individual data fetch errors.

        Covers lines 352-353, 359-360, 368-370, 378-380, 390-392."""
        mock_ticker = Mock(spec=yf.Ticker)

        # Mock failures for different data types
        mock_ticker.history.side_effect = Exception("Price history error")
        mock_ticker.info = Exception("Info error")  # This will cause an error when accessed
        mock_ticker.dividends = pd.Series(dtype=float)  # Empty dividends
        mock_ticker.splits = pd.Series(dtype=float)  # Empty splits
        mock_ticker.options = ()  # No options available

        mock_db = Mock()

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            with patch("yfinance.Ticker", return_value=mock_ticker):
                with patch("builtins.print") as mock_print:
                    # Mock individual methods to control their behavior
                    fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)

                    # Mock methods to raise specific exceptions
                    fetcher.get_stock_data = Mock(side_effect=Exception("Price history error"))
                    fetcher.get_info = Mock(side_effect=Exception("Info error"))
                    fetcher.get_dividends = Mock(return_value=pd.Series(dtype=float))  # Empty series
                    fetcher.get_splits = Mock(return_value=pd.Series(dtype=float))  # Empty series
                    fetcher.get_options_chain = Mock(side_effect=Exception("Options error"))

                    fetcher.fetch_and_store_all_data("TEST")

                    # Should print error messages for failed operations
                    print_calls = [call[0][0] for call in mock_print.call_args_list]
                    assert any("Error fetching price history" in call for call in print_calls)
                    assert any("Error fetching stock info" in call for call in print_calls)
                    assert any("No dividends found" in call for call in print_calls)
                    assert any("No splits found" in call for call in print_calls)
                    assert any("Error fetching options chain" in call for call in print_calls)


class TestDataFetcherUtilityMethods:
    """Test utility and management methods."""

    def test_get_database_stats_without_cache(self, mock_logging_manager):
        """Test get_database_stats without cache - covers lines 400-402."""
        fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
        stats = fetcher.get_database_stats()

        assert stats == {}

    def test_cleanup_old_data_without_cache(self, mock_logging_manager):
        """Test cleanup_old_data without cache - covers lines 410-414."""
        with patch("builtins.print") as mock_print:
            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            fetcher.cleanup_old_data(30)

            mock_print.assert_any_call("Warning: Caching is disabled, no data to clean up")

    def test_cleanup_old_data_with_cache(self, mock_logging_manager):
        """Test cleanup_old_data with cache enabled."""
        mock_db = Mock()

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            with patch("builtins.print") as mock_print:
                fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
                fetcher.cleanup_old_data(30)

                mock_db.cleanup_old_data.assert_called_once_with(30)
                mock_print.assert_called_with("Cleaned up data older than 30 days")

    def test_get_cached_symbols_without_cache(self, mock_logging_manager):
        """Test get_cached_symbols without cache - covers lines 422-424."""
        fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
        symbols = fetcher.get_cached_symbols()

        assert symbols == []

    def test_disable_cache(self, mock_logging_manager):
        """Test disable_cache method - covers lines 428-429."""
        mock_db = Mock()

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
            assert fetcher.use_cache is True
            assert fetcher.db is not None

            fetcher.disable_cache()

            assert fetcher.use_cache is False
            assert fetcher.db is None

    def test_enable_cache(self, mock_logging_manager):
        """Test enable_cache method - covers lines 437-438."""
        with patch("stockula.data.fetcher.DatabaseManager") as mock_db_class:
            mock_db = Mock()
            mock_db_class.return_value = mock_db

            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            assert fetcher.use_cache is False
            assert fetcher.db is None

            fetcher.enable_cache("custom.db")

            assert fetcher.use_cache is True
            assert fetcher.db is mock_db
            mock_db_class.assert_called_with("custom.db")

    def test_initialization_with_injected_database_manager(self, mock_logging_manager):
        """Test initialization with injected database manager - covers line 30."""
        mock_db = Mock()

        # Test with cache enabled and injected db manager
        fetcher = DataFetcher(use_cache=True, database_manager=mock_db, logging_manager=mock_logging_manager)
        assert fetcher.db is mock_db

        # Test with cache disabled and injected db manager
        fetcher = DataFetcher(use_cache=False, database_manager=mock_db, logging_manager=mock_logging_manager)
        assert fetcher.db is None


class TestDataFetcherTreasuryRates:
    """Test Treasury rate fetching functionality."""

    def test_treasury_tickers(self, mock_logging_manager):
        """Test Treasury ticker mappings."""
        fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
        assert fetcher.TREASURY_TICKERS["3_month"] == "^IRX"
        assert fetcher.TREASURY_TICKERS["13_week"] == "^IRX"
        assert fetcher.TREASURY_TICKERS["tbill_etf"] == "BIL"

    @patch("yfinance.Ticker")
    def test_get_treasury_rate_from_yfinance(self, mock_ticker, mock_logging_manager):
        """Test fetching single Treasury rate from yfinance."""
        # Mock yfinance response
        mock_ticker_obj = Mock()
        mock_data = pd.DataFrame({"Close": [5.25]}, index=[datetime(2024, 1, 15)])
        mock_ticker_obj.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_obj

        fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
        rate = fetcher.get_treasury_rate("2024-01-15", "3_month")

        assert rate == 0.0525  # Should convert from percentage to decimal
        mock_ticker.assert_called_once_with("^IRX")

    def test_get_treasury_rate_from_cache(self, mock_logging_manager):
        """Test fetching Treasury rate from cache."""
        # Setup mock to return cached data
        mock_db = Mock()
        mock_db.get_price_history.return_value = pd.DataFrame({"Close": [5.25]}, index=[datetime(2024, 1, 1)])

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
            rate = fetcher.get_treasury_rate("2024-01-01", "3_month")

            assert rate == 5.25  # From Close price
            mock_db.get_price_history.assert_called_once()

    @patch("yfinance.Ticker")
    def test_get_treasury_rates_range(self, mock_ticker, mock_logging_manager):
        """Test fetching Treasury rates for date range."""
        # Mock yfinance response
        mock_ticker_obj = Mock()
        dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="D")
        mock_data = pd.DataFrame(
            {
                "Close": [5.25, 5.26, 5.27, 5.28, 5.29, 5.30, 5.29, 5.28, 5.27, 5.26],
            },
            index=dates,
        )
        mock_ticker_obj.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_obj

        fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
        rates = fetcher.get_treasury_rates("2024-01-01", "2024-01-10", "3_month", force_refresh=True)

        assert len(rates) == 10
        assert rates.iloc[0] == 0.0525  # Should convert to decimal
        assert rates.iloc[-1] == 0.0526

    def test_get_average_treasury_rate(self, mock_logging_manager):
        """Test calculating average Treasury rate."""
        # Mock get_treasury_rates to return known values
        fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
        test_rates = pd.Series([0.0525, 0.0526, 0.0527, 0.0528, 0.0529])
        fetcher.get_treasury_rates = Mock(return_value=test_rates)

        avg_rate = fetcher.get_average_treasury_rate("2024-01-01", "2024-01-05", "3_month")

        assert avg_rate == pytest.approx(0.0527, rel=1e-4)

    def test_get_average_treasury_rate_empty(self, mock_logging_manager):
        """Test average rate when no data available."""
        # Mock get_treasury_rates to return empty series
        fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
        fetcher.get_treasury_rates = Mock(return_value=pd.Series(dtype=float))

        avg_rate = fetcher.get_average_treasury_rate("2024-01-01", "2024-01-05", "3_month")

        assert avg_rate is None

    @patch("yfinance.Ticker")
    def test_get_current_treasury_rate(self, mock_ticker, mock_logging_manager):
        """Test getting current Treasury rate."""
        # Mock yfinance response
        mock_ticker_obj = Mock()
        mock_data = pd.DataFrame({"Close": [5.42]}, index=[datetime.now()])
        mock_ticker_obj.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_obj

        fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
        rate = fetcher.get_current_treasury_rate("3_month")

        assert rate == 0.0542

    @patch("yfinance.Ticker")
    def test_treasury_rate_caching_behavior(self, mock_ticker, mock_logging_manager):
        """Test that rates are cached after fetching."""
        # Mock yfinance response
        mock_ticker_obj = Mock()
        mock_data = pd.DataFrame({"Close": [5.25]}, index=[datetime(2024, 1, 15)])
        mock_ticker_obj.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_obj

        # Mock database
        mock_db = Mock()
        mock_db.get_price_history.return_value = pd.DataFrame()

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
            fetcher.get_treasury_rate("2024-01-15", "3_month", force_refresh=True)

            # Verify caching was called
            mock_db.store_stock_info.assert_called_once()
            mock_db.store_price_history.assert_called_once()

            # Verify the stored data
            stored_df = mock_db.store_price_history.call_args[0][1]
            assert stored_df.iloc[0]["Close"] == 0.0525

    @patch("yfinance.Ticker")
    def test_treasury_rate_error_handling(self, mock_ticker, mock_logging_manager):
        """Test error handling when yfinance fails."""
        # Mock yfinance to raise exception
        mock_ticker.side_effect = Exception("Network error")

        fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
        rate = fetcher.get_treasury_rate("2024-01-15", "3_month", force_refresh=True)

        assert rate is None

    def test_treasury_percentage_vs_decimal_conversion(self, mock_logging_manager):
        """Test conversion between percentage and decimal formats."""
        # Create fetcher with mocked methods
        fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
        fetcher._fetch_rate_from_yfinance = Mock(return_value=0.0525)

        # Test as_decimal=True (default)
        rate_decimal = fetcher.get_treasury_rate("2024-01-15", "3_month")
        assert rate_decimal == 0.0525

        # Test as_decimal=False
        rate_percent = fetcher.get_treasury_rate("2024-01-15", "3_month", as_decimal=False)
        assert rate_percent == 5.25

    def test_different_treasury_duration_types(self, mock_logging_manager):
        """Test different Treasury duration types."""
        # Mock the fetch method
        fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
        fetcher._fetch_rate_from_yfinance = Mock(return_value=0.05)

        # Test different durations
        for duration in ["3_month", "13_week", "1_year", "tbill_etf", "sgov"]:
            rate = fetcher.get_treasury_rate("2024-01-15", duration)
            assert rate == 0.05

        # Test invalid duration (should default to 3_month)
        rate = fetcher.get_treasury_rate("2024-01-15", "invalid_duration")
        assert rate == 0.05


class TestDataFetcherProgressBars:
    """Test progress bar functionality for current prices."""

    def test_get_current_prices_with_progress_bar(self, mock_logging_manager):
        """Test current prices with progress bar for multiple symbols - covers lines 184-219."""
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]

        def side_effect(symbol):
            mock_ticker = Mock()
            mock_ticker.history.return_value = pd.DataFrame(
                {"Close": [150.0 + hash(symbol) % 50]}, index=pd.date_range("2023-01-01", periods=1)
            )
            return mock_ticker

        with patch("yfinance.Ticker", side_effect=side_effect):
            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            prices = fetcher.get_current_prices(symbols, show_progress=True)

            assert len(prices) == 4
            assert all(symbol in prices for symbol in symbols)

    def test_get_current_prices_progress_with_info_fallback(self, mock_logging_manager):
        """Test progress bar with info fallback - covers lines 208-215."""
        symbols = ["TEST1", "TEST2"]

        def side_effect(symbol):
            mock_ticker = Mock()
            mock_ticker.history.return_value = pd.DataFrame()  # Empty history
            if symbol == "TEST1":
                mock_ticker.info = {"currentPrice": 155.75}
            else:
                mock_ticker.info = {"regularMarketPrice": 160.25}
            return mock_ticker

        with patch("yfinance.Ticker", side_effect=side_effect):
            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            prices = fetcher.get_current_prices(symbols, show_progress=True)

            assert prices == {"TEST1": 155.75, "TEST2": 160.25}

    def test_get_current_prices_progress_with_warning(self, mock_logging_manager):
        """Test progress bar with price warning - covers line 215."""
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.history.return_value = pd.DataFrame()  # Empty history
        mock_ticker.info = {"longName": "Test Company"}  # No price fields

        with patch("yfinance.Ticker", return_value=mock_ticker):
            with patch("stockula.data.fetcher.console.print") as mock_console_print:
                fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
                prices = fetcher.get_current_prices(["TEST1", "TEST2"], show_progress=True)

                mock_console_print.assert_any_call("[yellow]Warning: Could not get current price for TEST1[/yellow]")
                mock_console_print.assert_any_call("[yellow]Warning: Could not get current price for TEST2[/yellow]")
                assert prices == {}

    def test_get_current_prices_progress_with_exception(self, mock_logging_manager):
        """Test progress bar with exception handling - covers lines 216-217."""
        mock_ticker = Mock(spec=yf.Ticker)
        mock_ticker.history.side_effect = Exception("API error")

        with patch("yfinance.Ticker", return_value=mock_ticker):
            with patch("stockula.data.fetcher.console.print") as mock_console_print:
                fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
                prices = fetcher.get_current_prices(["TEST1", "TEST2"], show_progress=True)

                mock_console_print.assert_any_call("[red]Error fetching price for TEST1: API error[/red]")
                mock_console_print.assert_any_call("[red]Error fetching price for TEST2: API error[/red]")
                assert prices == {}


class TestDataFetcherBatchOperations:
    """Test batch data fetching operations."""

    def test_get_stock_data_batch_all_cached(self, mock_logging_manager):
        """Test batch operation with all data in cache - covers lines 432-437."""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        cached_data = pd.DataFrame({"Close": [150.0]}, index=pd.date_range("2023-01-01", periods=1))

        mock_db = Mock()

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
            # Mock get_stock_data to return cached data
            fetcher.get_stock_data = Mock(return_value=cached_data)

            results = fetcher.get_stock_data_batch(symbols)

            assert len(results) == 3
            assert all(symbol in results for symbol in symbols)

    def test_get_stock_data_batch_single_symbol(self, mock_logging_manager):
        """Test batch operation with single symbol - covers lines 458-464."""
        mock_data = pd.DataFrame({"Close": [150.0]}, index=pd.date_range("2023-01-01", periods=1))

        mock_db = Mock()

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            with patch("yfinance.download", return_value=mock_data) as mock_download:
                fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
                # Override get_stock_data to force batch download path
                fetcher.get_stock_data = Mock(return_value=pd.DataFrame())  # Empty to force batch

                results = fetcher.get_stock_data_batch(["AAPL"])

                mock_download.assert_called_once()
                assert "AAPL" in results

    def test_get_stock_data_batch_multiple_symbols(self, mock_logging_manager):
        """Test batch operation with multiple symbols - covers lines 465-477."""
        symbols = ["AAPL", "GOOGL", "MSFT"]

        # Create multi-level DataFrame as yfinance.download returns
        columns = pd.MultiIndex.from_tuples([("AAPL", "Close"), ("GOOGL", "Close"), ("MSFT", "Close")])
        mock_data = pd.DataFrame([[150.0, 100.0, 300.0]], index=pd.date_range("2023-01-01", periods=1), columns=columns)

        mock_db = Mock()

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            with patch("yfinance.download", return_value=mock_data) as mock_download:
                fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
                # Override get_stock_data to force batch download path
                fetcher.get_stock_data = Mock(return_value=pd.DataFrame())  # Empty to force batch

                results = fetcher.get_stock_data_batch(symbols)

                mock_download.assert_called_once()
                assert len(results) == 3
                assert all(symbol in results for symbol in symbols)

    def test_get_stock_data_batch_with_missing_symbol(self, mock_logging_manager):
        """Test batch operation with missing symbol data - covers line 477."""
        symbols = ["AAPL", "INVALID"]

        # Create data only for AAPL
        columns = pd.MultiIndex.from_tuples([("AAPL", "Close")])
        mock_data = pd.DataFrame([[150.0]], index=pd.date_range("2023-01-01", periods=1), columns=columns)

        mock_db = Mock()

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            with patch("yfinance.download", return_value=mock_data):
                # Direct mock on injected logger
                fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
                fetcher.get_stock_data = Mock(return_value=pd.DataFrame())  # Empty to force batch

                results = fetcher.get_stock_data_batch(symbols)

                fetcher.logger.warning.assert_called_with("No data returned for INVALID")
                assert "AAPL" in results
                assert "INVALID" not in results

    def test_get_stock_data_batch_download_error_fallback(self, mock_logging_manager):
        """Test batch download error with individual fallback - covers lines 479-488."""
        symbols = ["AAPL", "GOOGL"]

        mock_db = Mock()
        fallback_data = pd.DataFrame({"Close": [150.0]}, index=pd.date_range("2023-01-01", periods=1))

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            with patch("yfinance.download", side_effect=Exception("Batch download failed")):
                # Direct mock on injected logger
                fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
                # Mock individual get_stock_data to return data
                fetcher.get_stock_data = Mock(return_value=fallback_data)

                # First call returns empty to force batch, subsequent calls return data
                def side_effect(*args, **kwargs):
                    if fetcher.get_stock_data.call_count <= 2:  # First 2 calls for cache check
                        return pd.DataFrame()
                    return fallback_data

                fetcher.get_stock_data.side_effect = side_effect

                results = fetcher.get_stock_data_batch(symbols)

                fetcher.logger.error.assert_called_with("Error in batch download: Batch download failed")
                # Should have fallback data for both symbols
                assert len(results) >= 1

    def test_get_stock_data_batch_individual_fallback_error(self, mock_logging_manager):
        """Test individual fallback error handling - covers line 488."""
        symbols = ["AAPL", "INVALID"]

        mock_db = Mock()

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            with patch("yfinance.download", side_effect=Exception("Batch download failed")):
                # Direct mock on injected logger
                fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)

                # Mock individual get_stock_data - first calls empty (cache check), then error
                # Mock individual get_stock_data properly
                with patch.object(fetcher, "get_stock_data") as mock_get_stock_data:

                    def side_effect(*args, **kwargs):
                        call_count = mock_get_stock_data.call_count
                        if call_count <= 2:  # Cache check calls
                            return pd.DataFrame()
                        if "INVALID" in args[0]:  # Individual fallback fails for INVALID
                            raise Exception("Error fetching INVALID")
                        return pd.DataFrame({"Close": [150]}, index=pd.date_range("2023-01-01", periods=1))

                    mock_get_stock_data.side_effect = side_effect

                fetcher.get_stock_data_batch(symbols)

                # Should log batch error
                fetcher.logger.error.assert_any_call("Error in batch download: Batch download failed")


class TestDataFetcherCurrentPricesBatch:
    """Test batch current prices functionality."""

    def test_get_current_prices_batch_with_fast_info(self, mock_logging_manager):
        """Test batch current prices with fast_info - covers lines 504-512."""
        symbols = ["AAPL", "GOOGL", "MSFT"]

        # Mock Tickers class
        mock_tickers = Mock()
        mock_ticker_instances = {}

        for symbol in symbols:
            mock_ticker = Mock()
            mock_fast_info = Mock()
            mock_fast_info.last_price = 150.0 + hash(symbol) % 50
            mock_ticker.fast_info = mock_fast_info
            mock_ticker_instances[symbol] = mock_ticker

        mock_tickers.tickers = mock_ticker_instances

        with patch("yfinance.Tickers", return_value=mock_tickers):
            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            prices = fetcher.get_current_prices_batch(symbols)

            assert len(prices) == 3
            assert all(symbol in prices for symbol in symbols)
            assert all(isinstance(price, float) for price in prices.values())

    def test_get_current_prices_batch_with_info_fallback(self, mock_logging_manager):
        """Test batch current prices with info fallback - covers lines 514-518."""
        symbols = ["AAPL", "GOOGL"]

        mock_tickers = Mock()
        mock_ticker_instances = {}

        for i, symbol in enumerate(symbols):
            mock_ticker = Mock()
            mock_fast_info = Mock()
            # First symbol has no last_price, second has no fast_info
            if i == 0:
                mock_fast_info.last_price = None
                mock_ticker.fast_info = mock_fast_info
                mock_ticker.info = {"regularMarketPrice": 155.0}
            else:
                mock_ticker.fast_info = Mock()
                del mock_ticker.fast_info.last_price  # AttributeError when accessed
                mock_ticker.info = {"previousClose": 160.0}
            mock_ticker_instances[symbol] = mock_ticker

        mock_tickers.tickers = mock_ticker_instances

        with patch("yfinance.Tickers", return_value=mock_tickers):
            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            prices = fetcher.get_current_prices_batch(symbols)

            assert prices["AAPL"] == 155.0
            assert prices["GOOGL"] == 160.0

    def test_get_current_prices_batch_with_error_and_individual_fallback(self, mock_logging_manager):
        """Test batch current prices with error and individual fallback - covers lines 519-527."""
        symbols = ["AAPL", "ERROR_SYMBOL"]

        mock_tickers = Mock()
        mock_ticker_instances = {}

        # Good symbol
        good_ticker = Mock()
        good_fast_info = Mock()
        good_fast_info.last_price = 150.0
        good_ticker.fast_info = good_fast_info
        mock_ticker_instances["AAPL"] = good_ticker

        # Error symbol
        error_ticker = Mock()
        error_ticker.fast_info = Mock()
        # This will raise an exception when accessed
        type(error_ticker.fast_info).last_price = PropertyMock(side_effect=Exception("Fast info error"))
        error_ticker.info = Mock()
        error_ticker.info.get.side_effect = Exception("Info error")
        mock_ticker_instances["ERROR_SYMBOL"] = error_ticker

        mock_tickers.tickers = mock_ticker_instances

        with patch("yfinance.Tickers", return_value=mock_tickers):
            # Direct mock on injected logger
            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            # Mock individual fallback to return data
            fetcher.get_current_prices = Mock(return_value={"ERROR_SYMBOL": 999.0})

            prices = fetcher.get_current_prices_batch(symbols)

            fetcher.logger.error.assert_called()  # Should log error for ERROR_SYMBOL
            assert prices["AAPL"] == 150.0
            assert prices["ERROR_SYMBOL"] == 999.0  # From individual fallback

    def test_get_current_prices_batch_complete_fallback_failure(self, mock_logging_manager):
        """Test batch current prices with complete fallback failure - covers lines 526-527."""
        symbols = ["ERROR_SYMBOL"]

        mock_tickers = Mock()
        mock_ticker = Mock()
        mock_ticker.fast_info = Mock()
        type(mock_ticker.fast_info).last_price = PropertyMock(side_effect=Exception("Fast info error"))
        mock_ticker.info = Mock()
        mock_ticker.info.get.side_effect = Exception("Info error")
        mock_tickers.tickers = {"ERROR_SYMBOL": mock_ticker}

        with patch("yfinance.Tickers", return_value=mock_tickers):
            # Direct mock on injected logger
            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            # Mock individual fallback to also fail
            fetcher.get_current_prices = Mock(side_effect=Exception("Individual fallback failed"))

            prices = fetcher.get_current_prices_batch(symbols)

            fetcher.logger.error.assert_called()  # Should log error
            assert "ERROR_SYMBOL" not in prices  # Symbol should not be in results


class TestDataFetcherLifecycleManagement:
    """Test DataFetcher lifecycle and cleanup methods."""

    def test_close_method_owns_db(self, mock_logging_manager):
        """Test close method when fetcher owns database - covers lines 61-63."""
        mock_db = Mock()

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
            assert fetcher._owns_db is True

            fetcher.close()

            mock_db.close.assert_called_once()
            assert fetcher.db is None

    def test_close_method_does_not_own_db(self, mock_logging_manager):
        """Test close method when fetcher doesn't own database."""
        mock_db = Mock()

        # Injected database manager
        fetcher = DataFetcher(use_cache=True, database_manager=mock_db, logging_manager=mock_logging_manager)
        assert fetcher._owns_db is False

        fetcher.close()

        # Should not close injected database
        mock_db.close.assert_not_called()

    def test_del_method_calls_close(self, mock_logging_manager):
        """Test __del__ method calls close - covers lines 67-71."""
        mock_db = Mock()

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
            close_method = Mock()
            fetcher.close = close_method

            fetcher.__del__()

            close_method.assert_called_once()

    def test_del_method_handles_exception(self, mock_logging_manager):
        """Test __del__ method handles exceptions gracefully - covers line 70-71."""
        mock_db = Mock()

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
            fetcher.close = Mock(side_effect=Exception("Cleanup error"))

            # Should not raise exception
            try:
                fetcher.__del__()
            except Exception:
                pytest.fail("__del__ should not raise exceptions")


class TestDataFetcherTreasuryAdvanced:
    """Test advanced Treasury rate functionality."""

    def test_get_treasury_rates_from_cache(self, mock_logging_manager):
        """Test treasury rates from cache - covers lines 750-753."""
        cached_rates = pd.Series([0.0525, 0.0526], index=pd.date_range("2024-01-01", periods=2))

        mock_db = Mock()

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
            fetcher._get_cached_rates = Mock(return_value=cached_rates)

            rates = fetcher.get_treasury_rates("2024-01-01", "2024-01-02", "3_month")

            assert rates.equals(cached_rates)

    def test_get_treasury_rates_with_progress(self, mock_logging_manager):
        """Test treasury rates with progress indication - covers line 756."""
        mock_rates = pd.Series([0.0525], index=pd.date_range("2024-01-01", periods=1))

        mock_db = Mock()

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
            fetcher._get_cached_rates = Mock(return_value=pd.Series(dtype=float))  # Empty cache
            fetcher._fetch_rates_from_yfinance = Mock(return_value=mock_rates)
            fetcher._cache_rates = Mock()

            rates = fetcher.get_treasury_rates("2024-01-01", "2024-01-02", "3_month", force_refresh=True)

            fetcher._fetch_rates_from_yfinance.assert_called_with(
                "^IRX", pd.to_datetime("2024-01-01"), pd.to_datetime("2024-01-02"), show_progress=True
            )
            assert rates.equals(mock_rates)

    def test_get_treasury_rates_caching(self, mock_logging_manager):
        """Test treasury rates caching behavior - covers lines 759-760."""
        mock_rates = pd.Series([0.0525], index=pd.date_range("2024-01-01", periods=1))

        mock_db = Mock()

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
            fetcher._get_cached_rates = Mock(return_value=pd.Series(dtype=float))  # Empty cache
            fetcher._fetch_rates_from_yfinance = Mock(return_value=mock_rates)
            fetcher._cache_rates = Mock()

            fetcher.get_treasury_rates("2024-01-01", "2024-01-02", "3_month", force_refresh=True)

            fetcher._cache_rates.assert_called_once_with("^IRX", mock_rates)

    def test_fetch_rates_from_yfinance_with_progress(self, mock_logging_manager):
        """Test _fetch_rates_from_yfinance with progress - covers lines 800-816."""
        mock_data = pd.DataFrame({"Close": [5.25]}, index=pd.date_range("2024-01-01", periods=1))

        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_data

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            rates = fetcher._fetch_rates_from_yfinance(
                "^IRX", datetime(2024, 1, 1), datetime(2024, 1, 2), show_progress=True
            )

            expected_rates = pd.Series([0.0525], index=mock_data.index)  # Converted to decimal
            assert rates.equals(expected_rates)

    def test_fetch_rates_from_yfinance_without_progress(self, mock_logging_manager):
        """Test _fetch_rates_from_yfinance without progress - covers lines 817-819."""
        mock_data = pd.DataFrame({"Close": [5.25]}, index=pd.date_range("2024-01-01", periods=1))

        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_data

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            rates = fetcher._fetch_rates_from_yfinance(
                "^IRX", datetime(2024, 1, 1), datetime(2024, 1, 2), show_progress=False
            )

            expected_rates = pd.Series([0.0525], index=mock_data.index)  # Converted to decimal
            assert rates.equals(expected_rates)

    def test_fetch_rates_from_yfinance_empty_data(self, mock_logging_manager):
        """Test _fetch_rates_from_yfinance with empty data - covers lines 821-822."""
        mock_ticker = Mock()
        mock_ticker.history.return_value = pd.DataFrame()  # Empty DataFrame

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            rates = fetcher._fetch_rates_from_yfinance("^IRX", datetime(2024, 1, 1), datetime(2024, 1, 2))

            assert rates.empty
            assert rates.dtype == float

    def test_fetch_rates_from_yfinance_non_irx_ticker(self, mock_logging_manager):
        """Test _fetch_rates_from_yfinance with non-IRX ticker - covers lines 827-829."""
        mock_data = pd.DataFrame({"Close": [0.0525]}, index=pd.date_range("2024-01-01", periods=1))

        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_data

        with patch("yfinance.Ticker", return_value=mock_ticker):
            fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
            rates = fetcher._fetch_rates_from_yfinance(
                "BIL",  # Not ^IRX
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
            )

            # Should not convert (already in decimal)
            expected_rates = pd.Series([0.0525], index=mock_data.index)
            assert rates.equals(expected_rates)

    def test_fetch_rates_from_yfinance_error_handling(self, mock_logging_manager):
        """Test _fetch_rates_from_yfinance error handling - covers lines 833-835."""
        with patch("yfinance.Ticker", side_effect=Exception("Network error")):
            with patch("stockula.data.fetcher.console.print") as mock_console_print:
                fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
                rates = fetcher._fetch_rates_from_yfinance("^IRX", datetime(2024, 1, 1), datetime(2024, 1, 2))

                mock_console_print.assert_called_with(
                    "[red]Error fetching Treasury rates for ^IRX: Network error[/red]"
                )
                assert rates.empty
                assert rates.dtype == float

    def test_get_cached_rate_no_db(self, mock_logging_manager):
        """Test _get_cached_rate with no database - covers lines 839-840."""
        fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
        rate = fetcher._get_cached_rate("^IRX", datetime(2024, 1, 1))

        assert rate is None

    def test_get_cached_rate_with_data(self, mock_logging_manager):
        """Test _get_cached_rate with data - covers lines 848-851."""
        mock_data = pd.DataFrame({"Close": [0.0525]}, index=pd.date_range("2024-01-01", periods=1))

        mock_db = Mock()
        mock_db.get_price_history.return_value = mock_data

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
            rate = fetcher._get_cached_rate("^IRX", datetime(2024, 1, 1))

            assert rate == 0.0525

    def test_get_cached_rate_db_none_check(self, mock_logging_manager):
        """Test _get_cached_rate with db None check - covers lines 846-847."""
        mock_db = Mock()

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
            fetcher.db = None  # Set db to None manually

            rate = fetcher._get_cached_rate("^IRX", datetime(2024, 1, 1))

            assert rate is None

    def test_get_cached_rates_no_db(self, mock_logging_manager):
        """Test _get_cached_rates with no database - covers lines 857-858."""
        fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
        rates = fetcher._get_cached_rates("^IRX", datetime(2024, 1, 1), datetime(2024, 1, 2))

        assert rates.empty
        assert rates.dtype == float

    def test_get_cached_rates_db_none_check(self, mock_logging_manager):
        """Test _get_cached_rates with db None check - covers lines 865-866."""
        mock_db = Mock()

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
            fetcher.db = None  # Set db to None manually

            rates = fetcher._get_cached_rates("^IRX", datetime(2024, 1, 1), datetime(2024, 1, 2))

            assert rates.empty
            assert rates.dtype == float

    def test_get_cached_rates_with_data(self, mock_logging_manager):
        """Test _get_cached_rates with data - covers lines 869-870."""
        mock_data = pd.DataFrame({"Close": [0.0525, 0.0526]}, index=pd.date_range("2024-01-01", periods=2))

        mock_db = Mock()
        mock_db.get_price_history.return_value = mock_data

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
            rates = fetcher._get_cached_rates("^IRX", datetime(2024, 1, 1), datetime(2024, 1, 2))

            expected_rates = mock_data["Close"]
            assert rates.equals(expected_rates)

    def test_cache_rate_no_db(self, mock_logging_manager):
        """Test _cache_rate with no database - covers lines 876-877."""
        fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
        # Should not raise exception
        fetcher._cache_rate("^IRX", datetime(2024, 1, 1), 0.0525)

    def test_cache_rate_db_none_check(self, mock_logging_manager):
        """Test _cache_rate with db None check - covers lines 892-893."""
        mock_db = Mock()

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
            fetcher.db = None  # Set db to None manually

            # Should not raise exception and not call store methods
            fetcher._cache_rate("^IRX", datetime(2024, 1, 1), 0.0525)

    def test_cache_rate_with_db(self, mock_logging_manager):
        """Test _cache_rate with database - covers lines 880-896."""
        mock_db = Mock()

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
            fetcher._cache_rate("^IRX", datetime(2024, 1, 1), 0.0525)

            # Should store both stock info and price history
            mock_db.store_stock_info.assert_called_once()
            mock_db.store_price_history.assert_called_once()

    def test_cache_rates_no_db_or_empty(self, mock_logging_manager):
        """Test _cache_rates with no db or empty data - covers lines 900-901."""
        fetcher = DataFetcher(use_cache=False, logging_manager=mock_logging_manager)
        empty_rates = pd.Series(dtype=float)

        # Should not raise exception
        fetcher._cache_rates("^IRX", empty_rates)

    def test_cache_rates_with_db(self, mock_logging_manager):
        """Test _cache_rates with database - covers lines 904-919."""
        mock_rates = pd.Series([0.0525, 0.0526], index=pd.date_range("2024-01-01", periods=2))
        mock_db = Mock()

        with patch("stockula.data.fetcher.DatabaseManager", return_value=mock_db):
            fetcher = DataFetcher(use_cache=True, logging_manager=mock_logging_manager)
            fetcher._cache_rates("^IRX", mock_rates)

            # Should store both stock info and price history
            mock_db.store_stock_info.assert_called_once()
            mock_db.store_price_history.assert_called_once()

            # Verify the DataFrame structure
            stored_df = mock_db.store_price_history.call_args[0][1]
            assert "Open" in stored_df.columns
            assert "High" in stored_df.columns
            assert "Low" in stored_df.columns
            assert "Close" in stored_df.columns
            assert "Volume" in stored_df.columns
