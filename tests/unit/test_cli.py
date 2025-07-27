"""Unit tests for database CLI module."""

import pytest
import argparse
import sys
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, call
from io import StringIO

from stockula.database.cli import (
    fetch_symbol_data,
    fetch_portfolio_data,
    show_database_stats,
    cleanup_database,
    query_symbol_data,
    main,
)


class TestFetchSymbolData:
    """Test fetch_symbol_data function."""

    @patch("stockula.database.cli.DataFetcher")
    def test_fetch_symbol_data_basic(self, mock_fetcher_class):
        """Test basic symbol data fetching."""
        mock_fetcher = Mock()
        mock_fetcher_class.return_value = mock_fetcher

        fetch_symbol_data("AAPL")

        # Verify fetcher was created with cache enabled
        mock_fetcher_class.assert_called_once_with(use_cache=True)

        # Verify fetch_and_store_all_data was called
        mock_fetcher.fetch_and_store_all_data.assert_called_once_with(
            "AAPL", None, None
        )

    @patch("stockula.database.cli.DataFetcher")
    def test_fetch_symbol_data_with_dates(self, mock_fetcher_class):
        """Test symbol data fetching with date range."""
        mock_fetcher = Mock()
        mock_fetcher_class.return_value = mock_fetcher

        fetch_symbol_data("TSLA", "2023-01-01", "2023-12-31")

        mock_fetcher.fetch_and_store_all_data.assert_called_once_with(
            "TSLA", "2023-01-01", "2023-12-31"
        )


class TestFetchPortfolioData:
    """Test fetch_portfolio_data function."""

    @patch("stockula.database.cli.DataFetcher")
    def test_fetch_portfolio_data_basic(self, mock_fetcher_class, capsys):
        """Test portfolio data fetching."""
        mock_fetcher = Mock()
        mock_fetcher_class.return_value = mock_fetcher

        symbols = ["AAPL", "GOOGL", "MSFT"]
        fetch_portfolio_data(symbols)

        # Verify fetcher was created
        mock_fetcher_class.assert_called_once_with(use_cache=True)

        # Verify each symbol was processed
        expected_calls = [
            call("AAPL", None, None),
            call("GOOGL", None, None),
            call("MSFT", None, None),
        ]
        mock_fetcher.fetch_and_store_all_data.assert_has_calls(expected_calls)

        # Check console output
        captured = capsys.readouterr()
        assert "Fetching data for 3 symbols" in captured.out
        assert "[1/3] Processing AAPL" in captured.out
        assert "[2/3] Processing GOOGL" in captured.out
        assert "[3/3] Processing MSFT" in captured.out
        assert "Completed fetching data for all 3 symbols" in captured.out

    @patch("stockula.database.cli.DataFetcher")
    def test_fetch_portfolio_data_with_dates(self, mock_fetcher_class):
        """Test portfolio data fetching with date range."""
        mock_fetcher = Mock()
        mock_fetcher_class.return_value = mock_fetcher

        symbols = ["AAPL", "TSLA"]
        fetch_portfolio_data(symbols, "2023-01-01", "2023-06-30")

        # Verify calls with dates
        expected_calls = [
            call("AAPL", "2023-01-01", "2023-06-30"),
            call("TSLA", "2023-01-01", "2023-06-30"),
        ]
        mock_fetcher.fetch_and_store_all_data.assert_has_calls(expected_calls)

    @patch("stockula.database.cli.DataFetcher")
    def test_fetch_portfolio_data_single_symbol(self, mock_fetcher_class, capsys):
        """Test portfolio data fetching with single symbol."""
        mock_fetcher = Mock()
        mock_fetcher_class.return_value = mock_fetcher

        symbols = ["AAPL"]
        fetch_portfolio_data(symbols)

        # Verify output for single symbol
        captured = capsys.readouterr()
        assert "Fetching data for 1 symbols" in captured.out
        assert "[1/1] Processing AAPL" in captured.out


class TestShowDatabaseStats:
    """Test show_database_stats function."""

    @patch("stockula.database.cli.DataFetcher")
    def test_show_database_stats_basic(self, mock_fetcher_class, capsys):
        """Test showing database statistics."""
        mock_fetcher = Mock()
        mock_stats = {
            "stocks": 25,
            "price_history": 12500,
            "dividends": 150,
            "stock_splits": 8,
        }
        mock_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]

        mock_fetcher.get_database_stats.return_value = mock_stats
        mock_fetcher.get_cached_symbols.return_value = mock_symbols
        mock_fetcher_class.return_value = mock_fetcher

        show_database_stats()

        # Check console output
        captured = capsys.readouterr()
        assert "Database Statistics:" in captured.out
        assert "stocks         : 25 records" in captured.out
        assert "price_history  : 12,500 records" in captured.out
        assert "dividends      : 150 records" in captured.out
        assert "stock_splits   : 8 records" in captured.out
        assert "Cached symbols (4):" in captured.out
        assert "AAPL" in captured.out
        assert "GOOGL" in captured.out

    @patch("stockula.database.cli.DataFetcher")
    def test_show_database_stats_no_symbols(self, mock_fetcher_class, capsys):
        """Test showing database statistics with no cached symbols."""
        mock_fetcher = Mock()
        mock_fetcher.get_database_stats.return_value = {"stocks": 0}
        mock_fetcher.get_cached_symbols.return_value = []
        mock_fetcher_class.return_value = mock_fetcher

        show_database_stats()

        captured = capsys.readouterr()
        assert "No symbols cached yet" in captured.out

    @patch("stockula.database.cli.DataFetcher")
    def test_show_database_stats_many_symbols(self, mock_fetcher_class, capsys):
        """Test showing database statistics with many symbols."""
        mock_fetcher = Mock()
        mock_fetcher.get_database_stats.return_value = {"stocks": 20}

        # Create 10 symbols to test line wrapping
        mock_symbols = [f"SYM{i:02d}" for i in range(10)]
        mock_fetcher.get_cached_symbols.return_value = mock_symbols
        mock_fetcher_class.return_value = mock_fetcher

        show_database_stats()

        captured = capsys.readouterr()
        assert "Cached symbols (10):" in captured.out
        # Should wrap after 8 symbols per line
        lines = captured.out.split("\n")
        symbol_lines = [line for line in lines if "SYM" in line]
        assert len(symbol_lines) >= 2  # Should wrap to multiple lines


class TestCleanupDatabase:
    """Test cleanup_database function."""

    @patch("stockula.database.cli.DataFetcher")
    def test_cleanup_database_default_days(self, mock_fetcher_class, capsys):
        """Test database cleanup with default days."""
        mock_fetcher = Mock()
        mock_fetcher_class.return_value = mock_fetcher

        cleanup_database()

        mock_fetcher.cleanup_old_data.assert_called_once_with(365)

        captured = capsys.readouterr()
        assert "Cleaning up data older than 365 days" in captured.out
        assert "Cleanup completed" in captured.out

    @patch("stockula.database.cli.DataFetcher")
    def test_cleanup_database_custom_days(self, mock_fetcher_class, capsys):
        """Test database cleanup with custom days."""
        mock_fetcher = Mock()
        mock_fetcher_class.return_value = mock_fetcher

        cleanup_database(180)

        mock_fetcher.cleanup_old_data.assert_called_once_with(180)

        captured = capsys.readouterr()
        assert "Cleaning up data older than 180 days" in captured.out


class TestQuerySymbolData:
    """Test query_symbol_data function."""

    @patch("stockula.database.cli.DatabaseManager")
    def test_query_symbol_data_complete(self, mock_db_class, capsys):
        """Test querying symbol data with all data types."""
        mock_db = Mock()

        # Mock stock info
        mock_info = {
            "longName": "Apple Inc.",
            "sector": "Technology",
            "marketCap": 3000000000000,
            "exchange": "NASDAQ",
        }
        mock_db.get_stock_info.return_value = mock_info

        # Mock price data
        price_data = pd.DataFrame(
            {"Close": [150.0, 151.0, 152.0]},
            index=pd.date_range("2023-01-01", periods=3),
        )
        mock_db.get_price_history.return_value = price_data

        # Mock dividends
        dividends = pd.Series(
            [0.25, 0.30], index=pd.date_range("2023-01-01", periods=2)
        )
        mock_db.get_dividends.return_value = dividends

        # Mock splits
        splits = pd.Series([2.0], index=pd.date_range("2023-06-01", periods=1))
        mock_db.get_splits.return_value = splits

        mock_db_class.return_value = mock_db

        query_symbol_data("AAPL")

        # Verify all queries were made
        mock_db.get_stock_info.assert_called_once_with("AAPL")
        mock_db.get_price_history.assert_called_once_with("AAPL")
        mock_db.get_dividends.assert_called_once_with("AAPL")
        mock_db.get_splits.assert_called_once_with("AAPL")

        # Check console output
        captured = capsys.readouterr()
        assert "Data for AAPL:" in captured.out
        assert "Name: Apple Inc." in captured.out
        assert "Sector: Technology" in captured.out
        assert "Market Cap: $3,000,000,000,000" in captured.out
        assert "Exchange: NASDAQ" in captured.out
        assert "Latest Price: $152.00" in captured.out
        assert "Price Data: 3 records" in captured.out
        assert "Dividends: 2 payments, total $0.55" in captured.out
        assert "Stock Splits: 1 splits" in captured.out

    @patch("stockula.database.cli.DatabaseManager")
    def test_query_symbol_data_no_data(self, mock_db_class, capsys):
        """Test querying symbol data with no data available."""
        mock_db = Mock()
        mock_db.get_stock_info.return_value = None
        mock_db.get_price_history.return_value = pd.DataFrame()
        mock_db.get_dividends.return_value = pd.Series(dtype=float)
        mock_db.get_splits.return_value = pd.Series(dtype=float)
        mock_db_class.return_value = mock_db

        query_symbol_data("INVALID")

        captured = capsys.readouterr()
        assert "Data for INVALID:" in captured.out
        assert "No price data available" in captured.out
        assert "No dividend data available" in captured.out
        assert "No split data available" in captured.out

    @patch("stockula.database.cli.DatabaseManager")
    def test_query_symbol_data_partial_info(self, mock_db_class, capsys):
        """Test querying symbol data with partial info."""
        mock_db = Mock()

        # Mock partial stock info (missing some fields)
        mock_info = {
            "longName": "Test Company",
            "marketCap": None,  # Missing market cap
        }
        mock_db.get_stock_info.return_value = mock_info
        mock_db.get_price_history.return_value = pd.DataFrame()
        mock_db.get_dividends.return_value = pd.Series(dtype=float)
        mock_db.get_splits.return_value = pd.Series(dtype=float)
        mock_db_class.return_value = mock_db

        query_symbol_data("TEST")

        captured = capsys.readouterr()
        assert "Name: Test Company" in captured.out
        assert "Sector: N/A" in captured.out
        assert "Market Cap: N/A" in captured.out
        assert "Exchange: N/A" in captured.out


class TestMainCLI:
    """Test main CLI function."""

    def test_main_no_command(self, capsys):
        """Test main function with no command."""
        with patch("sys.argv", ["cli"]):
            main()

        captured = capsys.readouterr()
        assert "usage:" in captured.out

    @patch("stockula.database.cli.fetch_symbol_data")
    def test_main_fetch_single_symbol(self, mock_fetch_symbol):
        """Test main function with fetch command for single symbol."""
        with patch("sys.argv", ["cli", "fetch", "AAPL"]):
            main()

        mock_fetch_symbol.assert_called_once_with("AAPL", None, None)

    @patch("stockula.database.cli.fetch_portfolio_data")
    def test_main_fetch_multiple_symbols(self, mock_fetch_portfolio):
        """Test main function with fetch command for multiple symbols."""
        with patch("sys.argv", ["cli", "fetch", "AAPL", "GOOGL", "MSFT"]):
            main()

        mock_fetch_portfolio.assert_called_once_with(
            ["AAPL", "GOOGL", "MSFT"], None, None
        )

    @patch("stockula.database.cli.fetch_symbol_data")
    def test_main_fetch_with_dates(self, mock_fetch_symbol):
        """Test main function with fetch command and date parameters."""
        with patch(
            "sys.argv",
            ["cli", "fetch", "AAPL", "--start", "2023-01-01", "--end", "2023-12-31"],
        ):
            main()

        mock_fetch_symbol.assert_called_once_with("AAPL", "2023-01-01", "2023-12-31")

    @patch("stockula.database.cli.show_database_stats")
    def test_main_stats_command(self, mock_show_stats):
        """Test main function with stats command."""
        with patch("sys.argv", ["cli", "stats"]):
            main()

        mock_show_stats.assert_called_once()

    @patch("stockula.database.cli.query_symbol_data")
    def test_main_query_command(self, mock_query):
        """Test main function with query command."""
        with patch("sys.argv", ["cli", "query", "AAPL"]):
            main()

        mock_query.assert_called_once_with("AAPL")

    @patch("stockula.database.cli.cleanup_database")
    def test_main_cleanup_command_default(self, mock_cleanup):
        """Test main function with cleanup command (default days)."""
        with patch("sys.argv", ["cli", "cleanup"]):
            main()

        mock_cleanup.assert_called_once_with(365)

    @patch("stockula.database.cli.cleanup_database")
    def test_main_cleanup_command_custom_days(self, mock_cleanup):
        """Test main function with cleanup command and custom days."""
        with patch("sys.argv", ["cli", "cleanup", "--days", "180"]):
            main()

        mock_cleanup.assert_called_once_with(180)

    @patch("stockula.database.cli.fetch_symbol_data")
    def test_main_keyboard_interrupt(self, mock_fetch_symbol, capsys):
        """Test main function handles KeyboardInterrupt."""
        mock_fetch_symbol.side_effect = KeyboardInterrupt()

        with patch("sys.argv", ["cli", "fetch", "AAPL"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Operation cancelled by user" in captured.out

    @patch("stockula.database.cli.fetch_symbol_data")
    def test_main_general_exception(self, mock_fetch_symbol, capsys):
        """Test main function handles general exceptions."""
        mock_fetch_symbol.side_effect = Exception("Database error")

        with patch("sys.argv", ["cli", "fetch", "AAPL"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error: Database error" in captured.out


class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    def test_argument_parsing_fetch(self):
        """Test argument parsing for fetch command."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        # Recreate the fetch parser as in main()
        fetch_parser = subparsers.add_parser("fetch")
        fetch_parser.add_argument("symbols", nargs="+")
        fetch_parser.add_argument("--start")
        fetch_parser.add_argument("--end")

        # Test parsing
        args = parser.parse_args(["fetch", "AAPL", "GOOGL", "--start", "2023-01-01"])

        assert args.command == "fetch"
        assert args.symbols == ["AAPL", "GOOGL"]
        assert args.start == "2023-01-01"
        assert args.end is None

    def test_argument_parsing_cleanup(self):
        """Test argument parsing for cleanup command."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        # Recreate the cleanup parser as in main()
        cleanup_parser = subparsers.add_parser("cleanup")
        cleanup_parser.add_argument("--days", type=int, default=365)

        # Test parsing with custom days
        args = parser.parse_args(["cleanup", "--days", "180"])

        assert args.command == "cleanup"
        assert args.days == 180

    def test_error_scenarios(self):
        """Test various error scenarios."""
        # Test that CLI functions are properly structured
        assert callable(fetch_symbol_data)
        assert callable(fetch_portfolio_data)
        assert callable(show_database_stats)
        assert callable(cleanup_database)
        assert callable(query_symbol_data)
        assert callable(main)

    @patch("builtins.print")
    def test_console_output_formatting(self, mock_print):
        """Test that console output is properly formatted."""
        # Test that functions use print for output
        with patch("stockula.database.cli.DataFetcher") as mock_fetcher_class:
            mock_fetcher = Mock()
            mock_fetcher.get_database_stats.return_value = {"stocks": 10}
            mock_fetcher.get_cached_symbols.return_value = ["AAPL"]
            mock_fetcher_class.return_value = mock_fetcher

            show_database_stats()

            # Verify print was called multiple times for formatted output
            assert mock_print.call_count > 0
