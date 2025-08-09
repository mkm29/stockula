"""Tests for main CLI module."""

import json
from io import StringIO
from unittest.mock import Mock, patch

from stockula.cli import print_results, run_stockula
from stockula.main import setup_logging


class TestLoggingSetup:
    """Test logging configuration."""

    def test_setup_logging_disabled(self, sample_stockula_config):
        """Test logging setup when disabled."""
        sample_stockula_config.logging.enabled = False

        # Create a mock logging manager
        mock_logging_manager = Mock()

        # Create container and override the logging manager
        from stockula.container import Container

        container = Container()
        container.logging_manager.override(mock_logging_manager)
        container.wire(modules=["stockula.main"])

        setup_logging(sample_stockula_config)

        # Verify setup was called on the logging manager
        mock_logging_manager.setup.assert_called_once_with(sample_stockula_config)

    def test_setup_logging_enabled(self, sample_stockula_config):
        """Test logging setup when enabled."""
        sample_stockula_config.logging.enabled = True
        sample_stockula_config.logging.level = "DEBUG"

        # Create a mock logging manager
        mock_logging_manager = Mock()

        # Create container and override the logging manager
        from stockula.container import Container

        container = Container()
        container.logging_manager.override(mock_logging_manager)
        container.wire(modules=["stockula.main"])

        setup_logging(sample_stockula_config)

        # Verify setup was called on the logging manager
        mock_logging_manager.setup.assert_called_once_with(sample_stockula_config)

    def test_setup_logging_with_file(self, sample_stockula_config):
        """Test logging setup with file output."""
        sample_stockula_config.logging.enabled = True
        sample_stockula_config.logging.log_to_file = True
        sample_stockula_config.logging.log_file = "test.log"

        # Create a mock logging manager
        mock_logging_manager = Mock()

        # Create container and override the logging manager
        from stockula.container import Container

        container = Container()
        container.logging_manager.override(mock_logging_manager)
        container.wire(modules=["stockula.main"])

        setup_logging(sample_stockula_config)

        # Verify setup was called with the correct config
        mock_logging_manager.setup.assert_called_once_with(sample_stockula_config)
        # The actual file handler creation is handled inside the LoggingManager


class TestPrintResults:
    """Test result printing."""

    def test_print_results_console(self):
        """Test console output format."""
        results = {
            "technical_analysis": [
                {
                    "ticker": "AAPL",
                    "indicators": {
                        "SMA_20": 150.0,
                        "RSI": 65.0,
                        "MACD": {"MACD": 0.5, "MACD_SIGNAL": 0.3},
                    },
                }
            ]
        }

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            print_results(results, "console")
            output = mock_stdout.getvalue()

            assert "Technical Analysis Results" in output
            assert "AAPL" in output
            # Rich tables format the output differently
            assert "SMA_20" in output
            assert "150.00" in output
            assert "RSI" in output
            assert "65.00" in output

    def test_print_results_json(self):
        """Test JSON output format."""
        results = {"technical_analysis": [{"ticker": "AAPL", "indicators": {"SMA_20": 150.0}}]}

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            print_results(results, "json")
            output = mock_stdout.getvalue()

            # Should be valid JSON
            parsed = json.loads(output)
            assert parsed["technical_analysis"][0]["ticker"] == "AAPL"

    def test_print_results_backtesting(self):
        """Test printing backtest results."""
        results = {
            "backtesting": [
                {
                    "ticker": "AAPL",
                    "strategy": "SMACross",
                    "parameters": {"fast_period": 10, "slow_period": 20},
                    "return_pct": 15.5,
                    "sharpe_ratio": 1.2,
                    "max_drawdown_pct": -10.0,
                    "num_trades": 25,
                    "win_rate": 60.0,
                    "initial_cash": 10000,
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                    "trading_days": 252,
                    "calendar_days": 365,
                }
            ]
        }

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            print_results(results, "console")
            output = mock_stdout.getvalue()

            assert "Backtesting Results" in output
            # Check for portfolio information display
            assert "Portfolio Information:" in output
            assert "Initial Capital: $10,000" in output
            assert "Start Date: 2023-01-01" in output
            assert "End Date: 2023-12-31" in output
            # Trading Days and Calendar Days are no longer displayed in the current format
            # Check for backtest result details (these are in table format now)
            assert "AAPL" in output
            assert "15.5" in output  # Return percentage in table
            assert "60.0" in output  # Win rate in table

    def test_print_results_forecasting(self):
        """Test printing forecast results."""
        results = {
            "forecasting": [
                {
                    "ticker": "AAPL",
                    "current_price": 150.0,
                    "forecast_price": 155.0,
                    "lower_bound": 145.0,
                    "upper_bound": 165.0,
                    "forecast_length": 30,
                    "best_model": "ARIMA",
                }
            ]
        }

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            print_results(results, "console")
            output = mock_stdout.getvalue()

            assert "Forecasting Results" in output
            # Rich tables format the output differently - check for price values
            assert "150" in output  # Current price
            assert "155" in output  # Forecast price
            assert "ARIMA" in output  # Best model


class TestRunStockula:
    """Test run_stockula function."""

    def test_run_stockula_with_config_file(self, temp_config_file, mock_container):
        """Test run_stockula with config file."""
        # Mock the manager methods
        mock_manager = Mock()
        mock_manager.create_portfolio.return_value = Mock()
        mock_manager.display_portfolio_summary = Mock()
        mock_manager.display_portfolio_holdings = Mock()
        mock_manager.run_main_processing.return_value = {"technical_analysis": [{"ticker": "AAPL", "indicators": {}}]}

        with patch("stockula.cli.create_container", return_value=mock_container):
            with patch("stockula.cli.StockulaManager", return_value=mock_manager):
                with patch("stockula.cli.print_results") as mock_print:
                    run_stockula(config=temp_config_file, mode="ta")

                    # Verify manager methods were called
                    mock_manager.create_portfolio.assert_called_once()
                    mock_manager.run_main_processing.assert_called_once()
                    mock_print.assert_called_once()

    def test_run_stockula_with_ticker_override(self, mock_container):
        """Test run_stockula with ticker override."""
        # Mock the manager methods
        mock_manager = Mock()
        mock_manager.create_portfolio.return_value = Mock()
        mock_manager.display_portfolio_summary = Mock()
        mock_manager.display_portfolio_holdings = Mock()
        mock_manager.run_main_processing.return_value = {"technical_analysis": [{"ticker": "TSLA", "indicators": {}}]}

        with patch("stockula.cli.create_container", return_value=mock_container):
            with patch("stockula.cli.StockulaManager", return_value=mock_manager):
                with patch("stockula.cli.print_results"):
                    run_stockula(ticker="TSLA", mode="ta")

                    # Verify config was updated with ticker
                    mock_config = mock_container.stockula_config()
                    assert len(mock_config.portfolio.tickers) == 1
                    assert mock_config.portfolio.tickers[0].symbol == "TSLA"

    def test_run_stockula_save_config(self, tmp_path, mock_container):
        """Test saving configuration."""
        config_path = str(tmp_path / "saved_.stockula.yaml")

        with patch("stockula.cli.create_container", return_value=mock_container):
            with patch("stockula.cli.save_config") as mock_save:
                run_stockula(save_config_path=config_path, mode="ta")

                # Check that save_config was called with the correct path
                mock_save.assert_called_once()
                assert mock_save.call_args[0][1] == config_path

    def test_run_stockula_optimize_allocation(self, mock_container):
        """Test optimize allocation mode."""
        # Mock the manager methods
        mock_manager = Mock()
        mock_manager.run_optimize_allocation.return_value = None

        with patch("stockula.cli.create_container", return_value=mock_container):
            with patch("stockula.cli.StockulaManager", return_value=mock_manager):
                run_stockula(mode="optimize-allocation")

                # Verify optimize allocation was called
                mock_manager.run_optimize_allocation.assert_called_once()

    def test_run_stockula_all_modes(self, mock_container):
        """Test running all analysis modes."""
        # Mock the manager methods
        mock_manager = Mock()
        mock_manager.create_portfolio.return_value = Mock()
        mock_manager.display_portfolio_summary = Mock()
        mock_manager.display_portfolio_holdings = Mock()
        mock_manager.run_main_processing.return_value = {
            "technical_analysis": [{"ticker": "AAPL", "indicators": {}}],
            "backtesting": [{"ticker": "AAPL", "strategy": "SMACross", "return_pct": 15.5}],
            "forecasting": [{"ticker": "AAPL", "forecast_price": 155.0}],
        }
        mock_manager.create_portfolio_backtest_results = Mock()

        with patch("stockula.cli.create_container", return_value=mock_container):
            with patch("stockula.cli.StockulaManager", return_value=mock_manager):
                with patch("stockula.cli.print_results") as mock_print:
                    with patch("stockula.cli.ResultsDisplay"):
                        run_stockula(mode="all")

                        # Verify all results were processed
                        mock_manager.run_main_processing.assert_called_once_with("all", mock_manager.create_portfolio())
                        mock_print.assert_called_once()
                        results = mock_print.call_args[0][0]
                        assert "technical_analysis" in results
                        assert "backtesting" in results
                        assert "forecasting" in results
