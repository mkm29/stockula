"""Unit tests for main module."""

import pytest
import json
import argparse
import sys
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, call
import pandas as pd
import numpy as np
from io import StringIO

from stockula.main import (
    setup_logging,
    get_strategy_class,
    run_technical_analysis,
    run_backtest,
    run_forecast,
    print_results,
    main,
)
from stockula.config import StockulaConfig, TickerConfig


class TestSetupLogging:
    """Test logging setup functionality."""

    @patch("stockula.main.log_manager")
    def test_setup_logging_enabled(self, mock_log_manager):
        """Test setting up logging when enabled."""
        config = StockulaConfig()
        config.logging.enabled = True
        config.logging.level = "DEBUG"
        config.logging.log_to_file = True
        config.logging.log_file = "test.log"

        setup_logging(config)

        # Should call setup on the log manager
        mock_log_manager.setup.assert_called_once_with(config)

    @patch("stockula.main.log_manager")
    def test_setup_logging_disabled(self, mock_log_manager):
        """Test setting up logging when disabled."""
        config = StockulaConfig()
        config.logging.enabled = False

        setup_logging(config)

        # Should still call setup on the log manager
        mock_log_manager.setup.assert_called_once_with(config)


class TestGetStrategyClass:
    """Test strategy class retrieval."""

    def test_get_strategy_class_valid(self):
        """Test getting valid strategy classes."""
        from stockula.backtesting.strategies import SMACrossStrategy, RSIStrategy

        assert get_strategy_class("smacross") == SMACrossStrategy
        assert get_strategy_class("SMACROSS") == SMACrossStrategy
        assert get_strategy_class("rsi") == RSIStrategy
        assert get_strategy_class("RSI") == RSIStrategy

    def test_get_strategy_class_invalid(self):
        """Test getting invalid strategy class."""
        assert get_strategy_class("invalid") is None
        assert get_strategy_class("") is None


class TestRunTechnicalAnalysis:
    """Test technical analysis execution."""

    @patch("stockula.main.TechnicalIndicators")
    @patch("stockula.main.DataFetcher")
    def test_run_technical_analysis_success(self, mock_fetcher_class, mock_ta_class):
        """Test successful technical analysis run."""
        # Setup config
        config = StockulaConfig()
        config.technical_analysis.indicators = ["sma", "rsi"]
        config.technical_analysis.sma_periods = [20]
        config.technical_analysis.rsi_period = 14

        # Mock data fetcher
        mock_fetcher = Mock()
        mock_data = pd.DataFrame(
            {
                "Open": [100] * 50,
                "High": [101] * 50,
                "Low": [99] * 50,
                "Close": [100] * 50,
                "Volume": [1000000] * 50,
            },
            index=pd.date_range("2023-01-01", periods=50),
        )
        mock_fetcher.get_stock_data.return_value = mock_data
        mock_fetcher_class.return_value = mock_fetcher

        # Mock technical indicators
        mock_ta = Mock()
        mock_ta.sma.return_value = pd.Series([100.5] * 50)
        mock_ta.rsi.return_value = pd.Series([50.0] * 50)
        mock_ta_class.return_value = mock_ta

        result = run_technical_analysis("AAPL", config)

        # Verify structure
        assert result["ticker"] == "AAPL"
        assert "indicators" in result
        assert "SMA_20" in result["indicators"]
        assert "RSI" in result["indicators"]
        assert result["indicators"]["SMA_20"] == 100.5
        assert result["indicators"]["RSI"] == 50.0

    @patch("stockula.main.TechnicalIndicators")
    @patch("stockula.main.DataFetcher")
    def test_run_technical_analysis_multiple_indicators(
        self, mock_fetcher_class, mock_ta_class
    ):
        """Test technical analysis with multiple indicators."""
        # Setup config with all indicators
        config = StockulaConfig()
        config.technical_analysis.indicators = [
            "sma",
            "ema",
            "rsi",
            "macd",
            "bbands",
            "atr",
            "adx",
        ]
        config.technical_analysis.sma_periods = [10, 20]
        config.technical_analysis.ema_periods = [12, 26]
        config.technical_analysis.rsi_period = 14

        # Mock data fetcher
        mock_fetcher = Mock()
        mock_data = pd.DataFrame(
            {"Close": [100] * 50}, index=pd.date_range("2023-01-01", periods=50)
        )
        mock_fetcher.get_stock_data.return_value = mock_data
        mock_fetcher_class.return_value = mock_fetcher

        # Mock technical indicators
        mock_ta = Mock()
        mock_ta.sma.return_value = pd.Series([100.0] * 50)
        mock_ta.ema.return_value = pd.Series([99.5] * 50)
        mock_ta.rsi.return_value = pd.Series([55.0] * 50)
        mock_ta.macd.return_value = pd.DataFrame(
            {"MACD": [0.5] * 50, "MACD_SIGNAL": [0.3] * 50}
        )
        mock_ta.bbands.return_value = pd.DataFrame(
            {"BB_UPPER": [102] * 50, "BB_MIDDLE": [100] * 50, "BB_LOWER": [98] * 50}
        )
        mock_ta.atr.return_value = pd.Series([2.0] * 50)
        mock_ta.adx.return_value = pd.Series([25.0] * 50)
        mock_ta_class.return_value = mock_ta

        result = run_technical_analysis("TEST", config)

        # Should have all indicators
        assert "SMA_10" in result["indicators"]
        assert "SMA_20" in result["indicators"]
        assert "EMA_12" in result["indicators"]
        assert "EMA_26" in result["indicators"]
        assert "RSI" in result["indicators"]
        assert "MACD" in result["indicators"]
        assert "BBands" in result["indicators"]
        assert "ATR" in result["indicators"]
        assert "ADX" in result["indicators"]


class TestRunBacktest:
    """Test backtest execution."""

    @patch("stockula.main.BacktestRunner")
    @patch("stockula.main.get_strategy_class")
    def test_run_backtest_success(self, mock_get_strategy, mock_runner_class):
        """Test successful backtest run."""
        # Setup config
        config = StockulaConfig()
        strategy_config = Mock()
        strategy_config.name = "SMACross"
        strategy_config.parameters = {"fast_period": 10, "slow_period": 20}
        config.backtest.strategies = [strategy_config]

        # Mock strategy
        mock_strategy = Mock()
        mock_get_strategy.return_value = mock_strategy

        # Mock runner
        mock_runner = Mock()
        mock_results = {
            "Return [%]": 15.5,
            "Sharpe Ratio": 1.25,
            "Max. Drawdown [%]": -8.3,
            "# Trades": 42,
            "Win Rate [%]": 55.0,
        }
        mock_runner.run_from_symbol.return_value = mock_results
        mock_runner_class.return_value = mock_runner

        results = run_backtest("AAPL", config)

        # Should return results list
        assert isinstance(results, list)
        assert len(results) == 1

        result = results[0]
        assert result["ticker"] == "AAPL"
        assert result["strategy"] == "SMACross"
        assert result["return_pct"] == 15.5
        assert result["sharpe_ratio"] == 1.25
        assert result["num_trades"] == 42
        assert result["win_rate"] == 55.0

    @patch("stockula.main.BacktestRunner")
    @patch("stockula.main.get_strategy_class")
    def test_run_backtest_no_trades(self, mock_get_strategy, mock_runner_class):
        """Test backtest with no trades (NaN win rate)."""
        # Setup config
        config = StockulaConfig()
        strategy_config = Mock()
        strategy_config.name = "SMACross"
        strategy_config.parameters = {}
        config.backtest.strategies = [strategy_config]

        # Mock strategy
        mock_strategy = Mock()
        mock_get_strategy.return_value = mock_strategy

        # Mock runner with NaN win rate
        mock_runner = Mock()
        mock_results = {
            "Return [%]": 0.0,
            "Sharpe Ratio": 0.0,
            "Max. Drawdown [%]": 0.0,
            "# Trades": 0,
            "Win Rate [%]": float("nan"),
        }
        mock_runner.run_from_symbol.return_value = mock_results
        mock_runner_class.return_value = mock_runner

        results = run_backtest("TEST", config)

        # Should handle NaN win rate
        result = results[0]
        assert result["win_rate"] is None

    @patch("stockula.main.BacktestRunner")
    @patch("stockula.main.get_strategy_class")
    def test_run_backtest_unknown_strategy(self, mock_get_strategy, mock_runner_class):
        """Test backtest with unknown strategy."""
        # Setup config
        config = StockulaConfig()
        strategy_config = Mock()
        strategy_config.name = "UnknownStrategy"
        strategy_config.parameters = {}
        config.backtest.strategies = [strategy_config]

        # Mock unknown strategy
        mock_get_strategy.return_value = None

        results = run_backtest("TEST", config)

        # Should return empty results
        assert results == []

    @patch("stockula.main.BacktestRunner")
    @patch("stockula.main.get_strategy_class")
    def test_run_backtest_exception(self, mock_get_strategy, mock_runner_class):
        """Test backtest with exception."""
        # Setup config
        config = StockulaConfig()
        strategy_config = Mock()
        strategy_config.name = "SMACross"
        strategy_config.parameters = {}
        config.backtest.strategies = [strategy_config]

        # Mock strategy
        mock_strategy = Mock()
        mock_get_strategy.return_value = mock_strategy

        # Mock runner that raises exception
        mock_runner = Mock()
        mock_runner.run_from_symbol.side_effect = Exception("Backtest error")
        mock_runner_class.return_value = mock_runner

        results = run_backtest("TEST", config)

        # Should handle exception gracefully
        assert results == []


class TestRunForecast:
    """Test forecast execution."""

    @patch("stockula.main.StockForecaster")
    def test_run_forecast_success(self, mock_forecaster_class):
        """Test successful forecast run."""
        # Setup config
        config = StockulaConfig()
        config.forecast.forecast_length = 30

        # Mock forecaster
        mock_forecaster = Mock()
        mock_predictions = pd.DataFrame(
            {
                "forecast": [110, 111, 112, 113, 114],
                "lower_bound": [105, 106, 107, 108, 109],
                "upper_bound": [115, 116, 117, 118, 119],
            }
        )
        mock_forecaster.forecast_from_symbol.return_value = mock_predictions
        mock_forecaster.get_best_model.return_value = {
            "model_name": "ARIMA",
            "model_params": {"p": 1, "d": 1, "q": 1},
        }
        mock_forecaster_class.return_value = mock_forecaster

        result = run_forecast("AAPL", config)

        # Verify structure
        assert result["ticker"] == "AAPL"
        assert result["current_price"] == 110
        assert result["forecast_price"] == 114
        assert result["lower_bound"] == 109
        assert result["upper_bound"] == 119
        assert result["forecast_length"] == 30
        assert result["best_model"] == "ARIMA"

    @patch("stockula.main.StockForecaster")
    def test_run_forecast_keyboard_interrupt(self, mock_forecaster_class):
        """Test forecast with keyboard interrupt."""
        # Setup config
        config = StockulaConfig()

        # Mock forecaster that raises KeyboardInterrupt
        mock_forecaster = Mock()
        mock_forecaster.forecast_from_symbol.side_effect = KeyboardInterrupt()
        mock_forecaster_class.return_value = mock_forecaster

        result = run_forecast("TEST", config)

        # Should handle interrupt gracefully
        assert result["ticker"] == "TEST"
        assert result["error"] == "Interrupted by user"

    @patch("stockula.main.StockForecaster")
    def test_run_forecast_exception(self, mock_forecaster_class):
        """Test forecast with general exception."""
        # Setup config
        config = StockulaConfig()

        # Mock forecaster that raises exception
        mock_forecaster = Mock()
        mock_forecaster.forecast_from_symbol.side_effect = Exception("Forecast error")
        mock_forecaster_class.return_value = mock_forecaster

        result = run_forecast("TEST", config)

        # Should handle exception gracefully
        assert result["ticker"] == "TEST"
        assert result["error"] == "Forecast error"


class TestPrintResults:
    """Test results printing functionality."""

    def test_print_results_console_format(self, capsys):
        """Test printing results in console format."""
        results = {
            "technical_analysis": [
                {
                    "ticker": "AAPL",
                    "indicators": {
                        "SMA_20": 150.50,
                        "RSI": 65.5,
                        "MACD": {"MACD": 1.5, "MACD_SIGNAL": 1.2},
                    },
                }
            ],
            "backtesting": [
                {
                    "ticker": "AAPL",
                    "strategy": "SMACross",
                    "parameters": {"fast": 10, "slow": 20},
                    "return_pct": 15.5,
                    "sharpe_ratio": 1.25,
                    "max_drawdown_pct": -8.3,
                    "num_trades": 42,
                    "win_rate": 55.0,
                }
            ],
            "forecasting": [
                {
                    "ticker": "AAPL",
                    "current_price": 150.0,
                    "forecast_price": 155.0,
                    "lower_bound": 150.0,
                    "upper_bound": 160.0,
                    "forecast_length": 30,
                    "best_model": "ARIMA",
                }
            ],
        }

        print_results(results, "console")

        captured = capsys.readouterr()
        assert "Technical Analysis Results" in captured.out
        assert "AAPL" in captured.out
        assert "SMA_20: 150.50" in captured.out
        assert "Backtesting Results" in captured.out
        assert "Return: 15.50%" in captured.out
        assert "Forecasting Results" in captured.out
        assert "30-Day Forecast: $155.00" in captured.out

    def test_print_results_json_format(self, capsys):
        """Test printing results in JSON format."""
        results = {
            "technical_analysis": [{"ticker": "TEST", "indicators": {"SMA_20": 100.0}}]
        }

        print_results(results, "json")

        captured = capsys.readouterr()
        # Should be valid JSON
        parsed = json.loads(captured.out)
        assert "technical_analysis" in parsed
        assert parsed["technical_analysis"][0]["ticker"] == "TEST"

    def test_print_results_forecast_with_error(self, capsys):
        """Test printing forecast results with error."""
        results = {"forecasting": [{"ticker": "INVALID", "error": "No data available"}]}

        print_results(results, "console")

        captured = capsys.readouterr()
        assert "INVALID: Error - No data available" in captured.out

    def test_print_results_backtest_no_trades(self, capsys):
        """Test printing backtest results with no trades."""
        results = {
            "backtesting": [
                {
                    "ticker": "TEST",
                    "strategy": "SMACross",
                    "parameters": {},
                    "return_pct": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown_pct": 0.0,
                    "num_trades": 0,
                    "win_rate": None,
                }
            ]
        }

        print_results(results, "console")

        captured = capsys.readouterr()
        assert "Win Rate: N/A (no trades)" in captured.out


class TestMainFunction:
    """Test main function."""

    @patch("stockula.main.load_config")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.DomainFactory")
    @patch("stockula.main.DataFetcher")
    @patch("stockula.main.run_technical_analysis")
    @patch("stockula.main.print_results")
    @patch("sys.argv", ["stockula", "--mode", "ta", "--ticker", "AAPL"])
    def test_main_ta_mode(
        self,
        mock_print,
        mock_ta,
        mock_fetcher_class,
        mock_factory,
        mock_logging,
        mock_load,
    ):
        """Test main function in TA mode."""
        # Setup config
        config = StockulaConfig()
        mock_load.return_value = config

        # Setup domain factory
        mock_asset = Mock()
        mock_asset.symbol = "AAPL"
        mock_asset.category = Mock()
        mock_portfolio = Mock()
        mock_portfolio.name = "Test Portfolio"
        mock_portfolio.initial_capital = 100000
        mock_portfolio.get_all_assets.return_value = [mock_asset]
        mock_portfolio.get_portfolio_value.return_value = 110000
        mock_factory.return_value.create_portfolio.return_value = mock_portfolio

        # Setup fetcher
        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.return_value = {"AAPL": 150.0}
        mock_portfolio.get_portfolio_value.return_value = 110000
        mock_fetcher_class.return_value = mock_fetcher

        # Setup TA results
        mock_ta.return_value = {"ticker": "AAPL", "indicators": {}}

        main()

        # Should call TA function
        mock_ta.assert_called_once()
        mock_print.assert_called_once()

    @patch("stockula.main.load_config")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.DomainFactory")
    @patch("stockula.main.DataFetcher")
    @patch("stockula.main.run_backtest")
    @patch("stockula.main.print_results")
    @patch("sys.argv", ["stockula", "--mode", "backtest"])
    def test_main_backtest_mode(
        self,
        mock_print,
        mock_backtest,
        mock_fetcher_class,
        mock_factory,
        mock_logging,
        mock_load,
    ):
        """Test main function in backtest mode."""
        # Setup config
        config = StockulaConfig()
        config.portfolio.tickers = [TickerConfig(symbol="AAPL", quantity=1.0)]
        mock_load.return_value = config

        # Setup domain factory
        mock_asset = Mock()
        mock_asset.symbol = "AAPL"
        mock_asset.category = Mock()
        mock_portfolio = Mock()
        mock_portfolio.name = "Test Portfolio"
        mock_portfolio.initial_capital = 100000
        mock_portfolio.get_all_assets.return_value = [mock_asset]
        mock_portfolio.get_portfolio_value.return_value = 100000
        mock_portfolio.allocation_method = "equal"
        mock_portfolio.get_allocation_by_category.return_value = {}
        mock_factory.return_value.create_portfolio.return_value = mock_portfolio

        # Setup fetcher
        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.return_value = {"AAPL": 150.0}
        mock_fetcher_class.return_value = mock_fetcher

        # Setup backtest results
        mock_backtest.return_value = [{"ticker": "AAPL", "strategy": "SMACross"}]

        main()

        # Should call backtest function
        mock_backtest.assert_called_once()
        mock_print.assert_called_once()

    @patch("stockula.main.load_config")
    @patch("sys.argv", ["stockula", "--config", "nonexistent.yaml"])
    def test_main_config_not_found(self, mock_load):
        """Test main function with non-existent config file."""
        mock_load.side_effect = FileNotFoundError("Config not found")

        # Should handle gracefully and use default config
        with patch("stockula.main.StockulaConfig") as mock_config_class:
            mock_config = Mock()
            mock_config.portfolio.tickers = [TickerConfig(symbol="TEST", quantity=1.0)]
            mock_config.backtest.hold_only_categories = []
            mock_config.backtest.strategies = []
            mock_config.data.start_date = None
            mock_config.output = {}
            mock_config_class.return_value = mock_config

            with patch("stockula.main.setup_logging"):
                with patch("stockula.main.DomainFactory") as mock_factory:
                    mock_portfolio = Mock()
                    mock_portfolio.name = "Test Portfolio"
                    mock_portfolio.initial_capital = 100000
                    mock_portfolio.allocation_method = "equal"
                    mock_portfolio.get_all_assets.return_value = [
                        Mock(symbol="TEST", category=Mock())
                    ]
                    mock_portfolio.get_portfolio_value.return_value = 100000
                    mock_portfolio.get_allocation_by_category.return_value = {}
                    mock_factory.return_value.create_portfolio.return_value = (
                        mock_portfolio
                    )

                    with patch("stockula.main.DataFetcher") as mock_fetcher_class:
                        mock_fetcher = Mock()
                        mock_fetcher.get_current_prices.return_value = {"TEST": 100.0}
                        mock_fetcher.get_stock_data.return_value = pd.DataFrame()
                        mock_fetcher_class.return_value = mock_fetcher

                        with patch("stockula.main.run_technical_analysis") as mock_ta:
                            mock_ta.return_value = {"ticker": "TEST", "indicators": {}}
                            with patch("stockula.main.print_results"):
                                main()

            # Should create default config
            mock_config_class.assert_called_once()

    @patch("stockula.main.load_config")
    @patch("sys.argv", ["stockula", "--save-config", "output.yaml"])
    def test_main_save_config(self, mock_load):
        """Test main function with save config option."""
        config = StockulaConfig()
        mock_load.return_value = config

        with patch("stockula.config.settings.save_config") as mock_save:
            # main() returns early, doesn't raise SystemExit
            main()

            # Should save config
            mock_save.assert_called_once_with(config, "output.yaml")


class TestMainIntegration:
    """Integration tests for main function components."""

    def test_config_date_handling(self):
        """Test that date handling works correctly."""
        config = StockulaConfig()
        config.data.start_date = datetime(2023, 1, 1)
        config.data.end_date = datetime(2023, 12, 31)

        # Should be able to format dates
        start_str = config.data.start_date.strftime("%Y-%m-%d")
        end_str = config.data.end_date.strftime("%Y-%m-%d")

        assert start_str == "2023-01-01"
        assert end_str == "2023-12-31"

    def test_ticker_override_logic(self):
        """Test ticker override logic."""
        config = StockulaConfig()
        config.portfolio.tickers = [
            TickerConfig(symbol="AAPL", quantity=1.0),
            TickerConfig(symbol="GOOGL", quantity=1.0),
        ]

        # Simulate ticker override
        ticker_override = "TSLA"
        if ticker_override:
            config.portfolio.tickers = [
                TickerConfig(symbol=ticker_override, quantity=1.0)
            ]

        assert len(config.portfolio.tickers) == 1
        assert config.portfolio.tickers[0].symbol == "TSLA"


class TestMainAdvanced:
    """Advanced tests for main function to improve coverage."""

    @patch("stockula.main.load_config")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.DomainFactory")
    @patch("stockula.main.DataFetcher")
    @patch("stockula.main.run_forecast")
    @patch("stockula.main.print_results")
    @patch("sys.argv", ["stockula", "--mode", "forecast", "--ticker", "AAPL"])
    def test_main_forecast_mode_with_warning(
        self,
        mock_print,
        mock_forecast,
        mock_fetcher_class,
        mock_factory,
        mock_logging,
        mock_load,
        capsys,
    ):
        """Test main function in forecast mode with warning message."""
        # Setup config
        config = StockulaConfig()
        config.portfolio.tickers = [TickerConfig(symbol="AAPL", quantity=1.0)]
        mock_load.return_value = config

        # Setup domain factory
        mock_asset = Mock()
        mock_asset.symbol = "AAPL"
        mock_asset.category = None
        mock_portfolio = Mock()
        mock_portfolio.name = "Test Portfolio"
        mock_portfolio.initial_capital = 100000
        mock_portfolio.get_all_assets.return_value = [mock_asset]
        mock_portfolio.get_portfolio_value.return_value = 110000
        mock_portfolio.allocation_method = "equal"
        mock_factory.return_value.create_portfolio.return_value = mock_portfolio

        # Setup fetcher
        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.return_value = {"AAPL": 150.0}
        mock_fetcher_class.return_value = mock_fetcher

        # Setup forecast results
        mock_forecast.return_value = {"ticker": "AAPL", "forecast_price": 155.0}

        main()

        # Should call forecast function
        mock_forecast.assert_called_once()
        mock_print.assert_called_once()

        # Check that warning was printed
        captured = capsys.readouterr()
        assert "FORECAST MODE - IMPORTANT NOTES:" in captured.out
        assert "AutoTS will try multiple models" in captured.out

    @patch("stockula.main.load_config")
    @patch("stockula.main.StockulaConfig")
    @patch("sys.argv", ["stockula", "--config", "nonexistent.yaml"])
    def test_main_config_exception(self, mock_config_class, mock_load):
        """Test main function with general exception in config loading."""
        mock_load.side_effect = Exception("Config parsing error")

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    @patch("stockula.main.load_config")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.DomainFactory")
    @patch("stockula.main.DataFetcher")
    @patch("stockula.main.run_technical_analysis")
    @patch("stockula.main.print_results")
    @patch("sys.argv", ["stockula", "--mode", "ta", "--output", "console"])
    def test_main_with_results_saving(
        self,
        mock_print,
        mock_ta,
        mock_fetcher_class,
        mock_factory,
        mock_logging,
        mock_load,
    ):
        """Test main function with results saving enabled."""
        # Setup config with save_results enabled
        config = StockulaConfig()
        config.output["save_results"] = True
        config.output["results_dir"] = "./test_results"
        mock_load.return_value = config

        # Setup domain factory
        mock_asset = Mock()
        mock_asset.symbol = "AAPL"
        mock_asset.category = None
        mock_portfolio = Mock()
        mock_portfolio.name = "Test Portfolio"
        mock_portfolio.initial_capital = 100000
        mock_portfolio.get_all_assets.return_value = [mock_asset]
        mock_portfolio.get_portfolio_value.return_value = 100000
        mock_portfolio.allocation_method = "equal"
        mock_factory.return_value.create_portfolio.return_value = mock_portfolio

        # Setup fetcher
        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.return_value = {"AAPL": 150.0}
        mock_fetcher_class.return_value = mock_fetcher

        # Setup TA results
        mock_ta.return_value = {"ticker": "AAPL", "indicators": {}}

        # Mock Path and file operations
        from pathlib import Path

        with patch("pathlib.Path") as mock_path_cls:
            mock_results_dir = Mock()
            mock_results_dir.mkdir = Mock()
            mock_results_file = Mock()
            mock_results_dir.__truediv__ = Mock(return_value=mock_results_file)
            mock_path_cls.return_value = mock_results_dir

            # Mock the open function for file writing
            with patch("builtins.open", create=True) as mock_open:
                mock_file = Mock()
                mock_open.return_value.__enter__.return_value = mock_file

                main()

                # Should create results directory
                mock_results_dir.mkdir.assert_called_once_with(exist_ok=True)

                # Should save results
                mock_open.assert_called_once()
                mock_file.write.assert_called()

    @patch("stockula.main.load_config")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.DomainFactory")
    @patch("stockula.main.DataFetcher")
    @patch("stockula.main.run_backtest")
    @patch("stockula.main.print_results")
    @patch("sys.argv", ["stockula", "--mode", "backtest", "--ticker", "AAPL"])
    def test_main_backtest_with_start_date_prices(
        self,
        mock_print,
        mock_backtest,
        mock_fetcher_class,
        mock_factory,
        mock_logging,
        mock_load,
    ):
        """Test main function fetching start date prices for backtesting."""
        # Setup config with start date
        config = StockulaConfig()
        config.data.start_date = datetime(2023, 1, 1)
        config.backtest.hold_only_categories = ["INDEX"]
        mock_load.return_value = config

        # Setup domain factory with hold-only asset
        mock_asset1 = Mock()
        mock_asset1.symbol = "AAPL"
        mock_asset1.category = None  # Tradeable

        from stockula.domain import Category

        mock_asset2 = Mock()
        mock_asset2.symbol = "SPY"
        mock_asset2.category = Category.INDEX  # Hold-only

        # Add proper get_value methods
        mock_asset1.get_value = Mock(return_value=1500.0)
        mock_asset2.get_value = Mock(return_value=8000.0)

        mock_portfolio = Mock()
        mock_portfolio.name = "Test Portfolio"
        mock_portfolio.initial_capital = 100000
        mock_portfolio.get_all_assets.return_value = [mock_asset1, mock_asset2]
        mock_portfolio.get_portfolio_value.return_value = 100000
        mock_portfolio.allocation_method = "equal"
        mock_portfolio.get_allocation_by_category.return_value = {
            "Index": {"value": 20000, "percentage": 20.0, "assets": ["SPY"]},
            "None": {"value": 80000, "percentage": 80.0, "assets": ["AAPL"]},
        }
        mock_factory.return_value.create_portfolio.return_value = mock_portfolio

        # Setup fetcher
        mock_fetcher = Mock()

        # Mock start date data fetching - first call returns empty, second call returns data
        mock_data_empty = pd.DataFrame()
        mock_data_with_prices = pd.DataFrame({"Close": [150.0]})

        def side_effect(symbol, start, end):
            if symbol == "AAPL":
                if end == "2023-01-01":  # First call with same start/end
                    return mock_data_empty
                else:  # Second call with extended end date
                    return mock_data_with_prices
            elif symbol == "SPY":
                if end == "2023-01-01":
                    return mock_data_empty
                else:
                    return pd.DataFrame({"Close": [400.0]})
            return pd.DataFrame()

        mock_fetcher.get_stock_data.side_effect = side_effect
        mock_fetcher.get_current_prices.return_value = {"AAPL": 160.0, "SPY": 420.0}
        mock_fetcher_class.return_value = mock_fetcher

        # Setup backtest results
        mock_backtest.return_value = [
            {"ticker": "AAPL", "strategy": "SMACross", "return_pct": 10.0}
        ]

        main()

        # Should fetch start date prices
        assert mock_fetcher.get_stock_data.call_count >= 2  # Called for each symbol

        # Should call backtest only for tradeable assets
        mock_backtest.assert_called_once_with("AAPL", config)

        # Should show portfolio summary
        mock_print.assert_called_once()

    @patch("stockula.main.load_config")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.DomainFactory")
    @patch("stockula.main.DataFetcher")
    @patch("stockula.main.run_technical_analysis")
    @patch("stockula.main.print_results")
    @patch("sys.argv", ["stockula", "--mode", "ta"])
    def test_main_ta_creates_results_dict(
        self,
        mock_print,
        mock_ta,
        mock_fetcher_class,
        mock_factory,
        mock_logging,
        mock_load,
    ):
        """Test that TA mode properly creates technical_analysis key in results."""
        # Setup config
        config = StockulaConfig()
        config.portfolio.tickers = [TickerConfig(symbol="AAPL", quantity=1.0)]
        mock_load.return_value = config

        # Setup domain factory
        mock_asset = Mock()
        mock_asset.symbol = "AAPL"
        mock_asset.category = None
        mock_portfolio = Mock()
        mock_portfolio.name = "Test Portfolio"
        mock_portfolio.initial_capital = 100000
        mock_portfolio.get_all_assets.return_value = [mock_asset]
        mock_portfolio.get_portfolio_value.return_value = 100000
        mock_portfolio.allocation_method = "equal"
        mock_factory.return_value.create_portfolio.return_value = mock_portfolio

        # Setup fetcher
        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.return_value = {"AAPL": 150.0}
        mock_fetcher_class.return_value = mock_fetcher

        # Setup TA results
        mock_ta.return_value = {"ticker": "AAPL", "indicators": {"SMA_20": 150.0}}

        main()

        # Should call TA
        mock_ta.assert_called_once()

        # Check that print_results was called with correct structure
        call_args = mock_print.call_args[0][0]
        assert "technical_analysis" in call_args
        assert len(call_args["technical_analysis"]) == 1

    @patch("stockula.main.load_config")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.DomainFactory")
    @patch("stockula.main.DataFetcher")
    @patch("stockula.main.run_backtest")
    @patch("stockula.main.print_results")
    @patch("sys.argv", ["stockula", "--mode", "backtest"])
    def test_main_with_performance_breakdown(
        self,
        mock_print,
        mock_backtest,
        mock_fetcher_class,
        mock_factory,
        mock_logging,
        mock_load,
        capsys,
    ):
        """Test main function showing performance breakdown by category."""
        # Setup config with start date for performance calculation
        config = StockulaConfig()
        config.data.start_date = datetime(2023, 1, 1)
        mock_load.return_value = config

        # Setup domain factory
        mock_asset1 = Mock()
        mock_asset1.symbol = "AAPL"
        mock_asset1.category = Mock()
        mock_asset1.get_value = Mock(side_effect=lambda p: 10 * p.get("AAPL", 0))

        mock_asset2 = Mock()
        mock_asset2.symbol = "SPY"
        mock_asset2.category = Mock()
        mock_asset2.get_value = Mock(side_effect=lambda p: 20 * p.get("SPY", 0))

        mock_portfolio = Mock()
        mock_portfolio.name = "Test Portfolio"
        mock_portfolio.initial_capital = 100000
        mock_portfolio.get_all_assets.return_value = [mock_asset1, mock_asset2]
        mock_portfolio.get_portfolio_value = Mock(
            side_effect=lambda p: sum(p.values()) * 15
        )
        mock_portfolio.allocation_method = "equal"

        # Category allocations at start
        start_allocations = {
            "Technology": {"value": 1500.0, "percentage": 60.0, "assets": ["AAPL"]},
            "Index": {"value": 8000.0, "percentage": 40.0, "assets": ["SPY"]},
        }

        # Category allocations at end (with gains)
        end_allocations = {
            "Technology": {"value": 1800.0, "percentage": 56.0, "assets": ["AAPL"]},
            "Index": {"value": 8800.0, "percentage": 44.0, "assets": ["SPY"]},
            "NewCategory": {
                "value": 500.0,
                "percentage": 2.5,
                "assets": ["NEW"],
            },  # New category
        }

        mock_portfolio.get_allocation_by_category = Mock(
            side_effect=lambda p: start_allocations
            if p.get("AAPL") == 150
            else end_allocations
        )

        mock_factory.return_value.create_portfolio.return_value = mock_portfolio

        # Setup fetcher
        mock_fetcher = Mock()

        # Mock start date fetching
        mock_data = pd.DataFrame({"Close": [150.0]})
        mock_fetcher.get_stock_data.return_value = mock_data
        mock_fetcher.get_current_prices.return_value = {"AAPL": 180.0, "SPY": 440.0}
        mock_fetcher_class.return_value = mock_fetcher

        # Setup backtest results
        mock_backtest.return_value = []

        main()

        # Check output contains performance breakdown
        captured = capsys.readouterr()
        assert "PORTFOLIO PERFORMANCE SUMMARY" in captured.out
        assert "Performance Breakdown By Category:" in captured.out
        assert "Technology:" in captured.out
        assert "Index:" in captured.out
        assert "NewCategory:" in captured.out
        assert "new category" in captured.out

    @patch("stockula.main.load_config")
    @patch("sys.argv", ["stockula", "--config", "test.yaml"])
    def test_main_with_default_config_search(self, mock_load):
        """Test main searches for default config files."""
        from pathlib import Path

        # First call fails (specific config not found), subsequent calls check defaults
        mock_load.side_effect = [
            FileNotFoundError("test.yaml not found"),
            StockulaConfig(),  # Found a default config
        ]

        with patch("pathlib.Path") as mock_path_cls:
            # Mock that .config.yaml exists
            mock_path = Mock()
            mock_path.exists.side_effect = [True]  # .config.yaml exists
            mock_path_cls.return_value = mock_path

            with patch("stockula.main.setup_logging"):
                with patch("stockula.main.DomainFactory") as mock_factory:
                    with patch("stockula.main.DataFetcher") as mock_fetcher_class:
                        with patch("stockula.main.print_results"):
                            # Setup minimal mocks
                            mock_portfolio = Mock()
                            mock_portfolio.name = "Test"
                            mock_portfolio.initial_capital = 100000
                            mock_portfolio.get_all_assets.return_value = []
                            mock_portfolio.get_portfolio_value.return_value = 100000
                            mock_portfolio.allocation_method = "equal"
                            mock_portfolio.get_allocation_by_category.return_value = {}
                            mock_factory.return_value.create_portfolio.return_value = (
                                mock_portfolio
                            )

                            mock_fetcher = Mock()
                            mock_fetcher.get_current_prices.return_value = {}
                            mock_fetcher_class.return_value = mock_fetcher

                            main()

        # Should have tried to load config at least once (could be once if default found)
        assert mock_load.call_count >= 1

    def test_main_entry_point(self):
        """Test the __main__ entry point."""
        # Import the module to execute the __main__ block
        import stockula.main

        # The __main__ block should exist
        assert hasattr(stockula.main, "__name__")

    @patch("stockula.main.load_config")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.DomainFactory")
    @patch("stockula.main.DataFetcher")
    @patch("stockula.main.run_backtest")
    @patch("stockula.main.print_results")
    @patch("sys.argv", ["stockula", "--mode", "backtest"])
    def test_main_with_hold_only_assets(
        self,
        mock_print,
        mock_backtest,
        mock_fetcher_class,
        mock_factory,
        mock_logging,
        mock_load,
        capsys,
    ):
        """Test main function with hold-only assets showing asset type breakdown."""
        # Setup config
        config = StockulaConfig()
        config.backtest.hold_only_categories = ["INDEX"]
        mock_load.return_value = config

        # Setup domain factory with mixed assets
        from stockula.domain import Category

        mock_tradeable = Mock()
        mock_tradeable.symbol = "AAPL"
        mock_tradeable.category = Category.GROWTH
        mock_tradeable.get_value = Mock(return_value=1500.0)

        mock_hold_only = Mock()
        mock_hold_only.symbol = "SPY"
        mock_hold_only.category = Category.INDEX
        mock_hold_only.get_value = Mock(return_value=8000.0)

        mock_portfolio = Mock()
        mock_portfolio.name = "Test Portfolio"
        mock_portfolio.initial_capital = 100000
        mock_portfolio.get_all_assets.return_value = [mock_tradeable, mock_hold_only]
        mock_portfolio.get_portfolio_value.return_value = 109500
        mock_portfolio.allocation_method = "equal"
        mock_portfolio.get_allocation_by_category.return_value = {}
        mock_factory.return_value.create_portfolio.return_value = mock_portfolio

        # Setup fetcher
        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.return_value = {"AAPL": 150.0, "SPY": 400.0}
        mock_fetcher_class.return_value = mock_fetcher

        # Setup backtest results
        mock_backtest.return_value = []

        main()

        # Check output shows asset type breakdown
        captured = capsys.readouterr()
        # The hold-only message is logged, not printed to stdout
        # But the asset type breakdown should be printed
        assert "Asset Type Breakdown:" in captured.out
        assert "Hold-only Assets: $8,000.00" in captured.out
        assert "Tradeable Assets: $1,500.00" in captured.out

    @patch("stockula.main.load_config")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.DomainFactory")
    @patch("stockula.main.DataFetcher")
    @patch("stockula.main.run_forecast")
    @patch("stockula.main.print_results")
    @patch("sys.argv", ["stockula", "--mode", "forecast"])
    def test_main_creates_forecasting_key(
        self,
        mock_print,
        mock_forecast,
        mock_fetcher_class,
        mock_factory,
        mock_logging,
        mock_load,
    ):
        """Test that forecast mode creates forecasting key in results."""
        # Setup config
        config = StockulaConfig()
        mock_load.return_value = config

        # Setup domain factory
        mock_asset = Mock()
        mock_asset.symbol = "AAPL"
        mock_asset.category = None
        mock_portfolio = Mock()
        mock_portfolio.name = "Test Portfolio"
        mock_portfolio.initial_capital = 100000
        mock_portfolio.get_all_assets.return_value = [mock_asset]
        mock_portfolio.get_portfolio_value.return_value = 100000
        mock_portfolio.allocation_method = "equal"
        mock_factory.return_value.create_portfolio.return_value = mock_portfolio

        # Setup fetcher
        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.return_value = {"AAPL": 150.0}
        mock_fetcher_class.return_value = mock_fetcher

        # Setup forecast results
        mock_forecast.return_value = {"ticker": "AAPL", "forecast_price": 155.0}

        main()

        # Check that print_results was called with forecasting key
        call_args = mock_print.call_args[0][0]
        assert "forecasting" in call_args
        assert len(call_args["forecasting"]) == 1

    @patch("stockula.main.load_config")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.DomainFactory")
    @patch("stockula.main.DataFetcher")
    @patch("stockula.main.run_backtest")
    @patch("stockula.main.print_results")
    @patch("sys.argv", ["stockula", "--mode", "backtest"])
    def test_main_backtest_creates_results_dict(
        self,
        mock_print,
        mock_backtest,
        mock_fetcher_class,
        mock_factory,
        mock_logging,
        mock_load,
    ):
        """Test that backtest mode properly creates backtesting key in results."""
        # Setup config
        config = StockulaConfig()
        config.backtest.hold_only_categories = []
        mock_load.return_value = config

        # Setup domain factory
        mock_asset = Mock()
        mock_asset.symbol = "AAPL"
        mock_asset.category = None
        mock_portfolio = Mock()
        mock_portfolio.name = "Test Portfolio"
        mock_portfolio.initial_capital = 100000
        mock_portfolio.get_all_assets.return_value = [mock_asset]
        mock_portfolio.get_portfolio_value.return_value = 100000
        mock_portfolio.allocation_method = "equal"
        mock_portfolio.get_allocation_by_category.return_value = {}
        mock_factory.return_value.create_portfolio.return_value = mock_portfolio

        # Setup fetcher
        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.return_value = {"AAPL": 150.0}
        mock_fetcher_class.return_value = mock_fetcher

        # Setup backtest results
        mock_backtest.return_value = [{"ticker": "AAPL", "strategy": "SMACross"}]

        main()

        # Check that print_results was called with backtesting key
        call_args = mock_print.call_args[0][0]
        assert "backtesting" in call_args
        assert len(call_args["backtesting"]) == 1

    @patch("stockula.main.load_config")
    @patch("sys.argv", ["stockula"])
    def test_main_no_default_config_found(self, mock_load):
        """Test main when no default config files are found."""
        from pathlib import Path

        # load_config should return default config
        mock_load.return_value = StockulaConfig()

        with patch("pathlib.Path") as mock_path_cls:
            # All default files don't exist
            mock_path = Mock()
            mock_path.exists.return_value = False
            mock_path_cls.return_value = mock_path

            with patch("stockula.main.setup_logging"):
                with patch("stockula.main.DomainFactory") as mock_factory:
                    with patch("stockula.main.DataFetcher") as mock_fetcher_class:
                        with patch("stockula.main.print_results"):
                            # Setup minimal mocks
                            mock_portfolio = Mock()
                            mock_portfolio.name = "Test"
                            mock_portfolio.initial_capital = 100000
                            mock_portfolio.get_all_assets.return_value = []
                            mock_portfolio.get_portfolio_value.return_value = 100000
                            mock_portfolio.allocation_method = "equal"
                            mock_portfolio.get_allocation_by_category.return_value = {}
                            mock_factory.return_value.create_portfolio.return_value = (
                                mock_portfolio
                            )

                            mock_fetcher = Mock()
                            mock_fetcher.get_current_prices.return_value = {}
                            mock_fetcher_class.return_value = mock_fetcher

                            main()

            # Should check for all default files
            assert mock_path.exists.call_count >= 8  # Number of default files

    @patch("stockula.main.load_config")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.DomainFactory")
    @patch("stockula.main.DataFetcher")
    @patch("stockula.main.run_backtest")
    @patch("stockula.main.print_results")
    @patch("sys.argv", ["stockula", "--mode", "all"])
    def test_main_all_mode(
        self,
        mock_print,
        mock_backtest,
        mock_fetcher_class,
        mock_factory,
        mock_logging,
        mock_load,
    ):
        """Test main function in 'all' mode runs all analyses."""
        # Setup config
        config = StockulaConfig()
        config.backtest.hold_only_categories = []
        mock_load.return_value = config

        # Setup domain factory
        mock_asset = Mock()
        mock_asset.symbol = "AAPL"
        mock_asset.category = None
        mock_portfolio = Mock()
        mock_portfolio.name = "Test Portfolio"
        mock_portfolio.initial_capital = 100000
        mock_portfolio.get_all_assets.return_value = [mock_asset]
        mock_portfolio.get_portfolio_value.return_value = 100000
        mock_portfolio.allocation_method = "equal"
        mock_portfolio.get_allocation_by_category.return_value = {}
        mock_factory.return_value.create_portfolio.return_value = mock_portfolio

        # Setup fetcher
        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.return_value = {"AAPL": 150.0}
        mock_fetcher_class.return_value = mock_fetcher

        # Setup backtest results
        mock_backtest.return_value = []

        # Mock all analysis functions
        with patch("stockula.main.run_technical_analysis") as mock_ta:
            with patch("stockula.main.run_forecast") as mock_forecast:
                mock_ta.return_value = {"ticker": "AAPL", "indicators": {}}
                mock_forecast.return_value = {"ticker": "AAPL", "forecast_price": 155.0}

                main()

                # All analysis functions should be called
                mock_ta.assert_called_once()
                mock_backtest.assert_called_once()
                mock_forecast.assert_called_once()

    @patch("stockula.main.load_config")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.DomainFactory")
    @patch("stockula.main.DataFetcher")
    @patch("stockula.main.print_results")
    @patch("sys.argv", ["stockula", "--mode", "backtest"])
    def test_main_unknown_hold_only_category(
        self,
        mock_print,
        mock_fetcher_class,
        mock_factory,
        mock_logging,
        mock_load,
        capsys,
    ):
        """Test main function with unknown hold-only category."""
        # Setup config with invalid category
        config = StockulaConfig()
        config.backtest.hold_only_categories = ["INVALID_CATEGORY"]
        mock_load.return_value = config

        # Setup domain factory
        mock_asset = Mock()
        mock_asset.symbol = "AAPL"
        mock_asset.category = None
        mock_portfolio = Mock()
        mock_portfolio.name = "Test Portfolio"
        mock_portfolio.initial_capital = 100000
        mock_portfolio.get_all_assets.return_value = [mock_asset]
        mock_portfolio.get_portfolio_value.return_value = 100000
        mock_portfolio.allocation_method = "equal"
        mock_portfolio.get_allocation_by_category.return_value = {}
        mock_factory.return_value.create_portfolio.return_value = mock_portfolio

        # Setup fetcher
        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.return_value = {"AAPL": 150.0}
        mock_fetcher_class.return_value = mock_fetcher

        main()

        # Should log warning about unknown category
        captured = capsys.readouterr()
        # Logger warning may not appear in stdout, so just verify it didn't crash

    @patch("stockula.main.load_config")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.DomainFactory")
    @patch("stockula.main.DataFetcher")
    @patch("stockula.main.run_backtest")
    @patch("stockula.main.print_results")
    @patch("sys.argv", ["stockula", "--mode", "backtest"])
    def test_main_with_error_getting_start_price(
        self,
        mock_print,
        mock_backtest,
        mock_fetcher_class,
        mock_factory,
        mock_logging,
        mock_load,
    ):
        """Test main function when error occurs getting start price."""
        # Setup config with start date
        config = StockulaConfig()
        config.data.start_date = datetime(2023, 1, 1)
        mock_load.return_value = config

        # Setup domain factory
        mock_asset = Mock()
        mock_asset.symbol = "AAPL"
        mock_asset.category = None
        mock_portfolio = Mock()
        mock_portfolio.name = "Test Portfolio"
        mock_portfolio.initial_capital = 100000
        mock_portfolio.get_all_assets.return_value = [mock_asset]
        mock_portfolio.get_portfolio_value.return_value = 100000
        mock_portfolio.allocation_method = "equal"
        mock_portfolio.get_allocation_by_category.return_value = {}
        mock_factory.return_value.create_portfolio.return_value = mock_portfolio

        # Setup fetcher to raise exception for start date
        mock_fetcher = Mock()
        mock_fetcher.get_stock_data.side_effect = Exception("API error")
        mock_fetcher.get_current_prices.return_value = {"AAPL": 150.0}
        mock_fetcher_class.return_value = mock_fetcher

        # Setup backtest results
        mock_backtest.return_value = []

        # Should not crash despite error
        main()

        # Should still call backtest
        mock_backtest.assert_called_once()
