"""Tests for main CLI module."""

import sys
import json
from unittest.mock import Mock, patch
from io import StringIO

from stockula.main import (
    main,
    setup_logging,
    get_strategy_class,
    run_technical_analysis,
    run_backtest,
    run_forecast,
    print_results,
)
from stockula.config import StockulaConfig


class TestLoggingSetup:
    """Test logging configuration."""

    def test_setup_logging_disabled(self, sample_stockula_config):
        """Test logging setup when disabled."""
        sample_stockula_config.logging.enabled = False

        with patch("stockula.main.logging") as mock_logging:
            setup_logging(sample_stockula_config)

            # Should set WARNING level when disabled
            root_logger = mock_logging.getLogger.return_value
            root_logger.setLevel.assert_called()

    def test_setup_logging_enabled(self, sample_stockula_config):
        """Test logging setup when enabled."""
        sample_stockula_config.logging.enabled = True
        sample_stockula_config.logging.level = "DEBUG"

        with patch("stockula.main.logging") as mock_logging:
            setup_logging(sample_stockula_config)

            # Should configure with DEBUG level
            assert mock_logging.getLogger.called

    def test_setup_logging_with_file(self, sample_stockula_config):
        """Test logging setup with file output."""
        sample_stockula_config.logging.enabled = True
        sample_stockula_config.logging.log_to_file = True
        sample_stockula_config.logging.log_file = "test.log"

        with patch("stockula.main.logging") as mock_logging:
            with patch("stockula.main.RotatingFileHandler") as mock_file_handler:
                setup_logging(sample_stockula_config)

                # Should create file handler
                mock_file_handler.assert_called_once_with(
                    filename="test.log",
                    maxBytes=sample_stockula_config.logging.max_log_size,
                    backupCount=sample_stockula_config.logging.backup_count,
                    encoding="utf-8",
                )


class TestStrategyClass:
    """Test strategy class retrieval."""

    def test_get_strategy_class_valid(self):
        """Test getting valid strategy classes."""
        from stockula.backtesting import SMACrossStrategy, RSIStrategy

        assert get_strategy_class("smacross") == SMACrossStrategy
        assert get_strategy_class("rsi") == RSIStrategy
        assert get_strategy_class("SMACROSS") == SMACrossStrategy  # Case insensitive

    def test_get_strategy_class_invalid(self):
        """Test getting invalid strategy class."""
        assert get_strategy_class("invalid_strategy") is None


class TestTechnicalAnalysis:
    """Test technical analysis functions."""

    def test_run_technical_analysis(self, sample_stockula_config, mock_data_fetcher):
        """Test running technical analysis."""
        with patch("stockula.main.DataFetcher", return_value=mock_data_fetcher):
            with patch("stockula.main.TechnicalIndicators") as mock_ta:
                # Setup mock indicators
                mock_ta_instance = Mock()
                mock_ta_instance.sma.return_value = Mock(
                    iloc=[-1], __getitem__=lambda x, y: 150.0
                )
                mock_ta_instance.rsi.return_value = Mock(
                    iloc=[-1], __getitem__=lambda x, y: 65.0
                )
                mock_ta_instance.macd.return_value = Mock(
                    iloc=[-1], to_dict=lambda: {"MACD": 0.5, "MACD_SIGNAL": 0.3}
                )
                mock_ta_instance.bbands.return_value = Mock(
                    iloc=[-1],
                    to_dict=lambda: {
                        "BB_UPPER": 155,
                        "BB_MIDDLE": 150,
                        "BB_LOWER": 145,
                    },
                )
                mock_ta_instance.atr.return_value = Mock(
                    iloc=[-1], __getitem__=lambda x, y: 2.5
                )
                mock_ta_instance.adx.return_value = Mock(
                    iloc=[-1], __getitem__=lambda x, y: 25.0
                )
                mock_ta.return_value = mock_ta_instance

                results = run_technical_analysis("AAPL", sample_stockula_config)

                assert results["ticker"] == "AAPL"
                assert "indicators" in results
                assert "SMA_20" in results["indicators"]
                assert results["indicators"]["RSI"] == 65.0


class TestBacktest:
    """Test backtesting functions."""

    def test_run_backtest(self, sample_stockula_config, mock_data_fetcher):
        """Test running backtest."""
        with patch("stockula.main.BacktestRunner") as mock_runner:
            mock_runner_instance = Mock()
            mock_runner_instance.run_from_symbol.return_value = {
                "Return [%]": 15.5,
                "Sharpe Ratio": 1.2,
                "Max. Drawdown [%]": -10.0,
                "# Trades": 25,
                "Win Rate [%]": 60.0,
            }
            mock_runner.return_value = mock_runner_instance

            results = run_backtest("AAPL", sample_stockula_config)

            assert len(results) > 0
            assert results[0]["ticker"] == "AAPL"
            assert results[0]["return_pct"] == 15.5
            assert results[0]["num_trades"] == 25

    def test_run_backtest_with_error(self, sample_stockula_config):
        """Test backtest error handling."""
        with patch("stockula.main.BacktestRunner") as mock_runner:
            mock_runner_instance = Mock()
            mock_runner_instance.run_from_symbol.side_effect = Exception(
                "Backtest failed"
            )
            mock_runner.return_value = mock_runner_instance

            results = run_backtest("AAPL", sample_stockula_config)

            # Should return empty list on error
            assert results == []


class TestForecast:
    """Test forecasting functions."""

    def test_run_forecast(self, sample_stockula_config):
        """Test running forecast."""
        with patch("stockula.main.StockForecaster") as mock_forecaster:
            mock_forecaster_instance = Mock()
            mock_forecaster_instance.forecast_from_symbol.return_value = {
                "forecast": Mock(
                    iloc=[
                        Mock(__getitem__=lambda x, y: 150.0),
                        Mock(__getitem__=lambda x, y: 155.0),
                    ]
                ),
                "lower_bound": Mock(iloc=[-1], __getitem__=lambda x, y: 145.0),
                "upper_bound": Mock(iloc=[-1], __getitem__=lambda x, y: 160.0),
            }
            mock_forecaster_instance.get_best_model.return_value = {
                "model_name": "ARIMA",
                "model_params": {"p": 1, "d": 1, "q": 1},
            }
            mock_forecaster.return_value = mock_forecaster_instance

            result = run_forecast("AAPL", sample_stockula_config)

            assert result["ticker"] == "AAPL"
            assert result["current_price"] == 150.0
            assert result["forecast_price"] == 155.0
            assert result["best_model"] == "ARIMA"

    def test_run_forecast_with_error(self, sample_stockula_config):
        """Test forecast error handling."""
        with patch("stockula.main.StockForecaster") as mock_forecaster:
            mock_forecaster_instance = Mock()
            mock_forecaster_instance.forecast_from_symbol.side_effect = Exception(
                "Forecast failed"
            )
            mock_forecaster.return_value = mock_forecaster_instance

            result = run_forecast("AAPL", sample_stockula_config)

            assert result["ticker"] == "AAPL"
            assert "error" in result
            assert result["error"] == "Forecast failed"


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
            assert "SMA_20: 150.00" in output
            assert "RSI: 65.00" in output

    def test_print_results_json(self):
        """Test JSON output format."""
        results = {
            "technical_analysis": [{"ticker": "AAPL", "indicators": {"SMA_20": 150.0}}]
        }

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
                }
            ]
        }

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            print_results(results, "console")
            output = mock_stdout.getvalue()

            assert "Backtesting Results" in output
            assert "Return: 15.50%" in output
            assert "Win Rate: 60.00%" in output

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
            assert "Current Price: $150.00" in output
            assert "30-Day Forecast: $155.00" in output


class TestMainFunction:
    """Test main entry point."""

    def test_main_with_config_file(self, temp_config_file):
        """Test main with config file."""
        test_args = ["stockula", "--config", temp_config_file, "--mode", "ta"]

        with patch.object(sys, "argv", test_args):
            with patch("stockula.main.load_config") as mock_load:
                with patch("stockula.main.DomainFactory") as mock_factory:
                    with patch("stockula.main.DataFetcher") as mock_fetcher:
                        with patch("stockula.main.run_technical_analysis") as mock_ta:
                            mock_load.return_value = StockulaConfig()
                            mock_ta.return_value = {"ticker": "AAPL", "indicators": {}}

                            main()

                            assert mock_load.called
                            assert mock_ta.called

    def test_main_with_ticker_override(self):
        """Test main with ticker override."""
        test_args = ["stockula", "--ticker", "TSLA", "--mode", "ta"]

        with patch.object(sys, "argv", test_args):
            with patch("stockula.main.load_config") as mock_load:
                with patch("stockula.main.DomainFactory") as mock_factory:
                    with patch("stockula.main.DataFetcher") as mock_fetcher:
                        with patch("stockula.main.run_technical_analysis") as mock_ta:
                            config = StockulaConfig()
                            mock_load.return_value = config
                            mock_ta.return_value = {"ticker": "TSLA", "indicators": {}}

                            main()

                            # Should override tickers with TSLA
                            assert config.portfolio.tickers[0].symbol == "TSLA"

    def test_main_save_config(self, tmp_path):
        """Test saving configuration."""
        config_path = str(tmp_path / "saved_config.yaml")
        test_args = ["stockula", "--save-config", config_path]

        with patch.object(sys, "argv", test_args):
            with patch("stockula.main.save_config") as mock_save:
                main()

                mock_save.assert_called_once()
                assert mock_save.call_args[0][1] == config_path

    def test_main_all_modes(self):
        """Test running all analysis modes."""
        test_args = ["stockula", "--mode", "all"]

        with patch.object(sys, "argv", test_args):
            with patch("stockula.main.load_config") as mock_load:
                with patch("stockula.main.DomainFactory") as mock_factory:
                    with patch("stockula.main.DataFetcher") as mock_fetcher:
                        with patch("stockula.main.run_technical_analysis") as mock_ta:
                            with patch("stockula.main.run_backtest") as mock_bt:
                                with patch("stockula.main.run_forecast") as mock_fc:
                                    mock_load.return_value = StockulaConfig()
                                    mock_ta.return_value = {
                                        "ticker": "AAPL",
                                        "indicators": {},
                                    }
                                    mock_bt.return_value = [{"ticker": "AAPL"}]
                                    mock_fc.return_value = {"ticker": "AAPL"}

                                    main()

                                    # All analysis functions should be called
                                    assert mock_ta.called
                                    assert mock_bt.called
                                    assert mock_fc.called
