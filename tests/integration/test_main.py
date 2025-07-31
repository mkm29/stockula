"""Tests for main CLI module."""

import json
import sys
from io import StringIO
from unittest.mock import Mock, patch

from stockula.main import (
    get_strategy_class,
    main,
    print_results,
    run_backtest,
    run_forecast,
    run_technical_analysis,
    setup_logging,
)


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


class TestStrategyClass:
    """Test strategy class retrieval."""

    def test_get_strategy_class_valid(self):
        """Test getting valid strategy classes."""
        from stockula.backtesting import RSIStrategy, SMACrossStrategy

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
        # Create container and override dependencies
        from stockula.container import Container

        container = Container()
        container.data_fetcher.override(mock_data_fetcher)
        container.wire(modules=["stockula.main"])

        with patch("stockula.main.TechnicalIndicators") as mock_ta:
            # Setup mock indicators
            mock_ta_instance = Mock()

            # Create iloc mock that behaves like a pandas object
            def create_iloc_mock(value):
                iloc_mock = Mock()
                iloc_mock.__getitem__ = Mock(return_value=value)
                return iloc_mock

            # Setup SMA mock
            sma_mock = Mock()
            sma_mock.iloc = create_iloc_mock(150.0)
            mock_ta_instance.sma.return_value = sma_mock

            # Setup EMA mock
            ema_mock = Mock()
            ema_mock.iloc = create_iloc_mock(151.0)
            mock_ta_instance.ema.return_value = ema_mock

            # Setup RSI mock
            rsi_mock = Mock()
            rsi_mock.iloc = create_iloc_mock(65.0)
            mock_ta_instance.rsi.return_value = rsi_mock
            # Setup MACD mock
            macd_mock = Mock()
            macd_iloc = Mock()
            macd_iloc.to_dict = Mock(return_value={"MACD": 0.5, "MACD_SIGNAL": 0.3})
            macd_mock.iloc = Mock()
            macd_mock.iloc.__getitem__ = Mock(return_value=macd_iloc)
            mock_ta_instance.macd.return_value = macd_mock

            # Setup Bollinger Bands mock
            bbands_mock = Mock()
            bbands_iloc = Mock()
            bbands_iloc.to_dict = Mock(
                return_value={
                    "BB_UPPER": 155,
                    "BB_MIDDLE": 150,
                    "BB_LOWER": 145,
                }
            )
            bbands_mock.iloc = Mock()
            bbands_mock.iloc.__getitem__ = Mock(return_value=bbands_iloc)
            mock_ta_instance.bbands.return_value = bbands_mock

            # Setup ATR mock
            atr_mock = Mock()
            atr_mock.iloc = create_iloc_mock(2.5)
            mock_ta_instance.atr.return_value = atr_mock

            # Setup ADX mock
            adx_mock = Mock()
            adx_mock.iloc = create_iloc_mock(25.0)
            mock_ta_instance.adx.return_value = adx_mock
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
        # Add a strategy to the config
        from stockula.config.models import StrategyConfig

        sample_stockula_config.backtest.strategies = [
            StrategyConfig(name="smacross", parameters={"fast_period": 10, "slow_period": 20})
        ]

        # Create mock runner
        mock_runner_instance = Mock()
        mock_runner_instance.run_from_symbol.return_value = {
            "Return [%]": 15.5,
            "Sharpe Ratio": 1.2,
            "Max. Drawdown [%]": -10.0,
            "# Trades": 25,
            "Win Rate [%]": 60.0,
        }

        # Create container and override dependencies
        from stockula.container import Container

        container = Container()
        container.data_fetcher.override(mock_data_fetcher)
        container.backtest_runner.override(mock_runner_instance)
        container.wire(modules=["stockula.main"])

        results = run_backtest("AAPL", sample_stockula_config)

        assert len(results) > 0
        assert results[0]["ticker"] == "AAPL"
        assert results[0]["return_pct"] == 15.5
        assert results[0]["num_trades"] == 25

    def test_run_backtest_with_error(self, sample_stockula_config):
        """Test backtest error handling."""
        # Create mock runner that raises an error
        mock_runner_instance = Mock()
        mock_runner_instance.run_from_symbol.side_effect = Exception("Backtest failed")

        # Create container and override dependencies
        from stockula.container import Container

        container = Container()
        container.backtest_runner.override(mock_runner_instance)
        container.wire(modules=["stockula.main"])

        results = run_backtest("AAPL", sample_stockula_config)

        # Should return empty list on error
        assert results == []


class TestForecast:
    """Test forecasting functions."""

    def test_run_forecast(self, sample_stockula_config):
        """Test running forecast."""
        # Create mock forecaster
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

        # Create container and override dependencies
        from stockula.container import Container

        container = Container()
        container.stock_forecaster.override(mock_forecaster_instance)
        container.wire(modules=["stockula.main"])

        result = run_forecast("AAPL", sample_stockula_config)

        assert result["ticker"] == "AAPL"
        assert result["current_price"] == 150.0
        assert result["forecast_price"] == 155.0
        assert result["best_model"] == "ARIMA"

    def test_run_forecast_with_error(self, sample_stockula_config):
        """Test forecast error handling."""
        # Create mock forecaster that raises an error
        mock_forecaster_instance = Mock()
        mock_forecaster_instance.forecast_from_symbol.side_effect = Exception("Forecast failed")

        # Create container and override dependencies
        from stockula.container import Container

        container = Container()
        container.stock_forecaster.override(mock_forecaster_instance)
        container.wire(modules=["stockula.main"])

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
            assert "Trading Days: 252" in output
            assert "Calendar Days: 365" in output
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
            # Rich tables format the output differently
            assert "Current Price" in output
            assert "150" in output
            assert "Forecast Price" in output
            assert "155" in output


class TestMainFunction:
    """Test main entry point."""

    def test_main_with_config_file(self, temp_config_file):
        """Test main with config file."""
        test_args = ["stockula", "--config", temp_config_file, "--mode", "ta"]

        with patch.object(sys, "argv", test_args):
            with patch("stockula.main.run_technical_analysis") as mock_ta:
                mock_ta.return_value = {"ticker": "AAPL", "indicators": {}}

                # Main will create its own container and load config
                main()

                # Check that technical analysis was called
                assert mock_ta.called

    def test_main_with_ticker_override(self):
        """Test main with ticker override."""
        test_args = ["stockula", "--ticker", "TSLA", "--mode", "ta"]

        with patch.object(sys, "argv", test_args):
            with patch("stockula.main.run_technical_analysis") as mock_ta:
                mock_ta.return_value = {"ticker": "TSLA", "indicators": {}}

                main()

                # Check that technical analysis was called with TSLA
                assert mock_ta.called
                # The ticker override happens inside main, so we can't check config here

    def test_main_save_config(self, tmp_path):
        """Test saving configuration."""
        config_path = str(tmp_path / "saved_config.yaml")
        test_args = ["stockula", "--save-config", config_path]

        with patch.object(sys, "argv", test_args):
            with patch("stockula.config.settings.save_config") as mock_save:
                main()

                # Check that save_config was called with the correct path
                mock_save.assert_called_once()
                assert mock_save.call_args[0][1] == config_path

    def test_main_all_modes(self):
        """Test running all analysis modes."""
        test_args = ["stockula", "--mode", "all"]

        with patch.object(sys, "argv", test_args):
            with patch("stockula.main.run_technical_analysis") as mock_ta:
                with patch("stockula.main.run_backtest") as mock_bt:
                    with patch("stockula.main.run_forecast") as mock_fc:
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
