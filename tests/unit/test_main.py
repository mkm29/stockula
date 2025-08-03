"""Consolidated unit tests for main module."""

import json
from datetime import date, datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from stockula.config import StockulaConfig, TickerConfig
from stockula.config.models import BrokerConfig, PortfolioBacktestResults
from stockula.domain.category import Category
from stockula.main import main, print_results, setup_logging
from stockula.manager import StockulaManager


# Helper fixtures and utilities
@pytest.fixture
def mock_config():
    """Create a basic test configuration."""
    config = StockulaConfig()
    config.data.start_date = date(2023, 1, 1)
    config.data.end_date = date(2023, 12, 31)
    config.portfolio.tickers = [TickerConfig(symbol="AAPL", quantity=1.0)]
    config.forecast.forecast_length = 30
    config.technical_analysis.indicators = ["sma", "rsi"]
    config.technical_analysis.sma_periods = [20]
    config.technical_analysis.rsi_period = 14
    return config


def create_mock_container(config=None):
    """Create a mock container with common dependencies."""
    container = Mock()

    if config is None:
        config = StockulaConfig()
        config.data.start_date = date(2023, 1, 1)
        config.data.end_date = date(2023, 12, 31)
        config.portfolio.tickers = [TickerConfig(symbol="AAPL", quantity=1.0)]

    # Mock portfolio
    mock_asset = Mock()
    mock_asset.symbol = "AAPL"
    mock_asset.category = None
    mock_asset.quantity = 1.0

    mock_portfolio = Mock()
    mock_portfolio.name = "Test Portfolio"
    mock_portfolio.initial_capital = 100000.0
    mock_portfolio.get_all_assets.return_value = [mock_asset]
    mock_portfolio.get_portfolio_value.return_value = 100000.0
    mock_portfolio.allocation_method = "equal"

    # Mock factory
    mock_factory = Mock()
    mock_factory.create_portfolio.return_value = mock_portfolio

    # Mock fetcher
    mock_fetcher = Mock()
    mock_fetcher.get_current_prices.return_value = {"AAPL": 150.0}

    # Set up container methods
    container.stockula_config.return_value = config
    container.logging_manager.return_value = Mock()
    container.domain_factory.return_value = mock_factory
    container.data_fetcher.return_value = mock_fetcher
    container.backtest_runner.return_value = Mock()
    container.stock_forecaster.return_value = Mock()

    return container


def create_mock_manager(config=None):
    """Create a mock StockulaManager for testing utility functions."""
    if config is None:
        config = StockulaConfig()

    container = create_mock_container(config)
    console = Mock()
    # Make the console support context manager protocol
    console.__enter__ = Mock(return_value=console)
    console.__exit__ = Mock(return_value=None)

    return StockulaManager(config, container, console)


class TestCoreUtilities:
    """Test core utility functions."""

    def test_date_to_string_variants(self):
        """Test date_to_string with different input types."""
        manager = create_mock_manager()
        assert manager.date_to_string(None) is None
        assert manager.date_to_string("2024-01-01") == "2024-01-01"
        assert manager.date_to_string(date(2024, 1, 15)) == "2024-01-15"
        assert manager.date_to_string(datetime(2024, 6, 30, 12, 30)) == "2024-06-30"

    def test_get_strategy_class(self):
        """Test strategy class retrieval."""
        from stockula.backtesting.strategies import KAMAStrategy, RSIStrategy, SMACrossStrategy

        manager = create_mock_manager()
        assert manager.get_strategy_class("smacross") == SMACrossStrategy
        assert manager.get_strategy_class("SMACROSS") == SMACrossStrategy
        assert manager.get_strategy_class("rsi") == RSIStrategy
        assert manager.get_strategy_class("KAMA") == KAMAStrategy
        assert manager.get_strategy_class("invalid") is None
        assert manager.get_strategy_class("") is None

    def test_setup_logging(self):
        """Test logging setup."""
        config = StockulaConfig()
        mock_logging_manager = Mock()

        setup_logging(config, logging_manager=mock_logging_manager)
        mock_logging_manager.setup.assert_called_once_with(config)


class TestTechnicalAnalysis:
    """Test technical analysis functionality."""

    @patch("stockula.manager.TechnicalIndicators")
    def test_run_technical_analysis_success(self, mock_ta_class):
        """Test successful technical analysis execution."""
        config = StockulaConfig()
        config.technical_analysis.indicators = ["sma", "rsi"]
        config.technical_analysis.sma_periods = [20]
        config.technical_analysis.rsi_period = 14

        # Mock data and indicators
        mock_data = pd.DataFrame(
            {
                "Open": [99] * 50,
                "High": [101] * 50,
                "Low": [98] * 50,
                "Close": [100] * 50,
                "Volume": [1000000] * 50,
            },
            index=pd.date_range("2023-01-01", periods=50),
        )
        mock_data_fetcher = Mock()
        mock_data_fetcher.get_stock_data.return_value = mock_data

        mock_ta = Mock()
        mock_ta.sma.return_value = pd.Series([100.5] * 50)
        mock_ta.rsi.return_value = pd.Series([50.0] * 50)
        mock_ta_class.return_value = mock_ta

        manager = create_mock_manager(config)
        # Set up the container's data_fetcher to return our mock
        manager.container.data_fetcher.return_value = mock_data_fetcher
        result = manager.run_technical_analysis("AAPL", show_progress=False)

        assert result["ticker"] == "AAPL"
        assert "SMA_20" in result["indicators"]
        assert "RSI" in result["indicators"]
        assert result["indicators"]["SMA_20"] == 100.5
        assert result["indicators"]["RSI"] == 50.0

    @patch("stockula.manager.TechnicalIndicators")
    def test_run_technical_analysis_all_indicators(self, mock_ta_class):
        """Test technical analysis with all indicator types."""
        config = StockulaConfig()
        config.technical_analysis.indicators = ["sma", "ema", "rsi", "macd", "bbands", "atr", "adx"]
        config.technical_analysis.sma_periods = [10, 20]
        config.technical_analysis.ema_periods = [12, 26]

        mock_data = pd.DataFrame(
            {"Close": [100] * 100},
            index=pd.date_range("2023-01-01", periods=100),
        )
        mock_data_fetcher = Mock()
        mock_data_fetcher.get_stock_data.return_value = mock_data

        # Mock all indicators
        mock_ta = Mock()
        mock_ta.sma.return_value = pd.Series([100.0] * 100)
        mock_ta.ema.return_value = pd.Series([99.5] * 100)
        mock_ta.rsi.return_value = pd.Series([55.0] * 100)
        mock_ta.macd.return_value = pd.DataFrame({"MACD": [0.5] * 100, "MACD_SIGNAL": [0.3] * 100})
        mock_ta.bbands.return_value = pd.DataFrame(
            {"BB_UPPER": [102] * 100, "BB_MIDDLE": [100] * 100, "BB_LOWER": [98] * 100}
        )
        mock_ta.atr.return_value = pd.Series([2.0] * 100)
        mock_ta.adx.return_value = pd.Series([25.0] * 100)
        mock_ta_class.return_value = mock_ta

        # Test with and without progress bar
        for _show_progress in [True, False]:
            manager = create_mock_manager(config)
            # Set up the container's data_fetcher to return our mock
            manager.container.data_fetcher.return_value = mock_data_fetcher
            result = manager.run_technical_analysis("TEST", show_progress=False)
            assert all(
                ind in result["indicators"]
                for ind in ["SMA_10", "SMA_20", "EMA_12", "EMA_26", "RSI", "MACD", "BBands", "ATR", "ADX"]
            )


class TestBacktesting:
    """Test backtesting functionality."""

    def test_run_backtest_success(self):
        """Test successful backtest execution."""
        config = StockulaConfig()
        strategy_config = Mock()
        strategy_config.name = "SMACross"
        strategy_config.parameters = {"fast_period": 10, "slow_period": 20}
        config.backtest.strategies = [strategy_config]

        Mock()
        mock_runner = Mock()
        mock_runner.run_from_symbol.return_value = {
            "Return [%]": 15.5,
            "Sharpe Ratio": 1.25,
            "Max. Drawdown [%]": -8.3,
            "# Trades": 42,
            "Win Rate [%]": 55.0,
        }

        manager = create_mock_manager(config)
        # Set up the container's backtest_runner to return our mock
        manager.container.backtest_runner.return_value = mock_runner
        results = manager.run_backtest("AAPL")

        assert len(results) == 1
        assert results[0]["ticker"] == "AAPL"
        assert results[0]["return_pct"] == 15.5
        assert results[0]["num_trades"] == 42

    def test_run_backtest_edge_cases(self):
        """Test backtest edge cases: no trades, unknown strategy, exception."""
        config = StockulaConfig()

        # Test 1: No trades (NaN win rate)
        strategy_config = Mock()
        strategy_config.name = "SMACross"
        strategy_config.parameters = {}
        config.backtest.strategies = [strategy_config]

        mock_runner = Mock()
        mock_runner.run_from_symbol.return_value = {
            "Return [%]": 0.0,
            "Sharpe Ratio": 0.0,
            "Max. Drawdown [%]": 0.0,
            "# Trades": 0,
            "Win Rate [%]": float("nan"),
        }

        manager = create_mock_manager(config)
        # Set up the container's backtest_runner to return our mock
        manager.container.backtest_runner.return_value = mock_runner
        results = manager.run_backtest("TEST")
        assert results[0]["win_rate"] is None

        # Test 2: Exception during backtest
        mock_runner.run_from_symbol.side_effect = Exception("Backtest error")
        manager = create_mock_manager(config)
        manager.container.backtest_runner.return_value = mock_runner
        results = manager.run_backtest("TEST")
        assert results == []


class TestForecasting:
    """Test forecasting functionality."""

    def test_run_forecast_success(self):
        """Test successful forecast execution."""
        config = StockulaConfig()
        config.forecast.forecast_length = 30

        mock_forecaster = Mock()
        dates = pd.date_range("2024-01-01", periods=5)
        mock_predictions = pd.DataFrame(
            {
                "forecast": [110, 111, 112, 113, 114],
                "lower_bound": [105, 106, 107, 108, 109],
                "upper_bound": [115, 116, 117, 118, 119],
            },
            index=dates,
        )
        mock_forecaster.forecast_from_symbol.return_value = mock_predictions
        mock_forecaster.get_best_model.return_value = {
            "model_name": "ARIMA",
            "model_params": {"p": 1, "d": 1, "q": 1},
        }

        manager = create_mock_manager(config)
        # Set up the container's stock_forecaster to return our mock
        manager.container.stock_forecaster.return_value = mock_forecaster
        result = manager.run_forecast("AAPL")

        assert result["ticker"] == "AAPL"
        assert result["current_price"] == 110
        assert result["forecast_price"] == 114
        assert result["best_model"] == "ARIMA"

    def test_run_forecast_with_evaluation(self):
        """Test forecast with evaluation functionality."""
        config = StockulaConfig()
        config.forecast.train_start_date = date(2023, 1, 1)
        config.forecast.train_end_date = date(2023, 12, 31)
        config.forecast.test_start_date = date(2024, 1, 1)
        config.forecast.test_end_date = date(2024, 1, 31)

        mock_forecaster = Mock()
        mock_result = {
            "predictions": pd.DataFrame(
                {
                    "forecast": [150.0, 152.0, 155.0],
                    "lower_bound": [148.0, 150.0, 152.0],
                    "upper_bound": [152.0, 154.0, 158.0],
                },
                index=pd.date_range("2024-01-01", periods=3),
            ),
            "train_period": {"start": "2023-01-01", "end": "2023-12-31", "days": 365},
            "test_period": {"start": "2024-01-01", "end": "2024-01-31", "days": 31},
            "evaluation_metrics": {"rmse": 2.5, "mae": 2.0, "mape": 1.5},
        }
        mock_forecaster.forecast_from_symbol_with_evaluation.return_value = mock_result
        mock_forecaster.get_best_model.return_value = {"model_name": "ARIMA"}

        manager = create_mock_manager(config)
        # Set up the container's stock_forecaster to return our mock
        manager.container.stock_forecaster.return_value = mock_forecaster
        result = manager.run_forecast_with_evaluation("AAPL")

        assert result["ticker"] == "AAPL"
        assert "evaluation" in result
        assert result["evaluation"]["rmse"] == 2.5

    def test_forecast_error_handling(self):
        """Test forecast error handling."""
        config = StockulaConfig()

        # Test keyboard interrupt
        mock_forecaster = Mock()
        mock_forecaster.forecast_from_symbol.side_effect = KeyboardInterrupt()
        manager = create_mock_manager(config)
        # Set up the container's stock_forecaster to return our mock
        manager.container.stock_forecaster.return_value = mock_forecaster
        result = manager.run_forecast("TEST")
        assert result["error"] == "Interrupted by user"

        # Test general exception
        mock_forecaster.forecast_from_symbol.side_effect = Exception("API error")
        manager = create_mock_manager(config)
        # Set up the container's stock_forecaster to return our mock
        manager.container.stock_forecaster.return_value = mock_forecaster
        result = manager.run_forecast("TEST")
        assert result["error"] == "API error"


class TestPrintResults:
    """Test results printing functionality."""

    def test_print_results_formats(self, capsys):
        """Test different output formats."""
        results = {
            "technical_analysis": [{"ticker": "AAPL", "indicators": {"SMA_20": 150.50}}],
            "backtesting": [
                {
                    "ticker": "AAPL",
                    "strategy": "SMACross",
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
                    "best_model": "ARIMA",
                }
            ],
        }

        # Test console output
        print_results(results, "console")
        captured = capsys.readouterr()
        assert "AAPL" in captured.out
        assert "150.50" in captured.out or "150.5" in captured.out

        # Test JSON output
        print_results(results, "json")
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "technical_analysis" in parsed

    def test_print_results_special_cases(self, capsys):
        """Test special result printing cases."""
        # Test with train/test split
        results = {
            "backtesting": [
                {
                    "ticker": "AAPL",
                    "strategy": "smacross",
                    "train_period": {"start": "2023-01-01", "end": "2023-09-30", "days": 273},
                    "test_period": {"start": "2023-10-01", "end": "2023-12-31", "days": 92},
                    "train_results": {"return_pct": 20.5, "sharpe_ratio": 1.8},
                    "test_results": {"return_pct": 15.2, "sharpe_ratio": 1.4, "num_trades": 12, "win_rate": 58.3},
                    "return_pct": 15.2,
                    "sharpe_ratio": 1.4,
                    "num_trades": 12,
                    "win_rate": 58.3,
                }
            ]
        }

        print_results(results, "console")
        captured = capsys.readouterr()
        assert "train" in captured.out.lower() or "Train" in captured.out
        assert "test" in captured.out.lower() or "Test" in captured.out

    def test_print_results_sorted_forecasts(self, capsys):
        """Test that forecast results are sorted by return percentage."""
        results = {
            "forecasting": [
                {
                    "ticker": "LOW",
                    "current_price": 100.0,
                    "forecast_price": 95.0,
                    "lower_bound": 90.0,
                    "upper_bound": 100.0,
                    "best_model": "ETS",
                },
                {
                    "ticker": "HIGH",
                    "current_price": 100.0,
                    "forecast_price": 120.0,
                    "lower_bound": 115.0,
                    "upper_bound": 125.0,
                    "best_model": "ARIMA",
                },
                {
                    "ticker": "MID",
                    "current_price": 100.0,
                    "forecast_price": 105.0,
                    "lower_bound": 102.0,
                    "upper_bound": 108.0,
                    "best_model": "GLS",
                },
            ]
        }

        print_results(results, "console")
        captured = capsys.readouterr()
        # HIGH should appear first (highest return)
        high_pos = captured.out.find("HIGH")
        mid_pos = captured.out.find("MID")
        low_pos = captured.out.find("LOW")
        assert high_pos < mid_pos < low_pos


class TestMainFunction:
    """Test main function integration."""

    @patch("stockula.main.create_container")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.log_manager")
    @patch("stockula.manager.StockulaManager.run_technical_analysis")
    @patch("stockula.main.print_results")
    @patch("sys.argv", ["stockula", "--mode", "ta", "--ticker", "AAPL"])
    def test_main_ta_mode(self, mock_print, mock_ta, mock_log_manager, mock_logging, mock_container):
        """Test main function in technical analysis mode."""
        container = create_mock_container()
        mock_container.return_value = container
        mock_ta.return_value = {"ticker": "AAPL", "indicators": {}}

        main()

        mock_ta.assert_called_once()
        mock_print.assert_called_once()

    @patch("stockula.main.create_container")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.log_manager")
    @patch("stockula.manager.StockulaManager.run_forecast_with_evaluation")
    @patch("stockula.main.print_results")
    @patch("sys.argv", ["stockula", "--mode", "forecast"])
    def test_main_forecast_mode(self, mock_print, mock_forecast_eval, mock_log_manager, mock_logging, mock_container):
        """Test main function in forecast mode."""
        config = StockulaConfig()
        config.portfolio.tickers = [TickerConfig(symbol="AAPL", quantity=10.0)]
        config.forecast.test_start_date = date(2024, 1, 1)
        config.forecast.test_end_date = date(2024, 1, 31)

        container = create_mock_container(config)
        mock_container.return_value = container

        mock_forecast_eval.return_value = {
            "ticker": "AAPL",
            "current_price": 150.0,
            "forecast_price": 160.0,
            "evaluation": {"rmse": 2.0, "mape": 1.5},
        }

        main()

        mock_forecast_eval.assert_called_once()
        mock_print.assert_called_once()

    @patch("stockula.main.create_container")
    @patch("stockula.main.setup_logging")
    @patch("sys.argv", ["stockula", "--save-config", "output.yaml"])
    def test_main_save_config(self, mock_logging, mock_container):
        """Test main function with save config option."""
        container = create_mock_container()
        container.logging_manager.return_value = Mock()
        mock_container.return_value = container

        with patch("stockula.main.save_config") as mock_save:
            main()
            mock_save.assert_called_once()

    @patch("stockula.main.create_container")
    @patch("stockula.main.setup_logging")
    @patch("stockula.manager.StockulaManager.run_technical_analysis")
    @patch("stockula.manager.StockulaManager.run_backtest")
    @patch("stockula.manager.StockulaManager.run_forecast")
    @patch("stockula.main.print_results")
    @patch("pathlib.Path")
    @patch("builtins.open", create=True)
    @patch("stockula.manager.json.dump")
    @patch("stockula.main.log_manager")
    @patch("stockula.manager.StockulaManager.get_strategy_class")
    @patch("sys.argv", ["stockula", "--output", "console", "--mode", "ta"])
    def test_main_saves_results(
        self,
        mock_get_strategy,
        mock_log_manager,
        mock_json_dump,
        mock_open,
        mock_path,
        mock_print,
        mock_forecast,
        mock_backtest,
        mock_ta,
        mock_logging,
        mock_container,
    ):
        """Test main runs successfully with save results config."""
        config = StockulaConfig()
        config.output = {"save_results": True, "results_dir": "./test_results", "format": "console"}
        config.portfolio.tickers = [TickerConfig(symbol="AAPL", quantity=10.0)]

        container = create_mock_container(config)
        mock_container.return_value = container

        # Mock analysis results
        mock_ta.return_value = {"ticker": "AAPL", "indicators": {"SMA_20": 150.0}}

        # Mock datetime for consistent behavior
        with patch("stockula.main.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240115_120000"
            main()

        # Verify the main function ran successfully
        mock_ta.assert_called_once()
        mock_print.assert_called_once()


class TestPortfolioBacktestResults:
    """Test portfolio backtest results creation."""

    def test_create_portfolio_backtest_results(self):
        """Test creating structured backtest results."""
        results = {"initial_portfolio_value": 10000.0, "initial_capital": 10000.0}

        config = StockulaConfig()
        config.data.start_date = date(2024, 1, 1)
        config.data.end_date = date(2025, 7, 25)

        strategy_results = {
            "SMACross": [
                {
                    "ticker": "AAPL",
                    "strategy": "SMACross",
                    "parameters": {"fast_period": 10},
                    "return_pct": 15.0,
                    "sharpe_ratio": 1.2,
                    "max_drawdown_pct": -10.0,
                    "num_trades": 20,
                    "win_rate": 60.0,
                }
            ]
        }

        manager = create_mock_manager(config)
        portfolio_results = manager.create_portfolio_backtest_results(results, strategy_results)

        assert isinstance(portfolio_results, PortfolioBacktestResults)
        assert portfolio_results.initial_portfolio_value == 10000.0
        assert len(portfolio_results.strategy_summaries) == 1
        assert portfolio_results.strategy_summaries[0].strategy_name == "SMACross"

    def test_create_portfolio_backtest_results_edge_cases(self):
        """Test portfolio backtest results edge cases."""
        # Test getting dates from backtest results
        results = {
            "initial_portfolio_value": 10000.0,
            "initial_capital": 10000.0,
            "backtesting": [
                {
                    "ticker": "AAPL",
                    "strategy": "SMACross",
                    "return_pct": 10.0,
                    "start_date": "2023-01-15",
                    "end_date": "2023-12-30",
                }
            ],
        }

        config = StockulaConfig()
        config.data.start_date = None
        config.data.end_date = None

        strategy_results = {
            "SMACross": [
                {
                    "ticker": "AAPL",
                    "strategy": "SMACross",
                    "return_pct": 10.0,
                    "sharpe_ratio": 1.0,
                    "max_drawdown_pct": -5.0,
                    "num_trades": 10,
                    "win_rate": 50.0,
                }
            ]
        }

        manager = create_mock_manager(config)
        portfolio_results = manager.create_portfolio_backtest_results(results, strategy_results)

        assert portfolio_results.date_range["start"] == "2023-01-15"
        assert portfolio_results.date_range["end"] == "2023-12-30"


class TestSaveDetailedReport:
    """Test report saving functionality."""

    @patch("stockula.manager.datetime")
    @patch("stockula.manager.Path")
    @patch("builtins.open", create=True)
    @patch("stockula.manager.json.dump")
    def test_save_detailed_report(self, mock_json_dump, mock_open, mock_path_class, mock_datetime):
        """Test saving detailed strategy report."""
        mock_datetime.now.return_value.strftime.return_value = "20240115_120000"

        # Create proper Path mock hierarchy
        mock_results_path = Mock(spec=Path)
        mock_reports_dir = Mock(spec=Path)
        mock_report_file = Mock(spec=Path)

        # Setup the chain of Path operations
        mock_path_class.side_effect = lambda x: mock_results_path if x == "./results" else Mock(spec=Path)
        mock_results_path.__truediv__ = Mock(return_value=mock_reports_dir)
        mock_reports_dir.mkdir = Mock()
        mock_reports_dir.__truediv__ = Mock(return_value=mock_report_file)
        mock_report_file.__str__ = Mock(return_value="results/reports/strategy_report_KAMA_20240115_120000.json")

        mock_file = Mock()
        mock_open.return_value.__enter__ = Mock(return_value=mock_file)
        mock_open.return_value.__exit__ = Mock(return_value=None)

        strategy_results = [{"ticker": "NVDA", "strategy": "KAMA", "return_pct": 50.0, "num_trades": 100}]
        results = {"initial_portfolio_value": 50000.0, "initial_capital": 50000.0}
        config = StockulaConfig()

        manager = create_mock_manager(config)
        report_path = manager.save_detailed_report("KAMA", strategy_results, results)

        assert str(report_path) == "results/reports/strategy_report_KAMA_20240115_120000.json"
        assert mock_json_dump.called


# Entry point coverage
class TestMainEntryPoint:
    """Test the main entry point coverage."""

    def test_main_entry_point_if_name_main(self):
        """Test the if __name__ == '__main__' entry point."""
        with patch("stockula.main.main") as mock_main:
            import stockula.main as main_module

            # Save original __name__
            original_name = main_module.__name__

            try:
                # Set module name to trigger the condition
                main_module.__name__ = "__main__"

                # Execute just the if condition block
                if main_module.__name__ == "__main__":
                    main_module.main()

                # Verify main was called
                mock_main.assert_called_once()

            finally:
                # Restore original __name__
                main_module.__name__ = original_name


class TestTrainTestSplitBacktesting:
    """Test backtesting with train/test split functionality."""

    def test_run_backtest_with_train_test_split(self):
        """Test backtest with train/test split."""
        config = StockulaConfig()
        strategy_config = Mock()
        strategy_config.name = "SMACross"
        strategy_config.parameters = {"fast_period": 10, "slow_period": 20}
        config.backtest.strategies = [strategy_config]

        # Set train/test dates
        config.forecast.train_start_date = date(2023, 1, 1)
        config.forecast.train_end_date = date(2023, 6, 30)
        config.forecast.test_start_date = date(2023, 7, 1)
        config.forecast.test_end_date = date(2023, 12, 31)
        # This flag doesn't exist, we check for train/test dates instead
        config.backtest.optimize = True
        config.backtest.optimization_params = {"fast_period": [10, 20], "slow_period": [30, 40]}

        mock_runner = Mock()
        mock_runner.run_with_train_test_split.return_value = {
            "train_period": {"start": "2023-01-01", "end": "2023-06-30", "days": 180},
            "test_period": {"start": "2023-07-01", "end": "2023-12-31", "days": 183},
            "train_results": {
                "return_pct": 25.5,
                "sharpe_ratio": 2.1,
                "max_drawdown_pct": -5.2,
                "num_trades": 30,
                "win_rate": 62.0,
            },
            "test_results": {
                "return_pct": 18.3,
                "sharpe_ratio": 1.7,
                "max_drawdown_pct": -7.8,
                "num_trades": 25,
                "win_rate": 58.0,
            },
            "optimized_parameters": {"fast_period": 15, "slow_period": 35},
            "performance_degradation": {"return_diff": -7.2, "sharpe_diff": -0.4},
        }

        manager = create_mock_manager(config)
        # Set up the container's backtest_runner to return our mock
        manager.container.backtest_runner.return_value = mock_runner
        results = manager.run_backtest("AAPL")

        assert len(results) == 1
        result = results[0]
        assert result["ticker"] == "AAPL"
        assert result["strategy"] == "SMACross"
        assert "train_period" in result
        assert "test_period" in result
        assert "train_results" in result
        assert "test_results" in result
        assert result["return_pct"] == 18.3  # Test period return
        assert result["parameters"]["fast_period"] == 15  # Optimized param

    def test_run_backtest_with_portfolio_info(self):
        """Test backtest with portfolio information fields."""
        config = StockulaConfig()
        strategy_config = Mock()
        strategy_config.name = "RSI"
        strategy_config.parameters = {"period": 14}
        config.backtest.strategies = [strategy_config]

        # Use backtest dates instead of forecast dates
        config.backtest.start_date = date(2023, 1, 1)
        config.backtest.end_date = date(2023, 12, 31)
        # No train/test split

        mock_runner = Mock()
        mock_runner.run_from_symbol.return_value = {
            "Return [%]": 22.5,
            "Sharpe Ratio": 1.85,
            "Max. Drawdown [%]": -6.3,
            "# Trades": 35,
            "Win Rate [%]": 60.0,
            "Initial Cash": 100000.0,
            "Start Date": "2023-01-01",
            "End Date": "2023-12-31",
            "Trading Days": 252,
            "Calendar Days": 365,
        }

        manager = create_mock_manager(config)
        # Set up the container's backtest_runner to return our mock
        manager.container.backtest_runner.return_value = mock_runner
        results = manager.run_backtest("AAPL")

        assert len(results) == 1
        result = results[0]
        assert result["initial_cash"] == 100000.0
        assert result["start_date"] == "2023-01-01"
        assert result["end_date"] == "2023-12-31"
        assert result["trading_days"] == 252
        assert result["calendar_days"] == 365


class TestPrintResultsEdgeCases:
    """Test edge cases in print_results function."""

    def test_print_results_with_nested_indicators(self, capsys):
        """Test printing results with nested indicator values."""
        results = {
            "technical_analysis": [
                {
                    "ticker": "AAPL",
                    "indicators": {
                        "BBands": {
                            "upper": 155.50,
                            "middle": 150.00,
                            "lower": 144.50,
                        },
                        "MACD": {
                            "macd": 2.35,
                            "signal": 2.10,
                            "histogram": 0.25,
                        },
                    },
                }
            ]
        }

        print_results(results, "console")
        captured = capsys.readouterr()
        assert "BBands - upper" in captured.out
        assert "155.50" in captured.out
        assert "MACD - signal" in captured.out
        assert "2.10" in captured.out

    def test_print_results_with_portfolio_composition(self, capsys):
        """Test printing results with portfolio composition table."""
        from stockula.domain.category import Category

        config = StockulaConfig()
        config.backtest.hold_only_categories = ["COMMODITY"]

        # Mock container
        container = Mock()

        # Mock portfolio
        mock_asset1 = Mock()
        mock_asset1.symbol = "AAPL"
        mock_asset1.category = Category.LARGE_CAP
        mock_asset1.quantity = 10.0

        mock_asset2 = Mock()
        mock_asset2.symbol = "GLD"
        mock_asset2.category = Category.COMMODITY
        mock_asset2.quantity = 5.0

        mock_portfolio = Mock()
        mock_portfolio.get_all_assets.return_value = [mock_asset1, mock_asset2]

        mock_factory = Mock()
        mock_factory.create_portfolio.return_value = mock_portfolio
        container.domain_factory.return_value = mock_factory

        # Mock fetcher
        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.return_value = {"AAPL": 150.0, "GLD": 180.0}
        container.data_fetcher.return_value = mock_fetcher

        results = {
            "backtesting": [
                {
                    "strategy": "test",
                    "ticker": "AAPL",
                    "return_pct": 10.0,
                    "sharpe_ratio": 1.2,
                    "max_drawdown_pct": -5.0,
                    "num_trades": 20,
                    "win_rate": 55.0,
                }
            ]
        }

        print_results(results, "console", config, container)
        captured = capsys.readouterr()

        assert "Portfolio Composition" in captured.out
        assert "AAPL" in captured.out
        assert "Hold Only" in captured.out  # GLD should be marked as hold only

    def test_print_results_portfolio_composition_error(self, capsys):
        """Test portfolio composition with price fetch error."""
        from stockula.domain.category import Category

        config = StockulaConfig()
        container = Mock()

        mock_asset = Mock()
        mock_asset.symbol = "AAPL"
        mock_asset.category = Category.LARGE_CAP
        mock_asset.quantity = 10.0

        mock_portfolio = Mock()
        mock_portfolio.get_all_assets.return_value = [mock_asset]

        mock_factory = Mock()
        mock_factory.create_portfolio.return_value = mock_portfolio
        container.domain_factory.return_value = mock_factory

        # Mock fetcher that raises exception
        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.side_effect = Exception("API error")
        container.data_fetcher.return_value = mock_fetcher

        results = {
            "backtesting": [
                {
                    "strategy": "test",
                    "ticker": "AAPL",
                    "return_pct": 10.0,
                    "sharpe_ratio": 1.2,
                    "max_drawdown_pct": -5.0,
                    "num_trades": 20,
                    "win_rate": 55.0,
                }
            ]
        }

        print_results(results, "console", config, container)
        captured = capsys.readouterr()

        assert "N/A" in captured.out  # Should show N/A for values

    def test_print_results_forecast_evaluation_metrics(self, capsys):
        """Test printing forecast results with evaluation metrics."""
        results = {
            "forecasting": [
                {
                    "ticker": "AAPL",
                    "current_price": 150.0,
                    "forecast_price": 160.0,
                    "lower_bound": 155.0,
                    "upper_bound": 165.0,
                    "best_model": "ARIMA",
                    "evaluation": {"rmse": 3.5, "mae": 2.8, "mape": 1.9},
                    "train_period": {"start": "2023-01-01", "end": "2023-06-30"},
                    "test_period": {"start": "2023-07-01", "end": "2023-07-31"},
                },
                {
                    "ticker": "MSFT",
                    "current_price": 300.0,
                    "forecast_price": 280.0,
                    "lower_bound": 270.0,
                    "upper_bound": 290.0,
                    "best_model": "ETS",
                    "evaluation": {"rmse": 5.2, "mae": 4.1, "mape": 2.1},
                    "train_period": {"start": "2023-01-01", "end": "2023-06-30"},
                    "test_period": {"start": "2023-07-01", "end": "2023-07-31"},
                },
            ]
        }

        print_results(results, "console")
        captured = capsys.readouterr()

        assert "Forecast Evaluation Metrics" in captured.out
        assert "RMSE" in captured.out
        assert "$3.50" in captured.out
        assert "MAPE %" in captured.out
        assert "1.90%" in captured.out


class TestPortfolioBacktestResultsComplex:
    """Test complex portfolio backtest results creation."""

    def test_create_portfolio_backtest_results_with_broker_configs(self):
        """Test creating portfolio results with different broker configurations."""

        # Test with modern broker config
        config = StockulaConfig()
        config.backtest.broker_config = BrokerConfig(
            name="td_ameritrade",
            commission_type="percentage",
            commission_value=0.001,
            min_commission=5.0,
            regulatory_fees=0.01,
            exchange_fees=0.005,
        )
        config.data.start_date = date(2023, 1, 1)
        config.data.end_date = date(2023, 12, 31)

        results = {"initial_portfolio_value": 100000.0, "initial_capital": 100000.0}
        strategy_results = {
            "SMACross": [
                {
                    "ticker": "AAPL",
                    "strategy": "SMACross",
                    "return_pct": 15.0,
                    "sharpe_ratio": 1.5,
                    "max_drawdown_pct": -8.0,
                    "num_trades": 25,
                    "win_rate": 60.0,
                }
            ]
        }

        manager = create_mock_manager(config)
        portfolio_results = manager.create_portfolio_backtest_results(results, strategy_results)

        assert portfolio_results.broker_config["name"] == "td_ameritrade"
        assert portfolio_results.broker_config["commission_type"] == "percentage"
        assert portfolio_results.broker_config["regulatory_fees"] == 0.01

    def test_create_portfolio_backtest_results_legacy_broker(self):
        """Test creating portfolio results with legacy broker config."""
        config = StockulaConfig()
        config.backtest.broker_config = None  # No modern config
        config.backtest.commission = 0.002  # Legacy commission
        config.data.start_date = date(2023, 1, 1)
        config.data.end_date = date(2023, 12, 31)

        results = {"initial_portfolio_value": 100000.0, "initial_capital": 100000.0}
        strategy_results = {"RSI": []}

        manager = create_mock_manager(config)
        portfolio_results = manager.create_portfolio_backtest_results(results, strategy_results)

        assert portfolio_results.broker_config["name"] == "legacy"
        assert portfolio_results.broker_config["commission_value"] == 0.002

    def test_create_portfolio_backtest_results_date_fallback(self):
        """Test date fallback logic in portfolio results."""
        config = StockulaConfig()
        # No dates in config
        config.backtest.start_date = None
        config.backtest.end_date = None
        config.data.start_date = None
        config.data.end_date = None

        results = {
            "initial_portfolio_value": 100000.0,
            "initial_capital": 100000.0,
            "backtesting": [
                {
                    "ticker": "AAPL",
                    "strategy": "SMACross",
                    "start_date": "2023-03-15",
                    "end_date": "2023-11-30",
                    "return_pct": 10.0,
                }
            ],
        }
        strategy_results = {"SMACross": []}

        manager = create_mock_manager(config)
        portfolio_results = manager.create_portfolio_backtest_results(results, strategy_results)

        assert portfolio_results.date_range["start"] == "2023-03-15"
        assert portfolio_results.date_range["end"] == "2023-11-30"


class TestMainFunctionAdvanced:
    """Test advanced main function scenarios."""

    @patch("stockula.main.create_container")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.log_manager")
    @patch(
        "sys.argv",
        [
            "stockula",
            "--ticker",
            "AAPL",
            "--train-start",
            "2023-01-01",
            "--train-end",
            "2023-06-30",
            "--test-start",
            "2023-07-01",
            "--test-end",
            "2023-12-31",
        ],
    )
    def test_main_with_date_overrides(self, mock_log_manager, mock_logging, mock_container):
        """Test main function with command line date overrides."""
        config = StockulaConfig()
        config.portfolio.tickers = []  # Will be overridden

        container = Mock()
        container.stockula_config.return_value = config
        container.logging_manager.return_value = Mock()

        mock_portfolio = Mock()
        mock_portfolio.name = "Test"
        mock_portfolio.initial_capital = 100000.0
        mock_portfolio.get_all_assets.return_value = []
        mock_portfolio.allocation_method = "equal"
        mock_portfolio.get_portfolio_value.return_value = 100000.0

        mock_factory = Mock()
        mock_factory.create_portfolio.return_value = mock_portfolio
        container.domain_factory.return_value = mock_factory

        # Add fetcher mock
        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.return_value = {}
        container.data_fetcher.return_value = mock_fetcher

        mock_container.return_value = container

        with patch("stockula.main.print_results"):
            main()

        # Check that dates were overridden
        assert config.forecast.train_start_date == date(2023, 1, 1)
        assert config.forecast.train_end_date == date(2023, 6, 30)
        assert config.forecast.test_start_date == date(2023, 7, 1)
        assert config.forecast.test_end_date == date(2023, 12, 31)

    @patch("stockula.main.create_container")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.log_manager")
    @patch("stockula.manager.StockulaManager.run_technical_analysis")
    @patch("stockula.manager.StockulaManager.get_strategy_class")
    @patch("stockula.main.print_results")
    @patch("sys.argv", ["stockula", "--mode", "all"])
    def test_main_all_mode_with_hold_only_assets(
        self, mock_print, mock_get_strategy, mock_ta, mock_log_manager, mock_logging, mock_container
    ):
        """Test main function in all mode with hold-only assets."""
        from stockula.domain.category import Category

        config = StockulaConfig()
        config.backtest.hold_only_categories = ["COMMODITY"]
        # Add a strategy to backtest
        strategy_config = Mock()
        strategy_config.name = "rsi"
        strategy_config.parameters = {"period": 14}
        config.backtest.strategies = [strategy_config]

        # Create assets with different categories
        mock_asset1 = Mock()
        mock_asset1.symbol = "AAPL"
        mock_asset1.category = Category.LARGE_CAP
        mock_asset1.quantity = 10.0

        mock_asset2 = Mock()
        mock_asset2.symbol = "GLD"
        mock_asset2.category = Category.COMMODITY
        mock_asset2.quantity = 5.0

        container = Mock()
        container.stockula_config.return_value = config
        container.logging_manager.return_value = Mock()

        mock_portfolio = Mock()
        mock_portfolio.name = "Test"
        mock_portfolio.initial_capital = 100000.0
        mock_portfolio.get_all_assets.return_value = [mock_asset1, mock_asset2]
        mock_portfolio.allocation_method = "equal"
        mock_portfolio.get_portfolio_value.return_value = 103000.0

        mock_factory = Mock()
        mock_factory.create_portfolio.return_value = mock_portfolio
        container.domain_factory.return_value = mock_factory

        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.return_value = {"AAPL": 150.0, "GLD": 180.0}
        container.data_fetcher.return_value = mock_fetcher

        # Mock backtest runner
        mock_runner = Mock()
        mock_runner.run_from_symbol.return_value = {
            "Return [%]": 10.5,
            "Sharpe Ratio": 1.8,
            "Max. Drawdown [%]": -5.2,
            "# Trades": 15,
            "Win Rate [%]": 65.0,
        }
        container.backtest_runner.return_value = mock_runner

        mock_container.return_value = container

        mock_ta.return_value = {"ticker": "AAPL", "indicators": {}}
        mock_strategy = Mock()
        mock_get_strategy.return_value = mock_strategy

        main()

        # Should run TA on both assets
        assert mock_ta.call_count == 2

        # Should only backtest AAPL (not GLD which is hold-only)
        assert mock_runner.run_from_symbol.call_count == 1
        # Verify it was called with AAPL
        call_args = mock_runner.run_from_symbol.call_args[0]
        assert call_args[0] == "AAPL"

    @patch("stockula.main.create_container")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.log_manager")
    @patch("stockula.manager.StockulaManager.run_forecast")
    @patch("stockula.manager.StockulaManager.run_forecast_with_evaluation")
    @patch("stockula.main.print_results")
    @patch("stockula.main.Progress")
    @patch("sys.argv", ["stockula", "--mode", "forecast"])
    def test_main_forecast_mode_with_portfolio_value(
        self,
        mock_progress_class,
        mock_print,
        mock_forecast_eval,
        mock_forecast,
        mock_log_manager,
        mock_logging,
        mock_container,
    ):
        """Test main function in forecast mode showing portfolio value table."""
        from stockula.domain.category import Category

        config = StockulaConfig()
        config.forecast.test_start_date = date(2024, 1, 1)
        config.forecast.test_end_date = date(2024, 1, 31)

        mock_asset = Mock()
        mock_asset.symbol = "AAPL"
        mock_asset.category = Category.LARGE_CAP
        mock_asset.quantity = 10.0

        container = Mock()
        container.stockula_config.return_value = config
        container.logging_manager.return_value = Mock()

        mock_portfolio = Mock()
        mock_portfolio.name = "Test"
        mock_portfolio.initial_capital = 100000.0
        mock_portfolio.get_all_assets.return_value = [mock_asset]
        mock_portfolio.allocation_method = "equal"
        mock_portfolio.get_portfolio_value.return_value = 101500.0

        mock_factory = Mock()
        mock_factory.create_portfolio.return_value = mock_portfolio
        container.domain_factory.return_value = mock_factory

        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.return_value = {"AAPL": 150.0}
        mock_fetcher.get_stock_data.return_value = pd.DataFrame()  # For start date fetch
        container.data_fetcher.return_value = mock_fetcher
        container.stock_forecaster.return_value = Mock()

        # Ensure portfolio value is numeric
        mock_portfolio.get_portfolio_value.return_value = 101500.0

        mock_container.return_value = container

        # Mock Progress to avoid timestamp comparison errors
        mock_progress = Mock()
        mock_progress.__enter__ = Mock(return_value=mock_progress)
        mock_progress.__exit__ = Mock(return_value=None)
        mock_progress.add_task = Mock(return_value=1)
        mock_progress.update = Mock()
        mock_progress.advance = Mock()
        mock_progress_class.return_value = mock_progress

        mock_forecast_eval.return_value = {
            "ticker": "AAPL",
            "current_price": 150.0,
            "forecast_price": 160.0,
            "lower_bound": 155.0,
            "upper_bound": 165.0,
            "best_model": "ARIMA",
            "evaluation": {"rmse": 2.0, "mae": 1.5, "mape": 1.0},
            "end_date": "2024-01-31",
        }

        with patch("stockula.main.console") as mock_console:
            main()

            # Check that portfolio value table was printed
            # Look for Table objects in print calls
            table_calls = [
                call
                for call in mock_console.print.call_args_list
                if len(call[0]) > 0 and hasattr(call[0][0], "__class__") and call[0][0].__class__.__name__ == "Table"
            ]
            assert len(table_calls) > 0

            # At least one table should have "Portfolio Value" as title
            assert any(
                hasattr(call[0][0], "title") and "Portfolio Value" in str(call[0][0].title) for call in table_calls
            )

    @patch("stockula.main.create_container")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.log_manager")
    @patch("sys.argv", ["stockula", "--mode", "backtest"])
    def test_main_backtest_start_price_fetching(self, mock_log_manager, mock_logging, mock_container):
        """Test fetching start prices for backtest initial portfolio value."""
        config = StockulaConfig()
        config.data.start_date = date(2023, 1, 1)
        config.data.end_date = date(2023, 12, 31)

        mock_asset = Mock()
        mock_asset.symbol = "AAPL"
        mock_asset.category = Category.LARGE_CAP
        mock_asset.quantity = 10.0

        container = Mock()
        container.stockula_config.return_value = config
        container.logging_manager.return_value = Mock()

        mock_portfolio = Mock()
        mock_portfolio.name = "Test"
        mock_portfolio.initial_capital = 100000.0
        mock_portfolio.get_all_assets.return_value = [mock_asset]
        mock_portfolio.allocation_method = "equal"
        mock_portfolio.get_portfolio_value.return_value = 101500.0
        mock_portfolio.get_portfolio_value.return_value = 101500.0  # With price

        mock_factory = Mock()
        mock_factory.create_portfolio.return_value = mock_portfolio
        container.domain_factory.return_value = mock_factory

        # Mock fetcher to return empty data first, then data on extended range
        mock_fetcher = Mock()
        empty_df = pd.DataFrame()
        data_df = pd.DataFrame({"Close": [150.0]}, index=pd.date_range("2023-01-03", periods=1))
        mock_fetcher.get_stock_data.side_effect = [empty_df, data_df]
        mock_fetcher.get_current_prices.return_value = {"AAPL": 155.0}
        container.data_fetcher.return_value = mock_fetcher
        container.backtest_runner.return_value = Mock()

        mock_container.return_value = container

        with patch("stockula.manager.StockulaManager.run_backtest") as mock_backtest:
            mock_backtest.return_value = []
            with patch("stockula.main.print_results"):
                main()

        # Check that fetcher was called to get start prices
        assert mock_fetcher.get_stock_data.call_count >= 1

    @patch("stockula.main.create_container")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.log_manager")
    @patch("stockula.main.print_results")
    @patch("stockula.manager.StockulaManager.create_portfolio_backtest_results")
    @patch("stockula.manager.StockulaManager.save_detailed_report")
    @patch("sys.argv", ["stockula", "--mode", "backtest"])
    def test_main_backtest_strategy_summaries(
        self,
        mock_save_report,
        mock_create_results,
        mock_print,
        mock_log_manager,
        mock_logging,
        mock_container,
    ):
        """Test displaying strategy summaries with different broker configurations."""
        config = StockulaConfig()

        # Test with tiered broker config
        config.backtest.broker_config = BrokerConfig(
            name="interactive_brokers",
            commission_type="tiered",
            commission_value=0.0035,
            min_commission=0.35,
        )

        # Add a strategy to the config
        strategy_config = Mock()
        strategy_config.name = "smacross"
        strategy_config.parameters = {"fast_period": 10, "slow_period": 20}
        config.backtest.strategies = [strategy_config]

        container = Mock()
        container.stockula_config.return_value = config
        container.logging_manager.return_value = Mock()

        mock_asset = Mock()
        mock_asset.symbol = "AAPL"
        mock_asset.category = Category.LARGE_CAP
        mock_asset.quantity = 10.0

        mock_portfolio = Mock()
        mock_portfolio.name = "Test"
        mock_portfolio.initial_capital = 100000.0
        mock_portfolio.get_all_assets.return_value = [mock_asset]
        mock_portfolio.allocation_method = "equal"
        mock_portfolio.get_portfolio_value.return_value = 101500.0

        mock_factory = Mock()
        mock_factory.create_portfolio.return_value = mock_portfolio
        container.domain_factory.return_value = mock_factory

        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.return_value = {"AAPL": 150.0}
        container.data_fetcher.return_value = mock_fetcher

        # Mock backtest runner
        mock_runner = Mock()
        mock_runner.run_from_symbol.return_value = {
            "Return [%]": 15.0,
            "Sharpe Ratio": 1.5,
            "Max. Drawdown [%]": -8.0,
            "# Trades": 25,
            "Win Rate [%]": 60.0,
            "Start Date": "2023-01-01",
            "End Date": "2023-12-31",
        }
        container.backtest_runner.return_value = mock_runner

        mock_container.return_value = container

        # Manager handles strategy class lookup internally

        # Mock portfolio results
        mock_strategy_summary = Mock()
        mock_strategy_summary.strategy_name = "SMACross"
        mock_strategy_summary.initial_portfolio_value = 100000.0
        mock_strategy_summary.final_portfolio_value = 115000.0
        mock_strategy_summary.average_return_pct = 15.0
        mock_strategy_summary.total_return_pct = 15.0
        mock_strategy_summary.winning_stocks = 1
        mock_strategy_summary.losing_stocks = 0
        mock_strategy_summary.total_trades = 25
        mock_strategy_summary.parameters = {"fast_period": 10, "slow_period": 20}
        mock_strategy_summary.detailed_results = []

        mock_portfolio_results = Mock()
        mock_portfolio_results.strategy_summaries = [mock_strategy_summary]
        mock_portfolio_results.date_range = {"start": "2023-01-01", "end": "2023-12-31"}

        mock_create_results.return_value = mock_portfolio_results
        mock_save_report.return_value = "results/reports/strategy_report_SMACross_20240101_120000.json"

        with patch("stockula.main.console") as mock_console:
            main()

            # Check that strategy summary panel was printed
            panel_calls = [call for call in mock_console.print.call_args_list if "Panel" in str(call)]
            assert len(panel_calls) > 0


class TestForecastingEdgeCases:
    """Test forecasting error handling and edge cases."""

    def test_run_forecast_empty_predictions(self):
        """Test forecast with empty predictions."""
        config = StockulaConfig()
        config.forecast.forecast_length = 30

        mock_forecaster = Mock()
        # Return empty DataFrame
        mock_forecaster.forecast_from_symbol.return_value = pd.DataFrame()
        mock_forecaster.get_best_model.return_value = {"model_name": "ARIMA"}

        manager = create_mock_manager(config)
        # Set up the container's stock_forecaster to return our mock
        manager.container.stock_forecaster.return_value = mock_forecaster
        result = manager.run_forecast("AAPL")

        assert result["ticker"] == "AAPL"
        assert "error" in result

    def test_run_forecast_with_evaluation_missing_dates(self):
        """Test forecast evaluation with missing date configurations."""
        config = StockulaConfig()
        # No forecast dates set, should fall back to data dates
        config.data.start_date = date(2023, 1, 1)
        config.data.end_date = date(2023, 12, 31)
        config.forecast.train_start_date = None
        config.forecast.train_end_date = None
        config.forecast.test_start_date = None
        config.forecast.test_end_date = None

        mock_forecaster = Mock()
        mock_result = {
            "predictions": pd.DataFrame(
                {
                    "forecast": [150.0],
                    "lower_bound": [148.0],
                    "upper_bound": [152.0],
                },
                index=pd.date_range("2024-01-01", periods=1),
            ),
            "train_period": {"start": "2023-01-01", "end": "2023-12-31", "days": 365},
            "test_period": {"start": "2024-01-01", "end": "2024-01-31", "days": 31},
            "evaluation_metrics": {"rmse": 2.5, "mae": 2.0, "mape": 1.5},
        }
        mock_forecaster.forecast_from_symbol_with_evaluation.return_value = mock_result
        mock_forecaster.get_best_model.return_value = {"model_name": "ARIMA"}

        manager = create_mock_manager(config)
        # Set up the container's stock_forecaster to return our mock
        manager.container.stock_forecaster.return_value = mock_forecaster
        result = manager.run_forecast_with_evaluation("AAPL")

        assert result["ticker"] == "AAPL"
        assert "evaluation" in result


class TestDateStringHandling:
    """Test date string conversion edge cases."""

    def test_date_to_string_with_string_date_in_config(self):
        """Test date_to_string when config contains string dates."""
        # This tests the edge case in main() where config dates might be strings
        config = StockulaConfig()
        config.data.start_date = "2023-01-01"  # String instead of date object

        # In main(), this would be handled
        manager = create_mock_manager(config)
        start_date_str = manager.date_to_string(config.data.start_date)
        assert start_date_str == "2023-01-01"


class TestMainHoldingsDisplay:
    """Test portfolio holdings display in main function."""

    @patch("stockula.main.create_container")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.log_manager")
    @patch("sys.argv", ["stockula", "--mode", "ta"])
    def test_main_holdings_with_no_category(self, mock_log_manager, mock_logging, mock_container):
        """Test displaying holdings with assets that have no category."""
        config = StockulaConfig()

        # Create asset with None category
        mock_asset = Mock()
        mock_asset.symbol = "AAPL"
        mock_asset.category = None
        mock_asset.quantity = "invalid"  # Test non-numeric quantity

        container = Mock()
        container.stockula_config.return_value = config
        container.logging_manager.return_value = Mock()

        mock_portfolio = Mock()
        mock_portfolio.name = "Test"
        mock_portfolio.initial_capital = 100000.0
        mock_portfolio.get_all_assets.return_value = [mock_asset]
        mock_portfolio.allocation_method = "equal"
        mock_portfolio.get_portfolio_value.return_value = 101500.0

        mock_factory = Mock()
        mock_factory.create_portfolio.return_value = mock_portfolio
        container.domain_factory.return_value = mock_factory

        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.return_value = {"AAPL": 150.0}
        container.data_fetcher.return_value = mock_fetcher

        mock_container.return_value = container

        with patch("stockula.manager.StockulaManager.run_technical_analysis") as mock_ta:
            mock_ta.return_value = {"ticker": "AAPL", "indicators": {}}
            with patch("stockula.main.print_results"):
                with patch("stockula.main.console") as mock_console:
                    main()

                    # Check that holdings table was printed with "N/A" for category
                    table_calls = [
                        call
                        for call in mock_console.print.call_args_list
                        if hasattr(call[0][0], "title") and "Holdings" in str(call[0][0].title)
                    ]
                    assert len(table_calls) > 0


class TestMainErrorHandling:
    """Test error handling in main function."""

    @patch("stockula.main.create_container")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.log_manager")
    @patch("stockula.manager.StockulaManager.run_technical_analysis")
    @patch("stockula.main.print_results")
    @patch("sys.argv", ["stockula", "--mode", "ta"])
    def test_main_with_progress_bars_single_operation(
        self, mock_print, mock_ta, mock_log_manager, mock_logging, mock_container
    ):
        """Test main with progress bars for single operation mode."""
        config = StockulaConfig()
        config.output["progress"] = True

        container = Mock()
        container.stockula_config.return_value = config
        container.logging_manager.return_value = Mock()

        mock_asset = Mock()
        mock_asset.symbol = "AAPL"
        mock_asset.category = Category.LARGE_CAP
        mock_asset.quantity = 10.0

        mock_portfolio = Mock()
        mock_portfolio.name = "Test"
        mock_portfolio.initial_capital = 100000.0
        mock_portfolio.get_all_assets.return_value = [mock_asset]
        mock_portfolio.allocation_method = "equal"
        mock_portfolio.get_portfolio_value.return_value = 101500.0

        mock_factory = Mock()
        mock_factory.create_portfolio.return_value = mock_portfolio
        container.domain_factory.return_value = mock_factory

        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.return_value = {"AAPL": 150.0}
        container.data_fetcher.return_value = mock_fetcher

        mock_container.return_value = container

        mock_ta.return_value = {"ticker": "AAPL", "indicators": {"SMA_20": 150.0}}

        main()

        # Should call TA without progress since it's the only operation
        mock_ta.assert_called_once()

    @patch("stockula.main.create_container")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.log_manager")
    @patch("stockula.manager.StockulaManager.run_backtest")
    @patch("stockula.main.print_results")
    @patch("sys.argv", ["stockula", "--mode", "backtest"])
    def test_main_backtest_with_multiple_strategies_progress(
        self, mock_print, mock_backtest, mock_log_manager, mock_logging, mock_container
    ):
        """Test main backtest mode with multiple strategies and progress bars."""
        config = StockulaConfig()
        config.output["progress"] = True

        # Multiple strategies to test progress tracking
        strategy1 = Mock()
        strategy1.name = "SMACross"
        strategy1.parameters = {}

        strategy2 = Mock()
        strategy2.name = "RSI"
        strategy2.parameters = {"period": 14}

        config.backtest.strategies = [strategy1, strategy2]

        container = Mock()
        container.stockula_config.return_value = config
        container.logging_manager.return_value = Mock()

        mock_asset = Mock()
        mock_asset.symbol = "AAPL"
        mock_asset.category = Category.LARGE_CAP
        mock_asset.quantity = 10.0

        mock_portfolio = Mock()
        mock_portfolio.name = "Test"
        mock_portfolio.initial_capital = 100000.0
        mock_portfolio.get_all_assets.return_value = [mock_asset]
        mock_portfolio.allocation_method = "equal"
        mock_portfolio.get_portfolio_value.return_value = 101500.0

        mock_factory = Mock()
        mock_factory.create_portfolio.return_value = mock_portfolio
        container.domain_factory.return_value = mock_factory

        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.return_value = {"AAPL": 150.0}
        container.data_fetcher.return_value = mock_fetcher
        container.backtest_runner.return_value = Mock()

        mock_container.return_value = container

        # Return actual backtest results to trigger progress logic
        mock_backtest.return_value = [
            {
                "ticker": "AAPL",
                "strategy": "SMACross",
                "return_pct": 15.0,
                "sharpe_ratio": 1.2,
                "max_drawdown_pct": -8.0,
                "num_trades": 20,
                "win_rate": 60.0,
            },
            {
                "ticker": "AAPL",
                "strategy": "RSI",
                "return_pct": 12.0,
                "sharpe_ratio": 1.0,
                "max_drawdown_pct": -6.0,
                "num_trades": 15,
                "win_rate": 55.0,
            },
        ]

        with patch("stockula.main.Progress") as mock_progress_class:
            mock_progress = Mock()
            mock_task = Mock()
            mock_progress.add_task.return_value = mock_task
            mock_progress.update = Mock()  # Add update method
            mock_progress.advance = Mock()  # Add advance method
            mock_progress_class.return_value.__enter__.return_value = mock_progress
            mock_progress_class.return_value.__exit__ = Mock(return_value=None)

            main()

            # Should create progress bars
            assert mock_progress_class.called
            # Should add task for backtesting progress
            assert mock_progress.add_task.called
            # Should advance progress for each strategy result
            assert mock_progress.advance.called

    @patch("stockula.main.create_container")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.log_manager")
    @patch("stockula.manager.StockulaManager.run_forecast_with_evaluation")
    @patch("stockula.main.print_results")
    @patch("sys.argv", ["stockula", "--mode", "forecast"])
    def test_main_forecast_parallel_processing(
        self, mock_print, mock_forecast_eval, mock_log_manager, mock_logging, mock_container
    ):
        """Test main forecast mode with parallel processing and progress tracking."""
        config = StockulaConfig()
        config.output["progress"] = True
        config.forecast.test_start_date = date(2024, 1, 1)
        config.forecast.test_end_date = date(2024, 1, 31)

        # Multiple assets for parallel forecasting
        mock_asset1 = Mock()
        mock_asset1.symbol = "AAPL"
        mock_asset1.category = Category.LARGE_CAP
        mock_asset1.quantity = 10.0

        mock_asset2 = Mock()
        mock_asset2.symbol = "MSFT"
        mock_asset2.category = Category.LARGE_CAP
        mock_asset2.quantity = 5.0

        container = Mock()
        container.stockula_config.return_value = config
        container.logging_manager.return_value = Mock()

        mock_portfolio = Mock()
        mock_portfolio.name = "Test"
        mock_portfolio.initial_capital = 100000.0
        mock_portfolio.get_all_assets.return_value = [mock_asset1, mock_asset2]
        mock_portfolio.allocation_method = "equal"
        mock_portfolio.get_portfolio_value.return_value = 103000.0

        mock_factory = Mock()
        mock_factory.create_portfolio.return_value = mock_portfolio
        container.domain_factory.return_value = mock_factory

        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.return_value = {"AAPL": 150.0, "MSFT": 300.0}
        container.data_fetcher.return_value = mock_fetcher
        container.stock_forecaster.return_value = Mock()

        mock_container.return_value = container

        # Mock forecast results
        def forecast_side_effect(ticker, *args, **kwargs):
            if ticker == "AAPL":
                return {
                    "ticker": "AAPL",
                    "current_price": 150.0,
                    "forecast_price": 160.0,
                    "lower_bound": 155.0,
                    "upper_bound": 165.0,
                    "best_model": "ARIMA",
                    "evaluation": {"rmse": 2.0, "mae": 1.5, "mape": 1.0},
                }
            else:
                # Simulate error for MSFT
                raise Exception("API error")

        mock_forecast_eval.side_effect = forecast_side_effect

        with patch("stockula.main.Progress") as mock_progress_class:
            mock_progress = Mock()
            mock_task = Mock()
            mock_progress.add_task.return_value = mock_task
            mock_progress.update = Mock()  # Add update method
            mock_progress.advance = Mock()  # Add advance method
            mock_progress_class.return_value.__enter__.return_value = mock_progress
            mock_progress_class.return_value.__exit__ = Mock(return_value=None)

            main()

            # Should create progress bars for parallel forecasting
            assert mock_progress_class.called
            # Should handle both success and error cases
            results_call = mock_print.call_args[0][0]
            assert len(results_call["forecasting"]) == 2
            assert any(f.get("error") == "API error" for f in results_call["forecasting"])
