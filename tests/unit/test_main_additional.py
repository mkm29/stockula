"""Additional tests for main module to improve coverage."""

from datetime import date, datetime
from unittest.mock import Mock, patch

import pandas as pd

from stockula.config import StockulaConfig
from stockula.main import (
    create_portfolio_backtest_results,
    date_to_string,
    run_forecast_with_evaluation,
    run_technical_analysis,
    save_detailed_report,
)


class TestDateToString:
    """Test date_to_string function."""

    def test_date_to_string_with_none(self):
        """Test date_to_string with None value."""
        assert date_to_string(None) is None

    def test_date_to_string_with_string(self):
        """Test date_to_string with string value."""
        assert date_to_string("2024-01-01") == "2024-01-01"

    def test_date_to_string_with_date(self):
        """Test date_to_string with date object."""
        test_date = date(2024, 1, 15)
        assert date_to_string(test_date) == "2024-01-15"

    def test_date_to_string_with_datetime(self):
        """Test date_to_string with datetime object."""
        test_datetime = datetime(2024, 6, 30, 12, 30, 45)
        assert date_to_string(test_datetime) == "2024-06-30"


class TestRunForecastWithEvaluation:
    """Test run_forecast_with_evaluation function."""

    @patch("stockula.main.log_manager")
    def test_run_forecast_with_evaluation_success(self, mock_log_manager):
        """Test successful forecast with evaluation."""
        # Setup config
        config = StockulaConfig()
        config.forecast.train_start_date = date(2023, 1, 1)
        config.forecast.train_end_date = date(2023, 12, 31)
        config.forecast.test_start_date = date(2024, 1, 1)
        config.forecast.test_end_date = date(2024, 1, 31)
        config.forecast.forecast_length = 30
        config.forecast.model_list = ["ARIMA"]
        config.forecast.ensemble = False
        config.forecast.max_generations = 1

        # Mock forecaster
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
            "evaluation_metrics": {
                "rmse": 2.5,
                "mae": 2.0,
                "mape": 1.5,
                "accuracy": 98.5,
            },
        }
        mock_forecaster.forecast_from_symbol_with_evaluation.return_value = mock_result
        mock_forecaster.get_best_model.return_value = {
            "model_name": "ARIMA",
            "model_params": {"p": 1, "d": 1, "q": 1},
        }

        # Run forecast
        result = run_forecast_with_evaluation("AAPL", config, stock_forecaster=mock_forecaster)

        # Verify results
        assert result["ticker"] == "AAPL"
        assert result["current_price"] == 150.0
        assert result["forecast_price"] == 155.0
        assert result["lower_bound"] == 152.0
        assert result["upper_bound"] == 158.0
        assert result["forecast_length"] == 30
        assert result["best_model"] == "ARIMA"
        assert "evaluation" in result
        assert result["evaluation"]["rmse"] == 2.5
        assert result["evaluation"]["mape"] == 1.5

        # Verify logging calls
        mock_log_manager.info.assert_called()

    @patch("stockula.main.log_manager")
    def test_run_forecast_with_evaluation_keyboard_interrupt(self, mock_log_manager):
        """Test forecast with evaluation interrupted by user."""
        config = StockulaConfig()
        config.forecast.test_start_date = date(2024, 1, 1)
        config.forecast.test_end_date = date(2024, 1, 31)

        # Mock forecaster that raises KeyboardInterrupt
        mock_forecaster = Mock()
        mock_forecaster.forecast_from_symbol_with_evaluation.side_effect = KeyboardInterrupt()

        result = run_forecast_with_evaluation("TEST", config, stock_forecaster=mock_forecaster)

        assert result["ticker"] == "TEST"
        assert result["error"] == "Interrupted by user"
        mock_log_manager.warning.assert_called_with("Forecast for TEST interrupted by user")

    @patch("stockula.main.log_manager")
    def test_run_forecast_with_evaluation_exception(self, mock_log_manager):
        """Test forecast with evaluation with general exception."""
        config = StockulaConfig()
        config.forecast.test_start_date = date(2024, 1, 1)
        config.forecast.test_end_date = date(2024, 1, 31)

        # Mock forecaster that raises exception
        mock_forecaster = Mock()
        mock_forecaster.forecast_from_symbol_with_evaluation.side_effect = Exception("API error")

        result = run_forecast_with_evaluation("TEST", config, stock_forecaster=mock_forecaster)

        assert result["ticker"] == "TEST"
        assert result["error"] == "API error"
        mock_log_manager.error.assert_called_with("Error forecasting TEST: API error")

    @patch("stockula.main.log_manager")
    def test_run_forecast_with_evaluation_fallback_dates(self, mock_log_manager):
        """Test forecast with evaluation using fallback dates from data config."""
        # Setup config with missing forecast dates but with data dates
        config = StockulaConfig()
        config.data.start_date = date(2023, 6, 1)
        config.data.end_date = date(2023, 12, 31)
        config.forecast.train_start_date = None  # Will use data.start_date
        config.forecast.train_end_date = None  # Will use data.end_date
        config.forecast.test_start_date = date(2024, 1, 1)
        config.forecast.test_end_date = date(2024, 1, 31)

        # Mock forecaster
        mock_forecaster = Mock()
        mock_result = {
            "predictions": pd.DataFrame(
                {"forecast": [100.0], "lower_bound": [95.0], "upper_bound": [105.0]},
                index=pd.date_range("2024-01-01", periods=1),
            ),
            "train_period": {"start": "2023-06-01", "end": "2023-12-31"},
            "test_period": {"start": "2024-01-01", "end": "2024-01-31"},
            "evaluation_metrics": None,
        }
        mock_forecaster.forecast_from_symbol_with_evaluation.return_value = mock_result
        mock_forecaster.get_best_model.return_value = {"model_name": "ETS"}

        run_forecast_with_evaluation("TEST", config, stock_forecaster=mock_forecaster)

        # Verify fallback dates were used
        mock_forecaster.forecast_from_symbol_with_evaluation.assert_called_with(
            "TEST",
            train_start_date="2023-06-01",
            train_end_date="2023-12-31",
            test_start_date="2024-01-01",
            test_end_date="2024-01-31",
            model_list="fast",
            ensemble="auto",
            max_generations=2,
        )


class TestRunTechnicalAnalysisEdgeCases:
    """Test edge cases for run_technical_analysis function."""

    @patch("stockula.main.TechnicalIndicators")
    def test_run_technical_analysis_no_progress_bar(self, mock_ta_class):
        """Test technical analysis without progress bar."""
        # Setup config with all indicators
        config = StockulaConfig()
        config.technical_analysis.indicators = ["sma", "ema", "rsi", "macd", "bbands", "atr", "adx"]
        config.technical_analysis.sma_periods = [20, 50]
        config.technical_analysis.ema_periods = [12, 26]
        config.technical_analysis.rsi_period = 14
        config.technical_analysis.atr_period = 14

        # Mock data fetcher
        mock_data_fetcher = Mock()
        mock_data = pd.DataFrame(
            {"Close": [100] * 100},
            index=pd.date_range("2023-01-01", periods=100),
        )
        mock_data_fetcher.get_stock_data.return_value = mock_data

        # Mock technical indicators
        mock_ta = Mock()
        mock_ta.sma.return_value = pd.Series([100.0] * 100)
        mock_ta.ema.return_value = pd.Series([99.5] * 100)
        mock_ta.rsi.return_value = pd.Series([50.0] * 100)
        mock_ta.macd.return_value = pd.DataFrame({"MACD": [0.5] * 100, "MACD_SIGNAL": [0.3] * 100})
        mock_ta.bbands.return_value = pd.DataFrame(
            {"BB_UPPER": [102] * 100, "BB_MIDDLE": [100] * 100, "BB_LOWER": [98] * 100}
        )
        mock_ta.atr.return_value = pd.Series([2.0] * 100)
        mock_ta.adx.return_value = pd.Series([25.0] * 100)
        mock_ta_class.return_value = mock_ta

        # Run without progress bar
        result = run_technical_analysis(
            "TEST",
            config,
            data_fetcher=mock_data_fetcher,
            show_progress=False,
        )

        # Verify all indicators were calculated
        assert "SMA_20" in result["indicators"]
        assert "SMA_50" in result["indicators"]
        assert "EMA_12" in result["indicators"]
        assert "EMA_26" in result["indicators"]
        assert "RSI" in result["indicators"]
        assert "MACD" in result["indicators"]
        assert "BBands" in result["indicators"]
        assert "ATR" in result["indicators"]
        assert "ADX" in result["indicators"]

    @patch("stockula.main.TechnicalIndicators")
    def test_run_technical_analysis_with_date_conversion(self, mock_ta_class):
        """Test technical analysis with date objects that need conversion."""
        config = StockulaConfig()
        config.technical_analysis.indicators = ["sma"]
        config.technical_analysis.sma_periods = [10]
        config.data.start_date = date(2023, 1, 1)  # date object, not string
        config.data.end_date = date(2023, 12, 31)  # date object, not string

        # Mock data fetcher
        mock_data_fetcher = Mock()
        mock_data = pd.DataFrame(
            {"Close": [100] * 50},
            index=pd.date_range("2023-01-01", periods=50),
        )
        mock_data_fetcher.get_stock_data.return_value = mock_data

        # Mock technical indicators
        mock_ta = Mock()
        mock_ta.sma.return_value = pd.Series([100.0] * 50)
        mock_ta_class.return_value = mock_ta

        run_technical_analysis("TEST", config, data_fetcher=mock_data_fetcher)

        # Verify date conversion worked
        mock_data_fetcher.get_stock_data.assert_called_with(
            "TEST",
            start="2023-01-01",
            end="2023-12-31",
            interval="1d",
        )


class TestCreatePortfolioBacktestResultsEdgeCases:
    """Test edge cases for create_portfolio_backtest_results function."""

    def test_create_portfolio_backtest_results_with_dates_from_results(self):
        """Test creating portfolio results getting dates from backtest results."""
        # Setup test data with no dates in config
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
        # No dates in config
        config.data.start_date = None
        config.data.end_date = None
        config.backtest.start_date = None
        config.backtest.end_date = None

        strategy_results = {
            "SMACross": [
                {
                    "ticker": "AAPL",
                    "strategy": "SMACross",
                    "parameters": {},
                    "return_pct": 10.0,
                    "sharpe_ratio": 1.0,
                    "max_drawdown_pct": -5.0,
                    "num_trades": 10,
                    "win_rate": 60.0,
                }
            ]
        }

        # Create portfolio results
        portfolio_results = create_portfolio_backtest_results(results, config, strategy_results)

        # Should get dates from backtest results
        assert portfolio_results.date_range["start"] == "2023-01-15"
        assert portfolio_results.date_range["end"] == "2023-12-30"

    def test_create_portfolio_backtest_results_backtest_dates_priority(self):
        """Test backtest dates take priority over data dates."""
        results = {"initial_portfolio_value": 10000.0, "initial_capital": 10000.0}

        config = StockulaConfig()
        # Both backtest and data dates set
        config.backtest.start_date = date(2023, 6, 1)
        config.backtest.end_date = date(2023, 8, 31)
        config.data.start_date = date(2023, 1, 1)
        config.data.end_date = date(2023, 12, 31)

        strategy_results = {}

        portfolio_results = create_portfolio_backtest_results(results, config, strategy_results)

        # Backtest dates should take priority
        assert portfolio_results.date_range["start"] == "2023-06-01"
        assert portfolio_results.date_range["end"] == "2023-08-31"

    def test_create_portfolio_backtest_results_with_broker_config(self):
        """Test creating portfolio results with detailed broker config."""
        results = {"initial_portfolio_value": 10000.0, "initial_capital": 10000.0}

        config = StockulaConfig()
        config.backtest.broker_config = Mock()
        config.backtest.broker_config.name = "td_ameritrade"
        config.backtest.broker_config.commission_type = "per_share"
        config.backtest.broker_config.commission_value = 0.01
        config.backtest.broker_config.min_commission = 0.5
        config.backtest.broker_config.regulatory_fees = 0.000166
        config.backtest.broker_config.exchange_fees = 0.000119

        strategy_results = {}

        portfolio_results = create_portfolio_backtest_results(results, config, strategy_results)

        # Check broker config
        broker_config = portfolio_results.broker_config
        assert broker_config["name"] == "td_ameritrade"
        assert broker_config["commission_type"] == "per_share"
        assert broker_config["commission_value"] == 0.01
        assert broker_config["min_commission"] == 0.5
        assert broker_config["regulatory_fees"] == 0.000166
        assert broker_config["exchange_fees"] == 0.000119


class TestSaveDetailedReportEdgeCases:
    """Test edge cases for save_detailed_report function."""

    @patch("stockula.main.datetime")
    @patch("stockula.main.Path")
    @patch("builtins.open", create=True)
    @patch("stockula.main.json.dump")
    def test_save_detailed_report_with_broker_tiers(self, mock_json_dump, mock_open, mock_path, mock_datetime):
        """Test saving report with tiered broker pricing."""
        # Mock datetime
        mock_datetime.now.return_value.strftime.return_value = "20240115_120000"

        # Setup mocks
        mock_reports_dir = Mock()
        mock_reports_dir.mkdir = Mock()
        mock_report_file = Mock()
        mock_report_file.__str__ = Mock(return_value="results/reports/strategy_report_KAMA_20240115_120000.json")
        mock_reports_dir.__truediv__ = Mock(return_value=mock_report_file)
        mock_path.return_value = mock_reports_dir

        mock_file = Mock()
        mock_open.return_value.__enter__ = Mock(return_value=mock_file)
        mock_open.return_value.__exit__ = Mock(return_value=None)

        # Test data with complex broker config
        strategy_results = [
            {
                "ticker": "NVDA",
                "strategy": "KAMA",
                "return_pct": 50.0,
                "num_trades": 100,
                "sharpe_ratio": 2.5,
                "win_rate": 75.0,
            }
        ]

        results = {"initial_portfolio_value": 50000.0, "initial_capital": 50000.0}

        config = StockulaConfig()
        config.data.start_date = date(2024, 1, 1)
        config.data.end_date = date(2024, 7, 31)
        config.backtest.broker_config = Mock()
        config.backtest.broker_config.name = "interactive_brokers"
        config.backtest.broker_config.commission_type = "tiered"
        config.backtest.broker_config.commission_value = 0.0035
        config.backtest.broker_config.min_commission = 0.35
        config.backtest.broker_config.regulatory_fees = 0.000166
        config.backtest.broker_config.per_share_commission = None

        # Call function
        report_path = save_detailed_report("KAMA", strategy_results, results, config)

        # Verify report structure
        assert report_path == "results/reports/strategy_report_KAMA_20240115_120000.json"
        assert mock_json_dump.called

        # Check the report data passed to json.dump
        report_data = mock_json_dump.call_args[0][0]
        assert report_data["strategy"] == "KAMA"
        assert report_data["broker_config"]["name"] == "interactive_brokers"
        assert report_data["broker_config"]["commission_type"] == "tiered"
        assert report_data["summary"]["total_trades"] == 100
        assert report_data["summary"]["winning_stocks"] == 1

    @patch("stockula.main.datetime")
    @patch("stockula.main.Path")
    @patch("builtins.open", create=True)
    @patch("stockula.main.json.dump")
    def test_save_detailed_report_with_per_share_commission(self, mock_json_dump, mock_open, mock_path, mock_datetime):
        """Test saving report with per-share commission from per_share_commission field."""
        # Mock datetime
        mock_datetime.now.return_value.strftime.return_value = "20240115_120000"

        # Setup mocks
        mock_reports_dir = Mock()
        mock_reports_dir.mkdir = Mock()
        mock_report_file = Mock()
        mock_report_file.__str__ = Mock(return_value="results/reports/strategy_report_TEST_20240115_120000.json")
        mock_reports_dir.__truediv__ = Mock(return_value=mock_report_file)
        mock_path.return_value = mock_reports_dir

        mock_file = Mock()
        mock_open.return_value.__enter__ = Mock(return_value=mock_file)
        mock_open.return_value.__exit__ = Mock(return_value=None)

        # Test data
        strategy_results = []
        results = {"initial_portfolio_value": 10000.0, "initial_capital": 10000.0}

        config = StockulaConfig()
        config.backtest.broker_config = Mock()
        config.backtest.broker_config.name = "custom_broker"
        config.backtest.broker_config.commission_type = "per_share"
        config.backtest.broker_config.commission_value = 0.005  # per_share uses commission_value
        config.backtest.broker_config.per_share_commission = 0.005  # Also set this
        config.backtest.broker_config.min_commission = 1.0
        config.backtest.broker_config.regulatory_fees = 0.0

        # Call function
        save_detailed_report("TEST", strategy_results, results, config)

        # Check the report data
        report_data = mock_json_dump.call_args[0][0]
        assert report_data["broker_config"]["commission_value"] == 0.005


class TestMainAdditionalEdgeCases:
    """Additional edge cases for main function."""

    @patch("stockula.main.log_manager")
    @patch("stockula.main.create_container")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.run_technical_analysis")
    @patch("stockula.main.print_results")
    @patch(
        "sys.argv",
        [
            "stockula",
            "--ticker",
            "AAPL",
            "--train-start",
            "2023-01-01",
            "--train-end",
            "2023-12-31",
            "--test-start",
            "2024-01-01",
            "--test-end",
            "2024-01-31",
        ],
    )
    def test_main_with_date_overrides(self, mock_print, mock_ta, mock_logging, mock_container, mock_log_manager):
        """Test main function with command line date overrides."""
        # Setup config
        config = StockulaConfig()

        # Setup container
        container = Mock()
        container.stockula_config.return_value = config
        container.logging_manager.return_value = Mock()
        mock_container.return_value = container

        # Setup minimal mocks
        mock_portfolio = Mock()
        mock_portfolio.name = "Test"
        mock_portfolio.initial_capital = 100000.0
        mock_portfolio.get_all_assets.return_value = []
        mock_portfolio.get_portfolio_value.return_value = 100000
        mock_portfolio.allocation_method = "equal"

        mock_factory = Mock()
        mock_factory.create_portfolio.return_value = mock_portfolio
        container.domain_factory.return_value = mock_factory

        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.return_value = {"AAPL": 150.0}
        container.data_fetcher.return_value = mock_fetcher
        container.backtest_runner.return_value = Mock()
        container.stock_forecaster.return_value = Mock()

        # Mock TA
        mock_ta.return_value = {"ticker": "AAPL", "indicators": {}}

        # Import and run main
        from stockula.main import main

        main()

        # Check that date overrides were applied
        assert config.forecast.train_start_date == date(2023, 1, 1)
        assert config.forecast.train_end_date == date(2023, 12, 31)
        assert config.forecast.test_start_date == date(2024, 1, 1)
        assert config.forecast.test_end_date == date(2024, 1, 31)

    @patch("stockula.main.create_container")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.print_results")
    @patch("stockula.main.Path")
    @patch("builtins.open", create=True)
    @patch("stockula.main.json.dump")
    @patch("sys.argv", ["stockula", "--output", "console"])
    def test_main_saves_results_when_configured(
        self, mock_json_dump, mock_open, mock_path, mock_print, mock_logging, mock_container
    ):
        """Test main saves results when save_results is enabled."""
        # Setup config with save_results enabled
        config = StockulaConfig()
        config.output["save_results"] = True
        config.output["results_dir"] = "./test_results"

        # Setup container
        container = Mock()
        container.stockula_config.return_value = config
        container.logging_manager.return_value = Mock()
        mock_container.return_value = container

        # Setup log manager
        import stockula.main

        stockula.main.log_manager = Mock()

        # Setup path mocks
        mock_results_dir = Mock()
        mock_results_dir.mkdir = Mock()
        mock_results_file = Mock()
        mock_results_file.__str__ = Mock(return_value="test_results/stockula_results_20240115_120000.json")
        mock_results_dir.__truediv__ = Mock(return_value=mock_results_file)
        mock_path.return_value = mock_results_dir

        # Setup file mock
        mock_file = Mock()
        mock_open.return_value.__enter__ = Mock(return_value=mock_file)
        mock_open.return_value.__exit__ = Mock(return_value=None)

        # Setup minimal mocks
        mock_portfolio = Mock()
        mock_portfolio.name = "Test"
        mock_portfolio.initial_capital = 100000.0
        mock_portfolio.get_all_assets.return_value = []
        mock_portfolio.get_portfolio_value.return_value = 100000
        mock_portfolio.allocation_method = "equal"

        mock_factory = Mock()
        mock_factory.create_portfolio.return_value = mock_portfolio
        container.domain_factory.return_value = mock_factory

        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.return_value = {}
        container.data_fetcher.return_value = mock_fetcher
        container.backtest_runner.return_value = Mock()
        container.stock_forecaster.return_value = Mock()

        # Import and run main
        from stockula.main import main

        main()

        # Verify results were saved
        mock_results_dir.mkdir.assert_called_with(exist_ok=True)
        mock_json_dump.assert_called_once()
        stockula.main.log_manager.info.assert_called()
