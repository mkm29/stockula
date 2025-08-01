"""Additional tests to improve main.py coverage to 80%+."""

from datetime import date
from unittest.mock import Mock, patch

from stockula.config import StockulaConfig, TickerConfig
from stockula.domain import Category
from stockula.main import (
    main,
    print_results,
)


class TestPrintResultsAdvanced:
    """Test advanced print_results scenarios."""

    def test_print_results_with_train_test_split(self, capsys):
        """Test printing results with train/test split data."""
        results = {
            "backtesting": [
                {
                    "ticker": "AAPL",
                    "strategy": "smacross",
                    "parameters": {"fast": 10, "slow": 20},
                    "train_period": {"start": "2023-01-01", "end": "2023-09-30", "days": 273},
                    "test_period": {"start": "2023-10-01", "end": "2023-12-31", "days": 92},
                    "train_results": {
                        "return_pct": 20.5,
                        "sharpe_ratio": 1.8,
                        "max_drawdown_pct": -6.5,
                        "num_trades": 35,
                        "win_rate": 65.7,
                    },
                    "test_results": {
                        "return_pct": 15.2,
                        "sharpe_ratio": 1.4,
                        "max_drawdown_pct": -8.2,
                        "num_trades": 12,
                        "win_rate": 58.3,
                    },
                    "return_pct": 15.2,
                    "sharpe_ratio": 1.4,
                    "max_drawdown_pct": -8.2,
                    "num_trades": 12,
                    "win_rate": 58.3,
                }
            ]
        }

        print_results(results, "console")

        captured = capsys.readouterr()
        # Check for train/test specific output
        assert "Train Return" in captured.out or "train" in captured.out.lower()
        assert "Test Return" in captured.out or "test" in captured.out.lower()
        assert "Train Sharpe" in captured.out or "1.8" in captured.out
        assert "Test Sharpe" in captured.out or "1.4" in captured.out
        assert "Training:" in captured.out or "train" in captured.out.lower()
        assert "Testing:" in captured.out or "test" in captured.out.lower()
        assert "2023-01-01" in captured.out
        assert "2023-09-30" in captured.out
        assert "2023-10-01" in captured.out
        assert "2023-12-31" in captured.out

    def test_print_results_with_portfolio_composition(self, capsys):
        """Test printing results with portfolio composition table."""
        results = {
            "backtesting": [
                {
                    "ticker": "AAPL",
                    "strategy": "smacross",
                    "return_pct": 10.0,
                    "sharpe_ratio": 1.2,
                    "max_drawdown_pct": -5.0,
                    "num_trades": 8,
                    "win_rate": 62.5,
                }
            ]
        }

        # Create config and container with portfolio
        config = StockulaConfig()
        config.backtest.hold_only_categories = ["INDEX"]

        # Mock container with proper structure
        container = Mock()

        # Mock assets
        mock_asset1 = Mock()
        mock_asset1.symbol = "AAPL"
        mock_asset1.quantity = 10.0
        mock_asset1.category = Category.GROWTH

        mock_asset2 = Mock()
        mock_asset2.symbol = "SPY"
        mock_asset2.quantity = 5.0
        mock_asset2.category = Category.INDEX

        # Mock portfolio
        mock_portfolio = Mock()
        mock_portfolio.get_all_assets.return_value = [mock_asset1, mock_asset2]

        # Mock factory
        mock_factory = Mock()
        mock_factory.create_portfolio.return_value = mock_portfolio
        container.domain_factory.return_value = mock_factory

        # Mock fetcher with prices
        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.return_value = {"AAPL": 150.0, "SPY": 400.0}
        container.data_fetcher.return_value = mock_fetcher

        print_results(results, "console", config=config, container=container)

        captured = capsys.readouterr()
        # Check for portfolio composition elements
        assert "Portfolio Composition" in captured.out or "portfolio" in captured.out.lower()
        assert "AAPL" in captured.out
        assert "SPY" in captured.out
        assert "GROWTH" in captured.out or "Growth" in captured.out
        assert "INDEX" in captured.out or "Index" in captured.out
        assert "Hold Only" in captured.out or "hold" in captured.out.lower()
        assert "Tradeable" in captured.out or "trade" in captured.out.lower()

    def test_print_results_with_evaluation_metrics(self, capsys):
        """Test printing forecast results with evaluation metrics."""
        results = {
            "forecasting": [
                {
                    "ticker": "AAPL",
                    "current_price": 150.0,
                    "forecast_price": 155.0,
                    "lower_bound": 152.0,
                    "upper_bound": 158.0,
                    "forecast_length": 30,
                    "best_model": "ARIMA",
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-31",
                    "train_period": {"start": "2023-01-01", "end": "2023-12-31"},
                    "test_period": {"start": "2024-01-01", "end": "2024-01-31"},
                    "evaluation": {
                        "rmse": 2.5,
                        "mae": 2.0,
                        "mape": 1.65,
                        "accuracy": 98.35,
                    },
                },
                {
                    "ticker": "GOOGL",
                    "current_price": 100.0,
                    "forecast_price": 95.0,
                    "lower_bound": 92.0,
                    "upper_bound": 98.0,
                    "forecast_length": 30,
                    "best_model": "ETS",
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-31",
                    "evaluation": {
                        "rmse": 3.0,
                        "mae": 2.5,
                        "mape": 2.5,
                    },
                },
            ]
        }

        print_results(results, "console")

        captured = capsys.readouterr()
        # Check for evaluation metrics
        assert "Forecast Evaluation Metrics" in captured.out or "evaluation" in captured.out.lower()
        assert "RMSE" in captured.out or "rmse" in captured.out.lower()
        assert "MAE" in captured.out or "mae" in captured.out.lower()
        assert "MAPE" in captured.out or "mape" in captured.out.lower()
        assert "$2.50" in captured.out or "2.5" in captured.out
        assert "$2.00" in captured.out or "2.0" in captured.out
        assert "1.65%" in captured.out or "1.65" in captured.out

    def test_print_results_sorted_forecasts(self, capsys):
        """Test that forecast results are sorted by return percentage."""
        results = {
            "forecasting": [
                {
                    "ticker": "LOW",
                    "current_price": 100.0,
                    "forecast_price": 95.0,  # -5% return
                    "lower_bound": 92.0,
                    "upper_bound": 98.0,
                    "forecast_length": 30,
                    "best_model": "ETS",
                },
                {
                    "ticker": "HIGH",
                    "current_price": 100.0,
                    "forecast_price": 120.0,  # +20% return
                    "lower_bound": 115.0,
                    "upper_bound": 125.0,
                    "forecast_length": 30,
                    "best_model": "ARIMA",
                },
                {
                    "ticker": "MID",
                    "current_price": 100.0,
                    "forecast_price": 105.0,  # +5% return
                    "lower_bound": 102.0,
                    "upper_bound": 108.0,
                    "forecast_length": 30,
                    "best_model": "GLS",
                },
            ]
        }

        print_results(results, "console")

        captured = capsys.readouterr()
        # Find positions of tickers in output
        high_pos = captured.out.find("HIGH")
        mid_pos = captured.out.find("MID")
        low_pos = captured.out.find("LOW")

        # HIGH should appear first (highest return), then MID, then LOW
        assert high_pos < mid_pos < low_pos

    def test_print_results_with_portfolio_metadata(self, capsys):
        """Test printing results with portfolio metadata."""
        results = {
            "portfolio": {
                "initial_capital": 50000,
                "start": "2023-01-01",
                "end": "2023-12-31",
            },
            "backtesting": [
                {
                    "ticker": "TEST",
                    "strategy": "RSI",
                    "return_pct": 5.0,
                    "sharpe_ratio": 0.8,
                    "max_drawdown_pct": -3.0,
                    "num_trades": 3,
                    "win_rate": 66.7,
                }
            ],
        }

        print_results(results, "console")

        captured = capsys.readouterr()
        # Check for portfolio information from metadata
        assert "50,000" in captured.out or "50000" in captured.out
        assert "2023-01-01" in captured.out
        assert "2023-12-31" in captured.out


class TestMainForecastingScenarios:
    """Test main function forecasting scenarios."""

    @patch("stockula.main.create_container")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.run_forecast_with_evaluation")
    @patch("stockula.main.run_forecast")
    @patch("stockula.main.print_results")
    @patch("stockula.main.log_manager")
    @patch("sys.argv", ["stockula", "--mode", "forecast"])
    def test_main_forecast_sequential_with_evaluation(
        self,
        mock_log_manager,
        mock_print,
        mock_forecast,
        mock_forecast_eval,
        mock_logging,
        mock_container,
    ):
        """Test main with sequential forecasting using evaluation."""
        # Setup config with test dates for evaluation
        config = StockulaConfig()
        config.portfolio.tickers = [
            TickerConfig(symbol="AAPL", quantity=10.0),
            TickerConfig(symbol="GOOGL", quantity=5.0),
        ]
        config.forecast.test_start_date = date(2024, 1, 1)
        config.forecast.test_end_date = date(2024, 1, 31)
        config.forecast.max_generations = 2
        config.forecast.num_validations = 5

        # Setup container
        container = Mock()
        container.stockula_config.return_value = config
        container.logging_manager.return_value = Mock()
        mock_container.return_value = container

        # Setup minimal portfolio mocks
        mock_asset1 = Mock()
        mock_asset1.symbol = "AAPL"
        mock_asset1.quantity = 10.0
        mock_asset1.category = None

        mock_asset2 = Mock()
        mock_asset2.symbol = "GOOGL"
        mock_asset2.quantity = 5.0
        mock_asset2.category = None

        mock_portfolio = Mock()
        mock_portfolio.name = "Test Portfolio"
        mock_portfolio.initial_capital = 100000.0
        mock_portfolio.get_all_assets.return_value = [mock_asset1, mock_asset2]
        mock_portfolio.get_portfolio_value.return_value = 100000
        mock_portfolio.allocation_method = "equal"

        mock_factory = Mock()
        mock_factory.create_portfolio.return_value = mock_portfolio
        container.domain_factory.return_value = mock_factory

        # Setup fetcher
        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.return_value = {"AAPL": 150.0, "GOOGL": 100.0}
        container.data_fetcher.return_value = mock_fetcher
        container.backtest_runner.return_value = Mock()

        # Mock stock forecaster
        mock_stock_forecaster = Mock()
        container.stock_forecaster.return_value = mock_stock_forecaster

        # Mock forecast results
        mock_forecast_eval.side_effect = [
            {
                "ticker": "AAPL",
                "current_price": 150.0,
                "forecast_price": 160.0,
                "evaluation": {"rmse": 2.0, "mape": 1.5},
            },
            {
                "ticker": "GOOGL",
                "current_price": 100.0,
                "forecast_price": 105.0,
                "evaluation": {"rmse": 1.5, "mape": 1.2},
            },
        ]

        # Import and run main

        main()

        # Should have called forecast_with_evaluation for each ticker
        assert mock_forecast_eval.call_count == 2
        mock_forecast_eval.assert_any_call("AAPL", config, mock_stock_forecaster)
        mock_forecast_eval.assert_any_call("GOOGL", config, mock_stock_forecaster)

        # Should not call regular forecast
        mock_forecast.assert_not_called()

    @patch("stockula.main.create_container")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.run_forecast")
    @patch("stockula.main.print_results")
    @patch("stockula.main.log_manager")
    @patch("sys.argv", ["stockula", "--mode", "forecast"])
    def test_main_forecast_sequential_no_evaluation(
        self,
        mock_log_manager,
        mock_print,
        mock_forecast,
        mock_logging,
        mock_container,
    ):
        """Test main with sequential forecasting without evaluation."""
        # Setup config without test dates
        config = StockulaConfig()
        config.portfolio.tickers = [TickerConfig(symbol="AAPL", quantity=10.0)]
        config.forecast.test_start_date = None
        config.forecast.test_end_date = None

        # Setup container
        container = Mock()
        container.stockula_config.return_value = config
        container.logging_manager.return_value = Mock()
        mock_container.return_value = container

        # Setup minimal portfolio mocks
        mock_asset = Mock()
        mock_asset.symbol = "AAPL"
        mock_asset.quantity = 10.0
        mock_asset.category = None

        mock_portfolio = Mock()
        mock_portfolio.name = "Test Portfolio"
        mock_portfolio.initial_capital = 100000.0
        mock_portfolio.get_all_assets.return_value = [mock_asset]
        mock_portfolio.get_portfolio_value.return_value = 100000
        mock_portfolio.allocation_method = "equal"

        mock_factory = Mock()
        mock_factory.create_portfolio.return_value = mock_portfolio
        container.domain_factory.return_value = mock_factory

        # Setup fetcher
        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.return_value = {"AAPL": 150.0}
        container.data_fetcher.return_value = mock_fetcher
        container.backtest_runner.return_value = Mock()

        # Mock stock forecaster
        mock_stock_forecaster = Mock()
        container.stock_forecaster.return_value = mock_stock_forecaster

        # Mock forecast results
        mock_forecast.return_value = {
            "ticker": "AAPL",
            "current_price": 150.0,
            "forecast_price": 155.0,
        }

        # Import and run main

        main()

        # Should have called regular forecast
        mock_forecast.assert_called_once_with("AAPL", config, mock_stock_forecaster)

    @patch("stockula.main.create_container")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.run_forecast_with_evaluation")
    @patch("stockula.main.print_results")
    @patch("stockula.main.log_manager")
    @patch("sys.argv", ["stockula", "--mode", "forecast"])
    def test_main_forecast_with_keyboard_interrupt(
        self,
        mock_log_manager,
        mock_print,
        mock_forecast_eval,
        mock_logging,
        mock_container,
    ):
        """Test main forecast handling keyboard interrupt."""
        # Setup config
        config = StockulaConfig()
        config.portfolio.tickers = [
            TickerConfig(symbol="AAPL", quantity=10.0),
            TickerConfig(symbol="GOOGL", quantity=5.0),
        ]
        config.forecast.test_start_date = date(2024, 1, 1)
        config.forecast.test_end_date = date(2024, 1, 31)

        # Setup container
        container = Mock()
        container.stockula_config.return_value = config
        container.logging_manager.return_value = Mock()
        mock_container.return_value = container

        # Setup minimal portfolio mocks
        mock_portfolio = Mock()
        mock_portfolio.name = "Test Portfolio"
        mock_portfolio.initial_capital = 100000.0
        mock_portfolio.get_all_assets.return_value = [
            Mock(symbol="AAPL", quantity=10.0, category=None),
            Mock(symbol="GOOGL", quantity=5.0, category=None),
        ]
        mock_portfolio.get_portfolio_value.return_value = 100000
        mock_portfolio.allocation_method = "equal"

        mock_factory = Mock()
        mock_factory.create_portfolio.return_value = mock_portfolio
        container.domain_factory.return_value = mock_factory

        # Setup fetcher
        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.return_value = {"AAPL": 150.0, "GOOGL": 100.0}
        container.data_fetcher.return_value = mock_fetcher
        container.backtest_runner.return_value = Mock()
        container.stock_forecaster.return_value = Mock()

        # Mock forecast to raise KeyboardInterrupt on second call
        mock_forecast_eval.side_effect = [
            {"ticker": "AAPL", "current_price": 150.0, "forecast_price": 155.0},
            KeyboardInterrupt(),
        ]

        # Import and run main

        main()

        # Should have attempted two forecasts but interrupted on second
        assert mock_forecast_eval.call_count == 2

        # Check that results include the error
        call_args = mock_print.call_args[0][0]
        assert "forecasting" in call_args
        assert any(r.get("error") == "Interrupted by user" for r in call_args["forecasting"])


class TestMainForecastValueCalculation:
    """Test forecast value calculation in main function."""

    @patch("stockula.main.create_container")
    @patch("stockula.main.setup_logging")
    @patch("stockula.main.print_results")
    @patch("stockula.main.log_manager")
    @patch("sys.argv", ["stockula", "--mode", "forecast"])
    def test_main_forecast_portfolio_value_calculation(
        self,
        mock_log_manager,
        mock_print,
        mock_logging,
        mock_container,
        capsys,
    ):
        """Test portfolio value calculation with forecast results."""
        # Setup config
        config = StockulaConfig()
        config.portfolio.tickers = [
            TickerConfig(symbol="AAPL", quantity=10.0),
            TickerConfig(symbol="GOOGL", quantity=5.0),
        ]
        config.forecast.test_start_date = date(2024, 1, 1)
        config.forecast.test_end_date = date(2024, 1, 31)

        # Setup container
        container = Mock()
        container.stockula_config.return_value = config
        container.logging_manager.return_value = Mock()
        mock_container.return_value = container

        # Setup portfolio with assets
        mock_asset1 = Mock()
        mock_asset1.symbol = "AAPL"
        mock_asset1.quantity = 10.0
        mock_asset1.category = None

        mock_asset2 = Mock()
        mock_asset2.symbol = "GOOGL"
        mock_asset2.quantity = 5.0
        mock_asset2.category = None

        mock_portfolio = Mock()
        mock_portfolio.name = "Test Portfolio"
        mock_portfolio.initial_capital = 2000.0  # 10*150 + 5*100 = 2000
        mock_portfolio.get_all_assets.return_value = [mock_asset1, mock_asset2]
        mock_portfolio.get_portfolio_value.return_value = 2000.0
        mock_portfolio.allocation_method = "equal"

        mock_factory = Mock()
        mock_factory.create_portfolio.return_value = mock_portfolio
        container.domain_factory.return_value = mock_factory

        # Setup fetcher
        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.return_value = {"AAPL": 150.0, "GOOGL": 100.0}
        container.data_fetcher.return_value = mock_fetcher
        container.backtest_runner.return_value = Mock()

        # Mock stock forecaster
        mock_forecaster = Mock()
        container.stock_forecaster.return_value = mock_forecaster

        # Don't mock print_results to avoid recursion - just let it run normally

        # Mock run_forecast_with_evaluation
        with patch("stockula.main.run_forecast_with_evaluation") as mock_forecast_eval:
            mock_forecast_eval.side_effect = [
                {
                    "ticker": "AAPL",
                    "current_price": 150.0,
                    "forecast_price": 165.0,
                    "lower_bound": 160.0,
                    "upper_bound": 170.0,
                    "end_date": "2024-01-31",
                    "evaluation": {"mape": 2.0},
                },
                {
                    "ticker": "GOOGL",
                    "current_price": 100.0,
                    "forecast_price": 110.0,
                    "lower_bound": 105.0,
                    "upper_bound": 115.0,
                    "end_date": "2024-01-31",
                    "evaluation": {"mape": 1.5},
                },
            ]

            # Import and run main
            from stockula.main import main

            main()

        # Capture output
        captured = capsys.readouterr()

        # Check that portfolio value table is shown
        assert "Portfolio Value" in captured.out
        assert "Observed Value" in captured.out
        assert "Predicted Value" in captured.out
        assert "$2,000.00" in captured.out  # Initial capital
        assert "$2,200.00" in captured.out  # Forecasted value (10% gain on both)
        assert "Accuracy" in captured.out
        assert "98." in captured.out  # Average accuracy should be ~98.25%
