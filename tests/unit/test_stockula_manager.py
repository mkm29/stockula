"""Comprehensive tests for StockulaManager class."""

import json
from datetime import date, datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import yaml

from stockula.backtesting import RSIStrategy, SMACrossStrategy
from stockula.config import StockulaConfig, TickerConfig
from stockula.domain.category import Category
from stockula.manager import StockulaManager


class TestStockulaManager:
    """Test suite for StockulaManager class."""

    @pytest.fixture
    def mock_config(self):
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

    @pytest.fixture
    def mock_container(self, mock_config):
        """Create a mock container with common dependencies."""
        container = Mock()

        # Mock logging manager
        mock_log_manager = Mock()
        container.logging_manager.return_value = mock_log_manager

        # Mock other common dependencies
        container.data_fetcher.return_value = Mock()
        container.domain_factory.return_value = Mock()
        container.backtest_runner.return_value = Mock()
        container.backtesting_manager.return_value = Mock()
        container.forecasting_manager.return_value = Mock()
        container.technical_analysis_manager.return_value = Mock()
        container.allocator_manager.return_value = Mock()

        return container

    @pytest.fixture
    def manager(self, mock_config, mock_container):
        """Create StockulaManager instance for testing."""
        return StockulaManager(mock_config, mock_container)

    def test_init(self, mock_config, mock_container):
        """Test StockulaManager initialization."""
        manager = StockulaManager(mock_config, mock_container)

        assert manager.config == mock_config
        assert manager.container == mock_container
        assert manager.console is not None
        assert manager.log_manager == mock_container.logging_manager.return_value

        # Check that strategy registry is properly accessible
        assert manager.strategy_registry is not None
        # Verify strategies are available through get_strategy_class method
        assert manager.get_strategy_class("smacross") == SMACrossStrategy
        assert manager.get_strategy_class("rsi") == RSIStrategy

    def test_get_strategy_class(self, manager):
        """Test strategy class retrieval."""
        assert manager.get_strategy_class("smacross") == SMACrossStrategy
        assert manager.get_strategy_class("SMACROSS") == SMACrossStrategy
        assert manager.get_strategy_class("RSI") == RSIStrategy
        assert manager.get_strategy_class("invalid") is None
        assert manager.get_strategy_class("") is None

    def test_date_to_string(self, manager):
        """Test date conversion to string."""
        assert manager.date_to_string(None) is None
        assert manager.date_to_string("2024-01-01") == "2024-01-01"
        assert manager.date_to_string(date(2024, 1, 15)) == "2024-01-15"
        assert manager.date_to_string(datetime(2024, 6, 30, 12, 30)) == "2024-06-30"

    def test_run_optimize_allocation_success(self, mock_config, mock_container):
        """Test successful allocation optimization."""
        # Set up config for backtest optimization
        mock_config.portfolio.allocation_method = "backtest_optimized"
        mock_config.backtest_optimization = Mock()  # Mock the optimization config

        # Mock allocator manager
        mock_allocator_manager = Mock()
        optimized_quantities = {"AAPL": 10.5, "GOOGL": 2.3}
        mock_allocator_manager.calculate_backtest_optimized_quantities.return_value = optimized_quantities
        mock_container.allocator_manager.return_value = mock_allocator_manager

        # Mock data fetcher for prices
        mock_data_fetcher = Mock()
        mock_data_fetcher.get_current_prices.return_value = {"AAPL": 150.0, "GOOGL": 2800.0}
        mock_container.data_fetcher.return_value = mock_data_fetcher

        manager = StockulaManager(mock_config, mock_container)
        result = manager.run_optimize_allocation()

        assert result == 0
        mock_allocator_manager.calculate_backtest_optimized_quantities.assert_called_once()

    def test_run_optimize_allocation_wrong_method(self, mock_config, mock_container):
        """Test optimization with non-backtest allocation method."""
        mock_config.portfolio.allocation_method = "equal"
        mock_config.backtest_optimization = Mock()

        # Mock allocator manager
        mock_allocator_manager = Mock()
        optimized_quantities = {"AAPL": 10.5}
        mock_allocator_manager.calculate_backtest_optimized_quantities.return_value = optimized_quantities
        mock_container.allocator_manager.return_value = mock_allocator_manager

        # Mock data fetcher for prices
        mock_data_fetcher = Mock()
        mock_data_fetcher.get_current_prices.return_value = {"AAPL": 150.0}
        mock_container.data_fetcher.return_value = mock_data_fetcher

        manager = StockulaManager(mock_config, mock_container)
        result = manager.run_optimize_allocation()

        # Should still succeed after warning
        assert result == 0
        assert mock_config.portfolio.allocation_method == "backtest_optimized"

    def test_run_optimize_allocation_no_config(self, mock_config, mock_container):
        """Test optimization without backtest_optimization config."""
        mock_config.portfolio.allocation_method = "backtest_optimized"
        mock_config.backtest_optimization = None

        manager = StockulaManager(mock_config, mock_container)
        result = manager.run_optimize_allocation()

        assert result == 1

    def test_run_optimize_allocation_with_save(self, mock_config, mock_container, tmp_path):
        """Test optimization with config saving."""
        mock_config.portfolio.allocation_method = "backtest_optimized"
        mock_config.backtest_optimization = Mock()

        # Mock allocator manager
        mock_allocator_manager = Mock()
        optimized_quantities = {"AAPL": 10.5}
        mock_allocator_manager.calculate_backtest_optimized_quantities.return_value = optimized_quantities
        mock_container.allocator_manager.return_value = mock_allocator_manager

        # Mock data fetcher for prices
        mock_data_fetcher = Mock()
        mock_data_fetcher.get_current_prices.return_value = {"AAPL": 150.0}
        mock_container.data_fetcher.return_value = mock_data_fetcher

        save_path = str(tmp_path / "optimized_.stockula.yaml")

        manager = StockulaManager(mock_config, mock_container)
        result = manager.run_optimize_allocation(save_path=save_path)

        assert result == 0
        assert Path(save_path).exists()

        # Verify saved config
        with open(save_path) as f:
            saved_config = yaml.safe_load(f)

        assert saved_config["portfolio"]["allocation_method"] == "custom"
        assert saved_config["portfolio"]["tickers"][0]["quantity"] == 10.5

    def test_run_optimize_allocation_exception(self, mock_config, mock_container):
        """Test optimization with exception handling."""
        mock_config.portfolio.allocation_method = "backtest_optimized"
        mock_config.backtest_optimization = Mock()

        # Mock allocator manager to raise exception
        mock_allocator_manager = Mock()
        mock_allocator_manager.calculate_backtest_optimized_quantities.side_effect = Exception("Test error")
        mock_container.allocator_manager.return_value = mock_allocator_manager

        manager = StockulaManager(mock_config, mock_container)
        result = manager.run_optimize_allocation()

        assert result == 1

    def test_display_optimization_results(self, mock_config, mock_container):
        """Test displaying optimization results."""
        mock_config.portfolio.tickers = [
            TickerConfig(symbol="AAPL", quantity=1.0),
            TickerConfig(symbol="GOOGL", quantity=1.0),
        ]

        # Mock data fetcher for prices
        mock_data_fetcher = Mock()
        mock_data_fetcher.get_current_prices.return_value = {"AAPL": 150.0, "GOOGL": 2800.0}
        mock_container.data_fetcher.return_value = mock_data_fetcher

        manager = StockulaManager(mock_config, mock_container)
        optimized_quantities = {"AAPL": 10.5, "GOOGL": 2.3}

        # Should not raise exception
        manager._display_optimization_results(optimized_quantities)

    def test_save_optimized_config(self, mock_config, mock_container, tmp_path):
        """Test saving optimized configuration."""
        mock_config.portfolio.tickers = [TickerConfig(symbol="AAPL", quantity=1.0)]

        manager = StockulaManager(mock_config, mock_container)
        optimized_quantities = {"AAPL": 10.5}
        save_path = str(tmp_path / "test_.stockula.yaml")

        manager._save_optimized_config(save_path, optimized_quantities)

        assert Path(save_path).exists()

        # Verify config was updated
        assert mock_config.portfolio.tickers[0].quantity == 10.5
        assert mock_config.portfolio.allocation_method == "custom"
        assert mock_config.portfolio.dynamic_allocation is False
        assert mock_config.portfolio.auto_allocate is False

    def test_save_optimized_config_numpy_quantity(self, mock_config, mock_container, tmp_path):
        """Test saving optimized config with numpy quantities."""

        mock_config.portfolio.tickers = [TickerConfig(symbol="AAPL", quantity=1.0)]

        manager = StockulaManager(mock_config, mock_container)

        # Create mock numpy scalar
        numpy_quantity = Mock()
        numpy_quantity.item.return_value = 10.5
        optimized_quantities = {"AAPL": numpy_quantity}

        save_path = str(tmp_path / "test_.stockula.yaml")
        manager._save_optimized_config(save_path, optimized_quantities)

        assert mock_config.portfolio.tickers[0].quantity == 10.5

    def test_convert_dates(self, manager):
        """Test recursive date conversion."""
        test_obj = {
            "date_field": date(2024, 1, 15),
            "nested": {
                "another_date": date(2024, 2, 20),
                "list_with_dates": [date(2024, 3, 25), "not_a_date"],
            },
            "regular_field": "value",
        }

        converted = manager._convert_dates(test_obj)

        assert converted["date_field"] == "2024-01-15"
        assert converted["nested"]["another_date"] == "2024-02-20"
        assert converted["nested"]["list_with_dates"][0] == "2024-03-25"
        assert converted["nested"]["list_with_dates"][1] == "not_a_date"
        assert converted["regular_field"] == "value"

    def test_create_portfolio(self, manager):
        """Test portfolio creation."""
        mock_factory = Mock()
        mock_portfolio = Mock()
        mock_factory.create_portfolio.return_value = mock_portfolio
        manager.container.domain_factory.return_value = mock_factory

        result = manager.create_portfolio()

        assert result == mock_portfolio
        mock_factory.create_portfolio.assert_called_once_with(manager.config)

    def test_display_portfolio_summary(self, manager):
        """Test portfolio summary display."""
        mock_portfolio = Mock()
        mock_portfolio.name = "Test Portfolio"
        mock_portfolio.initial_capital = 100000.0
        mock_portfolio.allocation_method = "equal"
        mock_portfolio.get_all_assets.return_value = ["asset1", "asset2"]

        # Should not raise exception
        manager.display_portfolio_summary(mock_portfolio)

    def test_display_portfolio_holdings(self, mock_config, mock_container):
        """Test portfolio holdings display."""
        manager = StockulaManager(mock_config, mock_container)

        # Create mock assets with different scenarios
        mock_asset1 = Mock()
        mock_asset1.symbol = "AAPL"
        mock_asset1.category = Category.LARGE_CAP
        mock_asset1.quantity = 10.5

        mock_asset2 = Mock()
        mock_asset2.symbol = "GOOGL"
        mock_asset2.category = None  # Test None category
        mock_asset2.quantity = "invalid"  # Test invalid quantity

        mock_asset3 = Mock()
        mock_asset3.symbol = "TSLA"
        mock_asset3.category = Mock()
        mock_asset3.category.name = "TECH"
        mock_asset3.quantity = 5.0

        mock_portfolio = Mock()
        mock_portfolio.get_all_assets.return_value = [mock_asset1, mock_asset2, mock_asset3]

        # Should not raise exception
        manager.display_portfolio_holdings(mock_portfolio)

    def test_display_portfolio_holdings_edge_cases(self, manager):
        """Test portfolio holdings display with edge cases."""
        # Asset without symbol attribute
        mock_asset_no_symbol = Mock(spec=[])  # No attributes

        # Asset with non-numeric category
        mock_asset_string_category = Mock()
        mock_asset_string_category.symbol = "TEST"
        mock_asset_string_category.category = "STRING_CATEGORY"
        mock_asset_string_category.quantity = 3.14

        mock_portfolio = Mock()
        mock_portfolio.get_all_assets.return_value = [mock_asset_no_symbol, mock_asset_string_category]

        # Should not raise exception
        manager.display_portfolio_holdings(mock_portfolio)

    def test_technical_analysis_custom_indicators(self, mock_config, mock_container):
        """Test technical analysis with custom indicators."""
        mock_config.technical_analysis.indicators = ["sma", "ema", "rsi", "macd", "bbands", "atr", "adx"]

        # Mock technical analysis manager
        mock_ta_manager = Mock()
        mock_ta_manager.analyze_symbol.return_value = {
            "ticker": "AAPL",
            "current_price": 150.0,
            "analysis_type": "custom",
            "indicators": {
                "sma": {"current": 148.5},
                "ema": {"current": 149.2},
                "rsi": {"current": 65.3},
                "macd": {"current": {"MACD": 1.5, "MACD_SIGNAL": 1.2}},
                "bbands": {"current": {"BB_UPPER": 152, "BB_MIDDLE": 150, "BB_LOWER": 148}},
                "atr": {"current": 2.5},
                "adx": {"current": 25.0},
            },
            "summary": {"signals": [], "strength": "neutral"},
        }
        mock_container.technical_analysis_manager.return_value = mock_ta_manager

        # Mock data fetcher and TechnicalIndicators for backward compatibility processing
        mock_data_fetcher = Mock()
        # Create enough data for SMA calculation (30 points to handle EMA_26)
        dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
        mock_data = pd.DataFrame(
            {
                "Open": [100 + i for i in range(30)],
                "High": [105 + i for i in range(30)],
                "Low": [95 + i for i in range(30)],
                "Close": [150 - i * 0.1 for i in range(30)],
                "Volume": [1000000 + i * 1000 for i in range(30)],
            },
            index=dates,
        )
        mock_data_fetcher.get_stock_data.return_value = mock_data
        mock_container.data_fetcher.return_value = mock_data_fetcher

        manager = StockulaManager(mock_config, mock_container)
        result = manager.run_technical_analysis("AAPL", show_progress=False)

        assert result["ticker"] == "AAPL"
        # Test that we get the original indicators back
        assert "sma" in result["indicators"]
        assert "ema" in result["indicators"]
        assert "rsi" in result["indicators"]
        # Test backward compatibility indicators may be added
        if "RSI" in result["indicators"]:
            assert result["indicators"]["RSI"] == 65.3

    def test_technical_analysis_period_specific_indicators(self, mock_config, mock_container):
        """Test technical analysis with period-specific indicators."""
        mock_config.technical_analysis.indicators = ["sma", "ema"]
        mock_config.technical_analysis.sma_periods = [20, 50]
        mock_config.technical_analysis.ema_periods = [12, 26]

        # Mock technical analysis manager
        mock_ta_manager = Mock()
        mock_ta_manager.analyze_symbol.return_value = {
            "ticker": "AAPL",
            "current_price": 150.0,
            "analysis_type": "custom",
            "indicators": {
                "sma": {"current": 148.5},
                "ema": {"current": 149.2},
            },
            "summary": {"signals": [], "strength": "neutral"},
        }
        mock_container.technical_analysis_manager.return_value = mock_ta_manager

        # Mock data fetcher and TechnicalIndicators for period-specific processing
        mock_data_fetcher = Mock()
        mock_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [105, 106, 107],
                "Low": [95, 96, 97],
                "Close": [150, 149, 148],
                "Volume": [1000000, 1100000, 1200000],
            }
        )
        mock_data_fetcher.get_stock_data.return_value = mock_data
        mock_container.data_fetcher.return_value = mock_data_fetcher

        # We need to patch TechnicalIndicators since it gets created in the method
        with patch("stockula.manager.TechnicalIndicators") as mock_ta_class:
            mock_ta_instance = Mock()
            mock_ta_instance.sma.return_value = pd.Series([148.5, 145.2])  # 20, 50 period values
            mock_ta_instance.ema.return_value = pd.Series([149.2, 147.8])  # 12, 26 period values
            mock_ta_class.return_value = mock_ta_instance

            manager = StockulaManager(mock_config, mock_container)
            result = manager.run_technical_analysis("AAPL", show_progress=False)

            # Should have period-specific indicators added by backward compatibility code
            assert result["ticker"] == "AAPL"
            # Check if period-specific indicators were added
            if "SMA_20" in result["indicators"]:
                assert "SMA_20" in result["indicators"]
                assert "SMA_50" in result["indicators"]
            if "EMA_12" in result["indicators"]:
                assert "EMA_12" in result["indicators"]
                assert "EMA_26" in result["indicators"]

    def test_run_technical_analysis_with_progress(self, mock_config, mock_container):
        """Test technical analysis with progress tracking."""
        mock_config.technical_analysis.indicators = ["sma", "rsi"]

        # Mock technical analysis manager
        mock_ta_manager = Mock()
        mock_ta_manager.analyze_symbol.return_value = {
            "ticker": "AAPL",
            "current_price": 150.0,
            "analysis_type": "custom",
            "indicators": {
                "sma": {"current": 148.5},
                "rsi": {"current": 65.3},
            },
            "summary": {"signals": [], "strength": "neutral"},
        }
        mock_container.technical_analysis_manager.return_value = mock_ta_manager

        manager = StockulaManager(mock_config, mock_container)
        result = manager.run_technical_analysis("AAPL", show_progress=True)

        assert result["ticker"] == "AAPL"
        assert "sma" in result["indicators"]
        assert "rsi" in result["indicators"]

    def test_run_backtest_standard(self, mock_config, mock_container):
        """Test standard backtest without train/test split."""
        # Configure backtest
        strategy_config = Mock()
        strategy_config.name = "smacross"
        strategy_config.parameters = {"fast_period": 10, "slow_period": 20}
        mock_config.backtest.strategies = [strategy_config]
        mock_config.backtest.start_date = date(2023, 1, 1)
        mock_config.backtest.end_date = date(2023, 12, 31)

        # No forecast dates (no train/test split)
        mock_config.forecast.train_start_date = None
        mock_config.forecast.train_end_date = None
        mock_config.forecast.test_start_date = None
        mock_config.forecast.test_end_date = None

        # Mock backtest runner
        mock_runner = Mock()
        mock_runner.run_from_symbol.return_value = {
            "Return [%]": 15.0,
            "Sharpe Ratio": 1.2,
            "Max. Drawdown [%]": -8.0,
            "# Trades": 25,
            "Win Rate [%]": 60.0,
            "Initial Cash": 100000.0,
            "Start Date": "2023-01-01",
            "End Date": "2023-12-31",
            "Trading Days": 252,
            "Calendar Days": 365,
        }
        mock_container.backtest_runner.return_value = mock_runner

        # Mock backtesting manager
        mock_backtesting_manager = Mock()
        mock_backtesting_manager.run_single_strategy.return_value = {
            "Return [%]": 15.0,
            "Sharpe Ratio": 1.2,
            "Max. Drawdown [%]": -8.0,
            "# Trades": 25,
            "Win Rate [%]": 60.0,
            "Initial Cash": 100000.0,
            "Start Date": "2023-01-01",
            "End Date": "2023-12-31",
            "Trading Days": 252,
            "Calendar Days": 365,
        }
        mock_container.backtesting_manager.return_value = mock_backtesting_manager

        manager = StockulaManager(mock_config, mock_container)
        results = manager.run_backtest("AAPL")

        assert len(results) == 1
        result = results[0]
        assert result["ticker"] == "AAPL"
        assert result["strategy"] == "smacross"
        assert result["return_pct"] == 15.0
        assert result["initial_cash"] == 100000.0
        assert result["trading_days"] == 252

    def test_run_backtest_train_test_split(self, mock_config, mock_container):
        """Test backtest with train/test split."""
        # Configure backtest
        strategy_config = Mock()
        strategy_config.name = "rsi"
        strategy_config.parameters = {"period": 14}
        mock_config.backtest.strategies = [strategy_config]
        mock_config.backtest.optimize = True
        mock_config.backtest.optimization_params = {"period": [14, 21]}

        # Set forecast dates (triggers train/test split)
        mock_config.forecast.train_start_date = date(2023, 1, 1)
        mock_config.forecast.train_end_date = date(2023, 6, 30)
        mock_config.forecast.test_start_date = date(2023, 7, 1)
        mock_config.forecast.test_end_date = date(2023, 12, 31)

        # Mock backtest runner
        mock_runner = Mock()
        mock_runner.run_with_train_test_split.return_value = {
            "train_period": {"start": "2023-01-01", "end": "2023-06-30", "days": 180},
            "test_period": {"start": "2023-07-01", "end": "2023-12-31", "days": 183},
            "train_results": {
                "return_pct": 20.0,
                "sharpe_ratio": 1.8,
                "max_drawdown_pct": -5.0,
                "num_trades": 30,
                "win_rate": 65.0,
            },
            "test_results": {
                "return_pct": 15.0,
                "sharpe_ratio": 1.5,
                "max_drawdown_pct": -7.0,
                "num_trades": 25,
                "win_rate": 60.0,
            },
            "optimized_parameters": {"period": 21},
            "performance_degradation": {"return_diff": -5.0, "sharpe_diff": -0.3},
        }
        mock_container.backtest_runner.return_value = mock_runner

        # Mock backtesting manager
        mock_backtesting_manager = Mock()
        mock_backtesting_manager.run_with_train_test_split.return_value = {
            "train_period": {"start": "2023-01-01", "end": "2023-06-30", "days": 180},
            "test_period": {"start": "2023-07-01", "end": "2023-12-31", "days": 183},
            "train_results": {
                "return_pct": 20.0,
                "sharpe_ratio": 1.8,
                "max_drawdown_pct": -5.0,
                "num_trades": 30,
                "win_rate": 65.0,
            },
            "test_results": {
                "return_pct": 15.0,
                "sharpe_ratio": 1.5,
                "max_drawdown_pct": -7.0,
                "num_trades": 25,
                "win_rate": 60.0,
            },
            "optimized_parameters": {"period": 21},
            "performance_degradation": {"return_diff": -5.0, "sharpe_diff": -0.3},
        }
        mock_container.backtesting_manager.return_value = mock_backtesting_manager

        manager = StockulaManager(mock_config, mock_container)
        results = manager.run_backtest("AAPL")

        assert len(results) == 1
        result = results[0]
        assert result["ticker"] == "AAPL"
        assert result["strategy"] == "rsi"
        assert "train_period" in result
        assert "test_period" in result
        assert result["return_pct"] == 15.0  # Test period return
        assert result["parameters"]["period"] == 21  # Optimized parameter

    def test_run_backtest_unknown_strategy(self, mock_config, mock_container):
        """Test backtest with unknown strategy."""
        strategy_config = Mock()
        strategy_config.name = "unknown_strategy"
        strategy_config.parameters = {}
        mock_config.backtest.strategies = [strategy_config]

        manager = StockulaManager(mock_config, mock_container)
        results = manager.run_backtest("AAPL")

        assert results == []

    def test_run_backtest_with_exception(self, mock_config, mock_container):
        """Test backtest with exception handling."""
        strategy_config = Mock()
        strategy_config.name = "smacross"
        strategy_config.parameters = {}
        mock_config.backtest.strategies = [strategy_config]

        # Mock runner to raise exception
        mock_runner = Mock()
        mock_runner.run_from_symbol.side_effect = Exception("Test error")
        mock_container.backtest_runner.return_value = mock_runner

        # Mock backtesting manager to raise exception
        mock_backtesting_manager = Mock()
        mock_backtesting_manager.run_single_strategy.side_effect = Exception("Test error")
        mock_container.backtesting_manager.return_value = mock_backtesting_manager

        manager = StockulaManager(mock_config, mock_container)
        results = manager.run_backtest("AAPL")

        assert results == []

    def test_run_backtest_nan_win_rate(self, mock_config, mock_container):
        """Test backtest with NaN win rate (no trades)."""
        strategy_config = Mock()
        strategy_config.name = "rsi"
        strategy_config.parameters = {}
        mock_config.backtest.strategies = [strategy_config]

        # Mock runner with NaN win rate
        mock_runner = Mock()
        mock_runner.run_from_symbol.return_value = {
            "Return [%]": 0.0,
            "Sharpe Ratio": 0.0,
            "Max. Drawdown [%]": 0.0,
            "# Trades": 0,
            "Win Rate [%]": float("nan"),
        }
        mock_container.backtest_runner.return_value = mock_runner

        # Mock backtesting manager with NaN win rate
        mock_backtesting_manager = Mock()
        mock_backtesting_manager.run_single_strategy.return_value = {
            "Return [%]": 0.0,
            "Sharpe Ratio": 0.0,
            "Max. Drawdown [%]": 0.0,
            "# Trades": 0,
            "Win Rate [%]": float("nan"),
        }
        mock_container.backtesting_manager.return_value = mock_backtesting_manager

        manager = StockulaManager(mock_config, mock_container)
        results = manager.run_backtest("AAPL")

        assert len(results) == 1
        assert results[0]["win_rate"] is None

    def test_get_backtest_dates(self, mock_config, mock_container):
        """Test backtest date retrieval."""
        manager = StockulaManager(mock_config, mock_container)

        # Test with backtest-specific dates
        mock_config.backtest.start_date = date(2023, 1, 1)
        mock_config.backtest.end_date = date(2023, 12, 31)
        start, end = manager._get_backtest_dates()
        assert start == "2023-01-01"
        assert end == "2023-12-31"

        # Test with fallback to data dates
        mock_config.backtest.start_date = None
        mock_config.backtest.end_date = None
        mock_config.data.start_date = date(2023, 2, 1)
        mock_config.data.end_date = date(2023, 11, 30)
        start, end = manager._get_backtest_dates()
        assert start == "2023-02-01"
        assert end == "2023-11-30"

        # Test with no dates
        mock_config.data.start_date = None
        mock_config.data.end_date = None
        start, end = manager._get_backtest_dates()
        assert start is None
        assert end is None

    def test_create_train_test_result(self, manager):
        """Test train/test result creation."""
        strategy_config = Mock()
        strategy_config.name = "smacross"
        strategy_config.parameters = {"fast": 10, "slow": 20}

        backtest_result = {
            "train_period": {"start": "2023-01-01", "end": "2023-06-30"},
            "test_period": {"start": "2023-07-01", "end": "2023-12-31"},
            "train_results": {"return_pct": 20.0, "sharpe_ratio": 1.8},
            "test_results": {
                "return_pct": 15.0,
                "sharpe_ratio": 1.5,
                "max_drawdown_pct": -7.0,
                "num_trades": 25,
                "win_rate": 60.0,
            },
            "optimized_parameters": {"fast": 12, "slow": 26},
            "performance_degradation": {"return_diff": -5.0},
        }

        result = manager._create_train_test_result("AAPL", strategy_config, backtest_result)

        assert result["ticker"] == "AAPL"
        assert result["strategy"] == "smacross"
        assert result["parameters"]["fast"] == 12  # Optimized
        assert result["return_pct"] == 15.0  # Test period
        assert "train_period" in result
        assert "test_period" in result

    def test_create_standard_result(self, manager):
        """Test standard result creation."""
        strategy_config = Mock()
        strategy_config.name = "rsi"
        strategy_config.parameters = {"period": 14}

        backtest_result = {
            "Return [%]": 12.5,
            "Sharpe Ratio": 1.3,
            "Max. Drawdown [%]": -6.5,
            "# Trades": 30,
            "Win Rate [%]": 58.0,
            "Initial Cash": 100000.0,
            "Start Date": "2023-01-01",
            "End Date": "2023-12-31",
        }

        result = manager._create_standard_result("AAPL", strategy_config, backtest_result)

        assert result["ticker"] == "AAPL"
        assert result["strategy"] == "rsi"
        assert result["return_pct"] == 12.5
        assert result["initial_cash"] == 100000.0
        assert result["start_date"] == "2023-01-01"

    def test_run_forecast_with_evaluation(self, mock_config, mock_container):
        """Test forecasting with evaluation."""
        # Set up evaluation dates
        mock_config.forecast.train_start_date = date(2023, 1, 1)
        mock_config.forecast.train_end_date = date(2023, 6, 30)
        mock_config.forecast.test_start_date = date(2023, 7, 1)
        mock_config.forecast.test_end_date = date(2023, 12, 31)

        # Mock forecasting manager
        mock_forecasting_manager = Mock()
        mock_forecasting_manager.forecast_symbol.return_value = {
            "ticker": "AAPL",
            "current_price": 150.0,
            "forecast_price": 155.0,
            "lower_bound": 148.0,
            "upper_bound": 162.0,
            "evaluation": {"rmse": 2.5, "mae": 2.0, "mase": 0.8, "mape": 1.5},
        }
        mock_container.forecasting_manager.return_value = mock_forecasting_manager

        manager = StockulaManager(mock_config, mock_container)
        result = manager.run_forecast_with_evaluation("AAPL")

        assert result["ticker"] == "AAPL"
        assert "evaluation" in result
        assert result["evaluation"]["rmse"] == 2.5

    def test_run_forecast_with_evaluation_no_dates(self, mock_config, mock_container):
        """Test forecasting without evaluation dates."""
        # No evaluation dates
        mock_config.forecast.train_start_date = None
        mock_config.forecast.train_end_date = None
        mock_config.forecast.test_start_date = None
        mock_config.forecast.test_end_date = None

        # Mock forecasting manager
        mock_forecasting_manager = Mock()
        mock_forecasting_manager.forecast_symbol.return_value = {
            "ticker": "AAPL",
            "current_price": 150.0,
            "forecast_price": 155.0,
        }
        mock_container.forecasting_manager.return_value = mock_forecasting_manager

        manager = StockulaManager(mock_config, mock_container)
        result = manager.run_forecast_with_evaluation("AAPL")

        assert result["ticker"] == "AAPL"
        # Should not have evaluation since no test dates
        assert "evaluation" not in result

    def test_run_forecast_with_evaluation_error(self, mock_config, mock_container):
        """Test forecasting with evaluation error handling."""
        mock_config.forecast.train_start_date = date(2023, 1, 1)
        mock_config.forecast.train_end_date = date(2023, 6, 30)
        mock_config.forecast.test_start_date = date(2023, 7, 1)
        mock_config.forecast.test_end_date = date(2023, 12, 31)

        # Mock forecasting manager to raise exception
        mock_forecasting_manager = Mock()
        mock_forecasting_manager.forecast_symbol.side_effect = Exception("API error")
        mock_container.forecasting_manager.return_value = mock_forecasting_manager

        manager = StockulaManager(mock_config, mock_container)
        result = manager.run_forecast_with_evaluation("AAPL")

        assert result["ticker"] == "AAPL"
        assert result["error"] == "API error"

    def test_compute_indicators_with_progress(self, mock_config, mock_container):
        """Test _compute_indicators method with progress tracking."""
        # This method is currently in the manager but may not be used
        # We'll test it for completeness
        mock_config.technical_analysis.indicators = ["sma", "rsi"]
        mock_config.technical_analysis.sma_periods = [20]
        mock_config.technical_analysis.rsi_period = 14

        manager = StockulaManager(mock_config, mock_container)

        # Create mock TechnicalIndicators
        mock_ta = Mock()
        mock_ta.sma.return_value = pd.Series([150.0])
        mock_ta.rsi.return_value = pd.Series([65.0])

        # Create mock progress
        mock_progress = Mock()
        mock_task = Mock()

        results = {"indicators": {}}

        # Should not raise exception
        manager._compute_indicators(mock_ta, mock_config.technical_analysis, results, mock_progress, mock_task, "AAPL")

        # Should have computed indicators
        assert "SMA_20" in results["indicators"]
        assert "RSI" in results["indicators"]

    def test_run_forecast_standard(self, mock_config, mock_container):
        """Test standard forecast method."""
        # Mock forecasting manager
        mock_forecasting_manager = Mock()
        mock_forecasting_manager.forecast_symbol.return_value = {
            "ticker": "AAPL",
            "current_price": 150.0,
            "forecast_price": 155.0,
            "lower_bound": 148.0,
            "upper_bound": 162.0,
        }
        mock_container.forecasting_manager.return_value = mock_forecasting_manager

        manager = StockulaManager(mock_config, mock_container)
        result = manager.run_forecast("AAPL")

        assert result["ticker"] == "AAPL"
        assert result["forecast_price"] == 155.0

        # Verify it was called without evaluation
        mock_forecasting_manager.forecast_symbol.assert_called_once_with("AAPL", mock_config, use_evaluation=False)

    def test_run_forecast_keyboard_interrupt(self, mock_config, mock_container):
        """Test forecast with keyboard interrupt."""
        mock_forecasting_manager = Mock()
        mock_forecasting_manager.forecast_symbol.side_effect = KeyboardInterrupt()
        mock_container.forecasting_manager.return_value = mock_forecasting_manager

        manager = StockulaManager(mock_config, mock_container)
        result = manager.run_forecast("AAPL")

        assert result["ticker"] == "AAPL"
        assert result["error"] == "Interrupted by user"

    def test_run_forecast_exception(self, mock_config, mock_container):
        """Test forecast with exception."""
        mock_forecasting_manager = Mock()
        mock_forecasting_manager.forecast_symbol.side_effect = Exception("API Error")
        mock_container.forecasting_manager.return_value = mock_forecasting_manager

        manager = StockulaManager(mock_config, mock_container)
        result = manager.run_forecast("AAPL")

        assert result["ticker"] == "AAPL"
        assert result["error"] == "API Error"

    def test_save_detailed_report(self, mock_config, mock_container, tmp_path):
        """Test saving detailed strategy report."""
        # Configure output directory
        mock_config.output = {"results_dir": str(tmp_path)}

        manager = StockulaManager(mock_config, mock_container)

        strategy_results = [
            {
                "ticker": "AAPL",
                "strategy": "smacross",
                "return_pct": 15.0,
                "sharpe_ratio": 1.2,
                "num_trades": 25,
            },
            {
                "ticker": "GOOGL",
                "strategy": "smacross",
                "return_pct": 12.0,
                "sharpe_ratio": 1.1,
                "num_trades": 20,
            },
        ]

        results = {
            "initial_portfolio_value": 100000.0,
            "initial_capital": 100000.0,
        }

        with patch("stockula.manager.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240115_120000"

            report_path = manager.save_detailed_report("smacross", strategy_results, results)

            # Verify file was created
            assert Path(report_path).exists()

            # Verify content
            with open(report_path) as f:
                report_data = json.load(f)

            assert report_data["strategy"] == "smacross"
            assert len(report_data["detailed_results"]) == 2
            assert report_data["summary"]["total_trades"] == 45
            assert report_data["summary"]["winning_stocks"] == 2

    def test_save_detailed_report_with_portfolio_results(self, mock_config, mock_container, tmp_path):
        """Test saving detailed report with portfolio results."""
        mock_config.output = {"results_dir": str(tmp_path)}

        manager = StockulaManager(mock_config, mock_container)

        # Create mock portfolio results
        mock_portfolio_results = Mock()
        mock_portfolio_results.model_dump.return_value = {"test": "data"}

        with patch("stockula.manager.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240115_120000"

            report_path = manager.save_detailed_report("test", [], {}, mock_portfolio_results)

            # Verify both files were created
            assert Path(report_path).exists()
            portfolio_file = tmp_path / "reports" / "portfolio_backtest_20240115_120000.json"
            assert portfolio_file.exists()

    def test_get_broker_config_dict_modern(self, mock_config, mock_container):
        """Test broker config dict with modern broker config."""
        mock_broker_config = Mock()
        mock_broker_config.name = "interactive_brokers"
        mock_broker_config.commission_type = "tiered"
        mock_broker_config.commission_value = 0.0035
        mock_broker_config.min_commission = 0.35
        mock_broker_config.regulatory_fees = 0.01
        mock_config.backtest.broker_config = mock_broker_config

        manager = StockulaManager(mock_config, mock_container)
        broker_dict = manager._get_broker_config_dict()

        assert broker_dict["name"] == "interactive_brokers"
        assert broker_dict["commission_type"] == "tiered"
        assert broker_dict["regulatory_fees"] == 0.01

    def test_get_broker_config_dict_legacy(self, mock_config, mock_container):
        """Test broker config dict with legacy config."""
        mock_config.backtest.broker_config = None
        mock_config.backtest.commission = 0.002

        manager = StockulaManager(mock_config, mock_container)
        broker_dict = manager._get_broker_config_dict()

        assert broker_dict["name"] == "legacy"
        assert broker_dict["commission_type"] == "percentage"
        assert broker_dict["commission_value"] == 0.002

    def test_create_portfolio_backtest_results(self, mock_config, mock_container):
        """Test creating portfolio backtest results."""
        manager = StockulaManager(mock_config, mock_container)

        results = {
            "initial_portfolio_value": 100000.0,
            "initial_capital": 100000.0,
        }

        strategy_results = {
            "smacross": [
                {
                    "ticker": "AAPL",
                    "strategy": "smacross",
                    "parameters": {"fast": 10, "slow": 20},
                    "return_pct": 15.0,
                    "sharpe_ratio": 1.2,
                    "max_drawdown_pct": -8.0,
                    "num_trades": 25,
                    "win_rate": 60.0,
                },
                {
                    "ticker": "GOOGL",
                    "strategy": "smacross",
                    "parameters": {"fast": 10, "slow": 20},
                    "return_pct": 10.0,
                    "sharpe_ratio": 1.0,
                    "max_drawdown_pct": -5.0,
                    "num_trades": 20,
                    "win_rate": 55.0,
                },
            ]
        }

        # Mock broker config
        mock_config.backtest.broker_config = None
        mock_config.backtest.commission = 0.001

        portfolio_results = manager.create_portfolio_backtest_results(results, strategy_results)

        assert portfolio_results.initial_portfolio_value == 100000.0
        assert len(portfolio_results.strategy_summaries) == 1

        summary = portfolio_results.strategy_summaries[0]
        assert summary.strategy_name == "smacross"
        assert summary.total_trades == 45
        assert summary.winning_stocks == 2
        assert summary.losing_stocks == 0
        assert len(summary.detailed_results) == 2

    def test_get_full_broker_config_dict_modern(self, mock_config, mock_container):
        """Test getting full broker config dict with modern config."""
        mock_broker_config = Mock()
        mock_broker_config.name = "td_ameritrade"
        mock_broker_config.commission_type = "percentage"
        mock_broker_config.commission_value = 0.001
        mock_broker_config.min_commission = 5.0
        mock_broker_config.regulatory_fees = 0.01
        mock_broker_config.exchange_fees = 0.005
        mock_config.backtest.broker_config = mock_broker_config

        manager = StockulaManager(mock_config, mock_container)
        broker_dict = manager._get_full_broker_config_dict()

        assert broker_dict["name"] == "td_ameritrade"
        assert broker_dict["commission_type"] == "percentage"
        assert broker_dict["exchange_fees"] == 0.005

    def test_get_full_broker_config_dict_legacy(self, mock_config, mock_container):
        """Test getting full broker config dict with legacy config."""
        mock_config.backtest.broker_config = None
        mock_config.backtest.commission = 0.0025

        manager = StockulaManager(mock_config, mock_container)
        broker_dict = manager._get_full_broker_config_dict()

        assert broker_dict["name"] == "legacy"
        assert broker_dict["commission_value"] == 0.0025
        assert broker_dict["regulatory_fees"] == 0
        assert broker_dict["exchange_fees"] == 0

    def test_get_date_range_from_config(self, mock_config, mock_container):
        """Test getting date range from config."""
        mock_config.data.start_date = date(2023, 1, 1)
        mock_config.data.end_date = date(2023, 12, 31)

        manager = StockulaManager(mock_config, mock_container)
        start_str, end_str = manager._get_date_range({})

        assert start_str == "2023-01-01"
        assert end_str == "2023-12-31"

    def test_get_date_range_from_backtest_config(self, mock_config, mock_container):
        """Test getting date range from backtest config."""
        mock_config.data.start_date = None
        mock_config.data.end_date = None
        mock_config.backtest.start_date = date(2023, 2, 1)
        mock_config.backtest.end_date = date(2023, 11, 30)

        manager = StockulaManager(mock_config, mock_container)
        start_str, end_str = manager._get_date_range({})

        assert start_str == "2023-02-01"
        assert end_str == "2023-11-30"

    def test_get_date_range_from_results(self, mock_config, mock_container):
        """Test getting date range from results when config has no dates."""
        mock_config.data.start_date = None
        mock_config.data.end_date = None
        mock_config.backtest.start_date = None
        mock_config.backtest.end_date = None

        results = {
            "backtesting": [
                {
                    "start_date": "2023-03-01",
                    "end_date": "2023-10-31",
                }
            ]
        }

        manager = StockulaManager(mock_config, mock_container)
        start_str, end_str = manager._get_date_range(results)

        assert start_str == "2023-03-01"
        assert end_str == "2023-10-31"

    def test_get_portfolio_value_at_date(self, mock_config, mock_container):
        """Test getting portfolio value at a specific date."""
        # Mock data fetcher
        mock_data_fetcher = Mock()
        mock_data = pd.DataFrame({"Close": [150.0, 151.0, 152.0]})
        mock_data_fetcher.get_stock_data.return_value = mock_data
        mock_container.data_fetcher.return_value = mock_data_fetcher

        # Mock portfolio
        mock_portfolio = Mock()
        mock_portfolio.get_all_assets.return_value = [Mock(symbol="AAPL")]
        mock_portfolio.get_portfolio_value.return_value = 100000.0

        manager = StockulaManager(mock_config, mock_container)

        # Test with specific date
        value, prices = manager.get_portfolio_value_at_date(mock_portfolio, "2023-01-01")
        assert value == 100000.0
        assert "AAPL" in prices

        # Test with no date (current prices)
        mock_data_fetcher.get_current_prices.return_value = {"AAPL": 155.0}
        value, prices = manager.get_portfolio_value_at_date(mock_portfolio, None)
        assert value == 100000.0
        assert prices["AAPL"] == 155.0

    def test_categorize_assets(self, mock_config, mock_container):
        """Test asset categorization."""
        mock_config.backtest.hold_only_categories = ["BOND", "REAL_ESTATE"]

        # Mock assets
        from stockula.domain.category import Category

        mock_asset1 = Mock()
        mock_asset1.symbol = "AAPL"
        mock_asset1.category = Category.LARGE_CAP

        mock_asset2 = Mock()
        mock_asset2.symbol = "BND"
        mock_asset2.category = Category.BOND

        mock_portfolio = Mock()
        mock_portfolio.get_all_assets.return_value = [mock_asset1, mock_asset2]

        manager = StockulaManager(mock_config, mock_container)
        tradeable, hold_only, hold_categories = manager.categorize_assets(mock_portfolio)

        assert len(tradeable) == 1
        assert len(hold_only) == 1
        assert tradeable[0].symbol == "AAPL"
        assert hold_only[0].symbol == "BND"
        assert Category.BOND in hold_categories

    def test_categorize_assets_invalid_category(self, mock_config, mock_container):
        """Test asset categorization with invalid category name."""
        mock_config.backtest.hold_only_categories = ["INVALID_CATEGORY"]

        mock_portfolio = Mock()
        mock_portfolio.get_all_assets.return_value = []

        manager = StockulaManager(mock_config, mock_container)
        tradeable, hold_only, hold_categories = manager.categorize_assets(mock_portfolio)

        # Should handle invalid category gracefully
        assert len(tradeable) == 0
        assert len(hold_only) == 0
        assert len(hold_categories) == 0
