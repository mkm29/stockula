"""Tests for BacktestingManager."""

from unittest.mock import MagicMock

import pytest

from stockula.backtesting.manager import BacktestingManager
from stockula.interfaces import ILoggingManager


@pytest.fixture
def mock_data_fetcher():
    """Mock DataFetcher for testing."""
    mock_fetcher = MagicMock()
    mock_fetcher.get_stock_data.return_value = MagicMock()
    return mock_fetcher


@pytest.fixture
def mock_logging_manager():
    """Mock LoggingManager for testing."""
    mock_logger = MagicMock(spec=ILoggingManager)
    return mock_logger


@pytest.fixture
def mock_backtest_runner():
    """Mock BacktestRunner for testing."""
    mock_runner = MagicMock()
    mock_runner.run_from_symbol.return_value = {
        "Return [%]": 15.5,
        "Sharpe Ratio": 1.2,
        "Max. Drawdown [%]": -8.5,
        "# Trades": 25,
        "Win Rate [%]": 65.0,
    }
    mock_runner.run_with_train_test_split.return_value = {
        "train_period": {"start": "2023-01-01", "end": "2023-07-01"},
        "test_period": {"start": "2023-07-01", "end": "2023-12-31"},
        "train_results": {"Return [%]": 18.0, "Sharpe Ratio": 1.4},
        "test_results": {"Return [%]": 12.0, "Sharpe Ratio": 1.0},
        "performance_degradation": {"return_degradation": -6.0},
    }
    return mock_runner


@pytest.fixture
def backtesting_manager(mock_data_fetcher, mock_logging_manager):
    """Create BacktestingManager instance for testing."""
    return BacktestingManager(mock_data_fetcher, mock_logging_manager)


class TestBacktestingManagerInitialization:
    """Test BacktestingManager initialization."""

    def test_initialization(self, mock_data_fetcher, mock_logging_manager):
        """Test manager initialization."""
        manager = BacktestingManager(mock_data_fetcher, mock_logging_manager)

        assert manager.data_fetcher == mock_data_fetcher
        assert manager.logger == mock_logging_manager
        assert manager._runner is None
        # Use public API methods instead of accessing private attributes
        strategy_groups = manager.get_strategy_groups()
        assert "basic" in strategy_groups
        assert "comprehensive" in strategy_groups
        strategy_presets = manager.get_strategy_presets()
        assert "smacross" in strategy_presets  # Note: normalized to smacross
        assert "rsi" in strategy_presets

    def test_strategy_groups_structure(self, backtesting_manager):
        """Test that strategy groups are properly structured."""
        groups = backtesting_manager.get_strategy_groups()

        assert isinstance(groups, dict)
        assert "basic" in groups
        assert "momentum" in groups
        assert "trend" in groups
        assert "advanced" in groups
        assert "comprehensive" in groups

        # Check that comprehensive contains all strategies
        comprehensive = groups["comprehensive"]
        basic = groups["basic"]

        for strategy in basic:
            assert strategy in comprehensive

    def test_strategy_presets_structure(self, backtesting_manager):
        """Test that strategy presets are properly structured."""
        presets = backtesting_manager.get_strategy_presets()

        assert isinstance(presets, dict)
        assert "smacross" in presets  # Note: normalized to smacross
        assert "rsi" in presets
        assert "macd" in presets

        # Check that presets contain proper parameters
        # Note: smacross is normalized to smacross
        sma_preset = presets["smacross"]
        assert "fast_period" in sma_preset
        assert "slow_period" in sma_preset

        rsi_preset = presets["rsi"]
        assert "period" in rsi_preset
        assert "oversold_threshold" in rsi_preset  # Correct key name
        assert "overbought_threshold" in rsi_preset  # Correct key name


class TestBacktestingManagerSetRunner:
    """Test BacktestingManager runner setup."""

    def test_set_runner(self, backtesting_manager, mock_backtest_runner):
        """Test setting the BacktestRunner."""
        backtesting_manager.set_runner(mock_backtest_runner)
        assert backtesting_manager._runner == mock_backtest_runner

    def test_operations_without_runner_raise_error(self, backtesting_manager):
        """Test that operations without runner raise ValueError."""
        with pytest.raises(ValueError, match="BacktestRunner not initialized"):
            backtesting_manager.run_single_strategy("AAPL", "smacross")

    def test_train_test_split_without_runner_raises_error(self, backtesting_manager):
        """Test that train/test split without runner raises ValueError."""
        with pytest.raises(ValueError, match="BacktestRunner not initialized"):
            backtesting_manager.run_with_train_test_split("AAPL", "smacross")


class TestBacktestingManagerSingleStrategy:
    """Test single strategy backtesting."""

    def test_run_single_strategy_success(self, backtesting_manager, mock_backtest_runner):
        """Test successful single strategy backtest."""
        backtesting_manager.set_runner(mock_backtest_runner)

        result = backtesting_manager.run_single_strategy("AAPL", "smacross")

        assert "Return [%]" in result
        assert result["Return [%]"] == 15.5
        assert result["Sharpe Ratio"] == 1.2
        mock_backtest_runner.run_from_symbol.assert_called_once()

    def test_run_single_strategy_with_custom_params(self, backtesting_manager, mock_backtest_runner):
        """Test single strategy backtest with custom parameters."""
        backtesting_manager.set_runner(mock_backtest_runner)
        custom_params = {"fast_period": 5, "slow_period": 20}

        result = backtesting_manager.run_single_strategy("AAPL", "smacross", strategy_params=custom_params)

        assert "Return [%]" in result
        mock_backtest_runner.run_from_symbol.assert_called_once()

    def test_run_single_strategy_with_dates(self, backtesting_manager, mock_backtest_runner):
        """Test single strategy backtest with custom date range."""
        backtesting_manager.set_runner(mock_backtest_runner)

        result = backtesting_manager.run_single_strategy(
            "AAPL", "smacross", start_date="2023-01-01", end_date="2023-12-31"
        )

        assert "Return [%]" in result
        mock_backtest_runner.run_from_symbol.assert_called_once()

    def test_run_single_strategy_error_handling(self, backtesting_manager, mock_backtest_runner, mock_logging_manager):
        """Test error handling in single strategy backtest."""
        backtesting_manager.set_runner(mock_backtest_runner)
        mock_backtest_runner.run_from_symbol.side_effect = Exception("Test error")

        result = backtesting_manager.run_single_strategy("AAPL", "smacross")

        assert "error" in result
        assert result["error"] == "Test error"
        assert result["ticker"] == "AAPL"
        assert result["strategy"] == "smacross"
        mock_logging_manager.error.assert_called_once()


class TestBacktestingManagerMultipleStrategies:
    """Test multiple strategies backtesting."""

    def test_run_multiple_strategies_basic_group(self, backtesting_manager, mock_backtest_runner):
        """Test running basic strategy group."""
        backtesting_manager.set_runner(mock_backtest_runner)

        results = backtesting_manager.run_multiple_strategies("AAPL", "basic")

        assert isinstance(results, dict)
        strategy_groups = backtesting_manager.get_strategy_groups()
        basic_strategies = strategy_groups["basic"]

        for strategy in basic_strategies:
            assert strategy in results
            assert "Return [%]" in results[strategy]

    def test_run_multiple_strategies_invalid_group(self, backtesting_manager, mock_backtest_runner):
        """Test running with invalid strategy group."""
        backtesting_manager.set_runner(mock_backtest_runner)

        with pytest.raises(ValueError, match="Unknown strategy group: invalid"):
            backtesting_manager.run_multiple_strategies("AAPL", "invalid")

    def test_run_multiple_strategies_comprehensive(self, backtesting_manager, mock_backtest_runner):
        """Test running comprehensive strategy group."""
        backtesting_manager.set_runner(mock_backtest_runner)

        results = backtesting_manager.run_multiple_strategies("AAPL", "comprehensive")

        assert isinstance(results, dict)
        strategy_groups = backtesting_manager.get_strategy_groups()
        comprehensive_strategies = strategy_groups["comprehensive"]

        assert len(results) == len(comprehensive_strategies)
        for strategy in comprehensive_strategies:
            assert strategy in results


class TestBacktestingManagerPortfolioBacktest:
    """Test portfolio backtesting."""

    def test_run_portfolio_backtest(self, backtesting_manager, mock_backtest_runner):
        """Test running backtest across multiple tickers."""
        backtesting_manager.set_runner(mock_backtest_runner)
        tickers = ["AAPL", "GOOGL", "MSFT"]

        results = backtesting_manager.run_portfolio_backtest(tickers, "smacross")

        assert isinstance(results, dict)
        assert len(results) == len(tickers)

        for ticker in tickers:
            assert ticker in results
            assert "Return [%]" in results[ticker]

    def test_run_portfolio_backtest_with_params(self, backtesting_manager, mock_backtest_runner):
        """Test portfolio backtest with custom parameters."""
        backtesting_manager.set_runner(mock_backtest_runner)
        tickers = ["AAPL", "GOOGL"]
        custom_params = {"period": 21}

        results = backtesting_manager.run_portfolio_backtest(tickers, "rsi", strategy_params=custom_params)

        assert len(results) == len(tickers)
        for ticker in tickers:
            assert ticker in results


class TestBacktestingManagerComprehensiveBacktest:
    """Test comprehensive backtesting."""

    def test_run_comprehensive_backtest(self, backtesting_manager, mock_backtest_runner):
        """Test comprehensive backtest across tickers and strategies."""
        backtesting_manager.set_runner(mock_backtest_runner)
        tickers = ["AAPL", "GOOGL"]

        results = backtesting_manager.run_comprehensive_backtest(tickers, "basic")

        assert isinstance(results, dict)
        assert len(results) == len(tickers)

        strategy_groups = backtesting_manager.get_strategy_groups()
        basic_strategies = strategy_groups["basic"]

        for ticker in tickers:
            assert ticker in results
            ticker_results = results[ticker]

            for strategy in basic_strategies:
                assert strategy in ticker_results
                assert "Return [%]" in ticker_results[strategy]


class TestBacktestingManagerTrainTestSplit:
    """Test train/test split backtesting."""

    def test_run_with_train_test_split(self, backtesting_manager, mock_backtest_runner):
        """Test train/test split backtest."""
        backtesting_manager.set_runner(mock_backtest_runner)

        result = backtesting_manager.run_with_train_test_split("AAPL", "smacross")

        assert "train_results" in result
        assert "test_results" in result
        assert "performance_degradation" in result
        mock_backtest_runner.run_with_train_test_split.assert_called_once()

    def test_train_test_split_with_optimization(self, backtesting_manager, mock_backtest_runner):
        """Test train/test split with parameter optimization."""
        backtesting_manager.set_runner(mock_backtest_runner)
        param_ranges = {"fast_period": [5, 10, 15]}

        result = backtesting_manager.run_with_train_test_split(
            "AAPL", "smacross", optimize_on_train=True, param_ranges=param_ranges
        )

        assert "train_results" in result
        assert "test_results" in result
        mock_backtest_runner.run_with_train_test_split.assert_called_once()

    def test_train_test_split_custom_ratio(self, backtesting_manager, mock_backtest_runner):
        """Test train/test split with custom train ratio."""
        backtesting_manager.set_runner(mock_backtest_runner)

        result = backtesting_manager.run_with_train_test_split("AAPL", "smacross", train_ratio=0.8)

        assert "train_results" in result
        mock_backtest_runner.run_with_train_test_split.assert_called_once()

    def test_train_test_split_error_handling(self, backtesting_manager, mock_backtest_runner, mock_logging_manager):
        """Test error handling in train/test split."""
        backtesting_manager.set_runner(mock_backtest_runner)
        mock_backtest_runner.run_with_train_test_split.side_effect = Exception("Split error")

        result = backtesting_manager.run_with_train_test_split("AAPL", "smacross")

        assert "error" in result
        assert result["error"] == "Split error"
        mock_logging_manager.error.assert_called_once()


class TestBacktestingManagerQuickBacktest:
    """Test quick backtesting functionality."""

    def test_quick_backtest_default_strategy(self, backtesting_manager, mock_backtest_runner):
        """Test quick backtest with default strategy."""
        backtesting_manager.set_runner(mock_backtest_runner)

        result = backtesting_manager.quick_backtest("AAPL")

        assert result["ticker"] == "AAPL"
        assert result["strategy"] == "smacross"
        assert "return_pct" in result
        assert "sharpe_ratio" in result
        assert "max_drawdown_pct" in result

    def test_quick_backtest_custom_strategy(self, backtesting_manager, mock_backtest_runner):
        """Test quick backtest with custom strategy."""
        backtesting_manager.set_runner(mock_backtest_runner)

        result = backtesting_manager.quick_backtest("AAPL", "rsi")

        assert result["strategy"] == "rsi"
        assert "return_pct" in result

    def test_quick_backtest_error_handling(self, backtesting_manager, mock_backtest_runner):
        """Test quick backtest error handling."""
        backtesting_manager.set_runner(mock_backtest_runner)
        mock_backtest_runner.run_from_symbol.side_effect = Exception("Quick error")

        result = backtesting_manager.quick_backtest("AAPL")

        assert "error" in result
        assert result["error"] == "Quick error"


class TestBacktestingManagerUtilityMethods:
    """Test utility methods."""

    def test_get_available_strategies(self, backtesting_manager):
        """Test getting available strategies."""
        strategies = backtesting_manager.get_available_strategies()

        assert isinstance(strategies, list)
        assert "smacross" in strategies  # Note: normalized name
        assert "rsi" in strategies
        assert "macd" in strategies

    def test_customize_strategy_parameters(self, backtesting_manager):
        """Test customizing strategy parameters."""
        # Get original parameters using the public API
        original_presets = backtesting_manager.get_strategy_presets()
        original_params = original_presets["smacross"].copy()  # Note: normalized name
        new_params = {"fast_period": 5}

        backtesting_manager.customize_strategy_parameters("smacross", new_params)

        # Get updated parameters using the public API
        updated_presets = backtesting_manager.get_strategy_presets()
        updated_params = updated_presets["smacross"]
        assert updated_params["fast_period"] == 5
        assert updated_params["slow_period"] == original_params["slow_period"]

    def test_customize_invalid_strategy_raises_error(self, backtesting_manager):
        """Test customizing invalid strategy raises error."""
        with pytest.raises(ValueError, match="Unknown strategy: invalid"):
            backtesting_manager.customize_strategy_parameters("invalid", {"param": 1})

    def test_create_custom_strategy_group(self, backtesting_manager):
        """Test creating custom strategy group."""
        custom_strategies = ["smacross", "rsi"]

        backtesting_manager.create_custom_strategy_group("my_group", custom_strategies)

        groups = backtesting_manager.get_strategy_groups()
        assert "my_group" in groups
        # Note: strategy names get normalized, so smacross becomes smacross
        assert groups["my_group"] == ["smacross", "rsi"]

    def test_create_custom_group_invalid_strategies_raises_error(self, backtesting_manager):
        """Test creating custom group with invalid strategies raises error."""
        invalid_strategies = ["smacross", "invalid_strategy"]

        with pytest.raises(ValueError, match="Invalid strategies"):
            backtesting_manager.create_custom_strategy_group("bad_group", invalid_strategies)

    def test_get_strategy_groups_returns_copy(self, backtesting_manager):
        """Test that get_strategy_groups returns a copy."""
        groups1 = backtesting_manager.get_strategy_groups()
        groups2 = backtesting_manager.get_strategy_groups()

        assert groups1 is not groups2
        assert groups1 == groups2

    def test_get_strategy_presets_returns_copy(self, backtesting_manager):
        """Test that get_strategy_presets returns a copy."""
        presets1 = backtesting_manager.get_strategy_presets()
        presets2 = backtesting_manager.get_strategy_presets()

        assert presets1 is not presets2
        assert presets1 == presets2
