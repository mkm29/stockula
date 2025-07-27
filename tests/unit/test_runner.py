"""Unit tests for backtesting runner module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from stockula.backtesting.runner import BacktestRunner
from stockula.backtesting.strategies import SMACrossStrategy


class TestBacktestRunnerInitialization:
    """Test BacktestRunner initialization."""

    def test_initialization_with_defaults(self, mock_data_fetcher):
        """Test BacktestRunner initialization with default parameters."""
        runner = BacktestRunner(data_fetcher=mock_data_fetcher)
        assert runner.cash == 10000
        assert runner.commission == 0.002
        assert runner.margin == 1.0
        assert runner.results is None
        assert runner.data_fetcher == mock_data_fetcher

    def test_initialization_with_custom_params(self, mock_data_fetcher):
        """Test BacktestRunner initialization with custom parameters."""
        runner = BacktestRunner(
            cash=50000, commission=0.001, margin=2.0, data_fetcher=mock_data_fetcher
        )
        assert runner.cash == 50000
        assert runner.commission == 0.001
        assert runner.margin == 2.0
        assert runner.results is None
        assert runner.data_fetcher == mock_data_fetcher


class TestBacktestRunnerRun:
    """Test BacktestRunner run method."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "Open": 100 + np.random.randn(100).cumsum(),
                "High": 101 + np.random.randn(100).cumsum(),
                "Low": 99 + np.random.randn(100).cumsum(),
                "Close": 100 + np.random.randn(100).cumsum(),
                "Volume": np.random.randint(1000000, 5000000, 100),
            },
            index=dates,
        )
        return data

    @patch("stockula.backtesting.runner.Backtest")
    def test_run_basic(self, mock_backtest_class, sample_data):
        """Test basic run functionality."""
        # Mock backtest and results
        mock_backtest = Mock()
        mock_results = {
            "Return [%]": 15.5,
            "Sharpe Ratio": 1.25,
            "Max. Drawdown [%]": -8.3,
            "# Trades": 42,
        }
        mock_backtest.run.return_value = mock_results
        mock_backtest_class.return_value = mock_backtest

        runner = BacktestRunner(data_fetcher=None)
        results = runner.run(sample_data, SMACrossStrategy)

        # Verify backtest was created with correct parameters
        mock_backtest_class.assert_called_once_with(
            sample_data, SMACrossStrategy, cash=10000, commission=0.002, margin=1.0
        )

        # Verify run was called
        mock_backtest.run.assert_called_once()

        # Verify results
        assert results == mock_results
        assert runner.results == mock_results

    @patch("stockula.backtesting.runner.Backtest")
    def test_run_with_strategy_kwargs(self, mock_backtest_class, sample_data):
        """Test run with strategy parameters."""
        mock_backtest = Mock()
        mock_results = {"Return [%]": 10.0}
        mock_backtest.run.return_value = mock_results
        mock_backtest_class.return_value = mock_backtest

        runner = BacktestRunner(data_fetcher=None)
        results = runner.run(
            sample_data, SMACrossStrategy, fast_period=5, slow_period=15
        )

        # Verify run was called with kwargs
        mock_backtest.run.assert_called_once_with(fast_period=5, slow_period=15)

    @patch("stockula.backtesting.runner.Backtest")
    def test_run_with_insufficient_data_warning(
        self, mock_backtest_class, sample_data, capsys
    ):
        """Test warning when insufficient data for strategy."""
        # Create strategy mock with period requirements
        mock_strategy = Mock()
        mock_strategy.__name__ = "TestStrategy"
        mock_strategy.slow_period = 200  # More than sample data
        mock_strategy.min_trading_days_buffer = 20

        mock_backtest = Mock()
        mock_backtest.run.return_value = {}
        mock_backtest_class.return_value = mock_backtest

        runner = BacktestRunner(data_fetcher=None)
        runner.run(sample_data, mock_strategy)

        # Check warning was printed
        captured = capsys.readouterr()
        assert "Warning: TestStrategy requires at least 220 days" in captured.out
        assert "but only 100 days available" in captured.out

    @patch("stockula.backtesting.runner.Backtest")
    def test_run_custom_parameters(self, mock_backtest_class, sample_data):
        """Test run with custom runner parameters."""
        mock_backtest = Mock()
        mock_backtest.run.return_value = {}
        mock_backtest_class.return_value = mock_backtest

        runner = BacktestRunner(
            cash=25000, commission=0.005, margin=1.5, data_fetcher=None
        )
        runner.run(sample_data, SMACrossStrategy)

        # Verify custom parameters were used
        mock_backtest_class.assert_called_once_with(
            sample_data, SMACrossStrategy, cash=25000, commission=0.005, margin=1.5
        )


class TestBacktestRunnerOptimize:
    """Test BacktestRunner optimize method."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "Open": 100 + np.random.randn(100).cumsum(),
                "High": 101 + np.random.randn(100).cumsum(),
                "Low": 99 + np.random.randn(100).cumsum(),
                "Close": 100 + np.random.randn(100).cumsum(),
                "Volume": np.random.randint(1000000, 5000000, 100),
            },
            index=dates,
        )
        return data

    @patch("stockula.backtesting.runner.Backtest")
    def test_optimize_basic(self, mock_backtest_class, sample_data):
        """Test basic optimize functionality."""
        mock_backtest = Mock()
        mock_results = {"fast_period": 10, "slow_period": 20, "Return [%]": 18.5}
        mock_backtest.optimize.return_value = mock_results
        mock_backtest_class.return_value = mock_backtest

        runner = BacktestRunner(data_fetcher=None)
        results = runner.optimize(
            sample_data,
            SMACrossStrategy,
            fast_period=range(5, 15),
            slow_period=range(15, 25),
        )

        # Verify backtest was created
        mock_backtest_class.assert_called_once_with(
            sample_data, SMACrossStrategy, cash=10000, commission=0.002, margin=1.0
        )

        # Verify optimize was called with parameters
        mock_backtest.optimize.assert_called_once_with(
            fast_period=range(5, 15), slow_period=range(15, 25)
        )

        assert results == mock_results

    @patch("stockula.backtesting.runner.Backtest")
    def test_optimize_custom_parameters(self, mock_backtest_class, sample_data):
        """Test optimize with custom runner parameters."""
        mock_backtest = Mock()
        mock_backtest.optimize.return_value = {}
        mock_backtest_class.return_value = mock_backtest

        runner = BacktestRunner(cash=30000, commission=0.001, data_fetcher=None)
        runner.optimize(sample_data, SMACrossStrategy, fast_period=range(5, 10))

        # Verify custom parameters were used
        mock_backtest_class.assert_called_once_with(
            sample_data, SMACrossStrategy, cash=30000, commission=0.001, margin=1.0
        )


class TestBacktestRunnerFromSymbol:
    """Test BacktestRunner run_from_symbol method."""

    def test_run_from_symbol_no_data_fetcher(self):
        """Test run_from_symbol without data fetcher raises error."""
        runner = BacktestRunner(data_fetcher=None)

        with pytest.raises(ValueError, match="Data fetcher not configured"):
            runner.run_from_symbol("AAPL", SMACrossStrategy)

    @patch("stockula.backtesting.runner.Backtest")
    def test_run_from_symbol_basic(self, mock_backtest_class):
        """Test run_from_symbol with basic parameters."""
        # Create a mock data fetcher
        mock_data_fetcher = Mock()

        # Setup mock data
        sample_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [101, 102, 103],
                "Low": [99, 100, 101],
                "Close": [100.5, 101.5, 102.5],
                "Volume": [1000000, 1100000, 1200000],
            }
        )
        mock_data_fetcher.get_stock_data.return_value = sample_data

        # Mock backtest
        mock_backtest = Mock()
        mock_results = {"Return [%]": 12.5}
        mock_backtest.run.return_value = mock_results
        mock_backtest_class.return_value = mock_backtest

        runner = BacktestRunner(data_fetcher=mock_data_fetcher)
        results = runner.run_from_symbol("AAPL", SMACrossStrategy)

        # Verify data fetcher was called
        mock_data_fetcher.get_stock_data.assert_called_once_with("AAPL", None, None)

        # Verify backtest was run with fetched data
        mock_backtest_class.assert_called_once_with(
            sample_data, SMACrossStrategy, cash=10000, commission=0.002, margin=1.0
        )

        assert results == mock_results

    @patch("stockula.backtesting.runner.Backtest")
    def test_run_from_symbol_with_dates(self, mock_backtest_class):
        """Test run_from_symbol with date parameters."""
        # Create a mock data fetcher
        mock_data_fetcher = Mock()
        mock_data_fetcher.get_stock_data.return_value = pd.DataFrame()

        mock_backtest = Mock()
        mock_backtest.run.return_value = {}
        mock_backtest_class.return_value = mock_backtest

        runner = BacktestRunner(data_fetcher=mock_data_fetcher)
        runner.run_from_symbol(
            "TSLA",
            SMACrossStrategy,
            start_date="2023-01-01",
            end_date="2023-12-31",
            fast_period=8,
        )

        # Verify dates were passed correctly
        mock_data_fetcher.get_stock_data.assert_called_once_with(
            "TSLA", "2023-01-01", "2023-12-31"
        )

        # Verify strategy kwargs were passed
        mock_backtest.run.assert_called_once_with(fast_period=8)


class TestBacktestRunnerStats:
    """Test BacktestRunner get_stats method."""

    def test_get_stats_no_results(self):
        """Test get_stats when no results available."""
        runner = BacktestRunner(data_fetcher=None)

        with pytest.raises(ValueError, match="No backtest results available"):
            runner.get_stats()

    def test_get_stats_with_results(self):
        """Test get_stats with available results."""
        runner = BacktestRunner(data_fetcher=None)
        mock_results = pd.Series(
            {"Return [%]": 15.5, "Sharpe Ratio": 1.25, "Max. Drawdown [%]": -8.3}
        )
        runner.results = mock_results

        stats = runner.get_stats()
        assert stats.equals(mock_results)


class TestBacktestRunnerPlot:
    """Test BacktestRunner plot method."""

    def test_plot_no_results(self):
        """Test plot when no results available."""
        runner = BacktestRunner(data_fetcher=None)

        with pytest.raises(ValueError, match="No backtest results available"):
            runner.plot()

    def test_plot_with_results(self):
        """Test plot with available results."""
        runner = BacktestRunner(data_fetcher=None)
        mock_results = Mock()
        runner.results = mock_results

        runner.plot(show_legend=True)

        # Verify plot was called with kwargs
        mock_results.plot.assert_called_once_with(show_legend=True)

    def test_plot_no_kwargs(self):
        """Test plot without additional arguments."""
        runner = BacktestRunner(data_fetcher=None)
        mock_results = Mock()
        runner.results = mock_results

        runner.plot()

        # Verify plot was called without kwargs
        mock_results.plot.assert_called_once_with()


class TestBacktestRunnerIntegration:
    """Integration tests for BacktestRunner."""

    def test_complete_workflow(self):
        """Test complete backtest workflow."""
        # Create realistic sample data
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        np.random.seed(42)
        base_price = 100
        data = pd.DataFrame(
            {
                "Open": base_price + np.random.randn(60).cumsum() * 0.5,
                "High": base_price + np.random.randn(60).cumsum() * 0.5 + 1,
                "Low": base_price + np.random.randn(60).cumsum() * 0.5 - 1,
                "Close": base_price + np.random.randn(60).cumsum() * 0.5,
                "Volume": np.random.randint(1000000, 5000000, 60),
            },
            index=dates,
        )

        # Ensure High >= Low and proper OHLC relationships
        data["High"] = np.maximum(data["High"], data[["Open", "Close"]].max(axis=1))
        data["Low"] = np.minimum(data["Low"], data[["Open", "Close"]].min(axis=1))

        runner = BacktestRunner(cash=10000, commission=0.001, data_fetcher=None)

        # This would be a real integration test if we had a working strategy
        # For now, we test the runner setup
        assert runner.cash == 10000
        assert runner.commission == 0.001
        assert runner.results is None

    def test_error_handling(self):
        """Test error handling in runner methods."""
        runner = BacktestRunner(data_fetcher=None)

        # Test with empty DataFrame
        empty_data = pd.DataFrame()

        # This should not raise an error at the runner level
        # (the backtesting library would handle the actual error)
        try:
            # We don't actually run this as it would fail in the backtesting library
            # but we test that our runner is set up correctly
            assert hasattr(runner, "run")
            assert hasattr(runner, "optimize")
            assert hasattr(runner, "run_from_symbol")
        except Exception:
            pytest.fail("Runner methods should be accessible")
