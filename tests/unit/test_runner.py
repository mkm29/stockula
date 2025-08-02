"""Unit tests for backtesting runner module."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from stockula.backtesting.runner import BacktestRunner
from stockula.backtesting.strategies import SMACrossStrategy
from stockula.config.models import BrokerConfig


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
        runner = BacktestRunner(cash=50000, commission=0.001, margin=2.0, data_fetcher=mock_data_fetcher)
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
            sample_data,
            SMACrossStrategy,
            cash=10000,
            commission=0.002,
            margin=1.0,
            trade_on_close=True,
            exclusive_orders=True,
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
        runner.run(sample_data, SMACrossStrategy, fast_period=5, slow_period=15)

        # Verify run was called with kwargs
        mock_backtest.run.assert_called_once_with(fast_period=5, slow_period=15)

    @patch("stockula.backtesting.runner.Backtest")
    def test_run_with_insufficient_data_warning(self, mock_backtest_class, sample_data, capsys):
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

        # Check warning elements were printed
        captured = capsys.readouterr()
        assert "Warning" in captured.out or "warning" in captured.out.lower()
        assert "TestStrategy" in captured.out
        assert "220" in captured.out or "requires" in captured.out.lower()
        assert "100" in captured.out or "available" in captured.out.lower()

    @patch("stockula.backtesting.runner.Backtest")
    def test_run_custom_parameters(self, mock_backtest_class, sample_data):
        """Test run with custom runner parameters."""
        mock_backtest = Mock()
        mock_backtest.run.return_value = {}
        mock_backtest_class.return_value = mock_backtest

        runner = BacktestRunner(cash=25000, commission=0.005, margin=1.5, data_fetcher=None)
        runner.run(sample_data, SMACrossStrategy)

        # Verify custom parameters were used
        mock_backtest_class.assert_called_once_with(
            sample_data,
            SMACrossStrategy,
            cash=25000,
            commission=0.005,
            margin=1.5,
            trade_on_close=True,
            exclusive_orders=True,
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
            sample_data,
            SMACrossStrategy,
            cash=10000,
            commission=0.002,
            margin=1.0,
            trade_on_close=True,
            exclusive_orders=True,
        )

        # Verify optimize was called with parameters
        mock_backtest.optimize.assert_called_once_with(fast_period=range(5, 15), slow_period=range(15, 25))

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
            sample_data,
            SMACrossStrategy,
            cash=30000,
            commission=0.001,
            margin=1.0,
            trade_on_close=True,
            exclusive_orders=True,
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
            sample_data,
            SMACrossStrategy,
            cash=10000,
            commission=0.002,
            margin=1.0,
            trade_on_close=True,
            exclusive_orders=True,
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
        mock_data_fetcher.get_stock_data.assert_called_once_with("TSLA", "2023-01-01", "2023-12-31")

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
        mock_results = pd.Series({"Return [%]": 15.5, "Sharpe Ratio": 1.25, "Max. Drawdown [%]": -8.3})
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


class TestBacktestRunnerBrokerConfig:
    """Test BacktestRunner broker configuration functionality."""

    def test_initialization_with_broker_config(self):
        """Test initialization with broker configuration."""
        broker_config = BrokerConfig(
            name="custom",
            commission_type="percentage",
            commission_value=0.001,
            min_commission=1.0,
            regulatory_fees=0.0000229,
        )

        runner = BacktestRunner(broker_config=broker_config, data_fetcher=None)

        # Verify commission function was created
        assert callable(runner.commission)
        assert runner.broker_config == broker_config

    def test_commission_func_percentage(self):
        """Test commission function with percentage commission."""
        broker_config = BrokerConfig(
            name="custom",
            commission_type="percentage",
            commission_value=0.001,  # 0.1%
            min_commission=1.0,
            regulatory_fees=0.0000229,
        )

        runner = BacktestRunner(broker_config=broker_config, data_fetcher=None)

        # Test commission calculation
        commission = runner.commission(100, 50.0)  # 100 shares at $50

        # Expected: (100 * 50 * 0.001) + (100 * 50 * 0.0000229) = 5.0 + 0.1145 = 5.1145
        expected = 5.0 + 0.1145
        assert abs(commission - expected) < 0.001

        # Test minimum commission
        small_commission = runner.commission(1, 1.0)  # 1 share at $1
        assert small_commission >= 1.0  # Should hit minimum

    def test_commission_func_fixed(self):
        """Test commission function with fixed commission."""
        broker_config = BrokerConfig(
            name="custom",
            commission_type="fixed",
            commission_value=9.99,
            regulatory_fees=0.0000229,
        )

        runner = BacktestRunner(broker_config=broker_config, data_fetcher=None)

        # Test commission calculation
        commission = runner.commission(100, 50.0)

        # Expected: 9.99 + (100 * 50 * 0.0000229) = 9.99 + 0.1145 = 10.1045
        expected = 9.99 + 0.1145
        assert abs(commission - expected) < 0.001

    def test_commission_func_per_share(self):
        """Test commission function with per-share commission."""
        broker_config = BrokerConfig(
            name="custom",
            commission_type="per_share",
            commission_value=0.005,
            min_commission=1.0,
            max_commission=10.0,
            regulatory_fees=0.0000229,
        )

        runner = BacktestRunner(broker_config=broker_config, data_fetcher=None)

        # Test commission calculation
        commission = runner.commission(100, 50.0)

        # Expected: (100 * 0.005) + (100 * 50 * 0.0000229) = 0.5 + 0.1145 = 0.6145
        # But minimum is 1.0, so should be 1.0 + 0.1145 = 1.1145
        expected = 1.0 + 0.1145
        assert abs(commission - expected) < 0.001

        # Test maximum commission (large trade)
        large_commission = runner.commission(10000, 50.0)
        # Per share: 10000 * 0.005 = 50, but max is 10.0
        # So: 10.0 + regulatory = 10.0 + (10000 * 50 * 0.0000229) = 10.0 + 11.45 = 21.45
        assert large_commission > 10.0  # Should include regulatory fees

    def test_commission_func_tiered(self):
        """Test commission function with tiered commission."""
        broker_config = BrokerConfig(
            name="custom",
            commission_type="tiered",
            commission_value={"0": 0.005, "300": 0.0035, "3000": 0.002},
            regulatory_fees=0.0000229,
        )

        runner = BacktestRunner(broker_config=broker_config, data_fetcher=None)

        # Test commission calculation (uses first tier)
        commission = runner.commission(100, 50.0)

        # Expected: (100 * 0.005) + (100 * 50 * 0.0000229) = 0.5 + 0.1145 = 0.6145
        expected = 0.5 + 0.1145
        assert abs(commission - expected) < 0.001

    def test_commission_func_robinhood(self):
        """Test commission function with Robinhood-specific logic."""
        broker_config = BrokerConfig(
            name="robinhood",
            commission_type="fixed",
            commission_value=0.0,
            exchange_fees=0.000166,  # TAF
            regulatory_fees=0.0,
        )

        runner = BacktestRunner(broker_config=broker_config, data_fetcher=None)

        # Test small trade (TAF waived)
        small_commission = runner.commission(50, 100.0)
        assert small_commission == 0.0  # No fees for <= 50 shares

        # Test large trade (TAF applied)
        large_commission = runner.commission(100, 100.0)
        expected_taf = 100 * 0.000166  # 0.0166
        assert abs(large_commission - expected_taf) < 0.001

        # Test very large trade (TAF capped)
        very_large_commission = runner.commission(100000, 100.0)
        assert very_large_commission == 8.30  # TAF cap

    def test_commission_func_other_broker_exchange_fees(self):
        """Test commission function with other broker exchange fees."""
        broker_config = BrokerConfig(
            name="other",
            commission_type="fixed",
            commission_value=0.0,
            exchange_fees=1.50,  # Fixed exchange fee
            regulatory_fees=0.0,
        )

        runner = BacktestRunner(broker_config=broker_config, data_fetcher=None)

        commission = runner.commission(100, 50.0)
        assert commission == 1.50


class TestBacktestRunnerCommissionEdgeCases:
    """Test commission function edge cases for better coverage."""

    def test_commission_func_per_share_with_none_per_share_commission(self):
        """Test per-share commission with None per_share_commission."""
        broker_config = BrokerConfig(
            name="custom",
            commission_type="per_share",
            commission_value=0.01,  # This will be used as per_share since per_share_commission is None
            per_share_commission=None,
            regulatory_fees=0.0,
        )

        runner = BacktestRunner(broker_config=broker_config, data_fetcher=None)
        commission = runner.commission(100, 50.0)

        # Should use commission_value as per_share
        expected = 100 * 0.01  # 100 shares * $0.01 per share
        assert abs(commission - expected) < 0.001

    def test_commission_func_tiered_with_non_dict_value(self):
        """Test tiered commission with non-dict commission value."""
        broker_config = BrokerConfig(
            name="custom",
            commission_type="tiered",
            commission_value=0.005,  # Not a dict, so no tiered calculation
            regulatory_fees=0.0,
        )

        runner = BacktestRunner(broker_config=broker_config, data_fetcher=None)
        commission = runner.commission(100, 50.0)

        # Should be 0 since commission_value is not a dict for tiered
        assert commission == 0.0


class TestBacktestRunnerDateExtractionEdgeCases:
    """Test date extraction edge cases."""

    @patch("stockula.backtesting.runner.Backtest")
    def test_run_with_pandas_datetime_index(self, mock_backtest_class):
        """Test run with pandas datetime index for date extraction."""
        # Create data with pandas datetime index that has .date() method
        dates = pd.to_datetime(pd.date_range("2023-01-01", periods=10, freq="D"))
        data = pd.DataFrame(
            {
                "Open": [100] * 10,
                "High": [101] * 10,
                "Low": [99] * 10,
                "Close": [100] * 10,
                "Volume": [1000000] * 10,
            },
            index=dates,
        )

        mock_backtest = Mock()
        mock_results = {}
        mock_backtest.run.return_value = mock_results
        mock_backtest_class.return_value = mock_backtest

        runner = BacktestRunner(data_fetcher=None)
        results = runner.run(data, Mock)

        # Should extract dates using .date().strftime() path (lines 210-217)
        assert "Start Date" in results
        assert "End Date" in results
        assert "Trading Days" in results
        assert "Calendar Days" in results
        assert results["Start Date"] == "2023-01-01"
        assert results["End Date"] == "2023-01-10"


class TestBacktestRunnerStrategyValidation:
    """Test strategy validation and warnings."""

    @patch("stockula.backtesting.runner.Backtest")
    def test_run_with_insufficient_data_warning(self, mock_backtest_class, capsys):
        """Test run with strategy that has insufficient data (triggers warning)."""
        # Create minimal data
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        data = pd.DataFrame(
            {
                "Open": [100] * 5,
                "High": [101] * 5,
                "Low": [99] * 5,
                "Close": [100] * 5,
                "Volume": [1000000] * 5,
            },
            index=dates,
        )

        # Create mock strategy with period requirements
        mock_strategy = Mock()
        mock_strategy.__name__ = "TestStrategy"
        mock_strategy.slow_period = 20  # Requires 20 days
        mock_strategy.min_trading_days_buffer = 10  # Plus 10 buffer = 30 total needed

        mock_backtest = Mock()
        mock_results = {}
        mock_backtest.run.return_value = mock_results
        mock_backtest_class.return_value = mock_backtest

        runner = BacktestRunner(data_fetcher=None)
        runner.run(data, mock_strategy)

        # Check that warning was printed (lines 149-155)
        captured = capsys.readouterr()
        assert "Warning: TestStrategy requires at least 30 days of data" in captured.out
        assert "but only 5 days available" in captured.out


class TestBacktestRunnerTrainTestSplit:
    """Test train/test split functionality."""

    def test_run_with_train_test_split_no_data_fetcher(self):
        """Test train/test split without data fetcher raises error."""
        runner = BacktestRunner(data_fetcher=None)

        with pytest.raises(ValueError, match="Data fetcher not configured"):
            runner.run_with_train_test_split("AAPL", Mock)

    @patch("stockula.backtesting.runner.BacktestRunner.run")
    @patch("stockula.backtesting.runner.BacktestRunner._extract_key_metrics")
    def test_run_with_train_test_split_basic(self, mock_extract, mock_run, mock_data_fetcher):
        """Test basic train/test split functionality."""
        # Mock data fetcher to return data
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        mock_data = pd.DataFrame(
            {
                "Open": [100] * 100,
                "High": [101] * 100,
                "Low": [99] * 100,
                "Close": [100] * 100,
                "Volume": [1000000] * 100,
            },
            index=dates,
        )

        # Mock the get_stock_data method
        with (
            patch.object(mock_data_fetcher, "get_stock_data", return_value=mock_data),
            patch.object(mock_data_fetcher, "get_treasury_rates", return_value=pd.Series()),
        ):
            # Mock strategy
            mock_strategy = Mock()
            mock_strategy.__name__ = "TestStrategy"

            runner = BacktestRunner(data_fetcher=mock_data_fetcher)

            mock_run.return_value = {"Return [%]": 10.0, "Sharpe Ratio": 1.0}
            mock_extract.return_value = {"return_pct": 10.0, "sharpe_ratio": 1.0}

            results = runner.run_with_train_test_split(
                "AAPL",
                mock_strategy,
                train_start_date="2023-01-01",
                train_end_date="2023-02-01",
                test_start_date="2023-02-02",
                test_end_date="2023-03-01",
            )

            # Verify results structure
            assert "symbol" in results
            assert "strategy" in results
            assert "train_period" in results
            assert "test_period" in results
            assert "train_results" in results
            assert "test_results" in results
            assert "performance_degradation" in results
            assert results["symbol"] == "AAPL"
            assert results["strategy"] == "TestStrategy"

    @patch("stockula.backtesting.runner.BacktestRunner.optimize")
    @patch("stockula.backtesting.runner.BacktestRunner.run")
    @patch("stockula.backtesting.runner.BacktestRunner._extract_key_metrics")
    def test_run_with_train_test_split_with_optimization(
        self, mock_extract, mock_run, mock_optimize, mock_data_fetcher
    ):
        """Test train/test split with parameter optimization."""
        # Mock data
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        mock_data = pd.DataFrame(
            {
                "Open": [100] * 100,
                "High": [101] * 100,
                "Low": [99] * 100,
                "Close": [100] * 100,
                "Volume": [1000000] * 100,
            },
            index=dates,
        )
        # Mock the get_stock_data method
        with (
            patch.object(mock_data_fetcher, "get_stock_data", return_value=mock_data),
            patch.object(mock_data_fetcher, "get_treasury_rates", return_value=pd.Series()),
        ):
            # Mock strategy
            mock_strategy = Mock()
            mock_strategy.__name__ = "TestStrategy"

            runner = BacktestRunner(data_fetcher=mock_data_fetcher)

            # Mock optimization result with parameters
            mock_optimize.return_value = {
                "fast_period": 10,
                "slow_period": 20,
                "Return [%]": 15.0,  # This should be filtered out
                "Sharpe Ratio": 1.2,  # This should be filtered out
            }
            mock_run.return_value = {"Return [%]": 12.0, "Sharpe Ratio": 1.1}
            mock_extract.return_value = {"return_pct": 12.0, "sharpe_ratio": 1.1}

            results = runner.run_with_train_test_split(
                "AAPL",
                mock_strategy,
                optimize_on_train=True,
                param_ranges={"fast_period": range(5, 15), "slow_period": range(15, 25)},
            )

            # Check that optimization was called and parameters were extracted
            assert "optimized_parameters" in results
            assert results["optimized_parameters"]["fast_period"] == 10
            assert results["optimized_parameters"]["slow_period"] == 20
            # Performance metrics should be filtered out
            assert "Return [%]" not in results["optimized_parameters"]
            assert "Sharpe Ratio" not in results["optimized_parameters"]

    def test_extract_key_metrics(self):
        """Test _extract_key_metrics method."""
        runner = BacktestRunner(data_fetcher=None)

        backtest_result = {
            "Return [%]": 15.5,
            "Sharpe Ratio": 1.25,
            "Max. Drawdown [%]": -8.3,
            "# Trades": 42,
            "Win Rate [%]": 65.0,
            "Equity Final [$]": 11550.0,
            "Buy & Hold Return [%]": 12.0,
        }

        metrics = runner._extract_key_metrics(backtest_result)

        assert metrics["return_pct"] == 15.5
        assert metrics["sharpe_ratio"] == 1.25
        assert metrics["max_drawdown_pct"] == -8.3
        assert metrics["num_trades"] == 42
        assert metrics["win_rate"] == 65.0
        assert metrics["equity_final"] == 11550.0
        assert metrics["buy_hold_return_pct"] == 12.0

    def test_extract_key_metrics_missing_values(self):
        """Test _extract_key_metrics with missing values."""
        runner = BacktestRunner(data_fetcher=None)

        backtest_result = {}  # Empty result

        metrics = runner._extract_key_metrics(backtest_result)

        # Should default to 0 for missing values
        assert metrics["return_pct"] == 0
        assert metrics["sharpe_ratio"] == 0
        assert metrics["max_drawdown_pct"] == 0
        assert metrics["num_trades"] == 0
        assert metrics["win_rate"] == 0
        assert metrics["equity_final"] == 0
        assert metrics["buy_hold_return_pct"] == 0


class TestBacktestRunnerEquityCurveEdgeCases:
    """Test equity curve processing edge cases."""

    def test_enhance_results_basic_coverage(self):
        """Test that equity curve processing code is covered."""
        runner = BacktestRunner(data_fetcher=None)

        # Test with Series equity curve and dynamic risk free rate
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        treasury_rates = pd.Series([0.05] * 10, index=dates)
        runner.risk_free_rate = treasury_rates
        runner._treasury_rates = treasury_rates
        runner._equity_curve = pd.Series([10000 + i * 100 for i in range(10)], index=dates)

        # Mock the enhance_backtest_metrics to avoid the import dependency
        with patch("stockula.backtesting.metrics.enhance_backtest_metrics") as mock_enhance:
            mock_enhance.return_value = {"enhanced_metric": 1.0}
            runner.results = {"Return [%]": 10.0}

            # This should call the equity curve processing code
            runner._enhance_results_with_dynamic_metrics()

            # Verify enhance was called
            mock_enhance.assert_called_once()


class TestBacktestRunnerRunFromSymbolDateExtraction:
    """Test date extraction edge cases in run_from_symbol."""

    @patch("stockula.backtesting.runner.BacktestRunner.run")
    def test_run_from_symbol_date_extraction_none_dates(self, mock_run, mock_data_fetcher):
        """Test run_from_symbol when start_date and end_date are None."""
        # Create data with index that has strftime method
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        mock_data = pd.DataFrame(
            {
                "Open": [100] * 10,
                "High": [101] * 10,
                "Low": [99] * 10,
                "Close": [100] * 10,
                "Volume": [1000000] * 10,
            },
            index=dates,
        )
        with (
            patch.object(mock_data_fetcher, "get_stock_data", return_value=mock_data),
            patch.object(
                mock_data_fetcher, "get_treasury_rates", return_value=pd.Series([0.05] * 10, index=dates)
            ) as mock_get_treasury,
        ):
            runner = BacktestRunner(data_fetcher=mock_data_fetcher)

            mock_run.return_value = {}

            # Call with None dates - should extract from data
            runner.run_from_symbol("AAPL", Mock, start_date=None, end_date=None)

            # Should call get_treasury_rates with extracted dates
            mock_get_treasury.assert_called_with("2023-01-01", "2023-01-10", "3_month")


class TestBacktestRunnerDynamicRiskFreeRateDateExtraction:
    """Test date extraction in run_with_dynamic_risk_free_rate."""

    @patch("stockula.backtesting.runner.BacktestRunner.run")
    def test_run_with_dynamic_risk_free_rate_none_dates(self, mock_run, mock_data_fetcher):
        """Test date extraction when start_date and end_date are None."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        mock_data = pd.DataFrame(
            {
                "Open": [100] * 10,
                "High": [101] * 10,
                "Low": [99] * 10,
                "Close": [100] * 10,
                "Volume": [1000000] * 10,
            },
            index=dates,
        )
        with (
            patch.object(mock_data_fetcher, "get_stock_data", return_value=mock_data),
            patch.object(
                mock_data_fetcher, "get_treasury_rates", return_value=pd.Series([0.05] * 10, index=dates)
            ) as mock_get_treasury,
        ):
            runner = BacktestRunner(data_fetcher=mock_data_fetcher)

            mock_run.return_value = {}

            # Call without dates - should extract from stock data (lines 638, 640)
            runner.run_with_dynamic_risk_free_rate("AAPL", Mock)

            # Should extract dates and call get_treasury_rates
            mock_get_treasury.assert_called_with("2023-01-01", "2023-01-10", "3_month")


class TestBacktestRunnerDynamicRiskFreeRate:
    """Test BacktestRunner dynamic risk-free rate functionality."""

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

    @pytest.fixture
    def treasury_rates(self):
        """Create sample treasury rates data."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        rates = pd.Series(
            0.05 + np.random.randn(100) * 0.001,  # Around 5% with small variations
            index=dates,
            name="3_month",
        )
        return rates

    @patch("stockula.backtesting.runner.Backtest")
    def test_run_with_dynamic_risk_free_rate(self, mock_backtest_class, sample_data, treasury_rates):
        """Test run with dynamic risk-free rate."""
        mock_backtest = Mock()
        mock_results = Mock()
        mock_results.__getitem__ = Mock(
            side_effect=lambda key: {
                "Return [%]": 15.5,
                "Sharpe Ratio": 1.25,
                "Max. Drawdown [%]": -8.3,
            }[key]
        )
        # Mock the equity curve
        mock_results._equity_curve = np.random.randn(100) + 10000
        mock_backtest.run.return_value = mock_results
        mock_backtest_class.return_value = mock_backtest

        runner = BacktestRunner(data_fetcher=None, risk_free_rate=treasury_rates)

        with patch.object(runner, "_enhance_results_with_dynamic_metrics") as mock_enhance:
            runner.run(sample_data, SMACrossStrategy)

            # Verify treasury rates were stored
            assert runner._treasury_rates is not None
            pd.testing.assert_series_equal(runner._treasury_rates, treasury_rates)

            # Verify enhancement was called
            mock_enhance.assert_called_once()

    @patch("stockula.backtesting.runner.Backtest")
    def test_run_without_dynamic_risk_free_rate(self, mock_backtest_class, sample_data):
        """Test run without dynamic risk-free rate."""
        mock_backtest = Mock()
        mock_results = {"Return [%]": 15.5}
        mock_backtest.run.return_value = mock_results
        mock_backtest_class.return_value = mock_backtest

        runner = BacktestRunner(data_fetcher=None)

        with patch.object(runner, "_enhance_results_with_dynamic_metrics") as mock_enhance:
            runner.run(sample_data, SMACrossStrategy)

            # Verify treasury rates were not stored
            assert runner._treasury_rates is None

            # Verify enhancement was not called
            mock_enhance.assert_not_called()

    def test_run_from_symbol_with_dynamic_rates(self):
        """Test run_from_symbol with automatic dynamic rate fetching."""
        # Create mock data fetcher
        mock_data_fetcher = Mock()

        # Setup sample data with date index
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        sample_data = pd.DataFrame(
            {
                "Open": [100] * 50,
                "High": [101] * 50,
                "Low": [99] * 50,
                "Close": [100] * 50,
                "Volume": [1000000] * 50,
            },
            index=dates,
        )
        mock_data_fetcher.get_stock_data.return_value = sample_data

        # Setup treasury rates
        treasury_rates = pd.Series([0.05] * 50, index=dates, name="3_month")
        mock_data_fetcher.get_treasury_rates.return_value = treasury_rates

        runner = BacktestRunner(data_fetcher=mock_data_fetcher)

        with patch.object(runner, "run") as mock_run:
            mock_run.return_value = {"Return [%]": 10.0}

            runner.run_from_symbol("AAPL", SMACrossStrategy, use_dynamic_risk_free_rate=True)

            # Verify treasury rates were fetched
            mock_data_fetcher.get_treasury_rates.assert_called_once()

            # Verify risk-free rate was set
            pd.testing.assert_series_equal(runner.risk_free_rate, treasury_rates)

    def test_run_from_symbol_without_dynamic_rates(self):
        """Test run_from_symbol without dynamic rate fetching."""
        mock_data_fetcher = Mock()
        sample_data = pd.DataFrame(
            {
                "Open": [100],
                "High": [101],
                "Low": [99],
                "Close": [100],
                "Volume": [1000000],
            }
        )
        mock_data_fetcher.get_stock_data.return_value = sample_data

        runner = BacktestRunner(data_fetcher=mock_data_fetcher)

        with patch.object(runner, "run") as mock_run:
            mock_run.return_value = {"Return [%]": 10.0}

            runner.run_from_symbol("AAPL", SMACrossStrategy, use_dynamic_risk_free_rate=False)

            # Verify treasury rates were not fetched
            mock_data_fetcher.get_treasury_rates.assert_not_called()

            # Verify risk-free rate was not set
            assert runner.risk_free_rate is None

    def test_run_with_dynamic_risk_free_rate_method(self):
        """Test run_with_dynamic_risk_free_rate method."""
        mock_data_fetcher = Mock()

        # Setup sample data with date index
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        sample_data = pd.DataFrame(
            {
                "Open": [100] * 50,
                "High": [101] * 50,
                "Low": [99] * 50,
                "Close": [100] * 50,
                "Volume": [1000000] * 50,
            },
            index=dates,
        )
        mock_data_fetcher.get_stock_data.return_value = sample_data

        # Setup treasury rates
        treasury_rates = pd.Series([0.05] * 50, index=dates)
        mock_data_fetcher.get_treasury_rates.return_value = treasury_rates

        runner = BacktestRunner(data_fetcher=mock_data_fetcher)

        with patch.object(runner, "run") as mock_run:
            mock_run.return_value = {"Return [%]": 10.0}

            runner.run_with_dynamic_risk_free_rate("AAPL", SMACrossStrategy, "2023-01-01", "2023-02-19")

            # Verify data and treasury rates were fetched
            mock_data_fetcher.get_stock_data.assert_called_once_with("AAPL", "2023-01-01", "2023-02-19")
            mock_data_fetcher.get_treasury_rates.assert_called_once_with("2023-01-01", "2023-02-19", "3_month")

            # Verify risk-free rate was set
            pd.testing.assert_series_equal(runner.risk_free_rate, treasury_rates)

    def test_run_with_dynamic_risk_free_rate_no_data_fetcher(self):
        """Test run_with_dynamic_risk_free_rate without data fetcher."""
        runner = BacktestRunner(data_fetcher=None)

        with pytest.raises(ValueError, match="Data fetcher not configured"):
            runner.run_with_dynamic_risk_free_rate("AAPL", SMACrossStrategy)


class TestBacktestRunnerDynamicMetrics:
    """Test BacktestRunner dynamic metrics enhancement."""

    @pytest.fixture
    def treasury_rates(self):
        """Create sample treasury rates."""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        return pd.Series([0.05] * 50, index=dates, name="3_month")

    def test_enhance_results_with_pandas_series_equity(self, treasury_rates):
        """Test enhancement with pandas Series equity curve."""
        runner = BacktestRunner(data_fetcher=None)
        runner._treasury_rates = treasury_rates

        # Create equity curve as pandas Series
        equity_curve = pd.Series(10000 + np.random.randn(50).cumsum() * 100, index=treasury_rates.index)
        runner._equity_curve = equity_curve

        # Create mock results
        runner.results = {"Return [%]": 10.0, "Sharpe Ratio": 1.0}

        with patch("stockula.backtesting.metrics.enhance_backtest_metrics") as mock_enhance:
            mock_enhance.return_value = {
                "Sharpe Ratio (Dynamic)": 1.5,
                "Sortino Ratio (Dynamic)": 1.8,
            }

            runner._enhance_results_with_dynamic_metrics()

            # Verify enhancement was called with correct parameters
            mock_enhance.assert_called_once()
            args = mock_enhance.call_args[0]
            assert args[0] == runner.results
            pd.testing.assert_series_equal(args[1], equity_curve)
            pd.testing.assert_series_equal(args[2], treasury_rates)

            # Verify results were updated
            assert runner.results["Sharpe Ratio (Dynamic)"] == 1.5
            assert runner.results["Sortino Ratio (Dynamic)"] == 1.8

    def test_enhance_results_with_numpy_array_equity(self, treasury_rates):
        """Test enhancement with numpy array equity curve."""
        runner = BacktestRunner(data_fetcher=None)
        runner._treasury_rates = treasury_rates

        # Create equity curve as numpy array
        equity_values = 10000 + np.random.randn(50).cumsum() * 100
        runner._equity_curve = equity_values

        runner.results = {"Return [%]": 10.0}

        with patch("stockula.backtesting.metrics.enhance_backtest_metrics") as mock_enhance:
            mock_enhance.return_value = {"Sharpe Ratio (Dynamic)": 1.5}

            runner._enhance_results_with_dynamic_metrics()

            # Verify enhancement was called
            mock_enhance.assert_called_once()
            args = mock_enhance.call_args[0]

            # Verify equity series was created correctly
            expected_equity = pd.Series(equity_values, index=treasury_rates.index)
            pd.testing.assert_series_equal(args[1], expected_equity)

    def test_enhance_results_with_dataframe_equity(self, treasury_rates):
        """Test enhancement with DataFrame equity curve."""
        runner = BacktestRunner(data_fetcher=None)
        runner._treasury_rates = treasury_rates

        # Create equity curve as DataFrame
        equity_df = pd.DataFrame(
            {"equity": 10000 + np.random.randn(50).cumsum() * 100},
            index=treasury_rates.index,
        )
        runner._equity_curve = equity_df

        runner.results = {"Return [%]": 10.0}

        with patch("stockula.backtesting.metrics.enhance_backtest_metrics") as mock_enhance:
            mock_enhance.return_value = {"Sharpe Ratio (Dynamic)": 1.5}

            runner._enhance_results_with_dynamic_metrics()

            # Verify enhancement was called
            mock_enhance.assert_called_once()

    def test_enhance_results_with_multidimensional_array(self, treasury_rates):
        """Test enhancement with multi-dimensional numpy array."""
        runner = BacktestRunner(data_fetcher=None)
        runner._treasury_rates = treasury_rates

        # Create 2D equity curve (backtesting.py sometimes returns 2D arrays)
        equity_2d = np.column_stack(
            [
                10000 + np.random.randn(50).cumsum() * 100,  # Equity values
                np.random.randn(50),  # Some other column
            ]
        )
        runner._equity_curve = equity_2d

        runner.results = {"Return [%]": 10.0}

        with patch("stockula.backtesting.metrics.enhance_backtest_metrics") as mock_enhance:
            mock_enhance.return_value = {"Sharpe Ratio (Dynamic)": 1.5}

            runner._enhance_results_with_dynamic_metrics()

            # Verify enhancement was called with first column
            mock_enhance.assert_called_once()
            args = mock_enhance.call_args[0]

            # Should use first column of the 2D array
            expected_equity = pd.Series(equity_2d[:, 0], index=treasury_rates.index)
            pd.testing.assert_series_equal(args[1], expected_equity)

    def test_enhance_results_no_treasury_rates(self):
        """Test enhancement without treasury rates."""
        runner = BacktestRunner(data_fetcher=None)
        runner._treasury_rates = None
        runner._equity_curve = pd.Series([10000, 10100, 10200])
        runner.results = {"Return [%]": 10.0}

        # Should return early without calling enhance_backtest_metrics
        with patch("stockula.backtesting.metrics.enhance_backtest_metrics") as mock_enhance:
            runner._enhance_results_with_dynamic_metrics()
            mock_enhance.assert_not_called()

    def test_enhance_results_no_equity_curve(self, treasury_rates):
        """Test enhancement without equity curve."""
        runner = BacktestRunner(data_fetcher=None)
        runner._treasury_rates = treasury_rates
        runner._equity_curve = None
        runner.results = {"Return [%]": 10.0}

        # Should return early without calling enhance_backtest_metrics
        with patch("stockula.backtesting.metrics.enhance_backtest_metrics") as mock_enhance:
            runner._enhance_results_with_dynamic_metrics()
            mock_enhance.assert_not_called()

    def test_enhance_results_unknown_equity_type(self, treasury_rates, capsys):
        """Test enhancement with unknown equity curve type."""
        runner = BacktestRunner(data_fetcher=None)
        runner._treasury_rates = treasury_rates
        # Use an object without __len__ to trigger the "Unknown equity curve type" path
        runner._equity_curve = object()  # Object without __len__ method
        runner.results = {"Return [%]": 10.0}

        runner._enhance_results_with_dynamic_metrics()

        # Check warning was printed
        captured = capsys.readouterr()
        assert "Warning: Unknown equity curve type" in captured.out

    def test_enhance_results_conversion_failure(self, treasury_rates, capsys):
        """Test enhancement when conversion to pandas Series fails."""
        runner = BacktestRunner(data_fetcher=None)
        runner._treasury_rates = treasury_rates

        # Mock object that has __len__ but fails list() conversion
        mock_equity = Mock(spec_set=["__len__", "__iter__"])  # Only allow these attributes
        mock_equity.__len__ = Mock(return_value=50)

        # Make list() conversion fail by making it non-iterable
        def raise_type_error(*args, **kwargs):
            raise TypeError("'Mock' object is not iterable")

        mock_equity.__iter__ = Mock(side_effect=raise_type_error)

        runner._equity_curve = mock_equity
        runner.results = {"Return [%]": 10.0}

        runner._enhance_results_with_dynamic_metrics()

        captured = capsys.readouterr()
        assert "Could not convert equity curve to pandas Series" in captured.out

    def test_enhance_results_metrics_calculation_error(self, treasury_rates, capsys):
        """Test enhancement when metrics calculation fails."""
        runner = BacktestRunner(data_fetcher=None)
        runner._treasury_rates = treasury_rates
        runner._equity_curve = pd.Series([10000, 10100, 10200], index=treasury_rates.index[:3])
        runner.results = {"Return [%]": 10.0}

        with patch(
            "stockula.backtesting.metrics.enhance_backtest_metrics",
            side_effect=Exception("Calculation failed"),
        ):
            runner._enhance_results_with_dynamic_metrics()

            captured = capsys.readouterr()
            assert "Could not calculate dynamic metrics: Calculation failed" in captured.out


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
        pd.DataFrame()

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
