"""Unit tests for BacktestOptimizedAllocator."""

from unittest.mock import Mock

import pandas as pd
import pytest

from stockula.allocation import BacktestOptimizedAllocator
from stockula.backtesting.strategies import MACDStrategy, RSIStrategy, SMACrossStrategy
from stockula.config import PortfolioConfig, StockulaConfig, TickerConfig


class TestBacktestOptimizedAllocator:
    """Test BacktestOptimizedAllocator functionality."""

    @pytest.fixture
    def mock_fetcher(self):
        """Create a mock data fetcher."""
        fetcher = Mock()
        return fetcher

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        logger = Mock()
        return logger

    @pytest.fixture
    def mock_backtest_runner(self):
        """Create a mock backtest runner."""
        runner = Mock()
        return runner

    @pytest.fixture
    def allocator(self, mock_fetcher, mock_logger, mock_backtest_runner):
        """Create allocator instance with mocks."""
        allocator = BacktestOptimizedAllocator(
            fetcher=mock_fetcher,
            logging_manager=mock_logger,
            backtest_runner=mock_backtest_runner,
        )
        return allocator

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration."""
        config = StockulaConfig(
            portfolio=PortfolioConfig(
                initial_capital=100000.0,
                allow_fractional_shares=True,
            )
        )
        return config

    @pytest.fixture
    def sample_tickers(self):
        """Create sample ticker configurations."""
        return [
            TickerConfig(symbol="AAPL", category="TECH"),
            TickerConfig(symbol="GOOGL", category="TECH"),
            TickerConfig(symbol="MSFT", category="TECH"),
        ]

    def test_find_best_strategies(self, allocator, mock_fetcher, mock_backtest_runner):
        """Test finding best strategies for each symbol."""
        # Setup mock data
        sample_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [102, 103, 104],
                "Low": [99, 100, 101],
                "Close": [101, 102, 103],
                "Volume": [1000000, 1100000, 1200000],
            },
            index=pd.date_range("2023-01-01", periods=3),
        )
        mock_fetcher.get_stock_data.return_value = sample_data

        # Mock backtest results - different strategies perform differently
        def mock_run(data, strategy):
            if strategy == SMACrossStrategy:
                return pd.Series({"Sharpe Ratio": 1.2, "Return [%]": 10.0})
            elif strategy == RSIStrategy:
                return pd.Series({"Sharpe Ratio": 1.5, "Return [%]": 12.0})
            elif strategy == MACDStrategy:
                return pd.Series({"Sharpe Ratio": 0.8, "Return [%]": 8.0})
            else:
                return pd.Series({"Sharpe Ratio": 1.0, "Return [%]": 9.0})

        mock_backtest_runner.run.side_effect = mock_run

        # Set default ranking metric (Return [%])
        allocator.ranking_metric = "Return [%]"

        # Run the method
        symbols = ["AAPL", "GOOGL"]
        best_strategies = allocator._find_best_strategies(symbols, "2023-01-01", "2023-06-30")

        # Verify results
        assert len(best_strategies) == 2
        # RSIStrategy should be best for both since it has highest Return [%] (12%)
        assert best_strategies["AAPL"] == RSIStrategy
        assert best_strategies["GOOGL"] == RSIStrategy

    def test_find_best_strategies_with_sharpe_ratio(self, allocator, mock_fetcher, mock_backtest_runner):
        """Test finding best strategies using Sharpe Ratio metric."""
        # Setup mock data
        sample_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [102, 103, 104],
                "Low": [99, 100, 101],
                "Close": [101, 102, 103],
                "Volume": [1000000, 1100000, 1200000],
            },
            index=pd.date_range("2023-01-01", periods=3),
        )
        mock_fetcher.get_stock_data.return_value = sample_data

        # Mock backtest results - SMACross has better Sharpe but lower return
        def mock_run(data, strategy):
            if strategy == SMACrossStrategy:
                return pd.Series({"Sharpe Ratio": 1.8, "Return [%]": 9.0})
            elif strategy == RSIStrategy:
                return pd.Series({"Sharpe Ratio": 1.5, "Return [%]": 12.0})
            elif strategy == MACDStrategy:
                return pd.Series({"Sharpe Ratio": 0.8, "Return [%]": 8.0})
            else:
                return pd.Series({"Sharpe Ratio": 1.0, "Return [%]": 10.0})

        mock_backtest_runner.run.side_effect = mock_run

        # Set ranking metric to Sharpe Ratio
        allocator.ranking_metric = "Sharpe Ratio"

        # Run the method
        symbols = ["AAPL", "GOOGL"]
        best_strategies = allocator._find_best_strategies(symbols, "2023-01-01", "2023-06-30")

        # Verify results
        assert len(best_strategies) == 2
        # SMACrossStrategy should be best for both since it has highest Sharpe Ratio (1.8)
        assert best_strategies["AAPL"] == SMACrossStrategy
        assert best_strategies["GOOGL"] == SMACrossStrategy

    def test_evaluate_test_performance(self, allocator, mock_fetcher, mock_backtest_runner):
        """Test evaluating performance on test data."""
        # Setup mock data
        sample_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [102, 103, 104],
                "Low": [99, 100, 101],
                "Close": [101, 102, 103],
                "Volume": [1000000, 1100000, 1200000],
            },
            index=pd.date_range("2023-07-01", periods=3),
        )
        mock_fetcher.get_stock_data.return_value = sample_data

        # Mock backtest results
        mock_backtest_runner.run.return_value = pd.Series({"Sharpe Ratio": 1.3, "Return [%]": 11.0})

        # Set default ranking metric
        allocator.ranking_metric = "Return [%]"

        # Setup best strategies
        best_strategies = {
            "AAPL": SMACrossStrategy,
            "GOOGL": RSIStrategy,
        }

        # Run the method
        performances = allocator._evaluate_test_performance(
            ["AAPL", "GOOGL"], best_strategies, "2023-07-01", "2023-12-31"
        )

        # Verify results
        assert len(performances) == 2
        assert performances["AAPL"] == 11.0  # Return [%] value
        assert performances["GOOGL"] == 11.0  # Return [%] value

    def test_calculate_performance_based_allocations(self, allocator):
        """Test calculating allocations based on performance."""
        # Set allocation constraints
        allocator.min_allocation_pct = 2.0
        allocator.max_allocation_pct = 40.0

        # Test with more assets to avoid hitting max allocation limits
        performances = {
            "AAPL": 3.0,  # Best performer
            "GOOGL": 2.5,  # Second best
            "MSFT": 2.0,  # Third
            "AMZN": 1.5,  # Fourth
            "META": 1.0,  # Fifth
            "TSLA": -0.5,  # Negative performer
        }

        allocations = allocator._calculate_performance_based_allocations(
            ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA"], performances, 100000
        )

        # Verify allocations sum to 100%
        assert abs(sum(allocations.values()) - 100.0) < 0.01

        # Verify relative allocations (with more assets, less likely to hit max cap)
        assert allocations["AAPL"] > allocations["GOOGL"]
        assert allocations["GOOGL"] > allocations["MSFT"]
        assert allocations["MSFT"] > allocations["META"]
        # TSLA should be close to min_allocation_pct (might be slightly different due to normalization)
        assert abs(allocations["TSLA"] - allocator.min_allocation_pct) < 0.1

        # Verify constraints (with small tolerance for normalization)
        for _symbol, pct in allocations.items():
            assert pct >= allocator.min_allocation_pct - 0.1  # Allow small tolerance
            assert pct <= allocator.max_allocation_pct

    def test_calculate_performance_based_allocations_no_positive(self, allocator):
        """Test allocation when no positive performers."""
        performances = {
            "AAPL": -1.0,
            "GOOGL": -0.5,
            "MSFT": 0.0,
        }

        allocations = allocator._calculate_performance_based_allocations(
            ["AAPL", "GOOGL", "MSFT"], performances, 100000
        )

        # Should use equal allocation
        expected_pct = 100.0 / 3
        for _symbol, pct in allocations.items():
            assert abs(pct - expected_pct) < 0.01

    def test_convert_allocations_to_quantities(self, allocator, sample_config, mock_fetcher):
        """Test converting allocation percentages to quantities."""

        # Mock price data
        def mock_get_stock_data(symbol, start, end):
            prices = {"AAPL": 150.0, "GOOGL": 2800.0, "MSFT": 300.0}
            return pd.DataFrame(
                {"Close": [prices.get(symbol, 100.0)]},
                index=pd.date_range(start, periods=1),
            )

        mock_fetcher.get_stock_data.side_effect = mock_get_stock_data

        # Setup allocations
        allocations = {
            "AAPL": 40.0,  # 40% = $40,000
            "GOOGL": 35.0,  # 35% = $35,000
            "MSFT": 25.0,  # 25% = $25,000
        }

        # Calculate quantities
        quantities = allocator._convert_allocations_to_quantities(
            sample_config, ["AAPL", "GOOGL", "MSFT"], allocations, "2023-12-31"
        )

        # Verify quantities are integers
        assert quantities["AAPL"] == 266  # int(40000 / 150)
        assert quantities["GOOGL"] == 12  # int(35000 / 2800)
        assert quantities["MSFT"] == 83  # int(25000 / 300)

    def test_convert_allocations_to_quantities_integer_shares(self, allocator, mock_fetcher):
        """Test quantity conversion always returns integers."""
        # Create config with fractional shares enabled
        config = StockulaConfig(
            portfolio=PortfolioConfig(
                initial_capital=100000.0,
                allow_fractional_shares=True,  # Even with fractional enabled
            )
        )

        # Mock price data
        mock_fetcher.get_stock_data.return_value = pd.DataFrame(
            {"Close": [150.0]}, index=pd.date_range("2023-12-31", periods=1)
        )

        allocations = {"AAPL": 100.0}  # 100% allocation

        quantities = allocator._convert_allocations_to_quantities(config, ["AAPL"], allocations, "2023-12-31")

        # Should always be integer for backtest_optimized
        assert quantities["AAPL"] == 666  # int(100000 / 150)
        assert isinstance(quantities["AAPL"], int)

    def test_calculate_backtest_optimized_quantities_integration(
        self, allocator, sample_config, sample_tickers, mock_fetcher, mock_backtest_runner
    ):
        """Test the full optimization process."""
        # Mock data fetching
        sample_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [102, 103, 104],
                "Low": [99, 100, 101],
                "Close": [101, 102, 103],
                "Volume": [1000000, 1100000, 1200000],
            },
            index=pd.date_range("2023-01-01", periods=3),
        )
        mock_fetcher.get_stock_data.return_value = sample_data

        # Mock backtest results

        def mock_run(data, strategy):
            # Determine which symbol based on call count
            call_count = mock_backtest_runner.run.call_count
            # This is a simplified mock - in reality would need more sophisticated logic
            return pd.Series({"Sharpe Ratio": 1.0 + (call_count % 3) * 0.1})

        mock_backtest_runner.run.side_effect = mock_run

        # Run the full process
        quantities = allocator.calculate_backtest_optimized_quantities(
            sample_config,
            sample_tickers,
            "2023-01-01",
            "2023-06-30",
            "2023-07-01",
            "2023-12-31",
        )

        # Verify we got quantities for all symbols
        assert len(quantities) == 3
        assert all(symbol in quantities for symbol in ["AAPL", "GOOGL", "MSFT"])
        assert all(qty >= 0 for qty in quantities.values())

    def test_error_handling_no_data(self, allocator, mock_fetcher, mock_backtest_runner):
        """Test handling when no data is available."""
        # Mock empty data
        mock_fetcher.get_stock_data.return_value = pd.DataFrame()

        # Should use default strategy
        best_strategies = allocator._find_best_strategies(["AAPL"], "2023-01-01", "2023-06-30")

        assert best_strategies["AAPL"] == SMACrossStrategy

    def test_error_handling_backtest_failure(self, allocator, mock_fetcher, mock_backtest_runner):
        """Test handling when backtest fails."""
        # Mock data
        sample_data = pd.DataFrame({"Close": [100, 101, 102]}, index=pd.date_range("2023-01-01", periods=3))
        mock_fetcher.get_stock_data.return_value = sample_data

        # Mock backtest failure
        mock_backtest_runner.run.side_effect = Exception("Backtest failed")

        # Should handle gracefully
        performances = allocator._evaluate_test_performance(
            ["AAPL"], {"AAPL": SMACrossStrategy}, "2023-07-01", "2023-12-31"
        )

        assert performances["AAPL"] == 0.0

    def test_calculate_with_config(self, allocator, mock_fetcher, mock_backtest_runner):
        """Test calculation with configuration from config file."""
        from stockula.config import BacktestOptimizationConfig

        # Create a config with backtest optimization settings
        config = StockulaConfig(
            portfolio=PortfolioConfig(
                initial_capital=100000.0,
                allocation_method="backtest_optimized",
                allow_fractional_shares=True,
            ),
            backtest_optimization=BacktestOptimizationConfig(
                train_start_date="2023-01-01",
                train_end_date="2023-06-30",
                test_start_date="2023-07-01",
                test_end_date="2023-12-31",
                ranking_metric="Sharpe Ratio",
                min_allocation_pct=5.0,
                max_allocation_pct=30.0,
                initial_allocation_pct=10.0,
            ),
        )

        # Mock data
        sample_data = pd.DataFrame({"Close": [100, 101, 102]}, index=pd.date_range("2023-01-01", periods=3))
        mock_fetcher.get_stock_data.return_value = sample_data

        # Mock backtest results
        mock_backtest_runner.run.return_value = pd.Series({"Sharpe Ratio": 1.5, "Return [%]": 10.0})

        # Run with config
        quantities = allocator.calculate_backtest_optimized_quantities(
            config=config,
            tickers_to_add=[
                TickerConfig(symbol="AAPL", category="TECH"),
                TickerConfig(symbol="GOOGL", category="TECH"),
            ],
        )

        # Verify that config settings were used
        assert allocator.ranking_metric == "Sharpe Ratio"
        assert allocator.min_allocation_pct == 5.0
        assert allocator.max_allocation_pct == 30.0

        # Verify we got quantities
        assert len(quantities) == 2
        assert all(qty >= 0 for qty in quantities.values())
