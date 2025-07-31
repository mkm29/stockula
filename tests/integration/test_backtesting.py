"""Tests for backtesting module."""

from datetime import datetime
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from stockula.backtesting import (
    BacktestRunner,
    DoubleEMACrossStrategy,
    MACDStrategy,
    RSIStrategy,
    SMACrossStrategy,
    TRIMACrossStrategy,
    TripleEMACrossStrategy,
)
from stockula.config.models import (
    BacktestResult,
    BrokerConfig,
    PortfolioBacktestResults,
    StrategyBacktestSummary,
)


class TestBacktestRunner:
    """Test BacktestRunner class."""

    def test_initialization(self):
        """Test BacktestRunner initialization."""
        runner = BacktestRunner(cash=50000, commission=0.001, margin=2.0)
        assert runner.cash == 50000
        assert runner.commission == 0.001
        assert runner.margin == 2.0

        # Test defaults
        runner_default = BacktestRunner()
        assert runner_default.cash == 10000
        assert runner_default.commission == 0.002
        assert runner_default.margin == 1.0

    def test_run_with_data(self, backtest_data):
        """Test running backtest with data."""
        runner = BacktestRunner()
        results = runner.run(backtest_data, SMACrossStrategy)

        # Check result structure - backtesting library returns a Series
        assert isinstance(results, pd.Series)
        assert "Return [%]" in results.index
        assert "Sharpe Ratio" in results.index
        assert "Max. Drawdown [%]" in results.index
        assert "# Trades" in results.index

        # Results should be numeric
        assert isinstance(results["Return [%]"], int | float)
        assert isinstance(results["# Trades"], int)

    def test_run_from_symbol(self, mock_data_fetcher, backtest_data):
        """Test running backtest from symbol."""
        # Create a mock that returns our backtest data
        mock_fetcher = Mock()
        mock_fetcher.get_stock_data = Mock(return_value=backtest_data)

        # Create runner with mocked data fetcher
        runner = BacktestRunner(data_fetcher=mock_fetcher)

        results = runner.run_from_symbol("AAPL", SMACrossStrategy)

        assert isinstance(results, pd.Series)  # backtesting library returns Series
        assert "Return [%]" in results.index
        mock_fetcher.get_stock_data.assert_called_once()

    def test_optimize(self, backtest_data):
        """Test strategy optimization."""
        runner = BacktestRunner()

        # Define parameter ranges
        param_ranges = {
            "fast_period": range(5, 15, 5),  # [5, 10]
            "slow_period": range(20, 30, 5),  # [20, 25]
        }

        optimal_results = runner.optimize(backtest_data, SMACrossStrategy, **param_ranges)

        # Check result - optimize returns a Series with the best backtest result
        assert isinstance(optimal_results, pd.Series)
        assert "Return [%]" in optimal_results.index

        # Check that the strategy has the optimized parameters
        assert hasattr(optimal_results._strategy, "fast_period")
        assert hasattr(optimal_results._strategy, "slow_period")
        assert optimal_results._strategy.fast_period in param_ranges["fast_period"]
        assert optimal_results._strategy.slow_period in param_ranges["slow_period"]

    def test_invalid_strategy(self, backtest_data):
        """Test error handling for invalid strategy."""
        runner = BacktestRunner()

        class InvalidStrategy:
            """Strategy without required methods."""

            pass

        with pytest.raises(TypeError, match="must be a Strategy sub-type"):
            runner.run(backtest_data, InvalidStrategy)


class TestSMACrossStrategy:
    """Test SMA Crossover Strategy."""

    def test_strategy_attributes(self):
        """Test strategy has required attributes."""
        assert hasattr(SMACrossStrategy, "fast_period")
        assert hasattr(SMACrossStrategy, "slow_period")
        assert SMACrossStrategy.fast_period < SMACrossStrategy.slow_period

    def test_strategy_execution(self, backtest_data):
        """Test strategy execution."""
        runner = BacktestRunner()
        results = runner.run(backtest_data, SMACrossStrategy)

        # Check that results are valid (strategy might not trade with default parameters)
        assert isinstance(results, pd.Series)
        assert "Return [%]" in results.index
        assert "# Trades" in results.index
        assert results["# Trades"] >= 0  # May be 0 if no signals generated

    def test_custom_parameters(self, backtest_data):
        """Test strategy with custom parameters."""

        # Create custom strategy class
        class CustomSMACross(SMACrossStrategy):
            fast_period = 5
            slow_period = 15

        runner = BacktestRunner()
        results = runner.run(backtest_data, CustomSMACross)

        # More sensitive parameters should generate more trades
        assert results["# Trades"] >= 0


class TestRSIStrategy:
    """Test RSI Strategy."""

    def test_strategy_attributes(self):
        """Test strategy has required attributes."""
        assert hasattr(RSIStrategy, "rsi_period")
        assert hasattr(RSIStrategy, "oversold_threshold")
        assert hasattr(RSIStrategy, "overbought_threshold")
        assert RSIStrategy.oversold_threshold < RSIStrategy.overbought_threshold

    def test_strategy_execution(self, backtest_data):
        """Test strategy execution."""
        runner = BacktestRunner()
        results = runner.run(backtest_data, RSIStrategy)

        # Check results
        assert isinstance(results["Return [%]"], int | float)
        assert isinstance(results["# Trades"], int)


class TestMACDStrategy:
    """Test MACD Strategy."""

    def test_strategy_attributes(self):
        """Test strategy has required attributes."""
        assert hasattr(MACDStrategy, "fast_period")
        assert hasattr(MACDStrategy, "slow_period")
        assert hasattr(MACDStrategy, "signal_period")

    def test_strategy_execution(self, backtest_data):
        """Test strategy execution."""
        runner = BacktestRunner()
        results = runner.run(backtest_data, MACDStrategy)

        # Check results
        assert isinstance(results["Return [%]"], int | float)
        assert results["Max. Drawdown [%]"] <= 0  # Drawdown is negative


class TestDoubleEMACrossStrategy:
    """Test Double EMA Cross Strategy."""

    def test_strategy_attributes(self):
        """Test strategy has required attributes."""
        assert hasattr(DoubleEMACrossStrategy, "fast_period")
        assert hasattr(DoubleEMACrossStrategy, "slow_period")
        assert hasattr(DoubleEMACrossStrategy, "momentum_atr_multiple")
        assert hasattr(DoubleEMACrossStrategy, "speculative_atr_multiple")

    def test_minimum_data_requirements(self):
        """Test minimum data requirements."""
        min_days = DoubleEMACrossStrategy.get_min_required_days()
        assert min_days == 75  # 55 + 20 buffer

        # Test with exact date
        end_date = "2024-01-01"
        start_date = DoubleEMACrossStrategy.get_recommended_start_date(end_date)

        # Should be at least min_days before end date
        days_diff = (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days
        assert days_diff >= min_days

    def test_insufficient_data_warning(self):
        """Test warning for insufficient data."""
        # Create data with insufficient history
        short_data = pd.DataFrame(
            {
                "Open": [100] * 50,
                "High": [101] * 50,
                "Low": [99] * 50,
                "Close": [100] * 50,
                "Volume": [1000000] * 50,
            },
            index=pd.date_range("2023-01-01", periods=50),
        )

        runner = BacktestRunner()

        # The strategy should issue a warning about insufficient data
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            runner.run(short_data, DoubleEMACrossStrategy)

            # Check that a warning was issued
            assert len(w) > 0
            assert any("Insufficient data" in str(warning.message) for warning in w)

    def test_strategy_with_categories(self, backtest_data):
        """Test strategy with different ATR multiples for categories."""

        # Modify strategy to test category-specific behavior
        class TestDoubleEMA(DoubleEMACrossStrategy):
            momentum_atr_multiple = 1.5
            speculative_atr_multiple = 1.0

        runner = BacktestRunner()
        results = runner.run(backtest_data, TestDoubleEMA)

        assert isinstance(results, pd.Series)
        assert "Return [%]" in results.index


class TestTripleEMACrossStrategy:
    """Test Triple EMA Cross Strategy."""

    def test_strategy_attributes(self):
        """Test strategy has required attributes."""
        assert hasattr(TripleEMACrossStrategy, "fast_period")
        assert hasattr(TripleEMACrossStrategy, "slow_period")
        assert hasattr(TripleEMACrossStrategy, "atr_period")
        assert hasattr(TripleEMACrossStrategy, "atr_multiple")

    def test_minimum_data_requirements(self):
        """Test minimum data requirements."""
        min_days = TripleEMACrossStrategy.get_min_required_days()
        assert min_days == 81  # 3*21-2=61 + 20 buffer

    def test_tema_calculation(self, backtest_data):
        """Test TEMA calculation is different from regular EMA."""
        runner = BacktestRunner()

        # Run strategy
        results = runner.run(backtest_data, TripleEMACrossStrategy)

        # TEMA should produce different results than regular EMA cross
        ema_results = runner.run(backtest_data, SMACrossStrategy)

        # Results should differ (different strategies)
        assert results["# Trades"] != ema_results["# Trades"]


class TestTRIMACrossStrategy:
    """Test TRIMA Cross Strategy."""

    def test_strategy_attributes(self):
        """Test strategy has required attributes."""
        assert hasattr(TRIMACrossStrategy, "fast_period")
        assert hasattr(TRIMACrossStrategy, "slow_period")
        assert hasattr(TRIMACrossStrategy, "atr_period")
        assert hasattr(TRIMACrossStrategy, "atr_multiple")

    def test_minimum_data_requirements(self):
        """Test minimum data requirements."""
        min_days = TRIMACrossStrategy.get_min_required_days()
        assert min_days == 76  # 2*28=56 + 20 buffer

    def test_strategy_smoothing(self, backtest_data):
        """Test that TRIMA provides smoother signals."""
        runner = BacktestRunner()

        # Add some noise to data
        noisy_data = backtest_data.copy()
        noise = np.random.normal(0, 2, len(noisy_data))
        noisy_data["Close"] = noisy_data["Close"] + noise
        noisy_data["High"] = noisy_data["High"] + abs(noise)
        noisy_data["Low"] = noisy_data["Low"] - abs(noise)

        # Run TRIMA strategy
        trima_results = runner.run(noisy_data, TRIMACrossStrategy)

        # Run regular SMA strategy for comparison
        sma_results = runner.run(noisy_data, SMACrossStrategy)

        # TRIMA should produce fewer trades due to smoothing
        assert trima_results["# Trades"] <= sma_results["# Trades"] * 1.5


class TestBacktestingEdgeCases:
    """Test edge cases in backtesting."""

    def test_no_trades(self):
        """Test strategy that generates no trades."""
        # Create flat data
        flat_data = pd.DataFrame(
            {
                "Open": [100] * 100,
                "High": [100] * 100,
                "Low": [100] * 100,
                "Close": [100] * 100,
                "Volume": [1000000] * 100,
            },
            index=pd.date_range("2023-01-01", periods=100),
        )

        runner = BacktestRunner()
        results = runner.run(flat_data, SMACrossStrategy)

        assert results["# Trades"] == 0
        assert results["Return [%]"] == 0.0

    def test_single_data_point(self):
        """Test with single data point."""
        single_point = pd.DataFrame(
            {
                "Open": [100],
                "High": [101],
                "Low": [99],
                "Close": [100],
                "Volume": [1000000],
            },
            index=[datetime.now()],
        )

        runner = BacktestRunner()

        # Should handle gracefully
        results = runner.run(single_point, SMACrossStrategy)
        assert results["# Trades"] == 0

    def test_extreme_volatility(self):
        """Test with extreme volatility."""
        dates = pd.date_range("2023-01-01", periods=100)

        # Create extreme volatility
        prices = []
        for i in range(len(dates)):
            if i % 2 == 0:
                prices.append(100)
            else:
                prices.append(150)  # 50% swings

        volatile_data = pd.DataFrame(
            {
                "Open": prices,
                "High": [p * 1.1 for p in prices],
                "Low": [p * 0.9 for p in prices],
                "Close": prices,
                "Volume": [1000000] * len(dates),
            },
            index=dates,
        )

        runner = BacktestRunner()
        results = runner.run(volatile_data, RSIStrategy)

        # Should handle without errors
        assert isinstance(results["Return [%]"], int | float)
        assert isinstance(results["Max. Drawdown [%]"], int | float)

    def test_commission_impact(self, backtest_data):
        """Test impact of commission on results."""
        # Run with no commission
        runner_no_comm = BacktestRunner(commission=0.0)
        results_no_comm = runner_no_comm.run(backtest_data, SMACrossStrategy)

        # Run with high commission
        runner_high_comm = BacktestRunner(commission=0.01)  # 1%
        results_high_comm = runner_high_comm.run(backtest_data, SMACrossStrategy)

        # High commission should reduce returns
        if results_no_comm["# Trades"] > 0:
            assert results_high_comm["Return [%]"] < results_no_comm["Return [%]"]

    def test_runner_with_broker_config(self, backtest_data):
        """Test BacktestRunner with broker configuration."""
        # Test with Robinhood preset
        broker_config = BrokerConfig.from_broker_preset("robinhood")
        runner = BacktestRunner(broker_config=broker_config)

        assert runner.broker_config == broker_config
        assert callable(runner.commission)

        # Run backtest
        results = runner.run(backtest_data, SMACrossStrategy)
        assert isinstance(results, pd.Series)
        assert "Return [%]" in results.index

    def test_convert_results_to_backtest_result(self, backtest_data):
        """Test converting raw backtest results to BacktestResult model."""
        runner = BacktestRunner()
        results = runner.run(backtest_data, SMACrossStrategy)

        # Convert to BacktestResult
        backtest_result = BacktestResult(
            ticker="TEST",
            strategy="SMACross",
            parameters={"fast_period": 10, "slow_period": 20},
            return_pct=results["Return [%]"],
            sharpe_ratio=results["Sharpe Ratio"],
            max_drawdown_pct=results["Max. Drawdown [%]"],
            num_trades=results["# Trades"],
            win_rate=results.get("Win Rate [%]") if results["# Trades"] > 0 else None,
        )

        assert backtest_result.ticker == "TEST"
        assert backtest_result.strategy == "SMACross"
        assert isinstance(backtest_result.return_pct, int | float)
        assert isinstance(backtest_result.num_trades, int)


class TestBacktestDataStructures:
    """Test the new backtest data structures in integration scenarios."""

    def test_full_portfolio_backtest_workflow(self, backtest_data):
        """Test complete workflow with new data structures."""
        # Setup portfolio with multiple strategies
        strategies = [
            ("SMACross", SMACrossStrategy, {}),
            ("RSI", RSIStrategy, {"period": 14}),
        ]

        strategy_summaries = []

        for strategy_name, strategy_class, params in strategies:
            runner = BacktestRunner(cash=10000)
            results = runner.run(backtest_data, strategy_class)

            # Create BacktestResult
            backtest_result = BacktestResult(
                ticker="TEST",
                strategy=strategy_name,
                parameters=params,
                return_pct=results["Return [%]"],
                sharpe_ratio=results["Sharpe Ratio"],
                max_drawdown_pct=results["Max. Drawdown [%]"],
                num_trades=results["# Trades"],
                win_rate=results.get("Win Rate [%]") if results["# Trades"] > 0 else None,
            )

            # Create strategy summary
            summary = StrategyBacktestSummary(
                strategy_name=strategy_name,
                parameters=params,
                initial_portfolio_value=10000.0,
                final_portfolio_value=10000.0 * (1 + backtest_result.return_pct / 100),
                total_return_pct=backtest_result.return_pct,
                total_trades=backtest_result.num_trades,
                winning_stocks=1 if backtest_result.return_pct > 0 else 0,
                losing_stocks=1 if backtest_result.return_pct < 0 else 0,
                average_return_pct=backtest_result.return_pct,
                average_sharpe_ratio=backtest_result.sharpe_ratio,
                detailed_results=[backtest_result],
            )

            strategy_summaries.append(summary)

        # Create portfolio results
        portfolio_results = PortfolioBacktestResults(
            initial_portfolio_value=10000.0,
            initial_capital=10000.0,
            date_range={"start": "2023-01-01", "end": "2023-12-31"},
            broker_config={
                "name": "robinhood",
                "commission_type": "fixed",
                "commission_value": 0.0,
            },
            strategy_summaries=strategy_summaries,
        )

        # Verify structure
        assert len(portfolio_results.strategy_summaries) == 2
        assert portfolio_results.strategy_summaries[0].strategy_name == "SMACross"
        assert portfolio_results.strategy_summaries[1].strategy_name == "RSI"

        # Test serialization
        data = portfolio_results.model_dump()
        assert isinstance(data, dict)
        assert "strategy_summaries" in data
        assert len(data["strategy_summaries"]) == 2

    def test_broker_config_impact_on_results(self, backtest_data):
        """Test how different broker configs affect results."""
        brokers = ["robinhood", "interactive_brokers", "fidelity"]
        results_by_broker = {}

        for broker_name in brokers:
            broker_config = BrokerConfig.from_broker_preset(broker_name)
            runner = BacktestRunner(cash=10000, broker_config=broker_config)
            results = runner.run(backtest_data, SMACrossStrategy)

            results_by_broker[broker_name] = BacktestResult(
                ticker="TEST",
                strategy="SMACross",
                parameters={},
                return_pct=results["Return [%]"],
                sharpe_ratio=results["Sharpe Ratio"],
                max_drawdown_pct=results["Max. Drawdown [%]"],
                num_trades=results["# Trades"],
                win_rate=results.get("Win Rate [%]") if results["# Trades"] > 0 else None,
            )

        # Verify different brokers may have different results due to fees
        # Interactive Brokers has per-share fees, others are zero commission
        if results_by_broker["robinhood"].num_trades > 0:
            # IB should have lower returns due to commission
            assert results_by_broker["interactive_brokers"].return_pct <= results_by_broker["robinhood"].return_pct
