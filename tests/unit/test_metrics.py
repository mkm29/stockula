"""Unit tests for backtesting metrics module."""

import numpy as np
import pandas as pd

from stockula.backtesting.metrics import (
    calculate_dynamic_sharpe_ratio,
    calculate_rolling_sharpe_ratio,
    calculate_sortino_ratio_dynamic,
    enhance_backtest_metrics,
)


class TestDynamicSharpeRatio:
    """Test dynamic Sharpe ratio calculation."""

    def test_calculate_dynamic_sharpe_ratio_basic(self):
        """Test basic dynamic Sharpe ratio calculation."""
        # Create sample data
        dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
        returns = pd.Series(np.random.normal(0.0005, 0.01, 252), index=dates)  # Daily returns
        risk_free_rates = pd.Series(0.04, index=dates)  # 4% annual rate

        sharpe = calculate_dynamic_sharpe_ratio(returns, risk_free_rates)

        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)

    def test_calculate_dynamic_sharpe_ratio_varying_rates(self):
        """Test with varying risk-free rates."""
        dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
        returns = pd.Series(np.random.normal(0.0005, 0.01, 252), index=dates)
        # Varying risk-free rates from 3% to 5%
        risk_free_rates = pd.Series(np.linspace(0.03, 0.05, 252), index=dates)

        sharpe = calculate_dynamic_sharpe_ratio(returns, risk_free_rates)

        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)

    def test_calculate_dynamic_sharpe_ratio_zero_volatility(self):
        """Test with zero volatility returns."""
        dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
        returns = pd.Series(0.001, index=dates)  # Constant returns
        risk_free_rates = pd.Series(0.04, index=dates)

        sharpe = calculate_dynamic_sharpe_ratio(returns, risk_free_rates)

        assert np.isnan(sharpe)  # Should return NaN for zero volatility

    def test_calculate_dynamic_sharpe_ratio_misaligned_dates(self):
        """Test with misaligned date ranges."""
        dates1 = pd.date_range(start="2023-01-01", periods=252, freq="D")
        dates2 = pd.date_range(start="2023-01-15", periods=252, freq="D")

        returns = pd.Series(np.random.normal(0.0005, 0.01, 252), index=dates1)
        risk_free_rates = pd.Series(0.04, index=dates2)

        sharpe = calculate_dynamic_sharpe_ratio(returns, risk_free_rates)

        # Should still calculate for overlapping dates
        assert isinstance(sharpe, float)

    def test_calculate_dynamic_sharpe_ratio_empty_data(self):
        """Test with empty data."""
        returns = pd.Series(dtype=float)
        risk_free_rates = pd.Series(dtype=float)

        sharpe = calculate_dynamic_sharpe_ratio(returns, risk_free_rates)

        assert np.isnan(sharpe)


class TestRollingSharpeRatio:
    """Test rolling Sharpe ratio calculation."""

    def test_calculate_rolling_sharpe_ratio(self):
        """Test basic rolling Sharpe ratio calculation."""
        dates = pd.date_range(start="2022-01-01", periods=504, freq="D")  # 2 years
        returns = pd.Series(np.random.normal(0.0005, 0.01, 504), index=dates)
        risk_free_rates = pd.Series(0.04, index=dates)

        rolling_sharpe = calculate_rolling_sharpe_ratio(returns, risk_free_rates, window=252)

        assert isinstance(rolling_sharpe, pd.Series)
        assert len(rolling_sharpe) == len(returns)
        # First 251 values should be NaN (window size - 1)
        assert rolling_sharpe.iloc[:251].isna().all()
        # Values after window should be calculated
        assert not rolling_sharpe.iloc[252:].isna().all()

    def test_calculate_rolling_sharpe_ratio_small_window(self):
        """Test with smaller window size."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        returns = pd.Series(np.random.normal(0.0005, 0.01, 100), index=dates)
        risk_free_rates = pd.Series(0.04, index=dates)

        rolling_sharpe = calculate_rolling_sharpe_ratio(returns, risk_free_rates, window=20)

        assert len(rolling_sharpe) == len(returns)
        assert rolling_sharpe.iloc[:19].isna().all()
        assert not rolling_sharpe.iloc[20:].isna().all()


class TestSortinoRatioDynamic:
    """Test dynamic Sortino ratio calculation."""

    def test_calculate_sortino_ratio_dynamic(self):
        """Test basic dynamic Sortino ratio calculation."""
        dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
        # Create returns with some negative values
        returns = pd.Series(np.random.normal(0.0005, 0.01, 252), index=dates)
        risk_free_rates = pd.Series(0.04, index=dates)

        sortino = calculate_sortino_ratio_dynamic(returns, risk_free_rates)

        assert isinstance(sortino, float)
        assert not np.isnan(sortino)

    def test_calculate_sortino_ratio_no_downside(self):
        """Test with no downside returns."""
        dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
        # All positive excess returns
        returns = pd.Series(0.002, index=dates)  # 0.2% daily returns
        risk_free_rates = pd.Series(0.04, index=dates)  # 4% annual

        sortino = calculate_sortino_ratio_dynamic(returns, risk_free_rates)

        # Should return NaN when no downside deviation
        assert np.isnan(sortino)


class TestEnhanceBacktestMetrics:
    """Test backtest metrics enhancement."""

    def test_enhance_backtest_metrics(self):
        """Test enhancing backtest results with dynamic metrics."""
        # Create mock backtest results
        original_results = pd.Series(
            {
                "Return [%]": 25.5,
                "Sharpe Ratio": 0.85,
                "Sortino Ratio": 1.2,
                "Max. Drawdown [%]": -15.3,
            }
        )

        # Create equity curve
        dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
        equity_curve = pd.Series(10000 * (1 + np.cumsum(np.random.normal(0.0005, 0.01, 252))), index=dates)

        # Create Treasury rates
        treasury_rates = pd.Series(np.linspace(0.03, 0.05, 252), index=dates)

        enhanced = enhance_backtest_metrics(original_results, equity_curve, treasury_rates)

        # Check that original metrics are preserved
        assert enhanced["Return [%]"] == 25.5
        assert enhanced["Max. Drawdown [%]"] == -15.3

        # Check that new metrics are added
        assert "Sharpe Ratio (Dynamic)" in enhanced
        assert "Sharpe Ratio (Static)" in enhanced
        assert "Sortino Ratio (Dynamic)" in enhanced
        assert "Sortino Ratio (Static)" in enhanced
        assert "Avg. Risk-Free Rate [%]" in enhanced
        assert "Risk-Free Rate Volatility [%]" in enhanced

        # Check that static values match original
        assert enhanced["Sharpe Ratio (Static)"] == 0.85
        assert enhanced["Sortino Ratio (Static)"] == 1.2

        # Check that average rate is calculated correctly
        expected_avg_rate = treasury_rates.mean() * 100
        assert abs(enhanced["Avg. Risk-Free Rate [%]"] - expected_avg_rate) < 0.001
