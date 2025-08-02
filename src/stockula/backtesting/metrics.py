"""Advanced performance metrics for backtesting."""

import numpy as np
import pandas as pd


def calculate_dynamic_sharpe_ratio(
    returns: pd.Series, risk_free_rates: pd.Series, periods_per_year: int = 252
) -> float:
    """Calculate Sharpe ratio using dynamic risk-free rates.

    Args:
        returns: Daily returns series (as decimals, e.g., 0.01 for 1%)
        risk_free_rates: Daily risk-free rates series (as decimals, e.g., 0.05 for 5% annual)
        periods_per_year: Number of trading periods per year (252 for daily)

    Returns:
        Dynamic Sharpe ratio
    """
    # Handle timezone mismatch
    if hasattr(returns.index, "tz") and returns.index.tz is not None:
        returns = returns.copy()
        returns.index = returns.index.tz_localize(None)

    if hasattr(risk_free_rates.index, "tz") and risk_free_rates.index.tz is not None:
        risk_free_rates = risk_free_rates.copy()
        risk_free_rates.index = risk_free_rates.index.tz_localize(None)

    # Align the series by date
    aligned_data = pd.DataFrame({"returns": returns, "rf_rates": risk_free_rates}).dropna()

    if len(aligned_data) == 0:
        return np.nan

    # Convert annual risk-free rates to daily
    daily_rf_rates = aligned_data["rf_rates"] / periods_per_year

    # Calculate excess returns
    excess_returns = aligned_data["returns"] - daily_rf_rates

    # Calculate annualized excess return and volatility
    mean_excess_return = excess_returns.mean() * periods_per_year
    volatility = excess_returns.std() * np.sqrt(periods_per_year)

    # Handle zero or near-zero volatility
    if volatility == 0 or np.isclose(volatility, 0, atol=1e-10):
        return np.nan

    return mean_excess_return / volatility


def calculate_rolling_sharpe_ratio(
    returns: pd.Series,
    risk_free_rates: pd.Series,
    window: int = 252,
    periods_per_year: int = 252,
) -> pd.Series:
    """Calculate rolling Sharpe ratio using dynamic risk-free rates.

    Args:
        returns: Daily returns series
        risk_free_rates: Daily risk-free rates series
        window: Rolling window size in days
        periods_per_year: Number of trading periods per year

    Returns:
        Rolling Sharpe ratio series
    """
    # Handle timezone mismatch
    if hasattr(returns.index, "tz") and returns.index.tz is not None:
        returns = returns.copy()
        returns.index = returns.index.tz_localize(None)

    if hasattr(risk_free_rates.index, "tz") and risk_free_rates.index.tz is not None:
        risk_free_rates = risk_free_rates.copy()
        risk_free_rates.index = risk_free_rates.index.tz_localize(None)

    # Align the series
    aligned_data = pd.DataFrame({"returns": returns, "rf_rates": risk_free_rates})

    # Convert annual risk-free rates to daily
    aligned_data["daily_rf"] = aligned_data["rf_rates"] / periods_per_year

    # Calculate excess returns
    aligned_data["excess_returns"] = aligned_data["returns"] - aligned_data["daily_rf"]

    # Calculate rolling statistics
    rolling_mean = aligned_data["excess_returns"].rolling(window).mean() * periods_per_year
    rolling_std = aligned_data["excess_returns"].rolling(window).std() * np.sqrt(periods_per_year)

    # Calculate rolling Sharpe ratio
    rolling_sharpe = rolling_mean / rolling_std

    return rolling_sharpe


def calculate_sortino_ratio_dynamic(
    returns: pd.Series, risk_free_rates: pd.Series, periods_per_year: int = 252
) -> float:
    """Calculate Sortino ratio using dynamic risk-free rates.

    Args:
        returns: Daily returns series
        risk_free_rates: Daily risk-free rates series
        periods_per_year: Number of trading periods per year

    Returns:
        Dynamic Sortino ratio
    """
    # Handle timezone mismatch
    if hasattr(returns.index, "tz") and returns.index.tz is not None:
        returns = returns.copy()
        returns.index = returns.index.tz_localize(None)

    if hasattr(risk_free_rates.index, "tz") and risk_free_rates.index.tz is not None:
        risk_free_rates = risk_free_rates.copy()
        risk_free_rates.index = risk_free_rates.index.tz_localize(None)

    # Align the series
    aligned_data = pd.DataFrame({"returns": returns, "rf_rates": risk_free_rates}).dropna()

    if len(aligned_data) == 0:
        return np.nan

    # Convert annual risk-free rates to daily
    daily_rf_rates = aligned_data["rf_rates"] / periods_per_year

    # Calculate excess returns
    excess_returns = aligned_data["returns"] - daily_rf_rates

    # Calculate downside returns (only negative excess returns)
    downside_returns = excess_returns.clip(upper=0)

    # Calculate annualized excess return and downside deviation
    mean_excess_return = excess_returns.mean() * periods_per_year
    downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(periods_per_year)

    # Handle zero or near-zero downside deviation
    if downside_deviation == 0 or np.isclose(downside_deviation, 0, atol=1e-10):
        return np.nan

    return mean_excess_return / downside_deviation


def enhance_backtest_metrics(
    backtest_results: pd.Series,
    equity_curve: pd.Series,
    treasury_rates: pd.Series,
    periods_per_year: int = 252,
) -> pd.Series:
    """Enhance backtest results with dynamic risk-free rate metrics.

    Args:
        backtest_results: Original backtest results from backtesting.py
        equity_curve: Equity curve time series
        treasury_rates: Treasury rate time series (annual rates as decimals)
        periods_per_year: Number of trading periods per year

    Returns:
        Enhanced backtest results with additional metrics
    """
    # Ensure equity_curve is a pandas Series
    if not isinstance(equity_curve, pd.Series):
        raise ValueError("equity_curve must be a pandas Series")

    # Ensure treasury_rates is a pandas Series
    if not isinstance(treasury_rates, pd.Series):
        raise ValueError("treasury_rates must be a pandas Series")

    # Handle timezone mismatch between equity_curve and treasury_rates
    # Convert both to timezone-naive for consistent comparison
    if hasattr(equity_curve.index, "tz") and equity_curve.index.tz is not None:
        equity_curve = equity_curve.copy()
        equity_curve.index = equity_curve.index.tz_localize(None)

    if hasattr(treasury_rates.index, "tz") and treasury_rates.index.tz is not None:
        treasury_rates = treasury_rates.copy()
        treasury_rates.index = treasury_rates.index.tz_localize(None)

    # Calculate returns from equity curve
    returns = equity_curve.pct_change(fill_method=None).dropna()

    # Calculate dynamic metrics
    dynamic_sharpe = calculate_dynamic_sharpe_ratio(returns, treasury_rates, periods_per_year)
    dynamic_sortino = calculate_sortino_ratio_dynamic(returns, treasury_rates, periods_per_year)

    # Create enhanced results
    enhanced_results = backtest_results.copy()

    # Add dynamic metrics
    enhanced_results["Sharpe Ratio (Dynamic)"] = dynamic_sharpe
    enhanced_results["Sharpe Ratio (Static)"] = backtest_results.get("Sharpe Ratio", np.nan)
    enhanced_results["Sortino Ratio (Dynamic)"] = dynamic_sortino
    enhanced_results["Sortino Ratio (Static)"] = backtest_results.get("Sortino Ratio", np.nan)

    # Calculate average risk-free rate for the period
    avg_rf_rate = treasury_rates.mean()
    enhanced_results["Avg. Risk-Free Rate [%]"] = avg_rf_rate * 100
    enhanced_results["Risk-Free Rate Volatility [%]"] = treasury_rates.std() * 100

    return enhanced_results
