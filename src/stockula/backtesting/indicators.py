"""Technical indicators used by trading strategies.

This module contains standalone implementations of technical indicators
that can be tested independently of the backtesting framework.
"""

import numpy as np
import pandas as pd


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average.

    Args:
        prices: Series of prices
        period: Number of periods for the average

    Returns:
        Series with SMA values
    """
    return prices.rolling(window=period).mean()


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average.

    Args:
        prices: Series of prices
        period: Number of periods for the average

    Returns:
        Series with EMA values
    """
    return prices.ewm(span=period, adjust=False).mean()


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index.

    Args:
        prices: Series of prices
        period: Number of periods for RSI calculation

    Returns:
        Series with RSI values (0-100)
    """
    prices = pd.Series(prices)
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence).

    Args:
        prices: Series of prices
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line EMA period

    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range.

    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        period: Number of periods for ATR

    Returns:
        Series with ATR values
    """
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_tema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Triple Exponential Moving Average.

    Args:
        prices: Series of prices
        period: Number of periods

    Returns:
        Series with TEMA values
    """
    ema1 = prices.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()

    tema = 3 * ema1 - 3 * ema2 + ema3
    return tema


def calculate_trima(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Triangular Moving Average.

    Args:
        prices: Series of prices
        period: Number of periods

    Returns:
        Series with TRIMA values
    """
    if period % 2 == 0:
        first_period = period // 2
        second_period = first_period + 1
    else:
        first_period = second_period = (period + 1) // 2

    sma_first = prices.rolling(window=first_period).mean()
    trima = sma_first.rolling(window=second_period).mean()
    return trima


def calculate_vidya(prices: pd.Series, cmo_period: int = 9, smoothing_period: int = 12) -> pd.Series:
    """Calculate Variable Index Dynamic Average.

    Args:
        prices: Series of prices
        cmo_period: Period for CMO calculation
        smoothing_period: Smoothing period

    Returns:
        Series with VIDYA values
    """
    prices = pd.Series(prices)

    # Calculate price changes
    changes = prices.diff()

    # Calculate CMO (Chande Momentum Oscillator)
    positive_sum = changes.where(changes > 0, 0).rolling(window=cmo_period).sum()
    negative_sum = abs(changes.where(changes < 0, 0).rolling(window=cmo_period).sum())
    total_sum = positive_sum + negative_sum

    # Avoid division by zero
    cmo = pd.Series(0.0, index=prices.index)
    mask = total_sum != 0
    cmo[mask] = ((positive_sum[mask] - negative_sum[mask]) / total_sum[mask]) * 100

    # Calculate Volatility Index (VI)
    vi = abs(cmo) / 100

    # Calculate adaptive alpha
    alpha = (2 / (smoothing_period + 1)) * vi

    # Initialize VIDYA with SMA
    vidya_values = prices.rolling(window=smoothing_period).mean()

    # Calculate VIDYA using adaptive alpha
    for i in range(smoothing_period, len(prices)):
        if pd.notna(vidya_values.iloc[i - 1]) and pd.notna(alpha.iloc[i]):
            vidya_values.iloc[i] = alpha.iloc[i] * prices.iloc[i] + (1 - alpha.iloc[i]) * vidya_values.iloc[i - 1]

    return vidya_values


def calculate_kama(prices: pd.Series, er_period: int = 10, fast_period: int = 2, slow_period: int = 30) -> pd.Series:
    """Calculate Kaufman's Adaptive Moving Average.

    Args:
        prices: Series of prices
        er_period: Efficiency ratio period
        fast_period: Fast EMA period
        slow_period: Slow EMA period

    Returns:
        Series with KAMA values
    """
    prices = pd.Series(prices)

    # Calculate Efficiency Ratio
    change = abs(prices - prices.shift(er_period))
    volatility = prices.diff().abs().rolling(window=er_period).sum()

    er = pd.Series(0.0, index=prices.index)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    er = er.clip(0, 1)

    # Calculate smoothing constants
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)

    # Calculate adaptive smoothing constant
    sc = ((er * (fast_sc - slow_sc)) + slow_sc) ** 2

    # Initialize KAMA
    kama = prices.copy()

    # Calculate KAMA
    for i in range(er_period, len(prices)):
        if pd.notna(kama.iloc[i - 1]):
            kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (prices.iloc[i] - kama.iloc[i - 1])

    return kama


def calculate_efficiency_ratio(prices: pd.Series, period: int = 10) -> pd.Series:
    """Calculate Efficiency Ratio for trend strength.

    Args:
        prices: Series of prices
        period: Period for calculation

    Returns:
        Series with efficiency ratio values (0-1)
    """
    prices = pd.Series(prices)

    # Direction (net change over period)
    change = abs(prices - prices.shift(period))

    # Volatility (sum of absolute changes)
    volatility = prices.diff().abs().rolling(window=period).sum()

    # Efficiency Ratio
    er = pd.Series(0.0, index=prices.index)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    er = er.clip(0, 1)

    return er


def calculate_fractal_dimension(prices: pd.Series, period: int = 16) -> pd.Series:
    """Calculate Fractal Dimension for FRAMA.

    Args:
        prices: Series of prices
        period: Period for calculation (should be even)

    Returns:
        Series with fractal dimension values (1-2)
    """
    prices = pd.Series(prices)
    half_period = period // 2

    # Calculate ranges for first and second half
    high1 = prices.rolling(window=half_period).max()
    low1 = prices.rolling(window=half_period).min()
    range1 = high1 - low1

    high2 = prices.shift(half_period).rolling(window=half_period).max()
    low2 = prices.shift(half_period).rolling(window=half_period).min()
    range2 = high2 - low2

    # Full period range
    high_full = prices.rolling(window=period).max()
    low_full = prices.rolling(window=period).min()
    range_full = high_full - low_full

    # Calculate dimension
    dimension = pd.Series(1.5, index=prices.index)  # Default

    # Valid calculation mask
    mask = (range1 > 0) & (range2 > 0) & (range_full > 0) & (range1 + range2 > 0)

    # Calculate fractal dimension where valid
    dimension[mask] = (np.log(range1[mask] + range2[mask]) - np.log(range_full[mask])) / np.log(2)

    # Ensure dimension is between 1 and 2
    dimension = dimension.clip(1, 2)

    return dimension


def calculate_frama(prices: pd.Series, period: int = 16) -> pd.Series:
    """Calculate Fractal Adaptive Moving Average.

    Args:
        prices: Series of prices
        period: Period for calculation

    Returns:
        Series with FRAMA values
    """
    prices = pd.Series(prices)

    # Calculate fractal dimension
    d = calculate_fractal_dimension(prices, period)

    # Calculate adaptive alpha
    alpha = np.exp(-4.6 * (d - 1))
    alpha = alpha.clip(0.01, 1)

    # Initialize FRAMA
    frama = prices.copy()

    # Calculate FRAMA
    for i in range(period, len(prices)):
        if pd.notna(frama.iloc[i - 1]) and pd.notna(alpha.iloc[i]):
            frama.iloc[i] = alpha.iloc[i] * prices.iloc[i] + (1 - alpha.iloc[i]) * frama.iloc[i - 1]

    return frama


def calculate_vama(prices: pd.Series, volumes: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Volume Adjusted Moving Average.

    Args:
        prices: Series of prices
        volumes: Series of volumes
        period: Period for calculation

    Returns:
        Series with VAMA values
    """
    prices = pd.Series(prices)
    volumes = pd.Series(volumes)

    # Volume ratio
    volume_ratio = volumes / volumes.rolling(window=period).mean()
    volume_ratio = volume_ratio.fillna(1)

    # Weighted moving average
    weighted_prices = prices * volume_ratio
    vama = weighted_prices.rolling(window=period).sum() / volume_ratio.rolling(window=period).sum()

    return vama


def detect_crossover(series1: pd.Series, series2: pd.Series, shift: int = 1) -> pd.Series:
    """Detect when series1 crosses above series2.

    Args:
        series1: First series
        series2: Second series
        shift: Number of periods to look back

    Returns:
        Boolean series indicating crossover points
    """
    return (series1 > series2) & (series1.shift(shift) <= series2.shift(shift))


def detect_crossunder(series1: pd.Series, series2: pd.Series, shift: int = 1) -> pd.Series:
    """Detect when series1 crosses below series2.

    Args:
        series1: First series
        series2: Second series
        shift: Number of periods to look back

    Returns:
        Boolean series indicating crossunder points
    """
    return (series1 < series2) & (series1.shift(shift) >= series2.shift(shift))
