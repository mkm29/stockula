"""Technical indicators using finta library."""

import pandas as pd
from finta import TA


class TechnicalIndicators:
    """Wrapper for finta technical indicators."""

    def __init__(self, df: pd.DataFrame):
        """Initialize with OHLCV DataFrame.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        """
        self.df = df
        self._validate_dataframe()

    def _validate_dataframe(self):
        """Validate that DataFrame has required columns."""
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")

    def sma(self, period: int = 20) -> pd.Series:
        """Calculate Simple Moving Average."""
        return TA.SMA(self.df, period)

    def ema(self, period: int = 20) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return TA.EMA(self.df, period)

    def rsi(self, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        return TA.RSI(self.df, period)

    def macd(
        self, period_fast: int = 12, period_slow: int = 26, signal: int = 9
    ) -> pd.DataFrame:
        """Calculate MACD indicator."""
        return TA.MACD(self.df, period_fast, period_slow, signal)

    def bbands(self, period: int = 20, std: int = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        return TA.BBANDS(self.df, period, std)

    def stoch(self, period: int = 14) -> pd.DataFrame:
        """Calculate Stochastic Oscillator."""
        return TA.STOCH(self.df, period)

    def atr(self, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        return TA.ATR(self.df, period)

    def adx(self, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        return TA.ADX(self.df, period)

    def williams_r(self, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        return TA.WILLIAMS(self.df, period)

    def cci(self, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index."""
        return TA.CCI(self.df, period)
