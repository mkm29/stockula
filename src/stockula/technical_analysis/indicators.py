"""Technical indicators using finta library."""

import pandas as pd
from finta import TA


class TechnicalIndicators:
    """Wrapper for finta technical indicators."""

    def __init__(self, df: pd.DataFrame):
        """Initialize with OHLCV DataFrame.

        Args:
            df: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
        """
        self.df = df
        self.data = df  # Alias for backward compatibility
        self._validate_dataframe()

    def _validate_dataframe(self):
        """Validate that DataFrame has required columns."""
        required_cols = ["Open", "High", "Low", "Close"]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")

    def sma(self, period: int = 20) -> pd.Series:
        """Calculate Simple Moving Average."""
        if period <= 0:
            raise ValueError("Period must be positive")
        if len(self.df) == 0:
            return pd.Series(dtype=float)
        if period > len(self.df):
            raise ValueError(f"Period {period} exceeds data length {len(self.df)}")
        return TA.SMA(self.df, period)

    def ema(self, period: int = 20) -> pd.Series:
        """Calculate Exponential Moving Average."""
        if period <= 0:
            raise ValueError("Period must be positive")
        if period > len(self.df):
            raise ValueError(f"Period {period} exceeds data length {len(self.df)}")
        return TA.EMA(self.df, period)

    def rsi(self, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        if period <= 0:
            raise ValueError("Period must be positive")
        if len(self.df) == 0:
            return pd.Series(dtype=float)
        return TA.RSI(self.df, period)

    def macd(self, period_fast: int = 12, period_slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD indicator."""
        if len(self.df) == 0:
            return pd.DataFrame(columns=["MACD", "MACD_SIGNAL", "MACD_DIFF"])
        result = TA.MACD(self.df, period_fast, period_slow, signal)
        # Add MACD_DIFF column (histogram)
        result["MACD_SIGNAL"] = result["SIGNAL"]
        result["MACD_DIFF"] = result["MACD"] - result["SIGNAL"]
        result = result.drop("SIGNAL", axis=1)
        return result

    def bbands(self, period: int = 20, std: int = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        return TA.BBANDS(self.df, period, std)

    def stoch(self, period: int = 14) -> pd.DataFrame:
        """Calculate Stochastic Oscillator."""
        # STOCH returns a Series, convert to DataFrame
        stoch_k = TA.STOCH(self.df, period)
        result = pd.DataFrame({"STOCH_K": stoch_k})
        # Calculate %D as 3-period SMA of %K
        result["STOCH_D"] = result["STOCH_K"].rolling(window=3).mean()
        return result

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

    def stochastic(self, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator (alias for stoch)."""
        return self.stoch(period)

    def obv(self) -> pd.Series:
        """Calculate On Balance Volume."""
        return TA.OBV(self.df)

    def ichimoku(self, tenkan: int = 9, kijun: int = 26, senkou: int = 52) -> pd.DataFrame:
        """Calculate Ichimoku Cloud."""
        result = TA.ICHIMOKU(self.df, tenkan, kijun, senkou)
        # Rename columns to match test expectations
        result_renamed = pd.DataFrame(
            {
                "ICHIMOKU_TENKAN": result["TENKAN"],
                "ICHIMOKU_KIJUN": result["KIJUN"],
                "ICHIMOKU_SENKOU_A": result["senkou_span_a"],
                "ICHIMOKU_SENKOU_B": result["SENKOU"],
            }
        )
        return result_renamed
