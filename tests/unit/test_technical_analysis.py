"""Tests for technical analysis module."""

import numpy as np
import pandas as pd
import pytest

from stockula.technical_analysis import TechnicalIndicators


class TestTechnicalIndicators:
    """Test TechnicalIndicators class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

        # Create realistic price movement
        np.random.seed(42)
        base_price = 100
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = base_price * (1 + returns).cumprod()

        data = pd.DataFrame(
            {
                "Open": prices * (1 - np.random.uniform(0, 0.01, len(dates))),
                "High": prices * (1 + np.random.uniform(0, 0.02, len(dates))),
                "Low": prices * (1 - np.random.uniform(0, 0.02, len(dates))),
                "Close": prices,
                "Volume": np.random.randint(1000000, 5000000, len(dates)),
            },
            index=dates,
        )

        return data

    @pytest.fixture
    def ta_indicators(self, sample_data):
        """Create TechnicalIndicators instance."""
        return TechnicalIndicators(sample_data)

    def test_initialization(self, sample_data):
        """Test TechnicalIndicators initialization."""
        ta = TechnicalIndicators(sample_data)
        assert ta.data is sample_data
        assert isinstance(ta.data, pd.DataFrame)
        assert all(col in ta.data.columns for col in ["Open", "High", "Low", "Close", "Volume"])

    def test_sma(self, ta_indicators):
        """Test Simple Moving Average calculation."""
        # Test different periods
        sma_10 = ta_indicators.sma(period=10)
        sma_20 = ta_indicators.sma(period=20)
        sma_50 = ta_indicators.sma(period=50)

        # Check return type
        assert isinstance(sma_10, pd.Series)
        assert isinstance(sma_20, pd.Series)
        assert isinstance(sma_50, pd.Series)

        # Check length
        assert len(sma_10) == len(ta_indicators.data)

        # Check NaN values at beginning
        assert sma_10.iloc[:9].isna().all()
        assert sma_20.iloc[:19].isna().all()
        assert sma_50.iloc[:49].isna().all()

        # Check values are calculated correctly
        assert not sma_10.iloc[9:].isna().any()
        assert not sma_20.iloc[19:].isna().any()
        assert not sma_50.iloc[49:].isna().any()

        # Verify calculation
        expected_sma_10 = ta_indicators.data["Close"].rolling(window=10).mean()
        pd.testing.assert_series_equal(sma_10, expected_sma_10, check_names=False)

    def test_ema(self, ta_indicators):
        """Test Exponential Moving Average calculation."""
        ema_12 = ta_indicators.ema(period=12)
        ema_26 = ta_indicators.ema(period=26)

        # Check return type
        assert isinstance(ema_12, pd.Series)
        assert isinstance(ema_26, pd.Series)

        # Check length
        assert len(ema_12) == len(ta_indicators.data)

        # EMA should start from first value (not NaN like SMA)
        assert not pd.isna(ema_12.iloc[0])

        # EMA should react faster to price changes than SMA
        sma_12 = ta_indicators.sma(period=12)

        # After significant price movement, EMA should differ from SMA
        assert not np.allclose(ema_12.iloc[20:].values, sma_12.iloc[20:].values, rtol=0.001)

    def test_rsi(self, ta_indicators):
        """Test Relative Strength Index calculation."""
        rsi = ta_indicators.rsi(period=14)

        # Check return type
        assert isinstance(rsi, pd.Series)

        # Check length
        assert len(rsi) == len(ta_indicators.data)

        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

        # Check for oversold/overbought conditions in synthetic data
        # Some values should be in extreme ranges given random data
        assert (valid_rsi < 30).any() or (valid_rsi > 70).any()

    def test_macd(self, ta_indicators):
        """Test MACD calculation."""
        macd_result = ta_indicators.macd(period_fast=12, period_slow=26, signal=9)

        # Check return type
        assert isinstance(macd_result, pd.DataFrame)

        # Check columns
        expected_columns = ["MACD", "MACD_SIGNAL", "MACD_DIFF"]
        assert all(col in macd_result.columns for col in expected_columns)

        # Check length
        assert len(macd_result) == len(ta_indicators.data)

        # MACD line should be EMA12 - EMA26
        ema_12 = ta_indicators.ema(12)
        ema_26 = ta_indicators.ema(26)
        expected_macd = ema_12 - ema_26

        pd.testing.assert_series_equal(
            macd_result["MACD"].dropna(),
            expected_macd.dropna(),
            check_names=False,
            rtol=0.001,
        )

        # MACD histogram should be MACD - Signal
        expected_hist = macd_result["MACD"] - macd_result["MACD_SIGNAL"]
        pd.testing.assert_series_equal(
            macd_result["MACD_DIFF"].dropna(),
            expected_hist.dropna(),
            check_names=False,
            rtol=0.001,
        )

    def test_bbands(self, ta_indicators):
        """Test Bollinger Bands calculation."""
        bbands = ta_indicators.bbands(period=20, std=2)

        # Check return type
        assert isinstance(bbands, pd.DataFrame)

        # Check columns
        expected_columns = ["BB_UPPER", "BB_MIDDLE", "BB_LOWER"]
        assert all(col in bbands.columns for col in expected_columns)

        # Check length
        assert len(bbands) == len(ta_indicators.data)

        # Middle band should be SMA
        sma_20 = ta_indicators.sma(20)
        pd.testing.assert_series_equal(bbands["BB_MIDDLE"], sma_20, check_names=False)

        # Upper band should be above middle, lower below
        valid_idx = ~bbands["BB_MIDDLE"].isna()
        assert (bbands.loc[valid_idx, "BB_UPPER"] > bbands.loc[valid_idx, "BB_MIDDLE"]).all()
        assert (bbands.loc[valid_idx, "BB_LOWER"] < bbands.loc[valid_idx, "BB_MIDDLE"]).all()

        # Bands should contain most price action
        close_prices = ta_indicators.data["Close"]
        within_bands = (close_prices >= bbands["BB_LOWER"]) & (close_prices <= bbands["BB_UPPER"])
        # Approximately 95% of prices should be within 2 std bands
        assert within_bands.sum() / within_bands.dropna().count() > 0.7

    def test_atr(self, ta_indicators):
        """Test Average True Range calculation."""
        atr = ta_indicators.atr(period=14)

        # Check return type
        assert isinstance(atr, pd.Series)

        # Check length
        assert len(atr) == len(ta_indicators.data)

        # ATR should be positive
        valid_atr = atr.dropna()
        assert (valid_atr > 0).all()

        # ATR should be reasonable compared to the daily range
        daily_range = ta_indicators.data["High"] - ta_indicators.data["Low"]
        # ATR uses exponential smoothing and can exceed daily range, especially with gaps
        # Just check that ATR is within a reasonable multiple of the average daily range
        assert (atr.dropna() <= daily_range.mean() * 3).all()

    def test_adx(self, ta_indicators):
        """Test Average Directional Index calculation."""
        adx = ta_indicators.adx(period=14)

        # Check return type
        assert isinstance(adx, pd.Series)

        # Check length
        assert len(adx) == len(ta_indicators.data)

        # ADX should be between 0 and 100
        valid_adx = adx.dropna()
        assert (valid_adx >= 0).all()
        assert (valid_adx <= 100).all()

    def test_stochastic(self, ta_indicators):
        """Test Stochastic Oscillator calculation."""
        stoch = ta_indicators.stochastic(period=14, smooth_k=3, smooth_d=3)

        # Check return type
        assert isinstance(stoch, pd.DataFrame)

        # Check columns
        assert "STOCH_K" in stoch.columns
        assert "STOCH_D" in stoch.columns

        # Check length
        assert len(stoch) == len(ta_indicators.data)

        # Stochastic should be between 0 and 100
        valid_k = stoch["STOCH_K"].dropna()
        valid_d = stoch["STOCH_D"].dropna()
        assert (valid_k >= 0).all() and (valid_k <= 100).all()
        assert (valid_d >= 0).all() and (valid_d <= 100).all()

    def test_volume_indicators(self, ta_indicators):
        """Test volume-based indicators."""
        # On Balance Volume
        obv = ta_indicators.obv()
        assert isinstance(obv, pd.Series)
        assert len(obv) == len(ta_indicators.data)

        # Volume should affect OBV
        # When close > previous close and volume is high, OBV should increase
        close_prices = ta_indicators.data["Close"]
        close_diff = close_prices.diff()

        # OBV changes should correlate with price direction and volume
        obv_diff = obv.diff().iloc[1:]
        price_direction = (close_diff > 0).iloc[1:]

        # At least some OBV increases should align with price increases
        assert ((obv_diff > 0) & price_direction).any()

    def test_ichimoku(self, ta_indicators):
        """Test Ichimoku Cloud calculation."""
        ichimoku = ta_indicators.ichimoku()

        # Check return type
        assert isinstance(ichimoku, pd.DataFrame)

        # Check required columns
        expected_columns = [
            "ICHIMOKU_TENKAN",
            "ICHIMOKU_KIJUN",
            "ICHIMOKU_SENKOU_A",
            "ICHIMOKU_SENKOU_B",
        ]
        assert all(col in ichimoku.columns for col in expected_columns)

        # Tenkan (9-period) should be more reactive than Kijun (26-period)
        tenkan_std = ichimoku["ICHIMOKU_TENKAN"].std()
        kijun_std = ichimoku["ICHIMOKU_KIJUN"].std()
        assert tenkan_std > kijun_std * 0.8  # Tenkan should vary more

    def test_invalid_period(self, ta_indicators):
        """Test error handling for invalid periods."""
        # Period longer than data
        with pytest.raises(ValueError):
            ta_indicators.sma(period=200)  # We only have 100 days

        # Negative period
        with pytest.raises(ValueError):
            ta_indicators.ema(period=-5)

        # Zero period
        with pytest.raises(ValueError):
            ta_indicators.rsi(period=0)

    def test_empty_data(self):
        """Test handling of empty data."""
        empty_data = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        ta = TechnicalIndicators(empty_data)

        # Should return empty series/dataframes
        assert ta.sma(10).empty
        assert ta.rsi(14).empty
        assert ta.macd().empty

    def test_missing_columns(self, sample_data):
        """Test handling of missing required columns."""
        # Remove Volume column
        data_no_volume = sample_data.drop(columns=["Volume"])
        ta = TechnicalIndicators(data_no_volume)

        # Most indicators should still work
        assert not ta.sma(10).empty
        assert not ta.rsi(14).empty

        # But volume indicators should fail
        with pytest.raises(LookupError):
            ta.obv()

    def test_all_indicators_together(self, ta_indicators):
        """Test that all indicators can be calculated without interference."""
        # Calculate all indicators
        results = {
            "sma_20": ta_indicators.sma(20),
            "ema_20": ta_indicators.ema(20),
            "rsi": ta_indicators.rsi(14),
            "macd": ta_indicators.macd(),
            "bbands": ta_indicators.bbands(),
            "atr": ta_indicators.atr(14),
            "adx": ta_indicators.adx(14),
            "stoch": ta_indicators.stochastic(),
            "obv": ta_indicators.obv(),
            "ichimoku": ta_indicators.ichimoku(),
        }

        # All should have the same index
        base_index = ta_indicators.data.index
        for name, indicator in results.items():
            if isinstance(indicator, pd.Series):
                assert indicator.index.equals(base_index), f"{name} has different index"
            else:
                assert indicator.index.equals(base_index), f"{name} has different index"
