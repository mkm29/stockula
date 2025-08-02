"""Trading strategies for backtesting.

⚠️ WARNING: This module has NOT been thoroughly tested due to complex dependencies
on the backtesting.py library. Unit testing is extremely difficult because of tight
coupling with the library's internal state management and indicator calculations.

IMPORTANT: These strategies should be considered EXPERIMENTAL and used with caution
in production environments. Comprehensive integration testing is planned for the future.

TODO: Implement integration tests with known data and expected outcomes (Q2 2024)
TODO: Consider refactoring to improve testability by separating calculations from logic
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover


class BaseStrategy(Strategy):
    """Base strategy class with common functionality."""

    def init(self):
        """Initialize strategy indicators."""
        pass

    def next(self):
        """Define trading logic."""
        pass


class SMACrossStrategy(BaseStrategy):
    """Simple Moving Average Crossover Strategy."""

    fast_period = 10
    slow_period = 20

    def init(self):
        """Initialize SMA indicators."""
        self.sma_fast = self.I(lambda x: pd.Series(x).rolling(self.fast_period).mean(), self.data.Close)
        self.sma_slow = self.I(lambda x: pd.Series(x).rolling(self.slow_period).mean(), self.data.Close)

    def next(self):
        """Execute trading logic on crossover."""
        if crossover(self.sma_fast, self.sma_slow):
            self.buy()
        elif crossover(self.sma_slow, self.sma_fast):
            self.position.close()


class RSIStrategy(BaseStrategy):
    """RSI-based trading strategy."""

    rsi_period = 14
    oversold_threshold = 30
    overbought_threshold = 70

    def init(self):
        """Initialize RSI indicator."""

        def rsi(prices, period=14):
            prices = pd.Series(prices)
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        self.rsi = self.I(rsi, self.data.Close, self.rsi_period)

    def next(self):
        """Execute RSI-based trading logic."""
        if self.rsi[-1] < self.oversold_threshold:
            if not self.position:
                self.buy()
        elif self.rsi[-1] > self.overbought_threshold:
            if self.position:
                self.position.close()


class MACDStrategy(BaseStrategy):
    """MACD-based trading strategy."""

    fast_period = 12
    slow_period = 26
    signal_period = 9

    def init(self):
        """Initialize MACD indicator."""

        def ema(prices, period):
            return pd.Series(prices).ewm(span=period, adjust=False).mean()

        def macd(prices):
            ema_fast = ema(prices, self.fast_period)
            ema_slow = ema(prices, self.slow_period)
            macd_line = ema_fast - ema_slow
            signal_line = ema(macd_line, self.signal_period)
            return macd_line, signal_line

        self.macd_line, self.signal_line = self.I(macd, self.data.Close, plot=False)

    def next(self):
        """Execute MACD-based trading logic."""
        if crossover(self.macd_line, self.signal_line):
            self.buy()
        elif crossover(self.signal_line, self.macd_line):
            if self.position:
                self.position.close()


class DoubleEMACrossStrategy(BaseStrategy):
    """Double Exponential Moving Average (EMA) Crossover Strategy.

    This strategy uses two EMAs (default 34 and 55 periods) to generate
    buy/sell signals based on crossover events. Designed for the portfolio
    strategy with broad-market core, momentum/large-cap growth, and
    speculative high-beta allocations.
    """

    fast_period = 34
    slow_period = 55

    # ATR-based stop loss multipliers for different asset classes
    momentum_atr_multiple = 1.25
    speculative_atr_multiple = 1.0
    atr_period = 14

    # Minimum required data buffer after slow period initialization
    min_trading_days_buffer = 20  # At least 20 days for signals after EMA warmup

    def init(self):
        """Initialize EMA and ATR indicators."""

        # Validate we have enough data
        total_data_points = len(self.data)
        required_data_points = self.slow_period + self.min_trading_days_buffer

        if total_data_points < required_data_points:
            import warnings

            warnings.warn(
                f"Insufficient data for DoubleEMACrossStrategy: "
                f"Have {total_data_points} days, need at least {required_data_points} days "
                f"({self.slow_period} for slow EMA + {self.min_trading_days_buffer} buffer). "
                f"Strategy may not generate any signals.",
                stacklevel=2,
            )

        def ema(prices, period):
            return pd.Series(prices).ewm(span=period, adjust=False).mean()

        def atr(high, low, close, period=14):
            """Calculate Average True Range."""
            high = pd.Series(high)
            low = pd.Series(low)
            close = pd.Series(close)

            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return atr

        # Initialize EMAs
        self.ema_fast = self.I(ema, self.data.Close, self.fast_period)
        self.ema_slow = self.I(ema, self.data.Close, self.slow_period)

        # Initialize ATR for stop loss calculation
        self.atr = self.I(atr, self.data.High, self.data.Low, self.data.Close, self.atr_period)

    def next(self):
        """Execute Double EMA crossover trading logic."""
        # Skip if we don't have enough data
        if len(self.data) < self.slow_period:
            return

        # Buy signal: Fast EMA crosses above Slow EMA
        if crossover(self.ema_fast, self.ema_slow):
            if not self.position:
                self.buy()

        # Sell signal: Fast EMA crosses below Slow EMA
        elif crossover(self.ema_slow, self.ema_fast):
            if self.position:
                self.position.close()

        # Check stop loss if we have a position
        elif self.position and self.trades:
            # Get the most recent trade entry price
            current_trade = self.trades[-1]
            # Use the appropriate ATR multiple based on asset volatility
            # In a real implementation, you'd categorize the asset
            atr_multiple = self.momentum_atr_multiple
            stop_loss_price = current_trade.entry_price - (atr_multiple * self.atr[-1])

            if self.data.Close[-1] <= stop_loss_price:
                self.position.close()

    @classmethod
    def get_min_required_days(cls):
        """Calculate minimum required trading days for this strategy."""
        return cls.slow_period + cls.min_trading_days_buffer

    @classmethod
    def get_recommended_start_date(cls, end_date: str) -> str:
        """Calculate recommended start date given an end date.

        Args:
            end_date: End date in YYYY-MM-DD format

        Returns:
            Recommended start date as string
        """
        # Convert to trading days (approximately 252 trading days per year)
        required_trading_days = cls.get_min_required_days()
        # Add 20% buffer for weekends/holidays
        required_calendar_days = int(required_trading_days * 1.4)

        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=required_calendar_days)

        return start_dt.strftime("%Y-%m-%d")


class VIDYAStrategy(BaseStrategy):
    """Variable Index Dynamic Average (VIDYA) Strategy.

    VIDYA is an adaptive moving average that adjusts its smoothing factor
    based on the Chande Momentum Oscillator (CMO). The CMO measures the
    relative strength of recent price changes, allowing VIDYA to adapt
    to market conditions - moving faster in trending markets and slower
    in sideways markets.

    Formula:
    CMO = (Sum of positive changes - Sum of negative changes) / (Sum of all changes) * 100
    VI = abs(CMO) / 100  (Volatility Index)
    Alpha = 2 / (N + 1) * VI
    VIDYA = Alpha * Close + (1 - Alpha) * Previous VIDYA
    """

    # CMO calculation period
    cmo_period = 9
    # VIDYA smoothing period (for alpha calculation)
    smoothing_period = 12

    # ATR-based stop loss
    atr_period = 14
    atr_multiple = 1.5

    # Minimum required data buffer
    min_trading_days_buffer = 20

    def init(self):
        """Initialize VIDYA and ATR indicators."""

        # Validate we have enough data
        total_data_points = len(self.data)
        required_data_points = max(self.cmo_period, self.smoothing_period) + self.min_trading_days_buffer

        if total_data_points < required_data_points:
            import warnings

            warnings.warn(
                f"Insufficient data for VIDYAStrategy: "
                f"Have {total_data_points} days, need at least {required_data_points} days "
                f"(max({self.cmo_period}, {self.smoothing_period})="
                f"{max(self.cmo_period, self.smoothing_period)} + "
                f"{self.min_trading_days_buffer} buffer). "
                f"Strategy may not generate any signals.",
                stacklevel=2,
            )

        def vidya(prices, cmo_period=9, smoothing_period=12):
            """Calculate Variable Index Dynamic Average (VIDYA)."""
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
                    vidya_values.iloc[i] = (
                        alpha.iloc[i] * prices.iloc[i] + (1 - alpha.iloc[i]) * vidya_values.iloc[i - 1]
                    )

            return vidya_values

        def atr(high, low, close, period=14):
            """Calculate Average True Range."""
            high = pd.Series(high)
            low = pd.Series(low)
            close = pd.Series(close)

            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return atr

        # Initialize VIDYA for two periods (fast and slow)
        self.vidya_fast = self.I(vidya, self.data.Close, self.cmo_period, self.smoothing_period)
        self.vidya_slow = self.I(vidya, self.data.Close, self.cmo_period * 2, self.smoothing_period * 2)

        # Initialize ATR for stop loss calculation
        self.atr = self.I(atr, self.data.High, self.data.Low, self.data.Close, self.atr_period)

    def next(self):
        """Execute VIDYA crossover trading logic."""
        # Skip if we don't have enough data
        if len(self.data) < max(self.cmo_period * 2, self.smoothing_period * 2):
            return

        # Buy signal: Fast VIDYA crosses above Slow VIDYA
        if crossover(self.vidya_fast, self.vidya_slow):
            if not self.position:
                self.buy()

        # Sell signal: Fast VIDYA crosses below Slow VIDYA
        elif crossover(self.vidya_slow, self.vidya_fast):
            if self.position:
                self.position.close()

        # Check stop loss if we have a position
        elif self.position and self.trades:
            # Get the most recent trade entry price
            current_trade = self.trades[-1]
            stop_loss_price = current_trade.entry_price - (self.atr_multiple * self.atr[-1])

            if self.data.Close[-1] <= stop_loss_price:
                self.position.close()

    @classmethod
    def get_min_required_days(cls):
        """Calculate minimum required trading days for this strategy."""
        return max(cls.cmo_period * 2, cls.smoothing_period * 2) + cls.min_trading_days_buffer

    @classmethod
    def get_recommended_start_date(cls, end_date: str) -> str:
        """Calculate recommended start date given an end date.

        Args:
            end_date: End date in YYYY-MM-DD format

        Returns:
            Recommended start date as string
        """
        # Convert to trading days (approximately 252 trading days per year)
        required_trading_days = cls.get_min_required_days()
        # Add 20% buffer for weekends/holidays
        required_calendar_days = int(required_trading_days * 1.4)

        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=required_calendar_days)

        return start_dt.strftime("%Y-%m-%d")


class KAMAStrategy(BaseStrategy):
    """Kaufman's Adaptive Moving Average (KAMA) Strategy.

    KAMA adapts its smoothing based on market noise and trend strength
    using the Efficiency Ratio (ER). It moves faster when markets trend
    and slower during consolidation periods.

    Formula:
    Direction = Close - Close[n periods ago]
    Volatility = Sum of abs(Close - Previous Close) over n periods
    ER = Direction / Volatility  (Efficiency Ratio)
    SC = [ER * (fast SC - slow SC) + slow SC]²  (Smoothing Constant)
    KAMA = Previous KAMA + SC * (Close - Previous KAMA)

    Where:
    - fast SC = 2/(fast period + 1)
    - slow SC = 2/(slow period + 1)
    """

    # Efficiency Ratio period
    er_period = 10
    # Fast EMA equivalent period
    fast_period = 2
    # Slow EMA equivalent period
    slow_period = 30

    # ATR-based stop loss
    atr_period = 14
    atr_multiple = 1.3

    # Minimum required data buffer
    min_trading_days_buffer = 20

    def init(self):
        """Initialize KAMA and ATR indicators."""

        # Validate we have enough data
        total_data_points = len(self.data)
        required_data_points = max(self.er_period, self.slow_period) + self.min_trading_days_buffer

        if total_data_points < required_data_points:
            import warnings

            warnings.warn(
                f"Insufficient data for KAMAStrategy: "
                f"Have {total_data_points} days, need at least {required_data_points} days "
                f"(max({self.er_period}, {self.slow_period})="
                f"{max(self.er_period, self.slow_period)} + "
                f"{self.min_trading_days_buffer} buffer). "
                f"Strategy may not generate any signals.",
                stacklevel=2,
            )

        def kama(prices, er_period=10, fast_period=2, slow_period=30):
            """Calculate Kaufman's Adaptive Moving Average (KAMA)."""
            prices = pd.Series(prices)

            # Calculate direction (change over period)
            direction = abs(prices - prices.shift(er_period))

            # Calculate volatility (sum of absolute changes)
            volatility = prices.diff().abs().rolling(window=er_period).sum()

            # Calculate Efficiency Ratio (ER)
            er = pd.Series(0.0, index=prices.index)
            mask = volatility != 0
            er[mask] = direction[mask] / volatility[mask]

            # Calculate Smoothing Constants
            fast_sc = 2 / (fast_period + 1)
            slow_sc = 2 / (slow_period + 1)

            # Calculate adaptive smoothing constant
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

            # Initialize KAMA with SMA
            kama_values = prices.rolling(window=er_period).mean()

            # Calculate KAMA
            for i in range(er_period, len(prices)):
                if pd.notna(kama_values.iloc[i - 1]) and pd.notna(sc.iloc[i]):
                    kama_values.iloc[i] = kama_values.iloc[i - 1] + sc.iloc[i] * (
                        prices.iloc[i] - kama_values.iloc[i - 1]
                    )

            return kama_values

        def atr(high, low, close, period=14):
            """Calculate Average True Range."""
            high = pd.Series(high)
            low = pd.Series(low)
            close = pd.Series(close)

            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return atr

        # Initialize KAMA for two different parameter sets
        self.kama_fast = self.I(kama, self.data.Close, self.er_period, self.fast_period, self.slow_period)
        self.kama_slow = self.I(
            kama,
            self.data.Close,
            self.er_period * 2,
            self.fast_period * 2,
            self.slow_period * 2,
        )

        # Initialize ATR for stop loss calculation
        self.atr = self.I(atr, self.data.High, self.data.Low, self.data.Close, self.atr_period)

    def next(self):
        """Execute KAMA crossover trading logic."""
        # Skip if we don't have enough data
        if len(self.data) < max(self.er_period * 2, self.slow_period * 2):
            return

        # Buy signal: Fast KAMA crosses above Slow KAMA
        if crossover(self.kama_fast, self.kama_slow):
            if not self.position:
                self.buy()

        # Sell signal: Fast KAMA crosses below Slow KAMA
        elif crossover(self.kama_slow, self.kama_fast):
            if self.position:
                self.position.close()

        # Check stop loss if we have a position
        elif self.position and self.trades:
            # Get the most recent trade entry price
            current_trade = self.trades[-1]
            stop_loss_price = current_trade.entry_price - (self.atr_multiple * self.atr[-1])

            if self.data.Close[-1] <= stop_loss_price:
                self.position.close()

    @classmethod
    def get_min_required_days(cls):
        """Calculate minimum required trading days for this strategy."""
        return max(cls.er_period * 2, cls.slow_period * 2) + cls.min_trading_days_buffer

    @classmethod
    def get_recommended_start_date(cls, end_date: str) -> str:
        """Calculate recommended start date given an end date.

        Args:
            end_date: End date in YYYY-MM-DD format

        Returns:
            Recommended start date as string
        """
        # Convert to trading days (approximately 252 trading days per year)
        required_trading_days = cls.get_min_required_days()
        # Add 20% buffer for weekends/holidays
        required_calendar_days = int(required_trading_days * 1.4)

        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=required_calendar_days)

        return start_dt.strftime("%Y-%m-%d")


class FRAMAStrategy(BaseStrategy):
    """Fractal Adaptive Moving Average (FRAMA) Strategy.

    FRAMA uses fractal geometry to dynamically adjust its smoothing factor.
    It calculates the fractal dimension of price movements to determine
    how much the market is trending versus ranging.

    The fractal dimension (D) ranges from 1 to 2:
    - D near 1: Strong trend (straight line)
    - D near 2: No trend (very jagged, fills the plane)

    Formula:
    D = (log(N1 + N2) - log(N3)) / log(2)
    Alpha = exp(-4.6 * (D - 1))
    FRAMA = Alpha * Close + (1 - Alpha) * Previous FRAMA

    Where N1, N2, N3 are calculated from highest high and lowest low
    over specific periods.
    """

    # FRAMA calculation period
    frama_period = 16  # Must be even, commonly 16

    # ATR-based stop loss
    atr_period = 14
    atr_multiple = 1.4

    # Minimum required data buffer
    min_trading_days_buffer = 20

    def init(self):
        """Initialize FRAMA and ATR indicators."""

        # Validate we have enough data
        total_data_points = len(self.data)
        required_data_points = self.frama_period * 2 + self.min_trading_days_buffer

        if total_data_points < required_data_points:
            import warnings

            warnings.warn(
                f"Insufficient data for FRAMAStrategy: "
                f"Have {total_data_points} days, need at least {required_data_points} days "
                f"({self.frama_period}*2={self.frama_period * 2} + {self.min_trading_days_buffer} buffer). "
                f"Strategy may not generate any signals.",
                stacklevel=2,
            )

        def frama(prices, period=16):
            """Calculate Fractal Adaptive Moving Average (FRAMA)."""

            prices = pd.Series(prices)

            # Ensure period is even
            if period % 2 != 0:
                period = period + 1

            half_period = period // 2

            # Initialize FRAMA with SMA
            frama_values = prices.rolling(window=period).mean()

            for i in range(period, len(prices)):
                # Get price window
                window = prices.iloc[i - period + 1 : i + 1]

                # Split window into halves
                first_half = window.iloc[:half_period]
                second_half = window.iloc[half_period:]

                # Calculate N1 (first half range)
                n1 = (first_half.max() - first_half.min()) / half_period

                # Calculate N2 (second half range)
                n2 = (second_half.max() - second_half.min()) / half_period

                # Calculate N3 (full period range)
                n3 = (window.max() - window.min()) / period

                # Avoid division by zero and log of zero/negative
                if n1 > 0 and n2 > 0 and n3 > 0:
                    # Calculate fractal dimension
                    if (n1 + n2) > 0 and n3 > 0:
                        d = (np.log(n1 + n2) - np.log(n3)) / np.log(2)
                    else:
                        d = 1.5  # Default to middle value

                    # Constrain D between 1 and 2
                    d = max(1.0, min(2.0, d))

                    # Calculate alpha
                    alpha = np.exp(-4.6 * (d - 1))

                    # Calculate FRAMA
                    if pd.notna(frama_values.iloc[i - 1]):
                        frama_values.iloc[i] = alpha * prices.iloc[i] + (1 - alpha) * frama_values.iloc[i - 1]

            return frama_values

        def atr(high, low, close, period=14):
            """Calculate Average True Range."""
            high = pd.Series(high)
            low = pd.Series(low)
            close = pd.Series(close)

            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return atr

        # Initialize FRAMA for two different periods
        self.frama_fast = self.I(frama, self.data.Close, self.frama_period)
        self.frama_slow = self.I(frama, self.data.Close, self.frama_period * 2)

        # Initialize ATR for stop loss calculation
        self.atr = self.I(atr, self.data.High, self.data.Low, self.data.Close, self.atr_period)

    def next(self):
        """Execute FRAMA crossover trading logic."""
        # Skip if we don't have enough data
        if len(self.data) < self.frama_period * 2:
            return

        # Buy signal: Fast FRAMA crosses above Slow FRAMA
        if crossover(self.frama_fast, self.frama_slow):
            if not self.position:
                self.buy()

        # Sell signal: Fast FRAMA crosses below Slow FRAMA
        elif crossover(self.frama_slow, self.frama_fast):
            if self.position:
                self.position.close()

        # Check stop loss if we have a position
        elif self.position and self.trades:
            # Get the most recent trade entry price
            current_trade = self.trades[-1]
            stop_loss_price = current_trade.entry_price - (self.atr_multiple * self.atr[-1])

            if self.data.Close[-1] <= stop_loss_price:
                self.position.close()

    @classmethod
    def get_min_required_days(cls):
        """Calculate minimum required trading days for this strategy."""
        return cls.frama_period * 2 + cls.min_trading_days_buffer

    @classmethod
    def get_recommended_start_date(cls, end_date: str) -> str:
        """Calculate recommended start date given an end date.

        Args:
            end_date: End date in YYYY-MM-DD format

        Returns:
            Recommended start date as string
        """
        # Convert to trading days (approximately 252 trading days per year)
        required_trading_days = cls.get_min_required_days()
        # Add 20% buffer for weekends/holidays
        required_calendar_days = int(required_trading_days * 1.4)

        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=required_calendar_days)

        return start_dt.strftime("%Y-%m-%d")


class TripleEMACrossStrategy(BaseStrategy):
    """Triple Exponential Moving Average (TEMA) Crossover Strategy.

    TEMA attempts to remove the inherent lag associated with moving averages
    by placing more weight on recent values. It uses a combination of single,
    double, and triple exponential smoothing.

    Formula: TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))

    This strategy generates buy/sell signals based on TEMA crossovers.
    """

    fast_period = 9
    slow_period = 21

    # ATR-based stop loss multipliers
    atr_multiple = 1.5
    atr_period = 14

    # Minimum required data buffer after slow TEMA initialization
    # TEMA needs 3 * period - 2 samples to start producing values
    min_trading_days_buffer = 20

    def init(self):
        """Initialize TEMA and ATR indicators."""

        # Validate we have enough data
        total_data_points = len(self.data)
        # TEMA needs 3 * period - 2 samples
        required_data_points = (3 * self.slow_period - 2) + self.min_trading_days_buffer

        if total_data_points < required_data_points:
            import warnings

            warnings.warn(
                f"Insufficient data for TripleEMACrossStrategy: "
                f"Have {total_data_points} days, need at least {required_data_points} days "
                f"(3*{self.slow_period}-2={3 * self.slow_period - 2} for slow TEMA + "
                f"{self.min_trading_days_buffer} buffer). "
                f"Strategy may not generate any signals.",
                stacklevel=2,
            )

        def tema(prices, period):
            """Calculate Triple Exponential Moving Average (TEMA)."""
            prices = pd.Series(prices)

            # Calculate EMAs
            ema1 = prices.ewm(span=period, adjust=False).mean()
            ema2 = ema1.ewm(span=period, adjust=False).mean()
            ema3 = ema2.ewm(span=period, adjust=False).mean()

            # TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
            tema_values = 3 * ema1 - 3 * ema2 + ema3

            return tema_values

        def atr(high, low, close, period=14):
            """Calculate Average True Range."""
            high = pd.Series(high)
            low = pd.Series(low)
            close = pd.Series(close)

            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return atr

        # Initialize TEMAs
        self.tema_fast = self.I(tema, self.data.Close, self.fast_period)
        self.tema_slow = self.I(tema, self.data.Close, self.slow_period)

        # Initialize ATR for stop loss calculation
        self.atr = self.I(atr, self.data.High, self.data.Low, self.data.Close, self.atr_period)

    def next(self):
        """Execute Triple EMA crossover trading logic."""
        # Skip if we don't have enough data
        # TEMA needs 3 * slow_period - 2 samples
        if len(self.data) < (3 * self.slow_period - 2):
            return

        # Buy signal: Fast TEMA crosses above Slow TEMA
        if crossover(self.tema_fast, self.tema_slow):
            if not self.position:
                self.buy()

        # Sell signal: Fast TEMA crosses below Slow TEMA
        elif crossover(self.tema_slow, self.tema_fast):
            if self.position:
                self.position.close()

        # Check stop loss if we have a position
        elif self.position and self.trades:
            # Get the most recent trade entry price
            current_trade = self.trades[-1]
            stop_loss_price = current_trade.entry_price - (self.atr_multiple * self.atr[-1])

            if self.data.Close[-1] <= stop_loss_price:
                self.position.close()

    @classmethod
    def get_min_required_days(cls):
        """Calculate minimum required trading days for this strategy."""
        # TEMA needs 3 * period - 2 samples
        return (3 * cls.slow_period - 2) + cls.min_trading_days_buffer

    @classmethod
    def get_recommended_start_date(cls, end_date: str) -> str:
        """Calculate recommended start date given an end date.

        Args:
            end_date: End date in YYYY-MM-DD format

        Returns:
            Recommended start date as string
        """
        # Convert to trading days (approximately 252 trading days per year)
        required_trading_days = cls.get_min_required_days()
        # Add 20% buffer for weekends/holidays
        required_calendar_days = int(required_trading_days * 1.4)

        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=required_calendar_days)

        return start_dt.strftime("%Y-%m-%d")


class TRIMACrossStrategy(BaseStrategy):
    """Triangular Moving Average (TRIMA) Crossover Strategy.

    TRIMA represents an average of prices but places more weight on the
    middle prices of the time period. It double-smooths the data using
    a window width that is one-half the length of the series.

    This strategy is ideal for filtering out short-term fluctuations
    and focusing on the overall trend direction.
    """

    fast_period = 14
    slow_period = 28

    # ATR-based stop loss
    atr_multiple = 1.2
    atr_period = 14

    # Minimum required data buffer
    # TRIMA needs 2 * period for full calculation
    min_trading_days_buffer = 20

    def init(self):
        """Initialize TRIMA and ATR indicators."""

        # Validate we have enough data
        total_data_points = len(self.data)
        # TRIMA needs 2 * period for full calculation
        required_data_points = (2 * self.slow_period) + self.min_trading_days_buffer

        if total_data_points < required_data_points:
            import warnings

            warnings.warn(
                f"Insufficient data for TRIMACrossStrategy: "
                f"Have {total_data_points} days, need at least {required_data_points} days "
                f"(2*{self.slow_period}={2 * self.slow_period} for slow TRIMA + "
                f"{self.min_trading_days_buffer} buffer). "
                f"Strategy may not generate any signals.",
                stacklevel=2,
            )

        def trima(prices, period):
            """Calculate Triangular Moving Average (TRIMA)."""
            prices = pd.Series(prices)

            # First calculate SMA
            sma = prices.rolling(window=period).mean()

            # Then calculate the rolling sum of SMA
            sma_sum = sma.rolling(window=period).sum()

            # Divide by period to get TRIMA
            trima_values = sma_sum / period

            return trima_values

        def atr(high, low, close, period=14):
            """Calculate Average True Range."""
            high = pd.Series(high)
            low = pd.Series(low)
            close = pd.Series(close)

            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return atr

        # Initialize TRIMAs
        self.trima_fast = self.I(trima, self.data.Close, self.fast_period)
        self.trima_slow = self.I(trima, self.data.Close, self.slow_period)

        # Initialize ATR for stop loss calculation
        self.atr = self.I(atr, self.data.High, self.data.Low, self.data.Close, self.atr_period)

    def next(self):
        """Execute TRIMA crossover trading logic."""
        # Skip if we don't have enough data
        # TRIMA needs 2 * slow_period samples
        if len(self.data) < (2 * self.slow_period):
            return

        # Buy signal: Fast TRIMA crosses above Slow TRIMA
        if crossover(self.trima_fast, self.trima_slow):
            if not self.position:
                self.buy()

        # Sell signal: Fast TRIMA crosses below Slow TRIMA
        elif crossover(self.trima_slow, self.trima_fast):
            if self.position:
                self.position.close()

        # Check stop loss if we have a position
        elif self.position and self.trades:
            # Get the most recent trade entry price
            current_trade = self.trades[-1]
            stop_loss_price = current_trade.entry_price - (self.atr_multiple * self.atr[-1])

            if self.data.Close[-1] <= stop_loss_price:
                self.position.close()

    @classmethod
    def get_min_required_days(cls):
        """Calculate minimum required trading days for this strategy."""
        # TRIMA needs 2 * period samples
        return (2 * cls.slow_period) + cls.min_trading_days_buffer

    @classmethod
    def get_recommended_start_date(cls, end_date: str) -> str:
        """Calculate recommended start date given an end date.

        Args:
            end_date: End date in YYYY-MM-DD format

        Returns:
            Recommended start date as string
        """
        # Convert to trading days (approximately 252 trading days per year)
        required_trading_days = cls.get_min_required_days()
        # Add 20% buffer for weekends/holidays
        required_calendar_days = int(required_trading_days * 1.4)

        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=required_calendar_days)

        return start_dt.strftime("%Y-%m-%d")


class VAMAStrategy(BaseStrategy):
    """Volume Adjusted Moving Average (VAMA) Strategy.

    VAMA calculates a volume-weighted moving average by adjusting the
    traditional moving average based on volume. It places more weight
    on periods with higher volume, making it more responsive to
    volume-driven price movements.

    Formula:
    Volume Price = Volume * Price
    Volume Sum = Rolling mean of Volume over period
    Volume Ratio = Volume Price / Volume Sum
    Cumulative Sum = Rolling sum of (Volume Ratio * Price) over period
    Cumulative Divisor = Rolling sum of Volume Ratio over period
    VAMA = Cumulative Sum / Cumulative Divisor
    """

    # VAMA calculation period
    vama_period = 8

    # Slow VAMA for crossover signals
    slow_vama_period = 21

    # ATR-based stop loss
    atr_period = 14
    atr_multiple = 1.5

    # Minimum required data buffer
    min_trading_days_buffer = 20

    def init(self):
        """Initialize VAMA and ATR indicators."""

        # Validate we have enough data
        total_data_points = len(self.data)
        required_data_points = self.slow_vama_period + self.min_trading_days_buffer

        if total_data_points < required_data_points:
            import warnings

            warnings.warn(
                f"Insufficient data for VAMAStrategy: "
                f"Have {total_data_points} days, need at least {required_data_points} days "
                f"({self.slow_vama_period} for slow VAMA + {self.min_trading_days_buffer} buffer). "
                f"Strategy may not generate any signals.",
                stacklevel=2,
            )

        def vama(prices, volumes, period=8):
            """Calculate Volume Adjusted Moving Average (VAMA)."""
            prices = pd.Series(prices)
            volumes = pd.Series(volumes)

            # Calculate volume * price
            vp = volumes * prices

            # Calculate rolling mean of volume
            volsum = volumes.rolling(window=period).mean()

            # Calculate volume ratio
            vol_ratio = pd.Series(0.0, index=prices.index)
            mask = volsum != 0
            vol_ratio[mask] = vp[mask] / volsum[mask]

            # Calculate cumulative sum and divisor
            cum_sum = (vol_ratio * prices).rolling(window=period).sum()
            cum_div = vol_ratio.rolling(window=period).sum()

            # Calculate VAMA
            vama_values = pd.Series(0.0, index=prices.index)
            div_mask = cum_div != 0
            vama_values[div_mask] = cum_sum[div_mask] / cum_div[div_mask]

            return vama_values

        def atr(high, low, close, period=14):
            """Calculate Average True Range."""
            high = pd.Series(high)
            low = pd.Series(low)
            close = pd.Series(close)

            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return atr

        # Initialize VAMAs for fast and slow periods
        self.vama_fast = self.I(vama, self.data.Close, self.data.Volume, self.vama_period)
        self.vama_slow = self.I(vama, self.data.Close, self.data.Volume, self.slow_vama_period)

        # Initialize ATR for stop loss calculation
        self.atr = self.I(atr, self.data.High, self.data.Low, self.data.Close, self.atr_period)

    def next(self):
        """Execute VAMA crossover trading logic."""
        # Skip if we dont have enough data
        if len(self.data) < self.slow_vama_period:
            return

        # Buy signal: Fast VAMA crosses above Slow VAMA
        if crossover(self.vama_fast, self.vama_slow):
            if not self.position:
                self.buy()

        # Sell signal: Fast VAMA crosses below Slow VAMA
        elif crossover(self.vama_slow, self.vama_fast):
            if self.position:
                self.position.close()

        # Check stop loss if we have a position
        elif self.position and self.trades:
            # Get the most recent trade entry price
            current_trade = self.trades[-1]
            stop_loss_price = current_trade.entry_price - (self.atr_multiple * self.atr[-1])

            if self.data.Close[-1] <= stop_loss_price:
                self.position.close()

    @classmethod
    def get_min_required_days(cls):
        """Calculate minimum required trading days for this strategy."""
        return cls.slow_vama_period + cls.min_trading_days_buffer

    @classmethod
    def get_recommended_start_date(cls, end_date: str) -> str:
        """Calculate recommended start date given an end date.

        Args:
            end_date: End date in YYYY-MM-DD format

        Returns:
            Recommended start date as string
        """
        # Convert to trading days (approximately 252 trading days per year)
        required_trading_days = cls.get_min_required_days()
        # Add 20% buffer for weekends/holidays
        required_calendar_days = int(required_trading_days * 1.4)

        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=required_calendar_days)

        return start_dt.strftime("%Y-%m-%d")


class KaufmanEfficiencyStrategy(BaseStrategy):
    """Kaufman Efficiency Ratio (ER) Strategy.

    The Efficiency Ratio measures the price change relative to the
    volatility over a period. Higher efficiency indicates strong trending
    markets, while lower efficiency indicates ranging/noisy markets.

    Formula:
    Direction = abs(Close - Close[n periods ago])
    Volatility = Sum of abs(Close - Previous Close) over n periods
    ER = Direction / Volatility

    Trading Logic:
    - Buy when ER is above upper threshold (trending up strongly)
    - Sell when ER is below lower threshold (losing trend strength)
    """

    # Efficiency Ratio calculation period
    er_period = 10

    # ER thresholds for trading signals
    er_upper_threshold = 0.5  # Enter long positions
    er_lower_threshold = 0.3  # Exit positions

    # Additional trend confirmation
    use_price_trend = True
    price_trend_period = 5

    # ATR-based stop loss
    atr_period = 14
    atr_multiple = 1.8

    # Minimum required data buffer
    min_trading_days_buffer = 20

    def init(self):
        """Initialize ER indicator and price trend confirmation."""

        # Validate we have enough data
        total_data_points = len(self.data)
        required_data_points = max(self.er_period, self.price_trend_period) + self.min_trading_days_buffer

        if total_data_points < required_data_points:
            import warnings

            warnings.warn(
                f"Insufficient data for KaufmanEfficiencyStrategy: "
                f"Have {total_data_points} days, need at least {required_data_points} days "
                f"(max({self.er_period}, {self.price_trend_period})="
                f"{max(self.er_period, self.price_trend_period)} + "
                f"{self.min_trading_days_buffer} buffer). "
                f"Strategy may not generate any signals.",
                stacklevel=2,
            )

        def efficiency_ratio(prices, period=10):
            """Calculate Kaufman Efficiency Ratio (ER)."""
            prices = pd.Series(prices)

            # Calculate absolute price change over period
            change = abs(prices - prices.shift(period))

            # Calculate volatility (sum of absolute daily changes)
            volatility = prices.diff().abs().rolling(window=period).sum()

            # Calculate Efficiency Ratio
            er = pd.Series(0.0, index=prices.index)
            mask = volatility != 0
            er[mask] = change[mask] / volatility[mask]

            return er

        def simple_trend(prices, period=5):
            """Simple price trend indicator - returns 1 for up, -1 for down."""
            prices = pd.Series(prices)
            trend = pd.Series(0, index=prices.index)

            for i in range(period, len(prices)):
                if prices.iloc[i] > prices.iloc[i - period]:
                    trend.iloc[i] = 1
                elif prices.iloc[i] < prices.iloc[i - period]:
                    trend.iloc[i] = -1

            return trend

        def atr(high, low, close, period=14):
            """Calculate Average True Range."""
            high = pd.Series(high)
            low = pd.Series(low)
            close = pd.Series(close)

            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return atr

        # Initialize Efficiency Ratio
        self.er = self.I(efficiency_ratio, self.data.Close, self.er_period)

        # Initialize price trend if enabled
        if self.use_price_trend:
            self.price_trend = self.I(simple_trend, self.data.Close, self.price_trend_period)

        # Initialize ATR for stop loss calculation
        self.atr = self.I(atr, self.data.High, self.data.Low, self.data.Close, self.atr_period)

    def next(self):
        """Execute Efficiency Ratio trading logic."""
        # Skip if we dont have enough data
        min_data_needed = max(self.er_period, self.price_trend_period) if self.use_price_trend else self.er_period
        if len(self.data) < min_data_needed:
            return

        current_er = self.er[-1]

        # Check for valid ER value
        if pd.isna(current_er):
            return

        # Buy signal: High efficiency (strong trend) + optional price trend confirmation
        if current_er > self.er_upper_threshold:
            # Additional trend confirmation if enabled
            trend_confirmed = True
            if self.use_price_trend and hasattr(self, "price_trend"):
                trend_confirmed = self.price_trend[-1] == 1

            if trend_confirmed and not self.position:
                self.buy()

        # Sell signal: Low efficiency (weak trend)
        elif current_er < self.er_lower_threshold:
            if self.position:
                self.position.close()

        # Check stop loss if we have a position
        elif self.position and self.trades:
            # Get the most recent trade entry price
            current_trade = self.trades[-1]
            stop_loss_price = current_trade.entry_price - (self.atr_multiple * self.atr[-1])

            if self.data.Close[-1] <= stop_loss_price:
                self.position.close()

    @classmethod
    def get_min_required_days(cls):
        """Calculate minimum required trading days for this strategy."""
        min_period = max(cls.er_period, cls.price_trend_period) if cls.use_price_trend else cls.er_period
        return min_period + cls.min_trading_days_buffer

    @classmethod
    def get_recommended_start_date(cls, end_date: str) -> str:
        """Calculate recommended start date given an end date.

        Args:
            end_date: End date in YYYY-MM-DD format

        Returns:
            Recommended start date as string
        """
        # Convert to trading days (approximately 252 trading days per year)
        required_trading_days = cls.get_min_required_days()
        # Add 20% buffer for weekends/holidays
        required_calendar_days = int(required_trading_days * 1.4)

        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=required_calendar_days)

        return start_dt.strftime("%Y-%m-%d")
