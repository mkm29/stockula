"""Trading strategies for backtesting."""

from backtesting import Strategy
from backtesting.lib import crossover
import pandas as pd
from datetime import datetime, timedelta


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
        self.sma_fast = self.I(
            lambda x: pd.Series(x).rolling(self.fast_period).mean(), self.data.Close
        )
        self.sma_slow = self.I(
            lambda x: pd.Series(x).rolling(self.slow_period).mean(), self.data.Close
        )

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
                f"Strategy may not generate any signals."
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
        self.atr = self.I(
            atr, self.data.High, self.data.Low, self.data.Close, self.atr_period
        )

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
                f"(3*{self.slow_period}-2={3 * self.slow_period - 2} for slow TEMA + {self.min_trading_days_buffer} buffer). "
                f"Strategy may not generate any signals."
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
        self.atr = self.I(
            atr, self.data.High, self.data.Low, self.data.Close, self.atr_period
        )

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
            stop_loss_price = current_trade.entry_price - (
                self.atr_multiple * self.atr[-1]
            )

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
