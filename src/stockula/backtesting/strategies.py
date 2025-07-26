"""Trading strategies for backtesting."""

from backtesting import Strategy
from backtesting.lib import crossover
import pandas as pd


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
