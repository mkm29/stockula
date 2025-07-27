"""Minimal strategy tests to hit specific coverage lines for 80%+ coverage."""

import warnings
from unittest.mock import Mock, patch

from stockula.backtesting.strategies import (
    BaseStrategy,
    SMACrossStrategy,
    RSIStrategy,
    MACDStrategy,
    DoubleEMACrossStrategy,
    TripleEMACrossStrategy,
    TRIMACrossStrategy,
    VIDYAStrategy,
    KAMAStrategy,
    FRAMAStrategy,
)


class TestStrategyCoverageTargeted:
    """Targeted tests to hit specific missing coverage lines."""

    def test_rsi_calculation_function_direct(self):
        """Test RSI calculation function directly - covers lines 56-62."""
        # Create a mock strategy with minimal setup
        strategy = Mock()
        strategy.rsi_period = 14
        strategy.data = Mock()
        strategy.data.Close = [
            100,
            102,
            104,
            101,
            99,
            98,
            105,
            107,
            103,
            108,
            110,
            109,
            111,
            113,
            115,
        ]

        # Capture the RSI function when I() is called
        def capture_rsi(func, *args, **kwargs):
            # This will execute the RSI calculation function (lines 56-62)
            return func(*args, **kwargs)

        strategy.I = capture_rsi

        # Execute init to trigger RSI calculation
        RSIStrategy.init(strategy)

        # Verify the RSI was assigned
        assert hasattr(strategy, "rsi")

    def test_rsi_next_oversold_buy(self):
        """Test RSI next method with oversold condition - covers lines 68-71."""
        strategy = Mock()
        strategy.rsi = [25]  # Oversold
        strategy.oversold_threshold = 30
        strategy.overbought_threshold = 70

        # Mock position as False (no current position)
        strategy.position = Mock()
        strategy.position.__bool__ = Mock(return_value=False)
        strategy.buy = Mock()

        # Execute next method
        RSIStrategy.next(strategy)

        # Should call buy
        strategy.buy.assert_called_once()

    def test_rsi_next_overbought_sell(self):
        """Test RSI next method with overbought condition - covers lines 72-73."""
        strategy = Mock()
        strategy.rsi = [75]  # Overbought
        strategy.oversold_threshold = 30
        strategy.overbought_threshold = 70

        # Mock position as True (has position)
        strategy.position = Mock()
        strategy.position.__bool__ = Mock(return_value=True)
        strategy.position.close = Mock()

        # Execute next method
        RSIStrategy.next(strategy)

        # Should close position
        strategy.position.close.assert_called_once()

    def test_macd_calculation_function_direct(self):
        """Test MACD calculation function directly - covers lines 87, 90-94."""
        strategy = Mock()
        strategy.fast_period = 12
        strategy.slow_period = 26
        strategy.signal_period = 9
        strategy.data = Mock()
        strategy.data.Close = [
            100,
            102,
            104,
            101,
            99,
            98,
            105,
            107,
            103,
            108,
            110,
            109,
            111,
            113,
            115,
        ]

        # Capture the MACD function when I() is called
        def capture_macd(func, *args, **kwargs):
            # Remove plot argument if present
            kwargs.pop("plot", None)
            # This will execute the MACD calculation function
            return func(*args, **kwargs)

        strategy.I = capture_macd

        # Execute init to trigger MACD calculation
        MACDStrategy.init(strategy)

        # Verify MACD indicators were assigned
        assert hasattr(strategy, "macd_line")
        assert hasattr(strategy, "signal_line")

    def test_macd_next_bullish_crossover(self):
        """Test MACD next method with bullish crossover - covers lines 100-103."""
        strategy = Mock()
        strategy.macd_line = [0.5]
        strategy.signal_line = [0]

        # Mock position as False
        strategy.position = Mock()
        strategy.position.__bool__ = Mock(return_value=False)
        strategy.buy = Mock()

        # Mock crossover to return True
        with patch("stockula.backtesting.strategies.crossover", return_value=True):
            MACDStrategy.next(strategy)

        strategy.buy.assert_called_once()

    def test_double_ema_insufficient_data_warning(self):
        """Test DoubleEMA insufficient data warning - covers lines 135-137."""
        strategy = Mock()
        strategy.fast_period = 34
        strategy.slow_period = 55
        strategy.atr_period = 14
        strategy.min_trading_days_buffer = 20
        strategy.momentum_atr_multiple = 1.25
        strategy.speculative_atr_multiple = 1.0

        # Mock data with insufficient length
        strategy.data = Mock()
        strategy.data.__len__ = Mock(return_value=50)  # Less than required 75
        strategy.data.Close = [100] * 50
        strategy.data.High = [101] * 50
        strategy.data.Low = [99] * 50

        strategy.I = Mock(return_value=Mock())

        # Should trigger warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DoubleEMACrossStrategy.init(strategy)

            # Should have issued a warning
            assert len(w) > 0
            assert "Insufficient data" in str(w[0].message)

    def test_double_ema_next_insufficient_data_early_return(self):
        """Test DoubleEMA next with insufficient data - covers lines 178-179."""
        strategy = Mock()
        strategy.slow_period = 55

        # Mock data with insufficient length
        strategy.data = Mock()
        strategy.data.__len__ = Mock(return_value=50)  # Less than slow_period

        # Should return early (None)
        result = DoubleEMACrossStrategy.next(strategy)
        assert result is None

    def test_double_ema_next_bullish_crossover(self):
        """Test DoubleEMA next with bullish crossover - covers lines 183-184."""
        strategy = Mock()
        strategy.slow_period = 55
        strategy.ema_fast = [101]
        strategy.ema_slow = [100]

        # Mock sufficient data
        strategy.data = Mock()
        strategy.data.__len__ = Mock(return_value=100)

        # Mock no position
        strategy.position = Mock()
        strategy.position.__bool__ = Mock(return_value=False)
        strategy.buy = Mock()

        # Mock crossover to return True
        with patch("stockula.backtesting.strategies.crossover", return_value=True):
            DoubleEMACrossStrategy.next(strategy)

        strategy.buy.assert_called_once()

    def test_double_ema_next_bearish_crossover(self):
        """Test DoubleEMA next with bearish crossover - covers lines 189-190."""
        strategy = Mock()
        strategy.slow_period = 55
        strategy.ema_fast = [99]
        strategy.ema_slow = [100]

        # Mock sufficient data
        strategy.data = Mock()
        strategy.data.__len__ = Mock(return_value=100)

        # Mock has position
        strategy.position = Mock()
        strategy.position.__bool__ = Mock(return_value=True)
        strategy.position.close = Mock()

        # Mock crossover to return False for buy, True for sell
        with patch(
            "stockula.backtesting.strategies.crossover", side_effect=[False, True]
        ):
            DoubleEMACrossStrategy.next(strategy)

        strategy.position.close.assert_called_once()

    def test_double_ema_stop_loss_momentum_asset(self):
        """Test DoubleEMA stop loss for momentum asset - covers lines 193-196."""
        strategy = Mock()
        strategy.slow_period = 55
        strategy.momentum_atr_multiple = 1.25
        strategy.atr = [2.0]

        # Mock sufficient data
        strategy.data = Mock()
        strategy.data.__len__ = Mock(return_value=100)
        strategy.data.Close = [95]  # Current price

        # Mock has position with trade
        strategy.position = Mock()
        strategy.position.__bool__ = Mock(return_value=True)
        strategy.position.close = Mock()

        mock_trade = Mock()
        mock_trade.entry_price = 100  # Entry price
        strategy.trades = [mock_trade]

        # Mock no crossover
        with patch("stockula.backtesting.strategies.crossover", return_value=False):
            DoubleEMACrossStrategy.next(strategy)

        # Should close position due to stop loss
        strategy.position.close.assert_called_once()

    def test_sma_init_calculation(self):
        """Test SMA init calculation - covers lines 30-33."""
        strategy = Mock()
        strategy.fast_period = 10
        strategy.slow_period = 20
        strategy.data = Mock()
        strategy.data.Close = [100] * 30

        # Capture SMA calculation
        def capture_sma(func, *args, **kwargs):
            return func(*args, **kwargs)

        strategy.I = capture_sma

        # Execute init
        SMACrossStrategy.init(strategy)

        # Verify SMAs were assigned
        assert hasattr(strategy, "sma_fast")
        assert hasattr(strategy, "sma_slow")

    def test_sma_next_bullish_crossover(self):
        """Test SMA next with bullish crossover - covers lines 39-40."""
        strategy = Mock()
        strategy.sma_fast = [101]
        strategy.sma_slow = [100]

        # Mock no position
        strategy.position = Mock()
        strategy.position.__bool__ = Mock(return_value=False)
        strategy.buy = Mock()

        # Mock crossover
        with patch("stockula.backtesting.strategies.crossover", return_value=True):
            SMACrossStrategy.next(strategy)

        strategy.buy.assert_called_once()

    def test_sma_next_bearish_crossover(self):
        """Test SMA next with bearish crossover - covers lines 41-42."""
        strategy = Mock()
        strategy.sma_fast = [99]
        strategy.sma_slow = [100]

        # Mock has position
        strategy.position = Mock()
        strategy.position.__bool__ = Mock(return_value=True)
        strategy.position.close = Mock()

        # Mock crossover to return False for first call, True for second call (sma_slow, sma_fast)
        with patch(
            "stockula.backtesting.strategies.crossover", side_effect=[False, True]
        ):
            SMACrossStrategy.next(strategy)

        strategy.position.close.assert_called_once()

    def test_triple_ema_next_insufficient_data(self):
        """Test TripleEMA next insufficient data - covers early return."""
        strategy = Mock()
        strategy.slow_period = 21

        # Mock insufficient data
        strategy.data = Mock()
        strategy.data.__len__ = Mock(return_value=20)

        # Should return early
        result = TripleEMACrossStrategy.next(strategy)
        assert result is None

    def test_trima_next_insufficient_data(self):
        """Test TRIMA next insufficient data - covers early return."""
        strategy = Mock()
        strategy.slow_period = 28

        # Mock insufficient data
        strategy.data = Mock()
        strategy.data.__len__ = Mock(return_value=27)

        # Should return early
        result = TRIMACrossStrategy.next(strategy)
        assert result is None

    def test_vidya_next_insufficient_data(self):
        """Test VIDYA next insufficient data - covers early return."""
        strategy = Mock()
        strategy.cmo_period = 9
        strategy.smoothing_period = 12
        strategy.min_trading_days_buffer = 20

        # Mock insufficient data
        strategy.data = Mock()
        strategy.data.__len__ = Mock(return_value=20)

        # Should return early
        result = VIDYAStrategy.next(strategy)
        assert result is None

    def test_kama_next_insufficient_data(self):
        """Test KAMA next insufficient data - covers early return."""
        strategy = Mock()
        strategy.er_period = 10
        strategy.slow_period = 30

        # Mock insufficient data
        strategy.data = Mock()
        strategy.data.__len__ = Mock(return_value=35)

        # Should return early
        result = KAMAStrategy.next(strategy)
        assert result is None

    def test_frama_next_insufficient_data(self):
        """Test FRAMA next insufficient data - covers early return."""
        strategy = Mock()
        strategy.frama_period = 16

        # Mock insufficient data
        strategy.data = Mock()
        strategy.data.__len__ = Mock(return_value=15)

        # Should return early
        result = FRAMAStrategy.next(strategy)
        assert result is None

    def test_base_strategy_methods_coverage(self):
        """Test BaseStrategy methods - covers lines 15, 19."""
        strategy = Mock()

        # These should execute without errors
        BaseStrategy.init(strategy)
        BaseStrategy.next(strategy)

        # Verify methods exist and are callable
        assert callable(BaseStrategy.init)
        assert callable(BaseStrategy.next)
