"""Final comprehensive tests for strategies.py to achieve 661+ lines coverage."""

import warnings
import numpy as np
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


class TestStrategiesFinalCoverage:
    """Final comprehensive tests targeting 661+ lines coverage."""

    def setup_method(self):
        """Set up test fixtures with realistic data."""
        np.random.seed(42)
        self.sample_close = [100 + i + np.random.randn() * 0.5 for i in range(100)]
        self.sample_high = [
            price + abs(np.random.randn() * 2) for price in self.sample_close
        ]
        self.sample_low = [
            price - abs(np.random.randn() * 2) for price in self.sample_close
        ]

    def create_mock_data(self, length=100):
        """Create properly mocked data object."""
        data = Mock()
        data.__len__ = Mock(return_value=length)
        data.Close = self.sample_close[:length]
        data.High = self.sample_high[:length]
        data.Low = self.sample_low[:length]
        return data

    def create_mock_strategy(self, strategy_class):
        """Create a properly configured mock strategy."""
        strategy = Mock()
        strategy.data = self.create_mock_data(100)

        # Set default attributes based on strategy type
        if strategy_class == VIDYAStrategy:
            strategy.cmo_period = 9
            strategy.smoothing_period = 12
            strategy.min_trading_days_buffer = 20
            strategy.atr_period = 14
        elif strategy_class == KAMAStrategy:
            strategy.er_period = 10
            strategy.slow_period = 30
            strategy.fast_period = 2
            strategy.min_trading_days_buffer = 20
            strategy.atr_period = 14
        elif strategy_class == FRAMAStrategy:
            strategy.frama_period = 16
            strategy.min_trading_days_buffer = 20
            strategy.atr_period = 14
        elif strategy_class == TripleEMACrossStrategy:
            strategy.fast_period = 8
            strategy.medium_period = 21
            strategy.slow_period = 55
            strategy.min_trading_days_buffer = 20
            strategy.atr_period = 14
        elif strategy_class == TRIMACrossStrategy:
            strategy.fast_period = 14
            strategy.slow_period = 28
            strategy.min_trading_days_buffer = 20
            strategy.atr_period = 14
        elif strategy_class == DoubleEMACrossStrategy:
            strategy.fast_period = 34
            strategy.slow_period = 55
            strategy.momentum_atr_multiple = 1.25
            strategy.speculative_atr_multiple = 1.0
            strategy.atr_period = 14
            strategy.min_trading_days_buffer = 20

        return strategy

    def test_vidya_complete_execution_cycle(self):
        """Test VIDYA strategy complete execution - covers lines 256-365."""
        strategy = self.create_mock_strategy(VIDYAStrategy)

        # Mock I() to return dummy results
        strategy.I = Mock(side_effect=lambda func, *args, **kwargs: [1.0] * 50)

        # Test init
        VIDYAStrategy.init(strategy)
        assert hasattr(strategy, "vidya")
        assert hasattr(strategy, "atr")

        # Test next with insufficient data
        strategy.data.__len__ = Mock(return_value=20)
        result = VIDYAStrategy.next(strategy)
        assert result is None

        # Test next with sufficient data and buy signal
        strategy.data.__len__ = Mock(return_value=50)
        strategy.data.Close = [101]
        strategy.vidya = [100, 101]
        strategy.position = Mock()
        strategy.position.__bool__ = Mock(return_value=False)
        strategy.buy = Mock()

        with patch("stockula.backtesting.strategies.crossover", return_value=True):
            VIDYAStrategy.next(strategy)
        strategy.buy.assert_called_once()

    def test_kama_complete_execution_cycle(self):
        """Test KAMA strategy complete execution - covers lines 433-566."""
        strategy = self.create_mock_strategy(KAMAStrategy)

        # Mock I() to return dummy results
        strategy.I = Mock(side_effect=lambda func, *args, **kwargs: [1.0] * 80)

        # Test init
        KAMAStrategy.init(strategy)
        assert hasattr(strategy, "kama")
        assert hasattr(strategy, "atr")

        # Test next with insufficient data
        strategy.data.__len__ = Mock(return_value=35)
        result = KAMAStrategy.next(strategy)
        assert result is None

        # Test next with sufficient data and sell signal
        strategy.data.__len__ = Mock(return_value=80)
        strategy.data.Close = [99]
        strategy.kama = [100, 99]
        strategy.position = Mock()
        strategy.position.__bool__ = Mock(return_value=True)
        strategy.position.close = Mock()

        # KAMA uses crossover(data.Close, kama) for sell signal
        with patch(
            "stockula.backtesting.strategies.crossover", side_effect=[False, True]
        ):
            KAMAStrategy.next(strategy)
        strategy.position.close.assert_called_once()

    def test_frama_complete_execution_cycle(self):
        """Test FRAMA strategy complete execution - covers lines 600-744."""
        strategy = self.create_mock_strategy(FRAMAStrategy)

        # Mock I() to return dummy results
        strategy.I = Mock(side_effect=lambda func, *args, **kwargs: [1.0] * 60)

        # Test init
        FRAMAStrategy.init(strategy)
        assert hasattr(strategy, "frama")
        assert hasattr(strategy, "atr")

        # Test next with insufficient data
        strategy.data.__len__ = Mock(return_value=15)
        result = FRAMAStrategy.next(strategy)
        assert result is None

        # Test next with sufficient data and trading
        strategy.data.__len__ = Mock(return_value=60)
        strategy.data.Close = [102]
        strategy.frama = [100, 102]
        strategy.position = Mock()
        strategy.position.__bool__ = Mock(return_value=False)
        strategy.buy = Mock()

        with patch("stockula.backtesting.strategies.crossover", return_value=True):
            FRAMAStrategy.next(strategy)
        strategy.buy.assert_called_once()

    def test_triple_ema_complete_execution_cycle(self):
        """Test TripleEMA strategy complete execution - covers lines 775-920."""
        strategy = self.create_mock_strategy(TripleEMACrossStrategy)

        # Mock I() to return dummy results
        strategy.I = Mock(side_effect=lambda func, *args, **kwargs: [1.0] * 100)

        # Test init
        TripleEMACrossStrategy.init(strategy)
        assert hasattr(strategy, "ema_fast")
        assert hasattr(strategy, "ema_medium")
        assert hasattr(strategy, "ema_slow")

        # Test next with insufficient data
        strategy.data.__len__ = Mock(return_value=50)
        result = TripleEMACrossStrategy.next(strategy)
        assert result is None

        # Test next with sufficient data and buy signal - need more data for TripleEMA
        strategy.data.__len__ = Mock(return_value=200)
        strategy.data = self.create_mock_data(200)
        strategy.ema_fast = [103]
        strategy.ema_medium = [102]
        strategy.ema_slow = [101]
        strategy.position = Mock()
        strategy.position.__bool__ = Mock(return_value=False)
        strategy.buy = Mock()

        # Mock crossover to return True for both checks
        with patch(
            "stockula.backtesting.strategies.crossover", side_effect=[True, True]
        ):
            TripleEMACrossStrategy.next(strategy)
        strategy.buy.assert_called_once()

    def test_trima_complete_execution_cycle(self):
        """Test TRIMA strategy complete execution - covers lines 1426-1572."""
        strategy = self.create_mock_strategy(TRIMACrossStrategy)

        # Mock I() to return dummy results
        strategy.I = Mock(side_effect=lambda func, *args, **kwargs: [1.0] * 80)

        # Test init
        TRIMACrossStrategy.init(strategy)
        assert hasattr(strategy, "trima_fast")
        assert hasattr(strategy, "trima_slow")

        # Test next with insufficient data
        strategy.data.__len__ = Mock(return_value=27)
        result = TRIMACrossStrategy.next(strategy)
        assert result is None

        # Test next with sufficient data and sell signal
        strategy.data.__len__ = Mock(return_value=80)
        strategy.trima_fast = [99]
        strategy.trima_slow = [100]
        strategy.position = Mock()
        strategy.position.__bool__ = Mock(return_value=True)
        strategy.position.close = Mock()

        with patch(
            "stockula.backtesting.strategies.crossover", side_effect=[False, True]
        ):
            TRIMACrossStrategy.next(strategy)
        strategy.position.close.assert_called_once()

    def test_insufficient_data_warnings_comprehensive(self):
        """Test insufficient data warnings across all strategies."""
        warning_test_cases = [
            (VIDYAStrategy, 20),
            (KAMAStrategy, 35),
            (FRAMAStrategy, 30),
            (TripleEMACrossStrategy, 80),
            (TRIMACrossStrategy, 50),
        ]

        for strategy_class, insufficient_length in warning_test_cases:
            strategy = self.create_mock_strategy(strategy_class)
            strategy.data.__len__ = Mock(return_value=insufficient_length)
            strategy.I = Mock(
                side_effect=lambda func, *args, **kwargs: [1.0] * insufficient_length
            )

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                strategy_class.init(strategy)

                # Should trigger warning for insufficient data
                assert len(w) > 0
                assert "Insufficient data" in str(w[0].message)

    def test_class_methods_comprehensive(self):
        """Test class methods and static methods across strategies."""
        # Test DoubleEMA class method
        min_days = DoubleEMACrossStrategy.get_min_required_days()
        assert isinstance(min_days, int)
        assert min_days > 0

        # Test get_recommended_start_date with string end_date
        if hasattr(DoubleEMACrossStrategy, "get_recommended_start_date"):
            end_date = "2023-12-31"
            start_date = DoubleEMACrossStrategy.get_recommended_start_date(end_date)
            assert isinstance(start_date, str)
            # Should be a valid date string
            assert len(start_date) == 10  # YYYY-MM-DD format

        # Test other strategy class methods
        for strategy_class in [
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
            VIDYAStrategy,
            KAMAStrategy,
            FRAMAStrategy,
        ]:
            if hasattr(strategy_class, "get_min_required_days"):
                min_days = strategy_class.get_min_required_days()
                assert isinstance(min_days, int)
                assert min_days > 0

    def test_calculation_functions_direct_execution(self):
        """Test direct execution of calculation functions - covers complex mathematical logic."""

        # Test VIDYA with CMO calculation
        strategy = self.create_mock_strategy(VIDYAStrategy)
        strategy.data.Close = [100.0] * 50  # Flat prices for edge case
        strategy.I = Mock(
            side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            VIDYAStrategy.init(strategy)

        # Test KAMA with efficiency ratio calculation
        strategy = self.create_mock_strategy(KAMAStrategy)
        strategy.data.Close = list(range(100, 180))  # Trending prices
        strategy.I = Mock(
            side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            KAMAStrategy.init(strategy)

        # Test FRAMA with fractal dimension calculation
        strategy = self.create_mock_strategy(FRAMAStrategy)
        strategy.data.Close = [
            100 + np.sin(i / 10) * 5 for i in range(60)
        ]  # Oscillating prices
        strategy.I = Mock(
            side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            FRAMAStrategy.init(strategy)

    def test_stop_loss_logic_comprehensive(self):
        """Test stop loss implementation across strategies."""
        strategy = self.create_mock_strategy(DoubleEMACrossStrategy)
        strategy.data.__len__ = Mock(return_value=100)
        strategy.data.Close = [95]  # Current price

        # Mock position with trade
        strategy.position = Mock()
        strategy.position.__bool__ = Mock(return_value=True)
        strategy.position.close = Mock()

        # Mock recent trade
        mock_trade = Mock()
        mock_trade.entry_price = 100
        strategy.trades = [mock_trade]
        strategy.atr = [2.0]

        # Mock no crossover (so stop loss logic is tested)
        with patch("stockula.backtesting.strategies.crossover", return_value=False):
            DoubleEMACrossStrategy.next(strategy)

        # Should close position due to stop loss
        strategy.position.close.assert_called_once()

    def test_trading_signals_comprehensive(self):
        """Test comprehensive trading signal coverage."""

        # Test RSI oversold/overbought
        for rsi_value, threshold, expected_action in [
            (25, 30, "buy"),  # Oversold
            (75, 70, "sell"),  # Overbought
        ]:
            strategy = Mock()
            strategy.rsi = [rsi_value]
            strategy.oversold_threshold = 30
            strategy.overbought_threshold = 70

            if expected_action == "buy":
                strategy.position = Mock()
                strategy.position.__bool__ = Mock(return_value=False)
                strategy.buy = Mock()

                RSIStrategy.next(strategy)
                strategy.buy.assert_called_once()
            else:
                strategy.position = Mock()
                strategy.position.__bool__ = Mock(return_value=True)
                strategy.position.close = Mock()

                RSIStrategy.next(strategy)
                strategy.position.close.assert_called_once()

    def test_crossover_conditions_comprehensive(self):
        """Test crossover conditions across different strategies."""

        crossover_strategies = [
            (SMACrossStrategy, {"sma_fast": [101], "sma_slow": [100]}),
            (MACDStrategy, {"macd_line": [0.5], "signal_line": [0]}),
        ]

        for strategy_class, indicators in crossover_strategies:
            # Test buy signal
            strategy = Mock()
            strategy.data = self.create_mock_data(60)

            for indicator, values in indicators.items():
                setattr(strategy, indicator, values)

            strategy.position = Mock()
            strategy.position.__bool__ = Mock(return_value=False)
            strategy.buy = Mock()

            with patch("stockula.backtesting.strategies.crossover", return_value=True):
                strategy_class.next(strategy)

            strategy.buy.assert_called_once()

            # Test sell signal
            strategy = Mock()
            strategy.data = self.create_mock_data(60)

            for indicator, values in indicators.items():
                setattr(strategy, indicator, values)

            strategy.position = Mock()
            strategy.position.__bool__ = Mock(return_value=True)
            strategy.position.close = Mock()

            with patch(
                "stockula.backtesting.strategies.crossover", side_effect=[False, True]
            ):
                strategy_class.next(strategy)

            strategy.position.close.assert_called_once()

    def test_base_strategy_coverage(self):
        """Test BaseStrategy methods for coverage."""
        strategy = Mock()

        # These should execute without errors
        BaseStrategy.init(strategy)
        BaseStrategy.next(strategy)

        # Verify methods exist and are callable
        assert callable(BaseStrategy.init)
        assert callable(BaseStrategy.next)

    def test_edge_case_data_scenarios(self):
        """Test edge case data scenarios across strategies."""

        # Test with extreme price movements
        extreme_data = [100, 200, 50, 300, 25, 400, 10, 500]

        for strategy_class in [VIDYAStrategy, KAMAStrategy, FRAMAStrategy]:
            strategy = self.create_mock_strategy(strategy_class)
            strategy.data.Close = extreme_data + [100] * 50
            strategy.data.High = [x + 10 for x in extreme_data] + [110] * 50
            strategy.data.Low = [x - 10 for x in extreme_data] + [90] * 50
            strategy.I = Mock(side_effect=lambda func, *args, **kwargs: [1.0] * 58)

            # Should handle extreme data without crashing
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                strategy_class.init(strategy)
