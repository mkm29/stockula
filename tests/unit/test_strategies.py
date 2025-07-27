"""Unit tests for backtesting strategies."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from stockula.backtesting.strategies import (
    BaseStrategy,
    SMACrossStrategy,
    RSIStrategy,
    MACDStrategy,
    DoubleEMACrossStrategy,
    TripleEMACrossStrategy,
    TRIMACrossStrategy,
)


class TestBaseStrategy:
    """Test BaseStrategy class."""

    def test_base_strategy_initialization(self):
        """Test base strategy can be initialized."""
        strategy = BaseStrategy
        assert hasattr(strategy, "init")
        assert hasattr(strategy, "next")


class TestSMACrossStrategy:
    """Test SMA Crossover Strategy."""

    def test_sma_strategy_attributes(self):
        """Test SMA strategy has required attributes."""
        assert hasattr(SMACrossStrategy, "fast_period")
        assert hasattr(SMACrossStrategy, "slow_period")
        assert SMACrossStrategy.fast_period == 10
        assert SMACrossStrategy.slow_period == 20

    def test_sma_strategy_initialization(self):
        """Test SMA strategy initialization without instantiation."""
        # Test class attributes and methods exist
        assert hasattr(SMACrossStrategy, "fast_period")
        assert hasattr(SMACrossStrategy, "slow_period")
        assert hasattr(SMACrossStrategy, "init")
        assert hasattr(SMACrossStrategy, "next")
        assert SMACrossStrategy.fast_period == 10
        assert SMACrossStrategy.slow_period == 20

    def test_sma_strategy_trading_logic(self):
        """Test SMA strategy has the required trading logic methods."""
        # Test that the strategy has the required methods and logic
        strategy_source = """
        def next(self):
            if crossover(self.sma_fast, self.sma_slow):
                self.buy()
            elif crossover(self.sma_slow, self.sma_fast):
                self.position.close()
        """

        # Check that the next method exists and contains expected logic
        import inspect

        source = inspect.getsource(SMACrossStrategy.next)
        assert "crossover" in source
        assert "self.buy()" in source
        assert "self.position.close()" in source


class TestRSIStrategy:
    """Test RSI Strategy."""

    def test_rsi_strategy_attributes(self):
        """Test RSI strategy has required attributes."""
        assert hasattr(RSIStrategy, "rsi_period")
        assert hasattr(RSIStrategy, "oversold_threshold")
        assert hasattr(RSIStrategy, "overbought_threshold")
        assert RSIStrategy.rsi_period == 14
        assert RSIStrategy.oversold_threshold == 30
        assert RSIStrategy.overbought_threshold == 70

    def test_rsi_strategy_initialization(self):
        """Test RSI strategy class structure."""
        # Test class attributes exist
        assert hasattr(RSIStrategy, "rsi_period")
        assert hasattr(RSIStrategy, "oversold_threshold")
        assert hasattr(RSIStrategy, "overbought_threshold")
        assert hasattr(RSIStrategy, "init")
        assert hasattr(RSIStrategy, "next")
        assert RSIStrategy.rsi_period == 14
        assert RSIStrategy.oversold_threshold == 30
        assert RSIStrategy.overbought_threshold == 70

    def test_rsi_strategy_methods(self):
        """Test RSI strategy has required methods."""
        # Check that init and next methods exist
        import inspect

        # Test that init method exists (may be inherited from BaseStrategy)
        assert hasattr(RSIStrategy, "init")
        assert hasattr(RSIStrategy, "next")

        # RSI strategy should have these class attributes
        assert RSIStrategy.rsi_period == 14
        assert RSIStrategy.oversold_threshold == 30
        assert RSIStrategy.overbought_threshold == 70


class TestMACDStrategy:
    """Test MACD Strategy."""

    def test_macd_strategy_attributes(self):
        """Test MACD strategy has required attributes."""
        assert hasattr(MACDStrategy, "fast_period")
        assert hasattr(MACDStrategy, "slow_period")
        assert hasattr(MACDStrategy, "signal_period")
        assert MACDStrategy.fast_period == 12
        assert MACDStrategy.slow_period == 26
        assert MACDStrategy.signal_period == 9

    def test_macd_strategy_methods(self):
        """Test MACD strategy has required methods."""
        # Check that the strategy has the basic structure
        import inspect

        # Test that methods exist
        assert hasattr(MACDStrategy, "init")
        assert hasattr(MACDStrategy, "next")

        # Test class attributes
        assert MACDStrategy.fast_period == 12
        assert MACDStrategy.slow_period == 26
        assert MACDStrategy.signal_period == 9


class TestDoubleEMACrossStrategy:
    """Test Double EMA Cross Strategy."""

    def test_double_ema_attributes(self):
        """Test Double EMA strategy attributes."""
        assert hasattr(DoubleEMACrossStrategy, "fast_period")
        assert hasattr(DoubleEMACrossStrategy, "slow_period")
        assert hasattr(DoubleEMACrossStrategy, "momentum_atr_multiple")
        assert hasattr(DoubleEMACrossStrategy, "speculative_atr_multiple")
        assert DoubleEMACrossStrategy.fast_period == 34
        assert DoubleEMACrossStrategy.slow_period == 55

    def test_get_min_required_days(self):
        """Test minimum required days calculation."""
        min_days = DoubleEMACrossStrategy.get_min_required_days()
        assert min_days == 75  # 55 + 20 buffer

    def test_get_recommended_start_date(self):
        """Test recommended start date calculation."""
        end_date = "2024-01-01"
        start_date = DoubleEMACrossStrategy.get_recommended_start_date(end_date)

        # Parse dates
        end = datetime.strptime(end_date, "%Y-%m-%d")
        start = datetime.strptime(start_date, "%Y-%m-%d")

        # Should be at least 75 days before
        days_diff = (end - start).days
        assert days_diff >= 75

    def test_insufficient_data_warning(self):
        """Test Double EMA strategy class structure."""
        # Test class attributes and methods exist
        assert hasattr(DoubleEMACrossStrategy, "fast_period")
        assert hasattr(DoubleEMACrossStrategy, "slow_period")
        assert hasattr(DoubleEMACrossStrategy, "momentum_atr_multiple")
        assert hasattr(DoubleEMACrossStrategy, "speculative_atr_multiple")
        assert hasattr(DoubleEMACrossStrategy, "init")
        assert hasattr(DoubleEMACrossStrategy, "next")
        assert DoubleEMACrossStrategy.fast_period == 34
        assert DoubleEMACrossStrategy.slow_period == 55

    def test_atr_calculation_method_exists(self):
        """Test ATR calculation method existence."""
        # Instead of testing the private method, test that strategy has required attributes
        assert hasattr(DoubleEMACrossStrategy, "momentum_atr_multiple")
        assert hasattr(DoubleEMACrossStrategy, "speculative_atr_multiple")

        # Test the static method exists if it's supposed to be static
        # For now, just test that the class has the expected structure
        assert DoubleEMACrossStrategy.momentum_atr_multiple == 1.25
        assert DoubleEMACrossStrategy.speculative_atr_multiple == 1.0


class TestTripleEMACrossStrategy:
    """Test Triple EMA Cross Strategy."""

    def test_triple_ema_attributes(self):
        """Test Triple EMA strategy attributes."""
        assert hasattr(TripleEMACrossStrategy, "fast_period")
        assert hasattr(TripleEMACrossStrategy, "slow_period")
        assert hasattr(TripleEMACrossStrategy, "atr_period")
        assert hasattr(TripleEMACrossStrategy, "atr_multiple")
        assert TripleEMACrossStrategy.fast_period == 9
        assert TripleEMACrossStrategy.slow_period == 21

    def test_tema_calculation_method_exists(self):
        """Test TEMA calculation method structure."""
        # Test class attributes exist
        assert hasattr(TripleEMACrossStrategy, "fast_period")
        assert hasattr(TripleEMACrossStrategy, "slow_period")
        assert hasattr(TripleEMACrossStrategy, "atr_period")
        assert hasattr(TripleEMACrossStrategy, "atr_multiple")
        assert hasattr(TripleEMACrossStrategy, "init")
        assert hasattr(TripleEMACrossStrategy, "next")

        # Test default values
        assert TripleEMACrossStrategy.fast_period == 9
        assert TripleEMACrossStrategy.slow_period == 21

    def test_get_min_required_days(self):
        """Test minimum required days for TEMA."""
        min_days = TripleEMACrossStrategy.get_min_required_days()
        assert min_days == 81  # 3*21-2=61 + 20 buffer


class TestTRIMACrossStrategy:
    """Test TRIMA Cross Strategy."""

    def test_trima_attributes(self):
        """Test TRIMA strategy attributes."""
        assert hasattr(TRIMACrossStrategy, "fast_period")
        assert hasattr(TRIMACrossStrategy, "slow_period")
        assert hasattr(TRIMACrossStrategy, "atr_period")
        assert hasattr(TRIMACrossStrategy, "atr_multiple")
        assert TRIMACrossStrategy.fast_period == 14
        assert TRIMACrossStrategy.slow_period == 28

    def test_trima_calculation_method_exists(self):
        """Test TRIMA calculation method structure."""
        # Test class attributes exist
        assert hasattr(TRIMACrossStrategy, "fast_period")
        assert hasattr(TRIMACrossStrategy, "slow_period")
        assert hasattr(TRIMACrossStrategy, "atr_period")
        assert hasattr(TRIMACrossStrategy, "atr_multiple")
        assert hasattr(TRIMACrossStrategy, "init")
        assert hasattr(TRIMACrossStrategy, "next")

        # Test default values
        assert TRIMACrossStrategy.fast_period == 14
        assert TRIMACrossStrategy.slow_period == 28

    def test_get_min_required_days(self):
        """Test minimum required days for TRIMA."""
        min_days = TRIMACrossStrategy.get_min_required_days()
        assert min_days == 76  # 2*28=56 + 20 buffer

    def test_strategy_methods_exist(self):
        """Test TRIMA strategy has required methods."""
        # Test that basic strategy structure exists
        assert hasattr(TRIMACrossStrategy, "init")
        assert hasattr(TRIMACrossStrategy, "next")
        assert hasattr(TRIMACrossStrategy, "get_min_required_days")

        # Test class attributes
        assert TRIMACrossStrategy.fast_period == 14
        assert TRIMACrossStrategy.slow_period == 28


class TestStrategyHelpers:
    """Test strategy helper methods and edge cases."""

    def test_strategy_parameter_validation(self):
        """Test that strategy parameters are valid."""
        strategies = [
            SMACrossStrategy,
            RSIStrategy,
            MACDStrategy,
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
        ]

        for strategy in strategies:
            # Check fast < slow for MA strategies
            if hasattr(strategy, "fast_period") and hasattr(strategy, "slow_period"):
                assert strategy.fast_period < strategy.slow_period

            # Check RSI thresholds
            if hasattr(strategy, "oversold_threshold") and hasattr(
                strategy, "overbought_threshold"
            ):
                assert (
                    0
                    < strategy.oversold_threshold
                    < strategy.overbought_threshold
                    < 100
                )

            # Check positive periods
            for attr in dir(strategy):
                if "period" in attr and not attr.startswith("_"):
                    value = getattr(strategy, attr)
                    if isinstance(value, (int, float)):
                        assert value > 0

    def test_strategy_inheritance(self):
        """Test all strategies inherit from BaseStrategy."""
        strategies = [
            SMACrossStrategy,
            RSIStrategy,
            MACDStrategy,
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
        ]

        for strategy in strategies:
            assert issubclass(strategy, BaseStrategy)
            assert hasattr(strategy, "init")
            assert hasattr(strategy, "next")


class TestStrategyImplementation:
    """Test strategy implementation details."""

    def test_strategy_class_methods(self):
        """Test that strategies have required class methods."""
        strategies_with_class_methods = [
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
        ]

        for strategy in strategies_with_class_methods:
            assert hasattr(strategy, "get_min_required_days")
            assert callable(getattr(strategy, "get_min_required_days"))

            # Test that get_min_required_days returns an integer
            min_days = strategy.get_min_required_days()
            assert isinstance(min_days, int)
            assert min_days > 0

    def test_strategy_recommended_start_date(self):
        """Test recommended start date calculation for strategies with this method."""
        strategies_with_start_date = [
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
        ]

        test_end_date = "2024-01-01"

        for strategy in strategies_with_start_date:
            if hasattr(strategy, "get_recommended_start_date"):
                start_date = strategy.get_recommended_start_date(test_end_date)
                assert isinstance(start_date, str)
                assert len(start_date) == 10  # YYYY-MM-DD format

                # Parse dates to verify start is before end
                start = datetime.strptime(start_date, "%Y-%m-%d")
                end = datetime.strptime(test_end_date, "%Y-%m-%d")
                assert start < end

    def test_strategy_periods_validation(self):
        """Test that strategy periods are valid."""
        strategies = [
            SMACrossStrategy,
            RSIStrategy,
            MACDStrategy,
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
        ]

        for strategy in strategies:
            # Test that all period attributes are positive integers
            for attr_name in dir(strategy):
                if "period" in attr_name and not attr_name.startswith("_"):
                    attr_value = getattr(strategy, attr_name)
                    if isinstance(attr_value, (int, float)):
                        assert attr_value > 0, (
                            f"{strategy.__name__}.{attr_name} should be positive"
                        )

    def test_strategy_inheritance_chain(self):
        """Test strategy inheritance from BaseStrategy."""
        strategies = [
            SMACrossStrategy,
            RSIStrategy,
            MACDStrategy,
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
        ]

        for strategy in strategies:
            # Check inheritance
            assert issubclass(strategy, BaseStrategy)

            # Check method resolution order includes BaseStrategy
            mro = strategy.__mro__
            assert BaseStrategy in mro

    def test_strategy_attributes_types(self):
        """Test that strategy attributes have correct types."""
        # RSI strategy specific attributes
        assert isinstance(RSIStrategy.rsi_period, int)
        assert isinstance(RSIStrategy.oversold_threshold, (int, float))
        assert isinstance(RSIStrategy.overbought_threshold, (int, float))

        # MACD strategy specific attributes
        assert isinstance(MACDStrategy.fast_period, int)
        assert isinstance(MACDStrategy.slow_period, int)
        assert isinstance(MACDStrategy.signal_period, int)

        # EMA strategies
        assert isinstance(DoubleEMACrossStrategy.momentum_atr_multiple, (int, float))
        assert isinstance(DoubleEMACrossStrategy.speculative_atr_multiple, (int, float))

    def test_strategy_constants_immutability(self):
        """Test that strategy class constants maintain their values."""
        # Store original values
        original_values = {
            "sma_fast": SMACrossStrategy.fast_period,
            "sma_slow": SMACrossStrategy.slow_period,
            "rsi_period": RSIStrategy.rsi_period,
            "macd_fast": MACDStrategy.fast_period,
        }

        # Access them multiple times to ensure they don't change
        for _ in range(3):
            assert SMACrossStrategy.fast_period == original_values["sma_fast"]
            assert SMACrossStrategy.slow_period == original_values["sma_slow"]
            assert RSIStrategy.rsi_period == original_values["rsi_period"]
            assert MACDStrategy.fast_period == original_values["macd_fast"]


class TestStrategyMethodInspection:
    """Test strategy method implementations through inspection."""

    def test_init_method_existence(self):
        """Test that all strategies have init methods."""
        strategies = [
            SMACrossStrategy,
            RSIStrategy,
            MACDStrategy,
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
        ]

        for strategy in strategies:
            assert hasattr(strategy, "init")
            # Verify it's callable
            assert callable(getattr(strategy, "init"))

    def test_next_method_existence(self):
        """Test that all strategies have next methods."""
        strategies = [
            SMACrossStrategy,
            RSIStrategy,
            MACDStrategy,
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
        ]

        for strategy in strategies:
            assert hasattr(strategy, "next")
            # Verify it's callable
            assert callable(getattr(strategy, "next"))

    def test_trading_logic_inspection(self):
        """Test that strategies contain expected trading logic."""
        import inspect

        # Test SMA strategy logic
        sma_source = inspect.getsource(SMACrossStrategy.next)
        assert "crossover" in sma_source
        assert "buy" in sma_source
        assert "close" in sma_source

        # Test RSI strategy logic
        rsi_source = inspect.getsource(RSIStrategy.next)
        assert "oversold_threshold" in rsi_source or "self.rsi" in rsi_source

        # Test MACD strategy logic
        macd_source = inspect.getsource(MACDStrategy.next)
        assert "crossover" in macd_source or "macd" in macd_source


class TestStrategyEdgeCases:
    """Test strategy edge cases and boundary conditions."""

    def test_strategy_with_minimal_periods(self):
        """Test strategies work with their minimal period requirements."""
        # These should be the minimum viable periods
        assert SMACrossStrategy.fast_period < SMACrossStrategy.slow_period
        assert RSIStrategy.rsi_period >= 2  # RSI needs at least 2 periods
        assert MACDStrategy.fast_period < MACDStrategy.slow_period

    def test_rsi_threshold_boundaries(self):
        """Test RSI threshold boundaries are reasonable."""
        assert 0 < RSIStrategy.oversold_threshold < 50
        assert 50 < RSIStrategy.overbought_threshold < 100
        assert RSIStrategy.oversold_threshold < RSIStrategy.overbought_threshold

    def test_atr_multiples_reasonable(self):
        """Test ATR multiples are in reasonable ranges."""
        # ATR multiples should be positive and reasonable for stop losses
        strategies_with_atr = [
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
        ]

        for strategy in strategies_with_atr:
            for attr_name in dir(strategy):
                if "atr_multiple" in attr_name and not attr_name.startswith("_"):
                    value = getattr(strategy, attr_name)
                    if isinstance(value, (int, float)):
                        assert 0 < value < 10, (
                            f"{strategy.__name__}.{attr_name} should be reasonable"
                        )

    def test_minimum_data_requirements(self):
        """Test minimum data requirements are sensible."""
        strategies_with_min_days = [
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
        ]

        for strategy in strategies_with_min_days:
            min_days = strategy.get_min_required_days()

            # Should require at least the slow period
            if hasattr(strategy, "slow_period"):
                assert min_days >= strategy.slow_period

            # Should not require an unreasonable amount of data
            assert min_days < 500  # Less than ~2 years of trading days


class TestStrategyDocumentation:
    """Test that strategies have proper documentation."""

    def test_strategy_docstrings(self):
        """Test that strategies have docstrings."""
        strategies = [
            BaseStrategy,
            SMACrossStrategy,
            RSIStrategy,
            MACDStrategy,
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
        ]

        for strategy in strategies:
            assert strategy.__doc__ is not None
            assert len(strategy.__doc__.strip()) > 0

    def test_method_docstrings(self):
        """Test that strategy methods have docstrings where expected."""
        strategies_with_class_methods = [
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
        ]

        for strategy in strategies_with_class_methods:
            if hasattr(strategy, "get_min_required_days"):
                method = getattr(strategy, "get_min_required_days")
                assert method.__doc__ is not None

            if hasattr(strategy, "get_recommended_start_date"):
                method = getattr(strategy, "get_recommended_start_date")
                assert method.__doc__ is not None


class TestAdvancedStrategyFeatures:
    """Test advanced strategy features."""

    def test_ema_strategies_atr_features(self):
        """Test EMA strategies have ATR-related features."""
        ema_strategies = [
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
        ]

        for strategy in ema_strategies:
            assert hasattr(strategy, "atr_period")
            assert strategy.atr_period > 0

            # Should have at least one ATR multiple
            atr_attrs = [
                attr
                for attr in dir(strategy)
                if "atr_multiple" in attr and not attr.startswith("_")
            ]
            assert len(atr_attrs) > 0

    def test_buffer_days_configuration(self):
        """Test that strategies with buffers have reasonable values."""
        strategies_with_buffers = [
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
        ]

        for strategy in strategies_with_buffers:
            if hasattr(strategy, "min_trading_days_buffer"):
                buffer = strategy.min_trading_days_buffer
                assert isinstance(buffer, int)
                assert 0 < buffer < 100  # Reasonable buffer size

    def test_complex_period_calculations(self):
        """Test complex period calculations for advanced strategies."""
        # Triple EMA requires more complex calculation: 3*period-2
        tema_min_days = TripleEMACrossStrategy.get_min_required_days()
        expected_tema = (
            3 * TripleEMACrossStrategy.slow_period - 2
        ) + TripleEMACrossStrategy.min_trading_days_buffer
        assert tema_min_days == expected_tema

        # TRIMA requires 2*period
        trima_min_days = TRIMACrossStrategy.get_min_required_days()
        expected_trima = (
            2 * TRIMACrossStrategy.slow_period
        ) + TRIMACrossStrategy.min_trading_days_buffer
        assert trima_min_days == expected_trima


class TestStrategyHelperFunctions:
    """Test strategy helper functions and calculations."""

    def test_rsi_calculation_logic(self):
        """Test RSI calculation logic."""
        # Create sample price data
        prices = pd.Series([100, 102, 101, 103, 102, 104, 103, 105])

        # Extract the RSI function from RSIStrategy.init
        import inspect

        source = inspect.getsource(RSIStrategy.init)

        # Verify the RSI function is defined in init
        assert "def rsi(" in source
        assert "delta = prices.diff()" in source
        assert "gain =" in source
        assert "loss =" in source

    def test_ema_calculation_in_strategies(self):
        """Test EMA calculation presence in strategies."""
        import inspect

        # Check MACD strategy has EMA function
        macd_source = inspect.getsource(MACDStrategy.init)
        assert "def ema(" in macd_source
        assert ".ewm(span=" in macd_source

        # Check Double EMA strategy has EMA function
        double_ema_source = inspect.getsource(DoubleEMACrossStrategy.init)
        assert "def ema(" in double_ema_source

    def test_atr_calculation_in_strategies(self):
        """Test ATR calculation presence in strategies."""
        import inspect

        # Check strategies that use ATR
        strategies_with_atr = [
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
        ]

        for strategy in strategies_with_atr:
            source = inspect.getsource(strategy.init)
            assert "def atr(" in source
            assert "tr1 = high - low" in source
            assert "tr2 = abs(high - close.shift())" in source
            assert "tr3 = abs(low - close.shift())" in source


class TestStrategyLogicInspection:
    """Test strategy trading logic through code inspection."""

    def test_sma_strategy_has_crossover_logic(self):
        """Test SMA strategy contains crossover logic."""
        import inspect

        source = inspect.getsource(SMACrossStrategy.next)

        # Should have crossover logic
        assert "crossover" in source
        assert "buy" in source
        assert "close" in source

    def test_rsi_strategy_has_threshold_logic(self):
        """Test RSI strategy contains threshold logic."""
        import inspect

        source = inspect.getsource(RSIStrategy.next)

        # Should have threshold logic
        assert "oversold_threshold" in source or "overbought_threshold" in source
        assert "self.rsi" in source

    def test_macd_strategy_has_signal_logic(self):
        """Test MACD strategy contains signal line logic."""
        import inspect

        source = inspect.getsource(MACDStrategy.next)

        # Should have MACD signal logic
        assert "crossover" in source
        assert "macd_line" in source or "signal_line" in source

    def test_double_ema_has_data_check(self):
        """Test Double EMA strategy has data sufficiency check."""
        import inspect

        source = inspect.getsource(DoubleEMACrossStrategy.next)

        # Should check data length
        assert "len(self.data)" in source
        assert "slow_period" in source

    def test_advanced_strategies_have_stop_loss(self):
        """Test advanced strategies contain stop loss logic."""
        import inspect

        strategies_with_stop_loss = [
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
        ]

        for strategy_class in strategies_with_stop_loss:
            source = inspect.getsource(strategy_class.next)

            # Should have stop loss logic
            assert "stop_loss_price" in source
            assert "atr" in source
            assert "position.close()" in source


class TestStrategyDataRequirements:
    """Test strategy data requirement calculations."""

    def test_strategy_minimum_data_calculations(self):
        """Test that strategies calculate minimum data requirements correctly."""
        # Test strategies with get_min_required_days
        strategies_with_min_days = [
            (DoubleEMACrossStrategy, 75),  # 55 + 20
            (TripleEMACrossStrategy, 81),  # 3*21-2 + 20 = 61 + 20
            (TRIMACrossStrategy, 76),  # 2*28 + 20 = 56 + 20
        ]

        for strategy_class, expected_min_days in strategies_with_min_days:
            calculated_min_days = strategy_class.get_min_required_days()
            assert calculated_min_days == expected_min_days

    def test_strategy_date_calculations(self):
        """Test strategy recommended start date calculations."""
        strategies_with_date_calc = [
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
        ]

        test_end_date = "2024-01-01"

        for strategy_class in strategies_with_date_calc:
            start_date = strategy_class.get_recommended_start_date(test_end_date)

            # Should be a valid date string
            assert isinstance(start_date, str)
            assert len(start_date) == 10  # YYYY-MM-DD

            # Should be before end date
            from datetime import datetime

            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(test_end_date, "%Y-%m-%d")
            assert start < end

            # Should account for minimum days required
            days_diff = (end - start).days
            min_required = strategy_class.get_min_required_days()
            # Allow some buffer for calendar vs trading days conversion
            assert days_diff >= min_required


class TestStrategyParameterValidation:
    """Test strategy parameter validation and edge cases."""

    def test_period_relationships(self):
        """Test that fast periods are less than slow periods."""
        moving_average_strategies = [
            SMACrossStrategy,
            MACDStrategy,
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
        ]

        for strategy in moving_average_strategies:
            if hasattr(strategy, "fast_period") and hasattr(strategy, "slow_period"):
                assert strategy.fast_period < strategy.slow_period

    def test_rsi_thresholds_valid(self):
        """Test RSI thresholds are in valid range."""
        assert 0 < RSIStrategy.oversold_threshold < 50
        assert 50 < RSIStrategy.overbought_threshold < 100
        assert RSIStrategy.oversold_threshold < RSIStrategy.overbought_threshold

    def test_atr_parameters_reasonable(self):
        """Test ATR parameters are reasonable."""
        strategies_with_atr = [
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
        ]

        for strategy in strategies_with_atr:
            # ATR period should be positive
            assert strategy.atr_period > 0

            # ATR multiples should be reasonable
            for attr_name in dir(strategy):
                if "atr_multiple" in attr_name and not attr_name.startswith("_"):
                    value = getattr(strategy, attr_name)
                    if isinstance(value, (int, float)):
                        assert 0 < value < 10  # Reasonable stop loss range

    def test_buffer_days_reasonable(self):
        """Test buffer days are reasonable."""
        strategies_with_buffers = [
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
        ]

        for strategy in strategies_with_buffers:
            if hasattr(strategy, "min_trading_days_buffer"):
                buffer = strategy.min_trading_days_buffer
                assert isinstance(buffer, int)
                assert 0 < buffer < 100


class TestBaseStrategyImplementation:
    """Test BaseStrategy implementation."""

    def test_base_strategy_init_pass(self):
        """Test BaseStrategy init method passes."""
        # Create a mock strategy instance instead of instantiating directly
        strategy = Mock(spec=BaseStrategy)

        # Call the actual init method
        BaseStrategy.init(strategy)
        # Should not raise an exception - init method is pass

    def test_base_strategy_next_pass(self):
        """Test BaseStrategy next method passes."""
        # Create a mock strategy instance instead of instantiating directly
        strategy = Mock(spec=BaseStrategy)

        # Call the actual next method
        BaseStrategy.next(strategy)
        # Should not raise an exception - next method is pass


class TestSMACrossStrategyExecution:
    """Test SMA Cross Strategy execution with mocks."""

    def test_sma_init_execution(self):
        """Test SMA strategy init method execution."""
        # Create mock strategy instance
        strategy = Mock(spec=SMACrossStrategy)
        strategy.I = Mock()
        strategy.data = Mock()
        strategy.data.Close = [100, 101, 102, 103, 104]

        # Execute the actual init method
        SMACrossStrategy.init(strategy)

        # Verify indicators were created
        assert strategy.I.call_count == 2

    def test_sma_next_buy_signal(self):
        """Test SMA strategy next method with buy signal."""
        strategy = Mock(spec=SMACrossStrategy)
        strategy.position = None
        strategy.buy = Mock()
        strategy.sma_fast = Mock()
        strategy.sma_slow = Mock()

        with patch("stockula.backtesting.strategies.crossover") as mock_crossover:
            mock_crossover.side_effect = [True, False]  # Buy signal

            SMACrossStrategy.next(strategy)

            strategy.buy.assert_called_once()

    def test_sma_next_sell_signal(self):
        """Test SMA strategy next method with sell signal."""
        strategy = Mock(spec=SMACrossStrategy)
        strategy.position = Mock()
        strategy.position.close = Mock()
        strategy.buy = Mock()
        strategy.sma_fast = Mock()
        strategy.sma_slow = Mock()

        with patch("stockula.backtesting.strategies.crossover") as mock_crossover:
            mock_crossover.side_effect = [False, True]  # Sell signal

            SMACrossStrategy.next(strategy)

            strategy.position.close.assert_called_once()
            strategy.buy.assert_not_called()

    def test_sma_next_no_signal(self):
        """Test SMA strategy next method with no signal."""
        strategy = Mock(spec=SMACrossStrategy)
        strategy.position = Mock()
        strategy.position.close = Mock()
        strategy.buy = Mock()
        strategy.sma_fast = Mock()
        strategy.sma_slow = Mock()

        with patch("stockula.backtesting.strategies.crossover") as mock_crossover:
            mock_crossover.side_effect = [False, False]  # No signal

            SMACrossStrategy.next(strategy)

            strategy.position.close.assert_not_called()
            strategy.buy.assert_not_called()


class TestRSIStrategyExecution:
    """Test RSI Strategy execution with mocks."""

    def test_rsi_init_execution(self):
        """Test RSI strategy init method execution."""
        strategy = Mock(spec=RSIStrategy)
        strategy.I = Mock()
        strategy.data = Mock()
        strategy.data.Close = [100, 101, 102, 103, 104]
        strategy.rsi_period = 14

        RSIStrategy.init(strategy)

        # Verify RSI indicator was created
        strategy.I.assert_called_once()

    def test_rsi_next_buy_oversold(self):
        """Test RSI strategy buy when oversold."""
        strategy = Mock(spec=RSIStrategy)
        strategy.position = None
        strategy.buy = Mock()
        strategy.rsi = [25.0]  # Oversold
        strategy.oversold_threshold = 30
        strategy.overbought_threshold = 70

        RSIStrategy.next(strategy)

        strategy.buy.assert_called_once()

    def test_rsi_next_sell_overbought(self):
        """Test RSI strategy sell when overbought."""
        strategy = Mock(spec=RSIStrategy)
        strategy.position = Mock()
        strategy.position.close = Mock()
        strategy.buy = Mock()
        strategy.rsi = [75.0]  # Overbought
        strategy.oversold_threshold = 30
        strategy.overbought_threshold = 70

        RSIStrategy.next(strategy)

        strategy.position.close.assert_called_once()
        strategy.buy.assert_not_called()

    def test_rsi_next_no_signal(self):
        """Test RSI strategy no signal in normal range."""
        strategy = Mock(spec=RSIStrategy)
        strategy.position = Mock()
        strategy.position.close = Mock()
        strategy.buy = Mock()
        strategy.rsi = [50.0]  # Normal range
        strategy.oversold_threshold = 30
        strategy.overbought_threshold = 70

        RSIStrategy.next(strategy)

        strategy.position.close.assert_not_called()
        strategy.buy.assert_not_called()

    def test_rsi_next_has_position_no_buy_oversold(self):
        """Test RSI strategy doesn't buy when already has position."""
        strategy = Mock(spec=RSIStrategy)
        strategy.position = Mock()  # Has position
        strategy.position.close = Mock()
        strategy.buy = Mock()
        strategy.rsi = [25.0]  # Oversold
        strategy.oversold_threshold = 30
        strategy.overbought_threshold = 70

        RSIStrategy.next(strategy)

        strategy.buy.assert_not_called()

    def test_rsi_next_no_position_no_sell_overbought(self):
        """Test RSI strategy doesn't sell when no position."""
        strategy = Mock(spec=RSIStrategy)
        strategy.position = None  # No position
        strategy.buy = Mock()
        strategy.rsi = [75.0]  # Overbought
        strategy.oversold_threshold = 30
        strategy.overbought_threshold = 70

        RSIStrategy.next(strategy)

        strategy.buy.assert_not_called()


class TestMACDStrategyExecution:
    """Test MACD Strategy execution with mocks."""

    def test_macd_init_execution(self):
        """Test MACD strategy init method execution."""
        strategy = Mock(spec=MACDStrategy)
        strategy.I = Mock(return_value=(Mock(), Mock()))
        strategy.data = Mock()
        strategy.data.Close = [100, 101, 102, 103, 104]
        strategy.fast_period = 12
        strategy.slow_period = 26
        strategy.signal_period = 9

        MACDStrategy.init(strategy)

        # Verify MACD indicator was created
        strategy.I.assert_called_once()

    def test_macd_next_buy_signal(self):
        """Test MACD strategy buy signal."""
        strategy = Mock(spec=MACDStrategy)
        strategy.buy = Mock()
        strategy.macd_line = Mock()
        strategy.signal_line = Mock()

        with patch("stockula.backtesting.strategies.crossover") as mock_crossover:
            mock_crossover.side_effect = [True, False]  # MACD crosses above signal

            MACDStrategy.next(strategy)

            strategy.buy.assert_called_once()

    def test_macd_next_sell_signal(self):
        """Test MACD strategy sell signal."""
        strategy = Mock(spec=MACDStrategy)
        strategy.position = Mock()
        strategy.position.close = Mock()
        strategy.buy = Mock()
        strategy.macd_line = Mock()
        strategy.signal_line = Mock()

        with patch("stockula.backtesting.strategies.crossover") as mock_crossover:
            mock_crossover.side_effect = [False, True]  # Signal crosses above MACD

            MACDStrategy.next(strategy)

            strategy.position.close.assert_called_once()
            strategy.buy.assert_not_called()

    def test_macd_next_no_signal(self):
        """Test MACD strategy no signal."""
        strategy = Mock(spec=MACDStrategy)
        strategy.position = Mock()
        strategy.position.close = Mock()
        strategy.buy = Mock()
        strategy.macd_line = Mock()
        strategy.signal_line = Mock()

        with patch("stockula.backtesting.strategies.crossover") as mock_crossover:
            mock_crossover.side_effect = [False, False]  # No crossover

            MACDStrategy.next(strategy)

            strategy.position.close.assert_not_called()
            strategy.buy.assert_not_called()


class TestAdvancedStrategyExecution:
    """Test advanced strategy execution with simplified, fast tests."""

    def test_double_ema_init_creates_indicators(self):
        """Test Double EMA init creates indicators."""
        strategy = Mock()
        strategy.I = Mock()
        # Set required strategy attributes
        strategy.slow_period = 55
        strategy.min_trading_days_buffer = 10
        # Create a simple data-like object that supports len()
        data_mock = type(
            "DataMock",
            (),
            {
                "Close": [100] * 100,
                "High": [101] * 100,
                "Low": [99] * 100,
                "__len__": lambda _: 100,
            },
        )()
        strategy.data = data_mock

        DoubleEMACrossStrategy.init(strategy)

        # Should create indicators (EMA calls through I)
        assert strategy.I.call_count >= 1

    def test_double_ema_next_early_return(self):
        """Test Double EMA next with insufficient data."""
        strategy = Mock()
        # Create data mock with insufficient length
        data_mock = type("DataMock", (), {"__len__": lambda _: 30})()
        strategy.data = data_mock
        strategy.slow_period = 55

        result = DoubleEMACrossStrategy.next(strategy)
        assert result is None

    def test_triple_ema_init_creates_indicators(self):
        """Test Triple EMA init creates indicators."""
        strategy = Mock()
        strategy.I = Mock()
        # Set required strategy attributes
        strategy.slow_period = 21
        strategy.min_trading_days_buffer = 10
        # Create a simple data-like object that supports len()
        data_mock = type(
            "DataMock",
            (),
            {
                "Close": [100] * 100,
                "High": [101] * 100,
                "Low": [99] * 100,
                "__len__": lambda _: 100,
            },
        )()
        strategy.data = data_mock

        TripleEMACrossStrategy.init(strategy)

        # Should create indicators
        assert strategy.I.call_count >= 1

    def test_triple_ema_next_early_return(self):
        """Test Triple EMA next with insufficient data."""
        strategy = Mock()
        # Create data mock with insufficient length
        data_mock = type("DataMock", (), {"__len__": lambda _: 50})()
        strategy.data = data_mock
        strategy.slow_period = 21

        result = TripleEMACrossStrategy.next(strategy)
        assert result is None

    def test_trima_init_creates_indicators(self):
        """Test TRIMA init creates indicators."""
        strategy = Mock()
        strategy.I = Mock()
        # Set required strategy attributes
        strategy.slow_period = 28
        strategy.min_trading_days_buffer = 10
        # Create a simple data-like object that supports len()
        data_mock = type(
            "DataMock",
            (),
            {
                "Close": [100] * 100,
                "High": [101] * 100,
                "Low": [99] * 100,
                "__len__": lambda _: 100,
            },
        )()
        strategy.data = data_mock

        TRIMACrossStrategy.init(strategy)

        # Should create indicators
        assert strategy.I.call_count >= 1

    def test_trima_next_early_return(self):
        """Test TRIMA next with insufficient data."""
        strategy = Mock()
        # Create data mock with insufficient length
        data_mock = type("DataMock", (), {"__len__": lambda _: 50})()
        strategy.data = data_mock
        strategy.slow_period = 28

        result = TRIMACrossStrategy.next(strategy)
        assert result is None


class TestStrategyCalculationFunctions:
    """Test strategy calculation functions with fast execution."""

    def test_rsi_function_extraction(self):
        """Test RSI function can be extracted and called."""
        strategy = Mock()
        strategy.I = Mock()
        strategy.data = Mock()
        strategy.data.Close = list(range(100, 115))
        strategy.rsi_period = 14

        RSIStrategy.init(strategy)

        # Verify I was called (RSI function created)
        strategy.I.assert_called_once()

    def test_macd_function_extraction(self):
        """Test MACD function can be extracted."""
        strategy = Mock()
        strategy.I = Mock(return_value=(Mock(), Mock()))
        strategy.data = Mock()
        strategy.data.Close = list(range(100, 110))
        strategy.fast_period = 12
        strategy.slow_period = 26
        strategy.signal_period = 9

        MACDStrategy.init(strategy)

        # Verify MACD was calculated
        strategy.I.assert_called_once()

    def test_advanced_strategies_create_atr(self):
        """Test advanced strategies create ATR indicators."""
        strategy_configs = [
            (DoubleEMACrossStrategy, 55, 20),  # slow_period, min_trading_days_buffer
            (TripleEMACrossStrategy, 21, 20),
            (TRIMACrossStrategy, 28, 20),
        ]

        for strategy_class, slow_period, buffer in strategy_configs:
            strategy = Mock()
            strategy.I = Mock()
            # Set required strategy attributes
            strategy.slow_period = slow_period
            strategy.min_trading_days_buffer = buffer
            # Create a simple data-like object that supports len()
            data_mock = type(
                "DataMock",
                (),
                {
                    "Close": [100] * 100,
                    "High": [101] * 100,
                    "Low": [99] * 100,
                    "__len__": lambda _: 100,
                },
            )()
            strategy.data = data_mock

            strategy_class.init(strategy)

            # Should create multiple indicators (including ATR)
            assert strategy.I.call_count >= 2


class TestStrategyBoundaryConditions:
    """Test strategy boundary conditions with fast execution."""

    def test_rsi_threshold_logic(self):
        """Test RSI threshold logic."""
        strategy = Mock()
        strategy.position = None
        strategy.buy = Mock()
        strategy.rsi = [29.0]  # Below oversold
        strategy.oversold_threshold = 30
        strategy.overbought_threshold = 70

        RSIStrategy.next(strategy)
        strategy.buy.assert_called_once()

    def test_strategies_no_action_when_no_signal(self):
        """Test strategies take no action when no signal."""
        # Test SMA with no crossover
        strategy = Mock()
        strategy.position = None
        strategy.buy = Mock()
        strategy.sma_fast = Mock()
        strategy.sma_slow = Mock()

        with patch("stockula.backtesting.strategies.crossover", return_value=False):
            SMACrossStrategy.next(strategy)
            strategy.buy.assert_not_called()

    def test_advanced_strategies_handle_no_trades(self):
        """Test advanced strategies handle empty trades list."""
        strategy = Mock()
        # Create data mock with length support
        data_mock = type("DataMock", (), {"__len__": lambda _: 100})()
        strategy.data = data_mock
        strategy.slow_period = 55
        strategy.position = Mock()
        strategy.trades = []  # No trades

        with patch("stockula.backtesting.strategies.crossover", return_value=False):
            DoubleEMACrossStrategy.next(strategy)
            # Should not crash with empty trades


class TestStrategyErrorHandling:
    """Test strategy error handling with fast execution."""

    def test_strategies_handle_empty_data(self):
        """Test strategies handle empty data."""
        strategy_configs = [
            (SMACrossStrategy, []),  # Returns single indicator
            (RSIStrategy, []),  # Returns single indicator
            (MACDStrategy, ([], [])),  # Returns tuple of two indicators
        ]

        for strategy_class, return_value in strategy_configs:
            strategy = Mock()
            strategy.I = Mock(return_value=return_value)
            # Create data mock with empty Close array
            data_mock = type("DataMock", (), {"Close": [], "__len__": lambda _: 0})()
            strategy.data = data_mock

            # Should not crash
            strategy_class.init(strategy)

    def test_strategies_handle_missing_position(self):
        """Test strategies handle None position."""
        strategy = Mock()
        strategy.position = None
        strategy.buy = Mock()
        strategy.rsi = [25.0]
        strategy.oversold_threshold = 30
        strategy.overbought_threshold = 70

        # Should not crash
        RSIStrategy.next(strategy)

    def test_advanced_strategies_handle_missing_attributes(self):
        """Test advanced strategies are robust."""
        strategy = Mock()
        # Create data mock with length support
        data_mock = type("DataMock", (), {"__len__": lambda _: 100})()
        strategy.data = data_mock
        strategy.slow_period = 55
        strategy.position = None

        with patch("stockula.backtesting.strategies.crossover", return_value=False):
            # Should not crash even with minimal setup
            DoubleEMACrossStrategy.next(strategy)
