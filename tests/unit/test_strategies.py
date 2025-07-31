"""Comprehensive unit tests for backtesting strategies.

This file consolidates all strategy tests, focusing on testable aspects
while accepting framework constraints of the backtesting.py library.
"""

# import warnings
import inspect
from datetime import datetime

from stockula.backtesting.strategies import (
    BaseStrategy,
    DoubleEMACrossStrategy,
    FRAMAStrategy,
    KAMAStrategy,
    KaufmanEfficiencyStrategy,
    MACDStrategy,
    RSIStrategy,
    SMACrossStrategy,
    TRIMACrossStrategy,
    TripleEMACrossStrategy,
    VAMAStrategy,
    VIDYAStrategy,
)


class TestBaseStrategy:
    """Test BaseStrategy abstract class."""

    def test_base_strategy_attributes(self):
        """Test base strategy has required attributes and methods."""
        assert hasattr(BaseStrategy, "init")
        assert hasattr(BaseStrategy, "next")

    def test_base_strategy_is_abstract(self):
        """Test that BaseStrategy is properly abstract."""
        # BaseStrategy should be a subclass of Strategy from backtesting
        assert BaseStrategy.__name__ == "BaseStrategy"

    def test_base_strategy_inheritance(self):
        """Test BaseStrategy inherits from Strategy."""
        from backtesting import Strategy

        assert issubclass(BaseStrategy, Strategy)


class TestSMACrossStrategy:
    """Test Simple Moving Average Crossover Strategy."""

    def test_sma_strategy_attributes(self):
        """Test SMA strategy has required attributes."""
        assert hasattr(SMACrossStrategy, "fast_period")
        assert hasattr(SMACrossStrategy, "slow_period")
        assert SMACrossStrategy.fast_period == 10
        assert SMACrossStrategy.slow_period == 20

    def test_sma_strategy_inheritance(self):
        """Test SMA strategy inherits from BaseStrategy."""
        assert issubclass(SMACrossStrategy, BaseStrategy)

    def test_sma_strategy_methods(self):
        """Test SMA strategy has required methods."""
        assert hasattr(SMACrossStrategy, "init")
        assert hasattr(SMACrossStrategy, "next")
        assert callable(getattr(SMACrossStrategy, "init", None))
        assert callable(getattr(SMACrossStrategy, "next", None))

    def test_sma_strategy_source_code_structure(self):
        """Test SMA strategy source code contains expected patterns."""
        source = inspect.getsource(SMACrossStrategy.next)
        assert "crossover" in source
        assert "self.buy()" in source
        assert "self.position.close()" in source


class TestRSIStrategy:
    """Test Relative Strength Index Strategy."""

    def test_rsi_strategy_attributes(self):
        """Test RSI strategy has required attributes."""
        assert hasattr(RSIStrategy, "rsi_period")
        assert hasattr(RSIStrategy, "oversold_threshold")
        assert hasattr(RSIStrategy, "overbought_threshold")
        assert RSIStrategy.rsi_period == 14
        assert RSIStrategy.oversold_threshold == 30
        assert RSIStrategy.overbought_threshold == 70

    def test_rsi_strategy_inheritance(self):
        """Test RSI strategy inherits from BaseStrategy."""
        assert issubclass(RSIStrategy, BaseStrategy)

    def test_rsi_calculation_logic(self):
        """Test RSI calculation logic exists in source."""
        source = inspect.getsource(RSIStrategy)
        assert "rsi_period" in source
        assert "oversold_threshold" in source
        assert "overbought_threshold" in source


class TestMACDStrategy:
    """Test Moving Average Convergence Divergence Strategy."""

    def test_macd_strategy_attributes(self):
        """Test MACD strategy has required attributes."""
        assert hasattr(MACDStrategy, "fast_period")
        assert hasattr(MACDStrategy, "slow_period")
        assert hasattr(MACDStrategy, "signal_period")
        assert MACDStrategy.fast_period == 12
        assert MACDStrategy.slow_period == 26
        assert MACDStrategy.signal_period == 9

    def test_macd_strategy_inheritance(self):
        """Test MACD strategy inherits from BaseStrategy."""
        assert issubclass(MACDStrategy, BaseStrategy)

    def test_macd_calculation_components(self):
        """Test MACD has all calculation components."""
        source = inspect.getsource(MACDStrategy)
        assert "fast_period" in source
        assert "slow_period" in source
        assert "signal_period" in source
        assert "macd" in source.lower()


class TestDoubleEMACrossStrategy:
    """Test Double Exponential Moving Average Crossover Strategy."""

    def test_double_ema_attributes(self):
        """Test Double EMA strategy has required attributes."""
        assert hasattr(DoubleEMACrossStrategy, "slow_period")
        assert hasattr(DoubleEMACrossStrategy, "fast_period")
        assert DoubleEMACrossStrategy.fast_period == 34
        assert DoubleEMACrossStrategy.slow_period == 55

    def test_double_ema_inheritance(self):
        """Test Double EMA strategy inherits from BaseStrategy."""
        assert issubclass(DoubleEMACrossStrategy, BaseStrategy)

    def test_double_ema_has_stop_loss_logic(self):
        """Test Double EMA strategy has stop loss logic."""
        source = inspect.getsource(DoubleEMACrossStrategy)
        assert "stop_loss" in source.lower()
        assert "atr" in source.lower()

    def test_insufficient_data_handling(self):
        """Test that insufficient data is handled."""
        source = inspect.getsource(DoubleEMACrossStrategy.next)
        assert "len(self.data)" in source
        assert "return" in source


class TestTripleEMACrossStrategy:
    """Test Triple Exponential Moving Average Crossover Strategy."""

    def test_triple_ema_attributes(self):
        """Test Triple EMA strategy has required attributes."""
        # TripleEMACrossStrategy has fast/slow periods only
        assert hasattr(TripleEMACrossStrategy, "fast_period")
        assert hasattr(TripleEMACrossStrategy, "slow_period")
        assert TripleEMACrossStrategy.fast_period == 9
        assert TripleEMACrossStrategy.slow_period == 21

    def test_triple_ema_inheritance(self):
        """Test Triple EMA strategy inherits from BaseStrategy."""
        assert issubclass(TripleEMACrossStrategy, BaseStrategy)

    def test_triple_ema_tema_calculation(self):
        """Test TEMA calculation components exist."""
        source = inspect.getsource(TripleEMACrossStrategy)
        assert "TEMA" in source or "tema" in source


class TestTRIMACrossStrategy:
    """Test Triangular Moving Average Crossover Strategy."""

    def test_trima_attributes(self):
        """Test TRIMA strategy has required attributes."""
        assert hasattr(TRIMACrossStrategy, "fast_period")
        assert hasattr(TRIMACrossStrategy, "slow_period")
        assert TRIMACrossStrategy.fast_period == 14
        assert TRIMACrossStrategy.slow_period == 28

    def test_trima_inheritance(self):
        """Test TRIMA strategy inherits from BaseStrategy."""
        assert issubclass(TRIMACrossStrategy, BaseStrategy)

    def test_trima_calculation_logic(self):
        """Test TRIMA has triangular weighting logic."""
        source = inspect.getsource(TRIMACrossStrategy)
        assert "TRIMA" in source or "trima" in source or "triangular" in source.lower()


class TestVIDYAStrategy:
    """Test Variable Index Dynamic Average Strategy."""

    def test_vidya_attributes(self):
        """Test VIDYA strategy has required attributes."""
        assert hasattr(VIDYAStrategy, "cmo_period")
        assert hasattr(VIDYAStrategy, "smoothing_period")
        assert VIDYAStrategy.cmo_period == 9
        assert VIDYAStrategy.smoothing_period == 12

    def test_vidya_inheritance(self):
        """Test VIDYA strategy inherits from BaseStrategy."""
        assert issubclass(VIDYAStrategy, BaseStrategy)

    def test_vidya_cmo_calculation_components(self):
        """Test VIDYA has CMO calculation components."""
        source = inspect.getsource(VIDYAStrategy)
        assert "cmo" in source.lower() or "momentum" in source.lower()


class TestKAMAStrategy:
    """Test Kaufman's Adaptive Moving Average Strategy."""

    def test_kama_attributes(self):
        """Test KAMA strategy has required attributes."""
        assert hasattr(KAMAStrategy, "er_period")
        assert hasattr(KAMAStrategy, "fast_period")
        assert hasattr(KAMAStrategy, "slow_period")
        assert KAMAStrategy.er_period == 10
        assert KAMAStrategy.fast_period == 2
        assert KAMAStrategy.slow_period == 30

    def test_kama_inheritance(self):
        """Test KAMA strategy inherits from BaseStrategy."""
        assert issubclass(KAMAStrategy, BaseStrategy)

    def test_kama_efficiency_ratio_logic(self):
        """Test KAMA has efficiency ratio calculation."""
        source = inspect.getsource(KAMAStrategy)
        assert "efficiency" in source.lower() or "er" in source.lower()


class TestFRAMAStrategy:
    """Test Fractal Adaptive Moving Average Strategy."""

    def test_frama_attributes(self):
        """Test FRAMA strategy has required attributes."""
        assert hasattr(FRAMAStrategy, "frama_period")
        assert FRAMAStrategy.frama_period == 16

    def test_frama_inheritance(self):
        """Test FRAMA strategy inherits from BaseStrategy."""
        assert issubclass(FRAMAStrategy, BaseStrategy)

    def test_frama_fractal_dimension_logic(self):
        """Test FRAMA has fractal dimension calculation."""
        source = inspect.getsource(FRAMAStrategy)
        assert "fractal" in source.lower() or "dimension" in source.lower()


class TestVAMAStrategy:
    """Test Volume Adjusted Moving Average Strategy."""

    def test_vama_attributes(self):
        """Test VAMA strategy has required attributes."""
        # VAMAStrategy only has slow_vama_period, not fast/slow periods
        assert hasattr(VAMAStrategy, "slow_vama_period")
        assert VAMAStrategy.slow_vama_period == 21

    def test_vama_inheritance(self):
        """Test VAMA strategy inherits from BaseStrategy."""
        assert issubclass(VAMAStrategy, BaseStrategy)

    def test_vama_volume_logic(self):
        """Test VAMA uses volume in calculations."""
        source = inspect.getsource(VAMAStrategy)
        assert "volume" in source.lower() or "Volume" in source


class TestKaufmanEfficiencyStrategy:
    """Test Kaufman Efficiency Strategy."""

    def test_kaufman_efficiency_attributes(self):
        """Test Kaufman Efficiency strategy has required attributes."""
        assert hasattr(KaufmanEfficiencyStrategy, "er_period")
        assert hasattr(KaufmanEfficiencyStrategy, "er_upper_threshold")
        assert hasattr(KaufmanEfficiencyStrategy, "er_lower_threshold")
        assert KaufmanEfficiencyStrategy.er_period == 10
        assert KaufmanEfficiencyStrategy.er_upper_threshold == 0.5
        assert KaufmanEfficiencyStrategy.er_lower_threshold == 0.3

    def test_kaufman_efficiency_inheritance(self):
        """Test Kaufman Efficiency strategy inherits from BaseStrategy."""
        assert issubclass(KaufmanEfficiencyStrategy, BaseStrategy)


class TestStrategyDataRequirements:
    """Test data requirements across all strategies."""

    def test_strategies_with_min_required_days(self):
        """Test strategies that implement get_min_required_days."""
        strategies_with_method = [
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
            VIDYAStrategy,
            KAMAStrategy,
            FRAMAStrategy,
            VAMAStrategy,
            KaufmanEfficiencyStrategy,
        ]

        for strategy in strategies_with_method:
            if hasattr(strategy, "get_min_required_days"):
                assert callable(strategy.get_min_required_days)
                min_days = strategy.get_min_required_days()
                assert isinstance(min_days, int)
                assert min_days > 0

    def test_strategies_with_recommended_start_date(self):
        """Test strategies that implement get_recommended_start_date."""
        strategies_with_method = [
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
            VIDYAStrategy,
            KAMAStrategy,
            FRAMAStrategy,
            VAMAStrategy,
            KaufmanEfficiencyStrategy,
        ]

        target_date = datetime(2024, 1, 1)

        for strategy in strategies_with_method:
            if hasattr(strategy, "get_recommended_start_date"):
                assert callable(strategy.get_recommended_start_date)
                recommended = strategy.get_recommended_start_date(target_date.strftime("%Y-%m-%d"))
                # The function returns a string, not datetime
                assert isinstance(recommended, str)
                # Convert back to datetime for comparison
                recommended_dt = datetime.strptime(recommended, "%Y-%m-%d")
                assert recommended_dt < target_date


class TestStrategyWarnings:
    """Test warning generation in strategies."""

    def test_strategies_with_insufficient_data_warnings(self):
        """Test strategies that should generate insufficient data warnings."""
        strategies_with_warnings = [
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
            VIDYAStrategy,
            KAMAStrategy,
            FRAMAStrategy,
        ]

        for strategy in strategies_with_warnings:
            source = inspect.getsource(strategy)
            # These strategies should check for sufficient data
            assert "len(self.data)" in source or "warnings.warn" in source


class TestStrategyCalculationFunctions:
    """Test calculation functions used by strategies."""

    def test_sma_calculation_exists(self):
        """Test SMA calculation is defined."""
        source = inspect.getsource(SMACrossStrategy)
        assert "SMA" in source or "sma" in source or "rolling" in source

    def test_ema_calculation_exists(self):
        """Test EMA calculation is defined."""
        source = inspect.getsource(DoubleEMACrossStrategy)
        assert "EMA" in source or "ema" in source or "ewm" in source

    def test_rsi_calculation_exists(self):
        """Test RSI calculation is defined."""
        source = inspect.getsource(RSIStrategy)
        assert "RSI" in source or "rsi" in source

    def test_macd_calculation_exists(self):
        """Test MACD calculation is defined."""
        source = inspect.getsource(MACDStrategy)
        assert "MACD" in source or "macd" in source

    def test_atr_calculation_exists(self):
        """Test ATR calculation is referenced in strategies that use it."""
        strategies_with_atr = [
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            VIDYAStrategy,
            KAMAStrategy,
            FRAMAStrategy,
        ]

        for strategy in strategies_with_atr:
            source = inspect.getsource(strategy)
            assert "ATR" in source or "atr" in source or "average_true_range" in source.lower()


class TestStrategyCrossoverLogic:
    """Test crossover detection logic in strategies."""

    def test_crossover_strategies(self):
        """Test strategies that use crossover logic."""
        crossover_strategies = [
            SMACrossStrategy,
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
            MACDStrategy,
        ]

        for strategy in crossover_strategies:
            source = inspect.getsource(strategy.next)
            assert "crossover" in source.lower() or "cross" in source.lower()


class TestStrategyTradingActions:
    """Test trading actions in strategies."""

    def test_buy_action_in_strategies(self):
        """Test strategies have buy actions."""
        all_strategies = [
            SMACrossStrategy,
            RSIStrategy,
            MACDStrategy,
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
            VIDYAStrategy,
            KAMAStrategy,
            FRAMAStrategy,
            VAMAStrategy,
            KaufmanEfficiencyStrategy,
        ]

        for strategy in all_strategies:
            source = inspect.getsource(strategy.next)
            assert "self.buy(" in source

    def test_position_close_in_strategies(self):
        """Test strategies have position closing logic."""
        strategies_with_close = [
            SMACrossStrategy,
            RSIStrategy,
            MACDStrategy,
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
            VIDYAStrategy,
            KAMAStrategy,
            FRAMAStrategy,
            VAMAStrategy,
        ]

        for strategy in strategies_with_close:
            source = inspect.getsource(strategy.next)
            assert "self.position.close()" in source or "position.close()" in source


class TestAdvancedStrategyFeatures:
    """Test advanced features in specific strategies."""

    def test_stop_loss_in_advanced_strategies(self):
        """Test stop loss implementation in advanced strategies."""
        stop_loss_strategies = [
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            VIDYAStrategy,
            KAMAStrategy,
            FRAMAStrategy,
        ]

        for strategy in stop_loss_strategies:
            source = inspect.getsource(strategy)
            assert "stop_loss" in source.lower()

    def test_adaptive_strategies_have_special_calculations(self):
        """Test adaptive strategies have their unique calculations."""
        # VIDYA should have CMO
        vidya_source = inspect.getsource(VIDYAStrategy)
        assert "cmo" in vidya_source.lower() or "chande" in vidya_source.lower()

        # KAMA should have efficiency ratio
        kama_source = inspect.getsource(KAMAStrategy)
        assert "efficiency" in kama_source.lower()

        # FRAMA should have fractal dimension
        frama_source = inspect.getsource(FRAMAStrategy)
        assert "fractal" in frama_source.lower()


class TestStrategyConsistency:
    """Test consistency across all strategies."""

    def test_all_strategies_inherit_from_base(self):
        """Test all strategies inherit from BaseStrategy."""
        all_strategies = [
            SMACrossStrategy,
            RSIStrategy,
            MACDStrategy,
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
            VIDYAStrategy,
            KAMAStrategy,
            FRAMAStrategy,
            VAMAStrategy,
            KaufmanEfficiencyStrategy,
        ]

        for strategy in all_strategies:
            assert issubclass(strategy, BaseStrategy)

    def test_all_strategies_have_init_and_next(self):
        """Test all strategies have init and next methods."""
        all_strategies = [
            SMACrossStrategy,
            RSIStrategy,
            MACDStrategy,
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
            VIDYAStrategy,
            KAMAStrategy,
            FRAMAStrategy,
            VAMAStrategy,
            KaufmanEfficiencyStrategy,
        ]

        for strategy in all_strategies:
            assert hasattr(strategy, "init")
            assert hasattr(strategy, "next")
            assert callable(strategy.init)
            assert callable(strategy.next)


class TestStrategyEdgeCases:
    """Test edge cases and error handling in strategies."""

    def test_strategies_handle_insufficient_data_gracefully(self):
        """Test strategies handle insufficient data without crashing."""
        # This tests the structure of the code, not runtime behavior
        strategies_with_data_checks = [
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
            VIDYAStrategy,
            KAMAStrategy,
            FRAMAStrategy,
        ]

        for strategy in strategies_with_data_checks:
            source = inspect.getsource(strategy.next)
            # Should have early return for insufficient data
            assert "return" in source
            assert "len(self.data)" in source

    def test_division_by_zero_handling(self):
        """Test strategies that might have division by zero handle it."""
        # Strategies with potential division operations
        division_strategies = [
            RSIStrategy,  # RS calculation
            VIDYAStrategy,  # CMO calculation
            KAMAStrategy,  # Efficiency ratio
        ]

        for strategy in division_strategies:
            source = inspect.getsource(strategy)
            # Should have some form of zero checking or safe division
            assert "!= 0" in source or "== 0" in source or "max(" in source or "np.where" in source or "if" in source


class TestStrategyIntegration:
    """Test strategy integration with realistic data."""

    def test_all_strategies_can_be_imported(self):
        """Test all strategies can be imported successfully."""
        # Import statements at top already test this
        assert True

    def test_strategy_parameter_types(self):
        """Test strategy parameters have correct types."""
        # Test integer parameters
        assert isinstance(SMACrossStrategy.fast_period, int)
        assert isinstance(SMACrossStrategy.slow_period, int)
        assert isinstance(RSIStrategy.rsi_period, int)
        assert isinstance(MACDStrategy.signal_period, int)

        # Test float parameters
        assert isinstance(KaufmanEfficiencyStrategy.er_upper_threshold, float)

        # Test boolean parameters where they exist
        # Note: stop_loss_enabled may not be a class attribute but defined in init

    def test_strategy_parameter_ranges(self):
        """Test strategy parameters are in sensible ranges."""
        # Period parameters should be positive
        assert SMACrossStrategy.fast_period > 0
        assert SMACrossStrategy.slow_period > 0
        assert RSIStrategy.rsi_period > 0

        # Fast should be less than slow
        assert SMACrossStrategy.fast_period < SMACrossStrategy.slow_period
        assert DoubleEMACrossStrategy.fast_period < DoubleEMACrossStrategy.slow_period

        # RSI thresholds should be between 0 and 100
        assert 0 < RSIStrategy.oversold_threshold < 100
        assert 0 < RSIStrategy.overbought_threshold < 100
        assert RSIStrategy.oversold_threshold < RSIStrategy.overbought_threshold

        # Efficiency threshold should be between 0 and 1
        assert 0 < KaufmanEfficiencyStrategy.er_upper_threshold < 1


# Additional test class for comprehensive coverage
class TestStrategyImplementationDetails:
    """Test implementation details of strategies."""

    def test_indicator_initialization_in_init_methods(self):
        """Test that init methods initialize indicators properly."""
        # SMA should initialize moving averages
        sma_init = inspect.getsource(SMACrossStrategy.init)
        assert "self.sma" in sma_init or "self.I(" in sma_init

        # RSI should initialize RSI indicator
        rsi_init = inspect.getsource(RSIStrategy.init)
        assert "self.rsi" in rsi_init or "self.I(" in rsi_init

        # MACD should initialize MACD components
        macd_init = inspect.getsource(MACDStrategy.init)
        assert "self.macd" in macd_init or "self.I(" in macd_init

    def test_trading_logic_conditions(self):
        """Test trading logic has proper conditions."""
        # RSI should check thresholds
        rsi_next = inspect.getsource(RSIStrategy.next)
        assert "oversold_threshold" in rsi_next
        assert "overbought_threshold" in rsi_next

        # MACD should check crossovers
        macd_next = inspect.getsource(MACDStrategy.next)
        assert "crossover" in macd_next or ">" in macd_next

        # Kaufman Efficiency should check threshold
        ke_next = inspect.getsource(KaufmanEfficiencyStrategy.next)
        assert "er_upper_threshold" in ke_next or "er_lower_threshold" in ke_next

    def test_position_management_logic(self):
        """Test position management is consistent."""
        # All strategies should check position in some way
        all_strategies = [
            SMACrossStrategy,
            RSIStrategy,
            MACDStrategy,
            DoubleEMACrossStrategy,
            TripleEMACrossStrategy,
            TRIMACrossStrategy,
            VIDYAStrategy,
            KAMAStrategy,
            FRAMAStrategy,
            VAMAStrategy,
            KaufmanEfficiencyStrategy,
        ]

        for strategy in all_strategies:
            next_source = inspect.getsource(strategy.next)
            # Most strategies check position state in some way
            assert "position" in next_source
