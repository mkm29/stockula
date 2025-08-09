"""Tests for the backtesting strategy registry."""

import pytest

from stockula.backtesting.registry import StrategyRegistry
from stockula.backtesting.strategies import (
    BaseStrategy,
    DoubleEMACrossStrategy,
    KaufmanEfficiencyStrategy,
    MACDStrategy,
    RSIStrategy,
    SMACrossStrategy,
)


class TestStrategyRegistry:
    """Test the StrategyRegistry class."""

    def test_normalize_strategy_name_pascal_case(self):
        """Test normalizing PascalCase strategy names."""
        assert StrategyRegistry.normalize_strategy_name("SMACross") == "smacross"
        assert StrategyRegistry.normalize_strategy_name("DoubleEMACross") == "double_ema_cross"
        assert StrategyRegistry.normalize_strategy_name("TripleEMACross") == "triple_ema_cross"
        assert StrategyRegistry.normalize_strategy_name("KaufmanEfficiency") == "kaufman_efficiency"
        assert StrategyRegistry.normalize_strategy_name("RSI") == "rsi"
        assert StrategyRegistry.normalize_strategy_name("MACD") == "macd"

    def test_normalize_strategy_name_snake_case(self):
        """Test normalizing snake_case strategy names (should remain unchanged)."""
        assert StrategyRegistry.normalize_strategy_name("smacross") == "smacross"
        assert StrategyRegistry.normalize_strategy_name("sma_cross") == "smacross"
        assert StrategyRegistry.normalize_strategy_name("double_ema_cross") == "double_ema_cross"
        assert StrategyRegistry.normalize_strategy_name("triple_ema_cross") == "triple_ema_cross"
        assert StrategyRegistry.normalize_strategy_name("kaufman_efficiency") == "kaufman_efficiency"

    def test_normalize_strategy_name_uppercase(self):
        """Test normalizing uppercase strategy names."""
        assert StrategyRegistry.normalize_strategy_name("RSI") == "rsi"
        assert StrategyRegistry.normalize_strategy_name("MACD") == "macd"
        assert StrategyRegistry.normalize_strategy_name("KAMA") == "kama"
        assert StrategyRegistry.normalize_strategy_name("FRAMA") == "frama"
        assert StrategyRegistry.normalize_strategy_name("VAMA") == "vama"
        assert StrategyRegistry.normalize_strategy_name("VIDYA") == "vidya"

    def test_normalize_strategy_name_alternative_names(self):
        """Test normalizing alternative strategy names."""
        assert StrategyRegistry.normalize_strategy_name("ER") == "kaufman_efficiency"
        assert StrategyRegistry.normalize_strategy_name("sma_cross") == "smacross"

    def test_normalize_strategy_name_unknown(self):
        """Test normalizing unknown strategy names (should return lowercase)."""
        assert StrategyRegistry.normalize_strategy_name("UnknownStrategy") == "unknownstrategy"
        assert StrategyRegistry.normalize_strategy_name("NEWSTRATEGY") == "newstrategy"
        assert StrategyRegistry.normalize_strategy_name("some_new_strategy") == "some_new_strategy"

    def test_get_strategy_class_valid(self):
        """Test getting valid strategy classes."""
        assert StrategyRegistry.get_strategy_class("SMACross") == SMACrossStrategy
        assert StrategyRegistry.get_strategy_class("smacross") == SMACrossStrategy
        assert StrategyRegistry.get_strategy_class("RSI") == RSIStrategy
        assert StrategyRegistry.get_strategy_class("rsi") == RSIStrategy
        assert StrategyRegistry.get_strategy_class("MACD") == MACDStrategy
        assert StrategyRegistry.get_strategy_class("DoubleEMACross") == DoubleEMACrossStrategy
        assert StrategyRegistry.get_strategy_class("double_ema_cross") == DoubleEMACrossStrategy
        assert StrategyRegistry.get_strategy_class("kaufman_efficiency") == KaufmanEfficiencyStrategy
        assert StrategyRegistry.get_strategy_class("ER") == KaufmanEfficiencyStrategy

    def test_get_strategy_class_all_strategies(self):
        """Test that all registered strategies can be retrieved."""
        for name, strategy_class in StrategyRegistry.STRATEGIES.items():
            assert StrategyRegistry.get_strategy_class(name) == strategy_class
            assert issubclass(strategy_class, BaseStrategy)

    def test_get_strategy_class_invalid(self):
        """Test getting invalid strategy returns None."""
        assert StrategyRegistry.get_strategy_class("InvalidStrategy") is None
        assert StrategyRegistry.get_strategy_class("NonExistent") is None
        assert StrategyRegistry.get_strategy_class("") is None

    def test_get_all_strategies(self):
        """Test getting all strategies."""
        strategies = StrategyRegistry.get_all_strategies()

        # Should return a copy
        assert strategies is not StrategyRegistry.STRATEGIES

        # Should contain all expected strategies
        assert len(strategies) == len(StrategyRegistry.STRATEGIES)
        assert "smacross" in strategies
        assert "rsi" in strategies
        assert "macd" in strategies
        assert "double_ema_cross" in strategies

        # All values should be strategy classes
        for _name, strategy_class in strategies.items():
            assert issubclass(strategy_class, BaseStrategy)

    def test_get_available_strategy_names(self):
        """Test getting list of available strategy names."""
        names = StrategyRegistry.get_available_strategy_names()

        # Should contain all expected strategies
        assert "smacross" in names
        assert "rsi" in names
        assert "macd" in names
        assert "double_ema_cross" in names
        assert "triple_ema_cross" in names
        assert "trima_cross" in names
        assert "kama" in names
        assert "frama" in names
        assert "vama" in names
        assert "vidya" in names
        assert "kaufman_efficiency" in names

        # Should match the keys in STRATEGIES
        assert set(names) == set(StrategyRegistry.STRATEGIES.keys())

    def test_get_strategy_groups(self):
        """Test getting strategy groups."""
        groups = StrategyRegistry.get_strategy_groups()

        # Should return a copy
        assert groups is not StrategyRegistry.STRATEGY_GROUPS

        # Should contain expected groups
        assert "basic" in groups
        assert "momentum" in groups
        assert "trend" in groups
        assert "advanced" in groups
        assert "comprehensive" in groups

        # Each group should contain valid strategies
        for _group_name, strategies in groups.items():
            for strategy_name in strategies:
                assert StrategyRegistry.is_valid_strategy(strategy_name)

    def test_get_strategy_presets(self):
        """Test getting all strategy presets."""
        presets = StrategyRegistry.get_strategy_presets()

        # Should return a copy
        assert presets is not StrategyRegistry.STRATEGY_PRESETS

        # Should contain presets for all strategies
        for strategy_name in StrategyRegistry.STRATEGIES.keys():
            assert strategy_name in presets

        # Check some specific presets
        assert presets["smacross"] == {"fast_period": 10, "slow_period": 20}
        assert presets["rsi"] == {"period": 14, "oversold_threshold": 30.0, "overbought_threshold": 70.0}
        assert presets["macd"] == {"fast_period": 12, "slow_period": 26, "signal_period": 9}

    def test_get_strategy_preset_valid(self):
        """Test getting preset for a specific strategy."""
        # Test with exact name
        preset = StrategyRegistry.get_strategy_preset("smacross")
        assert preset == {"fast_period": 10, "slow_period": 20}

        # Test with PascalCase (should normalize)
        preset = StrategyRegistry.get_strategy_preset("SMACross")
        assert preset == {"fast_period": 10, "slow_period": 20}

        # Test RSI preset
        preset = StrategyRegistry.get_strategy_preset("RSI")
        assert preset == {"period": 14, "oversold_threshold": 30.0, "overbought_threshold": 70.0}

        # Test MACD preset
        preset = StrategyRegistry.get_strategy_preset("MACD")
        assert preset == {"fast_period": 12, "slow_period": 26, "signal_period": 9}

    def test_get_strategy_preset_invalid(self):
        """Test getting preset for invalid strategy returns empty dict."""
        preset = StrategyRegistry.get_strategy_preset("InvalidStrategy")
        assert preset == {}

        preset = StrategyRegistry.get_strategy_preset("NonExistent")
        assert preset == {}

    def test_is_valid_strategy(self):
        """Test checking if strategy names are valid."""
        # Valid strategies (various formats)
        assert StrategyRegistry.is_valid_strategy("smacross") is True
        assert StrategyRegistry.is_valid_strategy("SMACross") is True
        assert StrategyRegistry.is_valid_strategy("RSI") is True
        assert StrategyRegistry.is_valid_strategy("rsi") is True
        assert StrategyRegistry.is_valid_strategy("DoubleEMACross") is True
        assert StrategyRegistry.is_valid_strategy("double_ema_cross") is True
        assert StrategyRegistry.is_valid_strategy("ER") is True  # Alternative name

        # Invalid strategies
        assert StrategyRegistry.is_valid_strategy("InvalidStrategy") is False
        assert StrategyRegistry.is_valid_strategy("NonExistent") is False
        assert StrategyRegistry.is_valid_strategy("") is False

    def test_is_valid_strategy_group(self):
        """Test checking if strategy group names are valid."""
        # Valid groups
        assert StrategyRegistry.is_valid_strategy_group("basic") is True
        assert StrategyRegistry.is_valid_strategy_group("momentum") is True
        assert StrategyRegistry.is_valid_strategy_group("trend") is True
        assert StrategyRegistry.is_valid_strategy_group("advanced") is True
        assert StrategyRegistry.is_valid_strategy_group("comprehensive") is True

        # Invalid groups
        assert StrategyRegistry.is_valid_strategy_group("invalid") is False
        assert StrategyRegistry.is_valid_strategy_group("nonexistent") is False
        assert StrategyRegistry.is_valid_strategy_group("") is False

    def test_get_strategies_in_group_valid(self):
        """Test getting strategies in a valid group."""
        # Basic group
        strategies = StrategyRegistry.get_strategies_in_group("basic")
        assert strategies == ["smacross", "rsi"]

        # Momentum group
        strategies = StrategyRegistry.get_strategies_in_group("momentum")
        assert strategies == ["rsi", "macd", "double_ema_cross"]

        # Trend group
        strategies = StrategyRegistry.get_strategies_in_group("trend")
        assert strategies == ["smacross", "triple_ema_cross", "trima_cross"]

        # Advanced group
        strategies = StrategyRegistry.get_strategies_in_group("advanced")
        assert strategies == ["kama", "frama", "vama", "vidya"]

        # Comprehensive group (should contain all strategies)
        strategies = StrategyRegistry.get_strategies_in_group("comprehensive")
        assert len(strategies) == 11
        for strategy in strategies:
            assert StrategyRegistry.is_valid_strategy(strategy)

    def test_get_strategies_in_group_returns_copy(self):
        """Test that get_strategies_in_group returns a copy."""
        strategies = StrategyRegistry.get_strategies_in_group("basic")
        strategies.append("new_strategy")

        # Original should be unchanged
        assert StrategyRegistry.STRATEGY_GROUPS["basic"] == ["smacross", "rsi"]

    def test_get_strategies_in_group_invalid(self):
        """Test getting strategies in an invalid group raises ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy group: invalid"):
            StrategyRegistry.get_strategies_in_group("invalid")

        with pytest.raises(ValueError, match="Unknown strategy group: nonexistent"):
            StrategyRegistry.get_strategies_in_group("nonexistent")

    def test_validate_strategies_all_valid(self):
        """Test validating a list of all valid strategies."""
        strategy_names = ["SMACross", "RSI", "MACD", "double_ema_cross"]
        valid, invalid = StrategyRegistry.validate_strategies(strategy_names)

        # All should be valid (normalized)
        assert set(valid) == {"smacross", "rsi", "macd", "double_ema_cross"}
        assert invalid == []

    def test_validate_strategies_all_invalid(self):
        """Test validating a list of all invalid strategies."""
        strategy_names = ["InvalidStrategy", "NonExistent", "FakeStrategy"]
        valid, invalid = StrategyRegistry.validate_strategies(strategy_names)

        assert valid == []
        assert set(invalid) == {"InvalidStrategy", "NonExistent", "FakeStrategy"}

    def test_validate_strategies_mixed(self):
        """Test validating a mixed list of valid and invalid strategies."""
        strategy_names = ["SMACross", "InvalidStrategy", "RSI", "NonExistent", "MACD"]
        valid, invalid = StrategyRegistry.validate_strategies(strategy_names)

        assert set(valid) == {"smacross", "rsi", "macd"}
        assert set(invalid) == {"InvalidStrategy", "NonExistent"}

    def test_validate_strategies_empty_list(self):
        """Test validating an empty list."""
        valid, invalid = StrategyRegistry.validate_strategies([])

        assert valid == []
        assert invalid == []

    def test_validate_strategies_with_duplicates(self):
        """Test validating a list with duplicate strategies."""
        strategy_names = ["SMACross", "smacross", "RSI", "rsi", "RSI"]
        valid, invalid = StrategyRegistry.validate_strategies(strategy_names)

        # Duplicates should be normalized and kept
        assert valid == ["smacross", "smacross", "rsi", "rsi", "rsi"]
        assert invalid == []

    def test_validate_strategies_with_alternative_names(self):
        """Test validating strategies with alternative names."""
        strategy_names = ["ER", "KaufmanEfficiency", "kaufman_efficiency"]
        valid, invalid = StrategyRegistry.validate_strategies(strategy_names)

        # All should map to kaufman_efficiency
        assert valid == ["kaufman_efficiency", "kaufman_efficiency", "kaufman_efficiency"]
        assert invalid == []

    def test_all_strategies_have_presets(self):
        """Test that all strategies have corresponding presets."""
        for strategy_name in StrategyRegistry.STRATEGIES.keys():
            assert strategy_name in StrategyRegistry.STRATEGY_PRESETS
            preset = StrategyRegistry.STRATEGY_PRESETS[strategy_name]
            assert isinstance(preset, dict)
            assert len(preset) > 0  # Should have at least one parameter

    def test_all_group_strategies_are_valid(self):
        """Test that all strategies in groups are valid."""
        for group_name, strategies in StrategyRegistry.STRATEGY_GROUPS.items():
            for strategy_name in strategies:
                assert StrategyRegistry.is_valid_strategy(strategy_name), (
                    f"Strategy '{strategy_name}' in group '{group_name}' is not valid"
                )

    def test_strategy_class_inheritance(self):
        """Test that all strategy classes inherit from BaseStrategy."""
        for name, strategy_class in StrategyRegistry.STRATEGIES.items():
            assert issubclass(strategy_class, BaseStrategy), (
                f"Strategy class for '{name}' does not inherit from BaseStrategy"
            )
