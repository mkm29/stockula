"""Strategy Registry - Centralized strategy management and mapping."""

from typing import Any, cast

from .strategies import (
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


class StrategyRegistry:
    """Central registry for trading strategies with normalization and mapping capabilities."""

    # All available strategies mapped to their classes
    STRATEGIES = {
        "smacross": SMACrossStrategy,
        "sma_cross": SMACrossStrategy,
        "rsi": RSIStrategy,
        "macd": MACDStrategy,
        "double_ema_cross": DoubleEMACrossStrategy,
        "triple_ema_cross": TripleEMACrossStrategy,
        "trima_cross": TRIMACrossStrategy,
        "kama": KAMAStrategy,
        "frama": FRAMAStrategy,
        "vama": VAMAStrategy,
        "vidya": VIDYAStrategy,
        "kaufman_efficiency": KaufmanEfficiencyStrategy,
    }

    # Strategy name normalization mapping (various formats to canonical snake_case)
    STRATEGY_NAME_MAPPING = {
        # PascalCase variants to snake_case
        "SMACross": "smacross",
        "RSI": "rsi",
        "MACD": "macd",
        "DoubleEMACross": "double_ema_cross",
        "TripleEMACross": "triple_ema_cross",
        "TRIMACross": "trima_cross",
        "KAMA": "kama",
        "FRAMA": "frama",
        "VAMA": "vama",
        "VIDYA": "vidya",
        "KaufmanEfficiency": "kaufman_efficiency",
        "ER": "kaufman_efficiency",  # Alternative name for KaufmanEfficiency
        # snake_case variants (already correct, but included for completeness)
        "smacross": "smacross",
        "sma_cross": "smacross",
        "rsi": "rsi",
        "macd": "macd",
        "double_ema_cross": "double_ema_cross",
        "triple_ema_cross": "triple_ema_cross",
        "trima_cross": "trima_cross",
        "kama": "kama",
        "frama": "frama",
        "vama": "vama",
        "vidya": "vidya",
        "kaufman_efficiency": "kaufman_efficiency",
    }

    # Predefined strategy groups for different trading approaches
    STRATEGY_GROUPS = {
        "basic": ["smacross", "rsi"],
        "momentum": ["rsi", "macd", "double_ema_cross"],
        "trend": ["smacross", "triple_ema_cross", "trima_cross"],
        "advanced": ["kama", "frama", "vama", "vidya"],
        "comprehensive": [
            "smacross",
            "rsi",
            "macd",
            "double_ema_cross",
            "triple_ema_cross",
            "trima_cross",
            "kama",
            "frama",
            "vama",
            "vidya",
            "kaufman_efficiency",
        ],
    }

    # Default parameter presets for strategies
    STRATEGY_PRESETS = {
        "smacross": {"fast_period": 10, "slow_period": 20},
        "sma_cross": {"fast_period": 10, "slow_period": 30},
        "rsi": {"period": 14, "oversold_threshold": 30.0, "overbought_threshold": 70.0},
        "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "double_ema_cross": {"fast_period": 34, "slow_period": 55},
        "triple_ema_cross": {"fast": 5, "medium": 10, "slow": 20},
        "trima_cross": {"fast_period": 14, "slow_period": 28},
        "kama": {"period": 14, "fast_sc": 2, "slow_sc": 30},
        "frama": {"period": 14, "frama_period": 16},
        "vama": {"period": 8, "vama_period": 8, "slow_vama_period": 21},
        "vidya": {"period": 14, "alpha": 0.2, "cmo_period": 9, "smoothing_period": 12},
        "kaufman_efficiency": {"period": 10, "fast_sc": 2, "slow_sc": 30},
    }

    @classmethod
    def normalize_strategy_name(cls, strategy_name: str) -> str:
        """Normalize strategy name to canonical snake_case format.

        Args:
            strategy_name: Strategy name in any format

        Returns:
            Normalized strategy name in snake_case

        Example:
            >>> StrategyRegistry.normalize_strategy_name("SMACross")
            'smacross'
            >>> StrategyRegistry.normalize_strategy_name("DoubleEMACross")
            'double_ema_cross'
        """
        return cls.STRATEGY_NAME_MAPPING.get(strategy_name, strategy_name.lower())

    @classmethod
    def get_strategy_class(cls, strategy_name: str) -> type[BaseStrategy] | None:
        """Get strategy class by name (with automatic normalization).

        Args:
            strategy_name: Strategy name in any format

        Returns:
            Strategy class or None if not found

        Example:
            >>> cls = StrategyRegistry.get_strategy_class("SMACross")
            >>> cls.__name__
            'SMACrossStrategy'
        """
        normalized_name = cls.normalize_strategy_name(strategy_name)
        return cls.STRATEGIES.get(normalized_name)

    @classmethod
    def get_all_strategies(cls) -> dict[str, type[BaseStrategy]]:
        """Get all available strategies.

        Returns:
            Dictionary mapping strategy names to strategy classes
        """
        return cls.STRATEGIES.copy()

    @classmethod
    def get_available_strategy_names(cls) -> list[str]:
        """Get list of all available strategy names (canonical snake_case).

        Returns:
            List of strategy names
        """
        return list(cls.STRATEGIES.keys())

    @classmethod
    def get_strategy_groups(cls) -> dict[str, list[str]]:
        """Get all available strategy groups.

        Returns:
            Dictionary of strategy groups and their constituent strategies
        """
        return cls.STRATEGY_GROUPS.copy()

    @classmethod
    def get_strategy_presets(cls) -> dict[str, dict[str, Any]]:
        """Get default parameter presets for all strategies.

        Returns:
            Dictionary of strategy names and their default parameters
        """
        return cast(dict[str, dict[str, Any]], cls.STRATEGY_PRESETS.copy())

    @classmethod
    def get_strategy_preset(cls, strategy_name: str) -> dict[str, Any]:
        """Get parameter preset for a specific strategy.

        Args:
            strategy_name: Strategy name (will be normalized)

        Returns:
            Dictionary of default parameters for the strategy
        """
        normalized_name = cls.normalize_strategy_name(strategy_name)
        return cast(dict[str, Any], cls.STRATEGY_PRESETS.get(normalized_name, {}))

    @classmethod
    def is_valid_strategy(cls, strategy_name: str) -> bool:
        """Check if a strategy name is valid.

        Args:
            strategy_name: Strategy name to check

        Returns:
            True if the strategy exists, False otherwise
        """
        normalized_name = cls.normalize_strategy_name(strategy_name)
        return normalized_name in cls.STRATEGIES

    @classmethod
    def is_valid_strategy_group(cls, group_name: str) -> bool:
        """Check if a strategy group name is valid.

        Args:
            group_name: Strategy group name to check

        Returns:
            True if the strategy group exists, False otherwise
        """
        return group_name in cls.STRATEGY_GROUPS

    @classmethod
    def get_strategies_in_group(cls, group_name: str) -> list[str]:
        """Get strategies in a specific group.

        Args:
            group_name: Name of the strategy group

        Returns:
            List of strategy names in the group

        Raises:
            ValueError: If the group name is not valid
        """
        if not cls.is_valid_strategy_group(group_name):
            available_groups = list(cls.STRATEGY_GROUPS.keys())
            raise ValueError(f"Unknown strategy group: {group_name}. Available: {available_groups}")

        return cls.STRATEGY_GROUPS[group_name].copy()

    @classmethod
    def validate_strategies(cls, strategy_names: list[str]) -> tuple[list[str], list[str]]:
        """Validate a list of strategy names.

        Args:
            strategy_names: List of strategy names to validate

        Returns:
            Tuple of (valid_strategies, invalid_strategies)
        """
        valid_strategies = []
        invalid_strategies = []

        for strategy_name in strategy_names:
            if cls.is_valid_strategy(strategy_name):
                valid_strategies.append(cls.normalize_strategy_name(strategy_name))
            else:
                invalid_strategies.append(strategy_name)

        return valid_strategies, invalid_strategies
