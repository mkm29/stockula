"""Strategy Repository - Repository for managing trading strategies with database support."""

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from sqlalchemy.orm import Session

from ..backtesting.strategies import (
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
from ..database.models import Strategy, StrategyPreset
from .repository import Repository

if TYPE_CHECKING:
    from ..database.manager import DatabaseManager


class StrategyRepository(Repository[type[BaseStrategy]]):
    """Repository for trading strategies with database persistence.

    This class manages all available trading strategies and provides methods
    for strategy discovery, validation, and configuration. It can operate
    in both in-memory mode and database-backed mode.
    """

    # Class property: mappings of strategy names to classes
    DEFAULT_MAPPINGS = {
        "smacross": SMACrossStrategy,
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

    # Default strategy groups
    DEFAULT_GROUPS = {
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

    # Default parameter presets
    DEFAULT_PRESETS = {
        "smacross": {"fast_period": 10, "slow_period": 20},
        "rsi": {"period": 14, "oversold_threshold": 30.0, "overbought_threshold": 70.0},
        "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "double_ema_cross": {
            "fast_period": 34,
            "slow_period": 55,
            "momentum_atr_multiple": 1.25,
            "speculative_atr_multiple": 1.0,
            "atr_period": 14,
        },
        "triple_ema_cross": {"fast": 5, "medium": 10, "slow": 20, "fast_period": 9, "slow_period": 21},
        "trima_cross": {"fast_period": 14, "slow_period": 28, "atr_multiple": 1.2},
        "kama": {"period": 14, "fast_sc": 2, "slow_sc": 30, "er_period": 10},
        "frama": {"period": 14, "frama_period": 16},
        "vama": {"period": 8, "vama_period": 8, "slow_vama_period": 21},
        "vidya": {"period": 14, "alpha": 0.2, "cmo_period": 9, "smoothing_period": 12},
        "kaufman_efficiency": {
            "period": 10,
            "fast_sc": 2,
            "slow_sc": 30,
            "er_upper_threshold": 0.5,
            "er_lower_threshold": 0.3,
        },
    }

    def __init__(self, db_manager: "DatabaseManager | None" = None):
        """Initialize the strategy repository.

        Args:
            db_manager: Optional database manager for persistence
        """
        super().__init__()
        self.db_manager = db_manager

        # Initialize groups and values (deep copies to prevent mutation)
        self.groups = deepcopy(self.DEFAULT_GROUPS)
        self._preset_values = deepcopy(self.DEFAULT_PRESETS)

        # Initialize with default strategies
        self._items = self.DEFAULT_MAPPINGS.copy()

        # Load from database if available
        if self.db_manager:
            self._load_from_database()

    def _load_from_database(self) -> None:
        """Load strategies and presets from the database."""
        if not self.db_manager:
            return

        with self.db_manager.get_session() as session:
            # Load active strategies
            strategies = session.query(Strategy).filter(Strategy.is_active).all()
            for strategy in strategies:
                # For now, we only load presets - actual strategy classes
                # would need to be dynamically imported from module_path
                self._load_strategy_presets(session, strategy)

    def _load_strategy_presets(self, session: Session, strategy: Strategy) -> None:
        """Load presets for a specific strategy from the database."""
        presets = session.query(StrategyPreset).filter(StrategyPreset.strategy_id == strategy.id).all()

        for preset in presets:
            if preset.is_default:
                self._preset_values[strategy.name] = preset.parameters

    def sync_to_database(self) -> None:
        """Sync current strategies and presets to the database."""
        if not self.db_manager:
            return

        from sqlalchemy.exc import IntegrityError, OperationalError

        try:
            with self.db_manager.get_session() as session:
                # Sync strategies
                for name, strategy_class in self._items.items():
                    existing = session.query(Strategy).filter(Strategy.name == name).first()
                    if not existing:
                        strategy = Strategy(
                            name=name,
                            class_name=strategy_class.__name__,
                            module_path=strategy_class.__module__,
                            description=strategy_class.__doc__.split("\n")[0] if strategy_class.__doc__ else None,
                            category=self._get_strategy_category(name),
                        )
                        session.add(strategy)
                        session.flush()  # Get the ID

                        # Add default preset
                        if name in self._preset_values:
                            preset = StrategyPreset(
                                strategy_id=strategy.id,
                                name="default",
                                is_default=True,
                            )
                            preset.set_parameters(self._preset_values[name])
                            session.add(preset)

                session.commit()
        except (IntegrityError, OperationalError):
            # Handle cases where:
            # - Strategy already exists (IntegrityError on unique constraint)
            # - Tables don't exist yet (OperationalError)
            # - Concurrent access from multiple processes/threads
            session.rollback()

    def _get_strategy_category(self, strategy_name: str) -> str | None:
        """Determine the category of a strategy based on groups."""
        for group_name, strategies in self.groups.items():
            if strategy_name in strategies and group_name != "comprehensive":
                return group_name
        return None

    @property
    def presets(self) -> dict[str, dict[str, Any]]:
        """Get immutable default presets.

        Returns:
            Dictionary of default presets (read-only deep copy)
        """
        return deepcopy(self.DEFAULT_PRESETS)

    @property
    def values(self) -> dict[str, dict[str, Any]]:
        """Get mutable preset values.

        Returns:
            Dictionary of current preset values (can be modified)
        """
        return self._preset_values

    def add(self, name: str, strategy_class: type[BaseStrategy]) -> None:
        """Add a strategy to the repository.

        Args:
            name: Strategy name (will be converted to lowercase)
            strategy_class: The strategy class
        """
        lowercase_name = name.lower()
        self._items[lowercase_name] = strategy_class

    def get(self, key: str, default=None):
        """Get a strategy by name (case-insensitive).

        Args:
            key: Strategy name in any case
            default: Default value if not found

        Returns:
            Strategy class or default
        """
        return self._items.get(key.lower(), default)

    def remove(self, key: str) -> None:
        """Remove a strategy from the repository.

        Args:
            key: Strategy name to remove

        Raises:
            KeyError: If strategy not found
        """
        lowercase_key = key.lower()
        if lowercase_key not in self._items:
            raise KeyError(f"Strategy '{key}' not found")
        del self._items[lowercase_key]

    def validate(self, name: str) -> None:
        """Validate that a strategy name exists.

        Args:
            name: Strategy name to validate

        Raises:
            ValueError: If the strategy name is not valid
        """
        lowercase_name = name.lower()
        if lowercase_name not in self._items:
            available = list(self._items.keys())
            raise ValueError(f"Unknown strategy: {name}. Available strategies: {available}")

    def get_strategy_class(self, strategy_name: str) -> type[BaseStrategy] | None:
        """Get a strategy class by name.

        Args:
            strategy_name: Strategy name (case-insensitive)

        Returns:
            Strategy class or None if not found
        """
        return self.get(strategy_name)

    def normalize_strategy_name(self, strategy_name: str) -> str:
        """Normalize strategy name to lowercase format.

        Args:
            strategy_name: Strategy name in any format

        Returns:
            Normalized (lowercase) strategy name
        """
        return strategy_name.lower()

    def get_available_strategy_names(self) -> list[str]:
        """Get list of all available strategy names.

        Returns:
            List of all strategy names
        """
        return list(self._items.keys())

    def is_valid_strategy(self, strategy_name: str) -> bool:
        """Check if a strategy name is valid.

        Args:
            strategy_name: Strategy name to validate

        Returns:
            True if valid, False otherwise
        """
        return strategy_name.lower() in self._items

    def get_strategy_groups(self) -> dict[str, list[str]]:
        """Get all strategy groups.

        Returns:
            Dictionary of group names to strategy lists
        """
        return self.groups.copy()

    def is_valid_strategy_group(self, group_name: str) -> bool:
        """Check if a strategy group is valid.

        Args:
            group_name: Group name to validate

        Returns:
            True if valid, False otherwise
        """
        return group_name in self.groups

    def get_strategies_in_group(self, group_name: str) -> list[str]:
        """Get strategies in a specific group.

        Args:
            group_name: Name of the group

        Returns:
            List of strategy names in the group

        Raises:
            ValueError: If group name is invalid
        """
        if not self.is_valid_strategy_group(group_name):
            available_groups = list(self.groups.keys())
            raise ValueError(f"Unknown strategy group: {group_name}. Available: {available_groups}")

        return self.groups[group_name].copy()

    def get_strategy_presets(self) -> dict[str, dict[str, Any]]:
        """Get all strategy parameter presets.

        Returns:
            Dictionary of strategy names to parameter presets
        """
        return self._preset_values.copy()

    def get_strategy_preset(self, strategy_name: str) -> dict[str, Any]:
        """Get parameter preset for a specific strategy.

        Args:
            strategy_name: Strategy name

        Returns:
            Dictionary of default parameters
        """
        normalized_name = self.normalize_strategy_name(strategy_name)
        return self._preset_values.get(normalized_name, {}).copy()

    def validate_strategies(self, strategy_names: list[str]) -> tuple[list[str], list[str]]:
        """Validate a list of strategy names.

        Args:
            strategy_names: List of strategy names to validate

        Returns:
            Tuple of (valid_strategies, invalid_strategies) - valid names are normalized to lowercase
        """
        valid = []
        invalid = []

        for name in strategy_names:
            lowercase_name = name.lower()
            if lowercase_name in self._items:
                valid.append(lowercase_name)
            else:
                invalid.append(name)

        return valid, invalid

    def add_strategy_group(self, group_name: str, strategies: list[str]) -> None:
        """Add or update a strategy group.

        Args:
            group_name: Name of the group
            strategies: List of strategy names

        Raises:
            ValueError: If any strategy is invalid
        """
        # Validate all strategies first
        for strategy in strategies:
            self.validate(strategy)

        # Store normalized (lowercase) names
        self.groups[group_name] = [s.lower() for s in strategies]

    def update_strategy_preset(self, strategy_name: str, parameters: dict[str, Any]) -> None:
        """Update parameters for a strategy preset.

        Args:
            strategy_name: Strategy name
            parameters: Parameters to update

        Raises:
            ValueError: If strategy is invalid
        """
        self.validate(strategy_name)

        lowercase_name = strategy_name.lower()
        if lowercase_name not in self._preset_values:
            self._preset_values[lowercase_name] = {}

        self._preset_values[lowercase_name].update(parameters)

        # Sync to database if available
        if self.db_manager:
            self._save_preset_to_database(lowercase_name, self._preset_values[lowercase_name])

    def _save_preset_to_database(self, strategy_name: str, parameters: dict[str, Any]) -> None:
        """Save a preset to the database."""
        if not self.db_manager:
            return

        with self.db_manager.get_session() as session:
            strategy = session.query(Strategy).filter(Strategy.name == strategy_name).first()
            if strategy:
                preset = (
                    session.query(StrategyPreset)
                    .filter(StrategyPreset.strategy_id == strategy.id, StrategyPreset.name == "default")
                    .first()
                )

                if preset:
                    preset.set_parameters(parameters)
                else:
                    preset = StrategyPreset(
                        strategy_id=strategy.id,
                        name="default",
                        is_default=True,
                    )
                    preset.set_parameters(parameters)
                    session.add(preset)

                session.commit()


# Create a singleton instance (without database for backward compatibility)
strategy_repository = StrategyRepository()
