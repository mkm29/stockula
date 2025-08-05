"""Unit tests for the StrategyRepository class."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from sqlalchemy.exc import IntegrityError, OperationalError

from stockula.backtesting.strategies import BaseStrategy, RSIStrategy, SMACrossStrategy
from stockula.data.strategy_repository import StrategyRepository


class TestStrategyRepository:
    """Test the StrategyRepository class."""

    def test_initialization_without_db(self):
        """Test initializing repository without database."""
        repo = StrategyRepository()

        assert repo.db_manager is None
        assert len(repo._items) == len(repo.DEFAULT_MAPPINGS)
        assert "smacross" in repo._items
        assert repo._items["smacross"] is SMACrossStrategy
        assert len(repo.groups) == len(repo.DEFAULT_GROUPS)
        assert len(repo._preset_values) == len(repo.DEFAULT_PRESETS)

    def test_initialization_with_db(self):
        """Test initializing repository with database."""
        mock_db = Mock()
        mock_session = MagicMock()
        mock_db.get_session.return_value = MagicMock(__enter__=Mock(return_value=mock_session))

        # Mock empty database
        mock_session.query.return_value.filter.return_value.all.return_value = []

        repo = StrategyRepository(mock_db)

        assert repo.db_manager is mock_db
        mock_db.get_session.assert_called_once()

    def test_presets_property_immutable(self):
        """Test that presets property returns immutable copy."""
        repo = StrategyRepository()

        # Get presets
        presets1 = repo.presets
        presets2 = repo.presets

        # Verify they are equal but not the same object
        assert presets1 == presets2
        assert presets1 is not presets2

        # Verify modifications don't affect original
        presets1["smacross"]["new_param"] = 999
        presets3 = repo.presets
        assert "new_param" not in presets3["smacross"]

    def test_values_property_mutable(self):
        """Test that values property returns mutable reference."""
        repo = StrategyRepository()

        # Get values and modify
        values = repo.values
        values["smacross"]["fast_period"] = 15

        # Verify change persists
        assert repo.values["smacross"]["fast_period"] == 15

    def test_add_strategy(self):
        """Test adding a new strategy."""
        repo = StrategyRepository()

        class TestStrategy(BaseStrategy):
            """Test strategy."""

            pass

        repo.add("teststrategy", TestStrategy)

        assert "teststrategy" in repo._items
        assert repo._items["teststrategy"] is TestStrategy

    def test_add_strategy_lowercase_conversion(self):
        """Test that strategy names are converted to lowercase."""
        repo = StrategyRepository()

        class TestStrategy(BaseStrategy):
            """Test strategy."""

            pass

        repo.add("TestStrategy", TestStrategy)

        assert "teststrategy" in repo._items
        assert "TestStrategy" not in repo._items

    def test_get_strategy(self):
        """Test getting a strategy."""
        repo = StrategyRepository()

        # Test case-insensitive retrieval
        assert repo.get("smacross") is SMACrossStrategy
        assert repo.get("SMACross") is SMACrossStrategy
        assert repo.get("SMACROSS") is SMACrossStrategy

        # Test non-existent strategy
        assert repo.get("nonexistent") is None
        assert repo.get("nonexistent", "default") == "default"

    def test_remove_strategy(self):
        """Test removing a strategy."""
        repo = StrategyRepository()

        assert "rsi" in repo._items
        repo.remove("rsi")
        assert "rsi" not in repo._items

    def test_remove_nonexistent_strategy_raises_error(self):
        """Test removing non-existent strategy raises error."""
        repo = StrategyRepository()

        with pytest.raises(KeyError, match="Strategy 'nonexistent' not found"):
            repo.remove("nonexistent")

    def test_validate_strategy(self):
        """Test validating strategy names."""
        repo = StrategyRepository()

        # Valid strategies should not raise
        repo.validate("smacross")
        repo.validate("SMACross")  # Case insensitive

        # Invalid strategy should raise
        with pytest.raises(ValueError, match="Unknown strategy: invalid"):
            repo.validate("invalid")

    def test_get_strategy_class(self):
        """Test get_strategy_class method."""
        repo = StrategyRepository()

        assert repo.get_strategy_class("smacross") is SMACrossStrategy
        assert repo.get_strategy_class("rsi") is RSIStrategy
        assert repo.get_strategy_class("invalid") is None

    def test_normalize_strategy_name(self):
        """Test normalizing strategy names."""
        repo = StrategyRepository()

        assert repo.normalize_strategy_name("SMACross") == "smacross"
        assert repo.normalize_strategy_name("SMACROSS") == "smacross"
        assert repo.normalize_strategy_name("smacross") == "smacross"

    def test_get_available_strategy_names(self):
        """Test getting all available strategy names."""
        repo = StrategyRepository()

        names = repo.get_available_strategy_names()
        assert isinstance(names, list)
        assert "smacross" in names
        assert "rsi" in names
        assert len(names) == len(repo._items)

    def test_is_valid_strategy(self):
        """Test checking if strategy is valid."""
        repo = StrategyRepository()

        assert repo.is_valid_strategy("smacross")
        assert repo.is_valid_strategy("SMACross")  # Case insensitive
        assert not repo.is_valid_strategy("invalid")

    def test_get_strategy_groups(self):
        """Test getting strategy groups."""
        repo = StrategyRepository()

        groups = repo.get_strategy_groups()
        assert isinstance(groups, dict)
        assert "basic" in groups
        assert "momentum" in groups
        assert isinstance(groups["basic"], list)

        # Verify it's a copy
        groups["new_group"] = ["test"]
        assert "new_group" not in repo.groups

    def test_is_valid_strategy_group(self):
        """Test checking if strategy group is valid."""
        repo = StrategyRepository()

        assert repo.is_valid_strategy_group("basic")
        assert repo.is_valid_strategy_group("momentum")
        assert not repo.is_valid_strategy_group("invalid")

    def test_get_strategies_in_group(self):
        """Test getting strategies in a group."""
        repo = StrategyRepository()

        strategies = repo.get_strategies_in_group("basic")
        assert isinstance(strategies, list)
        assert "smacross" in strategies
        assert "rsi" in strategies

        # Verify it's a copy
        strategies.append("new_strategy")
        assert "new_strategy" not in repo.groups["basic"]

    def test_get_strategies_in_invalid_group_raises_error(self):
        """Test getting strategies in invalid group raises error."""
        repo = StrategyRepository()

        with pytest.raises(ValueError, match="Unknown strategy group: invalid"):
            repo.get_strategies_in_group("invalid")

    def test_get_strategy_presets(self):
        """Test getting all strategy presets."""
        repo = StrategyRepository()

        presets = repo.get_strategy_presets()
        assert isinstance(presets, dict)
        assert "smacross" in presets
        assert presets["smacross"]["fast_period"] == 10

        # Verify it returns a copy by checking we can modify top level
        presets["new_strategy"] = {"param": 1}
        assert "new_strategy" not in repo._preset_values

        # Note: get_strategy_presets returns shallow copy, so nested dicts are shared
        # This is a potential issue but matches current implementation

    def test_get_strategy_preset(self):
        """Test getting preset for specific strategy."""
        repo = StrategyRepository()

        preset = repo.get_strategy_preset("smacross")
        assert isinstance(preset, dict)
        assert preset["fast_period"] == 10
        assert preset["slow_period"] == 20

        # Test case insensitive
        preset2 = repo.get_strategy_preset("SMACross")
        assert preset == preset2

        # Test non-existent
        preset3 = repo.get_strategy_preset("invalid")
        assert preset3 == {}

    def test_validate_strategies(self):
        """Test validating multiple strategies."""
        repo = StrategyRepository()

        valid, invalid = repo.validate_strategies(["smacross", "RSI", "invalid", "MACD"])

        assert "smacross" in valid
        assert "rsi" in valid  # Normalized to lowercase
        assert "macd" in valid
        assert "invalid" in invalid
        assert len(valid) == 3
        assert len(invalid) == 1

    def test_add_strategy_group(self):
        """Test adding a new strategy group."""
        repo = StrategyRepository()

        repo.add_strategy_group("custom", ["smacross", "rsi"])

        assert "custom" in repo.groups
        assert repo.groups["custom"] == ["smacross", "rsi"]

        # Test with case conversion
        repo.add_strategy_group("custom2", ["SMACross", "RSI"])
        assert repo.groups["custom2"] == ["smacross", "rsi"]

    def test_add_strategy_group_invalid_strategy_raises_error(self):
        """Test adding group with invalid strategy raises error."""
        repo = StrategyRepository()

        with pytest.raises(ValueError, match="Unknown strategy: invalid"):
            repo.add_strategy_group("custom", ["smacross", "invalid"])

    def test_update_strategy_preset(self):
        """Test updating strategy preset parameters."""
        repo = StrategyRepository()

        # Update existing preset
        repo.update_strategy_preset("smacross", {"fast_period": 15, "new_param": 100})

        assert repo._preset_values["smacross"]["fast_period"] == 15
        assert repo._preset_values["smacross"]["new_param"] == 100
        assert repo._preset_values["smacross"]["slow_period"] == 20  # Unchanged

        # Update non-existent preset
        repo.update_strategy_preset("kama", {"param": 50})
        assert repo._preset_values["kama"]["param"] == 50

    def test_update_strategy_preset_invalid_strategy_raises_error(self):
        """Test updating preset for invalid strategy raises error."""
        repo = StrategyRepository()

        with pytest.raises(ValueError, match="Unknown strategy: invalid"):
            repo.update_strategy_preset("invalid", {"param": 1})

    def test_sync_to_database_without_db_manager(self):
        """Test sync_to_database when no DB manager is set."""
        repo = StrategyRepository()
        repo.sync_to_database()  # Should not raise

    @patch("stockula.data.strategy_repository.Strategy")
    @patch("stockula.data.strategy_repository.StrategyPreset")
    def test_sync_to_database_with_new_strategies(self, mock_preset_model, mock_strategy_model):
        """Test syncing new strategies to database."""
        mock_db = Mock()
        mock_session = MagicMock()
        mock_db.get_session.return_value = MagicMock(__enter__=Mock(return_value=mock_session))

        # Mock no existing strategies
        mock_session.query.return_value.filter.return_value.first.return_value = None

        repo = StrategyRepository(mock_db)
        repo._items = {"test": SMACrossStrategy}
        repo._preset_values = {"test": {"param": 1}}

        # Create mock strategy instance
        mock_strategy = Mock()
        mock_strategy.id = 1
        mock_strategy_model.return_value = mock_strategy

        repo.sync_to_database()

        # Verify strategy was created
        mock_strategy_model.assert_called_once()
        mock_session.add.assert_called()
        mock_session.commit.assert_called_once()

    def test_sync_to_database_handles_integrity_error(self):
        """Test sync_to_database handles IntegrityError gracefully."""
        mock_db = Mock()
        mock_session = MagicMock()
        mock_db.get_session.return_value = MagicMock(__enter__=Mock(return_value=mock_session))

        # Mock IntegrityError
        mock_session.commit.side_effect = IntegrityError("test", "test", "test")

        repo = StrategyRepository(mock_db)
        repo.sync_to_database()  # Should not raise

        mock_session.rollback.assert_called_once()

    def test_sync_to_database_handles_operational_error(self):
        """Test sync_to_database handles OperationalError gracefully."""
        # Create repository without DB first
        repo = StrategyRepository()

        # Then add DB manager
        mock_db = Mock()
        mock_session = MagicMock()
        mock_db.get_session.return_value = MagicMock(__enter__=Mock(return_value=mock_session))

        # Mock OperationalError
        mock_session.query.side_effect = OperationalError("test", "test", "test")

        repo.db_manager = mock_db
        repo.sync_to_database()  # Should not raise

        mock_session.rollback.assert_called_once()

    def test_get_strategy_category(self):
        """Test determining strategy category."""
        repo = StrategyRepository()

        # smacross is in both "basic" and "trend", but "basic" comes first
        assert repo._get_strategy_category("smacross") in ["basic", "trend"]
        assert repo._get_strategy_category("rsi") in ["basic", "momentum"]
        assert repo._get_strategy_category("nonexistent") is None

    def test_load_from_database(self):
        """Test loading strategies from database."""
        mock_db = Mock()
        mock_session = MagicMock()
        mock_db.get_session.return_value = MagicMock(__enter__=Mock(return_value=mock_session))

        # Create mock strategy
        mock_strategy = Mock()
        mock_strategy.id = 1
        mock_strategy.name = "test_strategy"
        mock_strategy.is_active = True

        # Create mock preset
        mock_preset = Mock()
        mock_preset.is_default = True
        mock_preset.parameters = {"param": 123}

        # Setup query results
        mock_session.query.return_value.filter.side_effect = [
            Mock(all=Mock(return_value=[mock_strategy])),  # strategies query
            Mock(all=Mock(return_value=[mock_preset])),  # presets query
        ]

        repo = StrategyRepository(mock_db)

        # Verify preset was loaded
        assert repo._preset_values.get("test_strategy") == {"param": 123}

    def test_save_preset_to_database_update_existing(self):
        """Test updating existing preset in database."""
        mock_db = Mock()
        mock_session = MagicMock()
        mock_db.get_session.return_value = MagicMock(__enter__=Mock(return_value=mock_session))

        # Mock existing strategy and preset
        mock_strategy = Mock()
        mock_strategy.id = 1
        mock_preset = Mock()

        mock_session.query.return_value.filter.return_value.first.side_effect = [
            mock_strategy,  # Strategy exists
            mock_preset,  # Preset exists
        ]

        repo = StrategyRepository(mock_db)
        repo._save_preset_to_database("smacross", {"fast": 15})

        # Verify preset was updated
        mock_preset.set_parameters.assert_called_once_with({"fast": 15})
        mock_session.commit.assert_called_once()

    def test_update_strategy_preset_with_db_saves(self):
        """Test that update_strategy_preset saves to database when DB manager exists."""
        mock_db = Mock()
        mock_session = MagicMock()
        mock_db.get_session.return_value = MagicMock(__enter__=Mock(return_value=mock_session))
        # Mock empty database
        mock_session.query.return_value.filter.return_value.all.return_value = []

        repo = StrategyRepository(mock_db)

        with patch.object(repo, "_save_preset_to_database") as mock_save:
            repo.update_strategy_preset("smacross", {"fast_period": 15})

            mock_save.assert_called_once_with("smacross", repo._preset_values["smacross"])

    def test_repository_implements_base_methods(self):
        """Test that StrategyRepository implements all Repository methods."""
        repo = StrategyRepository()

        # Test inherited methods
        assert len(repo) == len(repo._items)
        assert "smacross" in repo
        assert repo.keys() == list(repo._items.keys())
        # Note: StrategyRepository has both a values property and inherits values() method
        # Access the base class method directly
        from stockula.data.repository import Repository

        values_list = Repository[type[BaseStrategy]].values(repo)
        assert len(values_list) == len(repo._items)
        assert all(issubclass(v, BaseStrategy) for v in values_list)
        assert len(list(repo.items())) == len(repo._items)

        # Test exists
        assert repo.exists("smacross")
        assert not repo.exists("invalid")
