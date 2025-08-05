"""Unit tests for the DataManager class."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from stockula.data.fetcher import DataFetcher
from stockula.data.manager import DataManager
from stockula.data.registry import Registry
from stockula.data.repository import DataRepository
from stockula.data.strategy_repository import StrategyRepository


class TestDataManager:
    """Test the DataManager class."""

    def setup_method(self):
        """Clear registry before each test."""
        # Import registry to clear it
        from stockula.data.registry import registry

        registry.clear()

    def test_initialization_without_db(self):
        """Test initializing DataManager without database."""
        manager = DataManager(use_cache=False)

        assert manager._db_manager is None
        assert manager._logging_manager is None
        assert isinstance(manager.fetcher, DataFetcher)
        assert isinstance(manager.registry, Registry)

        # Check strategy repository is registered
        assert "strategies" in manager.registry
        strategy_repo = manager.strategies
        assert isinstance(strategy_repo, StrategyRepository)
        assert strategy_repo.db_manager is None

    def test_initialization_with_db_and_logging(self):
        """Test initializing DataManager with database and logging."""
        mock_db = Mock()
        mock_session = MagicMock()
        mock_db.get_session.return_value = MagicMock(__enter__=Mock(return_value=mock_session))
        # Mock empty database
        mock_session.query.return_value.filter.return_value.all.return_value = []

        mock_logging = Mock()

        manager = DataManager(db_manager=mock_db, logging_manager=mock_logging, use_cache=True, db_path="test.db")

        assert manager._db_manager is mock_db
        assert manager._logging_manager is mock_logging

        # Check DataFetcher was initialized with correct params
        assert manager.fetcher.use_cache is True
        assert manager.fetcher.db is mock_db
        assert manager.fetcher.logger is mock_logging

    def test_fetcher_property(self):
        """Test fetcher property returns DataFetcher instance."""
        manager = DataManager(use_cache=False)
        fetcher = manager.fetcher

        assert isinstance(fetcher, DataFetcher)
        # Multiple calls return same instance
        assert manager.fetcher is fetcher

    def test_registry_property(self):
        """Test registry property returns Registry instance."""
        manager = DataManager(use_cache=False)
        registry = manager.registry

        assert isinstance(registry, Registry)
        # Multiple calls return same instance
        assert manager.registry is registry

    def test_get_repository(self):
        """Test getting repositories by name."""
        manager = DataManager(use_cache=False)

        # Get existing strategy repository
        strategy_repo = manager.get_repository("strategies")
        assert isinstance(strategy_repo, StrategyRepository)

        # Get non-existent repository
        assert manager.get_repository("nonexistent") is None

    def test_strategies_property(self):
        """Test strategies property returns StrategyRepository."""
        manager = DataManager(use_cache=False)

        repo = manager.strategies
        assert isinstance(repo, StrategyRepository)

        # Multiple calls return same instance
        assert manager.strategies is repo

    def test_strategies_property_auto_register(self):
        """Test strategy repository is auto-registered if not exists."""
        manager = DataManager(use_cache=False)

        # Remove strategy repository
        if "strategies" in manager.registry:
            manager.registry.remove_repository("strategies")

        # Should auto-register
        repo = manager.strategies
        assert isinstance(repo, StrategyRepository)
        assert "strategies" in manager.registry

    def test_register_repository(self):
        """Test registering a new repository."""
        manager = DataManager(use_cache=False)

        # Create and register a new repository
        test_repo = DataRepository[str]()
        manager.register_repository("test", test_repo)

        assert "test" in manager.registry
        assert manager.get_repository("test") is test_repo

    def test_register_duplicate_repository_raises_error(self):
        """Test registering duplicate repository raises error."""
        manager = DataManager(use_cache=False)

        test_repo = DataRepository[str]()
        manager.register_repository("test", test_repo)

        with pytest.raises(ValueError, match="Repository 'test' is already registered"):
            manager.register_repository("test", test_repo)

    def test_update_database_manager(self):
        """Test updating database manager updates all components."""
        # Start without DB
        manager = DataManager(db_manager=None, use_cache=False)
        strategy_repo = manager.strategies

        assert manager._db_manager is None
        assert manager.fetcher.db is None
        assert strategy_repo.db_manager is None

        # Update with new DB manager
        mock_db = Mock()
        mock_session = MagicMock()
        mock_db.get_session.return_value = MagicMock(__enter__=Mock(return_value=mock_session))

        with patch.object(strategy_repo, "sync_to_database") as mock_sync:
            manager.update_database_manager(mock_db)

            assert manager._db_manager is mock_db
            assert manager.fetcher.db is mock_db
            assert strategy_repo.db_manager is mock_db
            mock_sync.assert_called_once()

    def test_update_database_manager_multiple_repos(self):
        """Test updating database manager updates all repositories."""
        manager = DataManager(db_manager=None, use_cache=False)

        # Add multiple repositories with db_manager attribute
        class MockRepo:
            def __init__(self):
                self.db_manager = None
                self.sync_called = False

            def sync_to_database(self):
                self.sync_called = True

        repo1 = MockRepo()
        repo2 = MockRepo()
        repo_without_db = DataRepository[str]()  # No db_manager attribute

        manager.register_repository("repo1", repo1)
        manager.register_repository("repo2", repo2)
        manager.register_repository("repo3", repo_without_db)

        # Update DB manager
        mock_db = Mock()
        mock_session = MagicMock()
        mock_db.get_session.return_value = MagicMock(__enter__=Mock(return_value=mock_session))

        manager.update_database_manager(mock_db)

        # Check all repos with db_manager were updated
        assert repo1.db_manager is mock_db
        assert repo1.sync_called is True
        assert repo2.db_manager is mock_db
        assert repo2.sync_called is True

    def test_initialize_registry_idempotent(self):
        """Test that _initialize_registry is idempotent."""
        mock_db = Mock()
        mock_session = MagicMock()
        mock_db.get_session.return_value = MagicMock(__enter__=Mock(return_value=mock_session))
        # Mock empty database
        mock_session.query.return_value.filter.return_value.all.return_value = []

        manager = DataManager(db_manager=mock_db)

        # Get initial repository
        repo1 = manager.strategies

        # Call _initialize_registry again
        manager._initialize_registry()

        # Should still have same repository instance
        repo2 = manager.strategies
        assert repo1 is repo2

    def test_initialize_registry_updates_existing_repo_db(self):
        """Test _initialize_registry updates existing repo's DB manager."""
        # Create manager without DB
        manager1 = DataManager(db_manager=None, use_cache=False)
        repo = manager1.strategies
        assert repo.db_manager is None

        # Create another manager with DB using same registry
        mock_db = Mock()
        mock_db.get_session = MagicMock()

        with patch.object(repo, "sync_to_database") as mock_sync:
            DataManager(db_manager=mock_db, use_cache=False)

            # Should have updated existing repo
            assert repo.db_manager is mock_db
            mock_sync.assert_called_once()

    def test_data_manager_with_custom_db_path(self):
        """Test DataManager with custom database path."""
        manager = DataManager(use_cache=True, db_path="custom.db")

        # DataFetcher should have been initialized with the custom path
        # Note: DataFetcher doesn't expose _db_path directly, but we can check it was created
        assert manager.fetcher.use_cache is True

    def test_data_manager_shares_global_registry(self):
        """Test that multiple DataManager instances share the same registry."""
        manager1 = DataManager(use_cache=False)
        manager2 = DataManager(use_cache=False)

        # Both should have same registry instance
        assert manager1.registry is manager2.registry

        # Repository registered in one is visible in other
        test_repo = DataRepository[str]()
        manager1.register_repository("shared", test_repo)

        assert manager2.get_repository("shared") is test_repo
