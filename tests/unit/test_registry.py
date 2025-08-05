"""Unit tests for the Registry pattern."""

import pytest

from stockula.data.registry import Registry
from stockula.data.repository import DataRepository


class MockRepository(DataRepository[str]):
    """Mock repository for testing."""

    def __init__(self):
        """Initialize the mock repository."""
        super().__init__()
        self.test_value = "test"


class TestRegistry:
    """Test the Registry class."""

    def test_initialization(self):
        """Test registry initialization."""
        registry = Registry()
        assert registry._repositories == {}
        assert len(registry.repositories) == 0

    def test_register_repository(self):
        """Test registering a repository."""
        registry = Registry()
        repo = MockRepository()

        registry.register_repository("test", repo)

        assert "test" in registry
        assert registry.get_repository("test") is repo
        assert registry["test"] is repo

    def test_register_duplicate_repository_raises_error(self):
        """Test that registering a duplicate repository raises ValueError."""
        registry = Registry()
        repo1 = MockRepository()
        repo2 = MockRepository()

        registry.register_repository("test", repo1)

        with pytest.raises(ValueError, match="Repository 'test' is already registered"):
            registry.register_repository("test", repo2)

    def test_get_repository(self):
        """Test getting a repository by name."""
        registry = Registry()
        repo = MockRepository()
        registry.register_repository("test", repo)

        # Test successful retrieval
        assert registry.get_repository("test") is repo

        # Test non-existent repository returns None
        assert registry.get_repository("nonexistent") is None

    def test_get_repository_with_subscript(self):
        """Test getting a repository using subscript notation."""
        registry = Registry()
        repo = MockRepository()
        registry.register_repository("test", repo)

        # Test successful retrieval
        assert registry["test"] is repo

        # Test non-existent repository raises KeyError
        with pytest.raises(KeyError, match="Repository 'nonexistent' not found"):
            _ = registry["nonexistent"]

    def test_remove_repository(self):
        """Test removing a repository."""
        registry = Registry()
        repo = MockRepository()
        registry.register_repository("test", repo)

        assert "test" in registry
        registry.remove_repository("test")
        assert "test" not in registry
        assert registry.get_repository("test") is None

    def test_remove_nonexistent_repository_raises_error(self):
        """Test removing a non-existent repository raises KeyError."""
        registry = Registry()

        with pytest.raises(KeyError, match="Repository 'nonexistent' not found"):
            registry.remove_repository("nonexistent")

    def test_list_repositories(self):
        """Test listing all repository names."""
        registry = Registry()
        repo1 = MockRepository()
        repo2 = MockRepository()

        assert registry.list_repositories() == []

        registry.register_repository("repo1", repo1)
        registry.register_repository("repo2", repo2)

        repos = registry.list_repositories()
        assert len(repos) == 2
        assert "repo1" in repos
        assert "repo2" in repos

    def test_clear(self):
        """Test clearing all repositories."""
        registry = Registry()
        repo1 = MockRepository()
        repo2 = MockRepository()

        registry.register_repository("repo1", repo1)
        registry.register_repository("repo2", repo2)

        assert len(registry.repositories) == 2

        registry.clear()

        assert len(registry.repositories) == 0
        assert "repo1" not in registry
        assert "repo2" not in registry

    def test_repositories_property(self):
        """Test the repositories property returns a copy."""
        registry = Registry()
        repo = MockRepository()
        registry.register_repository("test", repo)

        # Get repositories
        repos = registry.repositories
        assert "test" in repos
        assert repos["test"] is repo

        # Verify it's a copy by modifying it
        repos["new"] = MockRepository()
        assert "new" not in registry

    def test_contains_method(self):
        """Test the __contains__ method."""
        registry = Registry()
        repo = MockRepository()

        assert "test" not in registry

        registry.register_repository("test", repo)

        assert "test" in registry
        assert "nonexistent" not in registry

    def test_multiple_repository_types(self):
        """Test registering different types of repositories."""
        registry = Registry()

        # Create different repository types
        string_repo = DataRepository[str]()
        int_repo = DataRepository[int]()

        registry.register_repository("strings", string_repo)
        registry.register_repository("integers", int_repo)

        assert registry["strings"] is string_repo
        assert registry["integers"] is int_repo
        assert len(registry.repositories) == 2
