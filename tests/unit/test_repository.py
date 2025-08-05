"""Unit tests for the Repository pattern."""

import pytest

from stockula.data.repository import DataRepository, Repository


class ConcreteRepository(Repository[str]):
    """Concrete implementation of Repository for testing."""

    def add(self, key: str, item: str) -> None:
        """Add an item to the repository."""
        self._items[key] = item

    def get(self, key: str, default: str | None = None) -> str | None:
        """Get an item by key."""
        return self._items.get(key, default)

    def remove(self, key: str) -> None:
        """Remove an item from the repository."""
        if key not in self._items:
            raise KeyError(f"Key '{key}' not found")
        del self._items[key]


class TestRepository:
    """Test the abstract Repository class."""

    def test_initialization(self):
        """Test repository initialization."""
        repo = ConcreteRepository()
        assert repo._items == {}
        assert len(repo) == 0

    def test_add_and_get(self):
        """Test adding and getting items."""
        repo = ConcreteRepository()

        repo.add("key1", "value1")
        repo.add("key2", "value2")

        assert repo.get("key1") == "value1"
        assert repo.get("key2") == "value2"
        assert repo.get("nonexistent") is None
        assert repo.get("nonexistent", "default") == "default"

    def test_remove(self):
        """Test removing items."""
        repo = ConcreteRepository()
        repo.add("key1", "value1")

        assert repo.exists("key1")
        repo.remove("key1")
        assert not repo.exists("key1")
        assert repo.get("key1") is None

    def test_remove_nonexistent_raises_error(self):
        """Test removing non-existent item raises error."""
        repo = ConcreteRepository()

        with pytest.raises(KeyError, match="Key 'nonexistent' not found"):
            repo.remove("nonexistent")

    def test_exists(self):
        """Test checking if items exist."""
        repo = ConcreteRepository()

        assert not repo.exists("key1")
        repo.add("key1", "value1")
        assert repo.exists("key1")

    def test_keys(self):
        """Test getting all keys."""
        repo = ConcreteRepository()

        assert repo.keys() == []

        repo.add("key1", "value1")
        repo.add("key2", "value2")

        keys = repo.keys()
        assert len(keys) == 2
        assert "key1" in keys
        assert "key2" in keys

    def test_values(self):
        """Test getting all values."""
        repo = ConcreteRepository()

        assert repo.values() == []

        repo.add("key1", "value1")
        repo.add("key2", "value2")

        values = repo.values()
        assert len(values) == 2
        assert "value1" in values
        assert "value2" in values

    def test_items(self):
        """Test getting all key-value pairs."""
        repo = ConcreteRepository()

        assert repo.items() == []

        repo.add("key1", "value1")
        repo.add("key2", "value2")

        items = repo.items()
        assert len(items) == 2
        assert ("key1", "value1") in items
        assert ("key2", "value2") in items

    def test_clear(self):
        """Test clearing all items."""
        repo = ConcreteRepository()
        repo.add("key1", "value1")
        repo.add("key2", "value2")

        assert len(repo) == 2
        repo.clear()
        assert len(repo) == 0
        assert repo.keys() == []

    def test_len(self):
        """Test __len__ method."""
        repo = ConcreteRepository()

        assert len(repo) == 0

        repo.add("key1", "value1")
        assert len(repo) == 1

        repo.add("key2", "value2")
        assert len(repo) == 2

        repo.remove("key1")
        assert len(repo) == 1

    def test_contains(self):
        """Test __contains__ method."""
        repo = ConcreteRepository()

        assert "key1" not in repo

        repo.add("key1", "value1")
        assert "key1" in repo
        assert "key2" not in repo


class TestDataRepository:
    """Test the DataRepository implementation."""

    def test_initialization(self):
        """Test DataRepository initialization."""
        repo = DataRepository[str]()
        assert repo._items == {}
        assert len(repo) == 0

    def test_add(self):
        """Test adding items to DataRepository."""
        repo = DataRepository[str]()

        repo.add("key1", "value1")
        assert repo._items["key1"] == "value1"

        # Test overwriting
        repo.add("key1", "new_value")
        assert repo._items["key1"] == "new_value"

    def test_get(self):
        """Test getting items from DataRepository."""
        repo = DataRepository[str]()
        repo.add("key1", "value1")

        assert repo.get("key1") == "value1"
        assert repo.get("nonexistent") is None
        assert repo.get("nonexistent", "default") == "default"

    def test_remove(self):
        """Test removing items from DataRepository."""
        repo = DataRepository[str]()
        repo.add("key1", "value1")

        repo.remove("key1")
        assert "key1" not in repo._items

    def test_remove_nonexistent_raises_error(self):
        """Test removing non-existent item raises KeyError."""
        repo = DataRepository[str]()

        with pytest.raises(KeyError, match="Key 'nonexistent' not found in repository"):
            repo.remove("nonexistent")

    def test_type_safety_with_different_types(self):
        """Test DataRepository with different types."""
        # String repository
        string_repo = DataRepository[str]()
        string_repo.add("key", "value")
        assert string_repo.get("key") == "value"

        # Integer repository
        int_repo = DataRepository[int]()
        int_repo.add("key", 42)
        assert int_repo.get("key") == 42

        # Complex type repository
        dict_repo = DataRepository[dict]()
        test_dict = {"nested": "value"}
        dict_repo.add("key", test_dict)
        assert dict_repo.get("key") == test_dict

    def test_inherited_methods(self):
        """Test that DataRepository inherits all Repository methods."""
        repo = DataRepository[str]()

        # Test all inherited methods work
        repo.add("key1", "value1")
        repo.add("key2", "value2")

        assert repo.exists("key1")
        assert repo.keys() == ["key1", "key2"]
        assert set(repo.values()) == {"value1", "value2"}
        assert len(repo.items()) == 2
        assert len(repo) == 2
        assert "key1" in repo

        repo.clear()
        assert len(repo) == 0
