"""Base repository pattern for managing collections of items."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class Repository(Generic[T], ABC):
    """Abstract base class for repositories."""

    def __init__(self):
        """Initialize the repository."""
        self._items: dict[str, T] = {}

    @abstractmethod
    def add(self, key: str, item: T) -> None:
        """Add an item to the repository.

        Args:
            key: Unique identifier for the item
            item: The item to add
        """
        pass

    @abstractmethod
    def get(self, key: str, default: T | None = None) -> T | None:
        """Get an item by key.

        Args:
            key: The item key
            default: Default value if not found

        Returns:
            The item or default value
        """
        pass

    @abstractmethod
    def remove(self, key: str) -> None:
        """Remove an item from the repository.

        Args:
            key: The item key

        Raises:
            KeyError: If the key doesn't exist
        """
        pass

    def exists(self, key: str) -> bool:
        """Check if an item exists.

        Args:
            key: The item key

        Returns:
            True if exists, False otherwise
        """
        return key in self._items

    def keys(self) -> list[str]:
        """Get all keys in the repository.

        Returns:
            List of all keys
        """
        return list(self._items.keys())

    def values(self) -> list[T]:
        """Get all values in the repository.

        Returns:
            List of all values
        """
        return list(self._items.values())

    def items(self) -> list[tuple[str, T]]:
        """Get all key-value pairs.

        Returns:
            List of (key, value) tuples
        """
        return list(self._items.items())

    def clear(self) -> None:
        """Clear all items from the repository."""
        self._items.clear()

    def __len__(self) -> int:
        """Get the number of items in the repository."""
        return len(self._items)

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the repository."""
        return key in self._items


class DataRepository(Repository[T]):
    """Simple in-memory repository implementation."""

    def add(self, key: str, item: T) -> None:
        """Add an item to the repository."""
        self._items[key] = item

    def get(self, key: str, default: T | None = None) -> T | None:
        """Get an item by key."""
        return self._items.get(key, default)

    def remove(self, key: str) -> None:
        """Remove an item from the repository."""
        if key not in self._items:
            raise KeyError(f"Key '{key}' not found in repository")
        del self._items[key]
