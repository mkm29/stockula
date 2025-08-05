"""Base Repository class for managing collections of items."""

from collections.abc import Iterator, MutableMapping
from typing import TypeVar

T = TypeVar("T")


class Repository[T](MutableMapping[str, T]):
    """Generic repository class that acts as a dictionary-like container.

    This class provides a flexible base for creating repositories of different types
    of objects. It uses a simple mapping approach with alias support.

    Example:
        class StrategyRepository(Repository[Type[BaseStrategy]]):
            def add_strategy(self, name: str, strategy_class: Type[BaseStrategy]):
                self.add(name, strategy_class)
    """

    def __init__(self):
        """Initialize the repository."""
        self._items: dict[str, T] = {}  # Primary storage
        self._mappings: dict[str, str] = {}  # Alias -> primary key mappings

    def __getitem__(self, key: str) -> T:
        """Get an item by key or alias."""
        # First check primary storage
        if key in self._items:
            return self._items[key]

        # Then check if it's an alias
        if key in self._mappings:
            primary_key = self._mappings[key]
            return self._items[primary_key]

        raise KeyError(key)

    def __setitem__(self, key: str, value: T) -> None:
        """Set an item in the registry."""
        self._items[key] = value

    def __delitem__(self, key: str) -> None:
        """Delete an item from the registry."""
        # If it's an alias, remove the alias
        if key in self._mappings:
            del self._mappings[key]
        # If it's a primary key, remove it and all its aliases
        elif key in self._items:
            del self._items[key]
            # Remove all aliases pointing to this key
            aliases_to_remove = [alias for alias, pk in self._mappings.items() if pk == key]
            for alias in aliases_to_remove:
                del self._mappings[alias]
        else:
            raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        """Iterate over primary keys."""
        return iter(self._items)

    def __len__(self) -> int:
        """Return the number of primary items."""
        return len(self._items)

    def add(self, key: str, value: T, aliases: list[str] | None = None) -> None:
        """Add an item to the registry.

        Args:
            key: The primary key for the item
            value: The item to add
            aliases: Optional list of aliases that map to the same key
        """
        self[key] = value

        # Register aliases if provided
        if aliases:
            for alias in aliases:
                self._mappings[alias] = key

    def get_by_alias(self, alias: str) -> T | None:
        """Get an item by its alias.

        Args:
            alias: The alias to look up

        Returns:
            The item if found, None otherwise
        """
        try:
            return self[alias]
        except KeyError:
            return None

    def keys(self, include_aliases: bool = False):
        """Return a view of the registry's keys.

        Args:
            include_aliases: Whether to include aliases in the result

        Returns:
            View of all keys (and optionally aliases)
        """
        if include_aliases:
            # Return a list that includes both primary keys and aliases
            return list(self._items.keys()) + list(self._mappings.keys())
        # Default behavior - just return the primary keys
        return self._items.keys()

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the registry (including aliases).

        This allows using 'in' operator: 'smacross' in registry

        Args:
            key: The key to check

        Returns:
            True if the key exists (as primary key or alias), False otherwise
        """
        return key in self._items or key in self._mappings

    def has_key(self, key: str, check_aliases: bool = True) -> bool:
        """Check if a key exists in the registry.

        Args:
            key: The key to check
            check_aliases: Whether to also check aliases

        Returns:
            True if the key exists, False otherwise
        """
        if check_aliases:
            return key in self  # Uses __contains__
        return key in self._items

    def get_aliases(self, key: str) -> list[str]:
        """Get all aliases for a given key.

        Args:
            key: The key to get aliases for

        Returns:
            List of aliases for the key
        """
        return [alias for alias, primary in self._mappings.items() if primary == key]

    def clear_aliases(self) -> None:
        """Clear all aliases."""
        self._mappings.clear()

    def update_with_aliases(self, other: dict[str, T], aliases: dict[str, str] | None = None) -> None:
        """Update the registry with items and their aliases.

        Args:
            other: Dictionary of items to add
            aliases: Optional dictionary mapping aliases to primary keys
        """
        self.update(other)
        if aliases:
            self._mappings.update(aliases)

    def get(self, key: str, default=None):
        """Get an item by key or alias, returning default if not found."""
        try:
            return self[key]
        except KeyError:
            return default
