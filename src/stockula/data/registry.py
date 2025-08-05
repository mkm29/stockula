"""Registry pattern for managing multiple repositories."""

from typing import Any, TypeVar

from .repository import Repository

T = TypeVar("T")


class Registry:
    """Central registry for managing multiple repositories.

    The Registry serves as a centralized location for accessing
    various repositories in the application.
    """

    def __init__(self):
        """Initialize the registry."""
        self._repositories: dict[str, Repository[Any]] = {}

    def register_repository(self, name: str, repository: Repository[Any]) -> None:
        """Register a repository.

        Args:
            name: Name to register the repository under
            repository: The repository instance

        Raises:
            ValueError: If a repository with the same name already exists
        """
        if name in self._repositories:
            raise ValueError(f"Repository '{name}' is already registered")
        self._repositories[name] = repository

    def get_repository(self, name: str) -> Repository[Any] | None:
        """Get a repository by name.

        Args:
            name: Name of the repository

        Returns:
            The repository instance or None if not found
        """
        return self._repositories.get(name)

    def remove_repository(self, name: str) -> None:
        """Remove a repository from the registry.

        Args:
            name: Name of the repository to remove

        Raises:
            KeyError: If the repository doesn't exist
        """
        if name not in self._repositories:
            raise KeyError(f"Repository '{name}' not found")
        del self._repositories[name]

    def list_repositories(self) -> list[str]:
        """List all registered repository names.

        Returns:
            List of repository names
        """
        return list(self._repositories.keys())

    def clear(self) -> None:
        """Clear all repositories from the registry."""
        self._repositories.clear()

    @property
    def repositories(self) -> dict[str, Repository[Any]]:
        """Get all repositories.

        Returns:
            Dictionary of all repositories
        """
        return self._repositories.copy()

    def __contains__(self, name: str) -> bool:
        """Check if a repository is registered.

        Args:
            name: Name of the repository

        Returns:
            True if registered, False otherwise
        """
        return name in self._repositories

    def __getitem__(self, name: str) -> Repository[Any]:
        """Get a repository using subscript notation.

        Args:
            name: Name of the repository

        Returns:
            The repository instance

        Raises:
            KeyError: If the repository doesn't exist
        """
        if name not in self._repositories:
            raise KeyError(f"Repository '{name}' not found")
        return self._repositories[name]


# Global registry instance
registry = Registry()
