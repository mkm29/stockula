from typing import Any, TypeVar, cast

T = TypeVar("T")


class UniversalRepositoryRegistry:
    """A registry that can hold repositories of different types."""

    def __init__(self) -> None:
        self._repositories: dict[str, Any] = {}
        self._types: dict[str, type] = {}

    def register(self, name: str, repository: Any) -> None:
        """Register a repository with type tracking."""
        self._repositories[name] = repository
        self._types[name] = type(repository)

    def get(self, name: str, expected_type: type[T] = None) -> T | None:
        """Get a repository with optional type checking."""
        repo = self._repositories.get(name)
        if repo is None:
            return None

        if expected_type and not isinstance(repo, expected_type):
            raise TypeError(f"Repository '{name}' is {type(repo).__name__}, expected {expected_type.__name__}")

        return cast(T, repo) if expected_type else repo

    def get_type(self, name: str) -> type | None:
        """Get the type of a registered repository."""
        return self._types.get(name)


class UserRepository:
    def find_by_id(self, user_id: int):
        pass


class ProductRepository:
    def find_by_sku(self, sku: str):
        pass


# Usage:
registry = UniversalRepositoryRegistry()
registry.register("users", UserRepository())
registry.register("products", ProductRepository())

# Type-safe retrieval
user_repo = registry.get("users", UserRepository)
product_repo = registry.get("products", ProductRepository)
