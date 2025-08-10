"""Centralized data management module.

This module provides a DataManager class that manages instances of DataFetcher,
Registry, and Repository components to ensure consistency across the application.
"""

from typing import TYPE_CHECKING, Any, cast

from .fetcher import DataFetcher
from .registry import Registry, registry
from .repository import Repository
from .strategy_repository import StrategyRepository

if TYPE_CHECKING:
    from stockula.database.manager import DatabaseManager
    from stockula.utils.logging_manager import LoggingManager


class DataManager:
    """Centralized manager for data-related components.

    This class provides a single point of access for DataFetcher, Registry,
    and Repository instances, ensuring consistency and proper initialization.

    Attributes:
        _fetcher: The DataFetcher instance for retrieving market data
        _registry: The Registry instance for managing repositories
        _db_manager: Optional database manager for persistence
        _logging_manager: Optional logging manager for logging
    """

    def __init__(
        self,
        db_manager: "DatabaseManager | None" = None,
        logging_manager: "LoggingManager | None" = None,
        use_cache: bool = True,
        db_path: str | None = None,
    ) -> None:
        """Initialize the DataManager.

        Args:
            db_manager: Optional database manager for persistence
            logging_manager: Optional logging manager for logging
            use_cache: Whether to use caching in DataFetcher
            db_path: Optional database path for DataFetcher
        """
        self._db_manager = db_manager
        self._logging_manager = logging_manager

        # Initialize DataFetcher
        self._fetcher = DataFetcher(
            use_cache=use_cache,
            db_path=db_path or "stockula.db",
            database_manager=db_manager,
            logging_manager=cast(Any, logging_manager),
        )

        # Use the global registry instance
        self._registry = registry

        # Initialize registry with repositories
        self._initialize_registry()

    def _initialize_registry(self) -> None:
        """Initialize the registry with default repositories."""
        # Register strategy repository if not already registered
        if "strategies" not in self._registry:
            strategy_repo = StrategyRepository(self._db_manager)
            self._registry.register_repository("strategies", strategy_repo)
        else:
            # Update existing strategy repository with DB manager if needed
            existing_repo = self._registry["strategies"]
            if hasattr(existing_repo, "db_manager") and existing_repo.db_manager != self._db_manager:
                existing_repo.db_manager = self._db_manager
                if self._db_manager is not None:
                    existing_repo.sync_to_database()  # type: ignore[attr-defined]

    @property
    def fetcher(self) -> DataFetcher:
        """Get the DataFetcher instance.

        Returns:
            The DataFetcher instance
        """
        return cast(DataFetcher, self._fetcher)

    @property
    def registry(self) -> Registry:
        """Get the Registry instance.

        Returns:
            The Registry instance
        """
        return self._registry

    @property
    def strategies(self) -> StrategyRepository:
        """Get the strategy repository.

        Returns:
            The StrategyRepository instance
        """
        repo = self._registry.get_repository("strategies")
        if repo is None:
            # Auto-register if not exists
            repo = StrategyRepository(self._db_manager)
            self._registry.register_repository("strategies", repo)
        return repo  # type: ignore[return-value]

    def get_repository(self, name: str) -> Repository[Any] | None:
        """Get a repository by name.

        Args:
            name: The name of the repository

        Returns:
            The repository instance or None if not found
        """
        return self._registry.get_repository(name)

    def register_repository(self, name: str, repository: Repository[Any]) -> None:
        """Register a new repository.

        Args:
            name: The name to register the repository under
            repository: The repository instance to register

        Raises:
            ValueError: If a repository with the same name already exists
        """
        self._registry.register_repository(name, repository)

    def update_database_manager(self, db_manager: "DatabaseManager | None") -> None:
        """Update the database manager and sync all repositories.

        Args:
            db_manager: The new database manager or None
        """
        self._db_manager = db_manager

        # Update DataFetcher's database manager
        self._fetcher.db = db_manager

        # Update all repositories that support database persistence
        for _name, repo in self._registry.repositories.items():
            if hasattr(repo, "db_manager"):
                repo.db_manager = db_manager
                if db_manager is not None and hasattr(repo, "sync_to_database"):
                    repo.sync_to_database()
