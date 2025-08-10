# DataManager API Reference

The `DataManager` is a centralized class that manages instances of DataFetcher, Registry, and Repository components to
ensure consistency across the application. It provides a single point of access for all data-related operations in
Stockula.

## Overview

The DataManager pattern simplifies data management by:

- Providing centralized access to DataFetcher, Registry, and Repository instances
- Ensuring consistent initialization of data components
- Managing database connections across all data components
- Automatically initializing the strategy repository with the global registry

## Import

```python
from stockula.data import DataManager
```

## Class Initialization

### `__init__(db_manager=None, logging_manager=None, use_cache=True, db_path=None)`

Creates a new DataManager instance.

**Parameters:**

- `db_manager` (DatabaseManager | None): Optional database manager for persistence
- `logging_manager` (LoggingManager | None): Optional logging manager for logging
- `use_cache` (bool): Whether to use caching in DataFetcher (default: True)
- `db_path` (str | None): Optional database path for DataFetcher

**Example:**

```python
from stockula.data import DataManager
from stockula.database import DatabaseManager
from stockula.utils import LoggingManager

# Basic initialization
manager = DataManager()

# With database and logging
db_manager = DatabaseManager("stockula.db")
logging_manager = LoggingManager("stockula")
manager = DataManager(
    db_manager=db_manager,
    logging_manager=logging_manager,
    use_cache=True,
    db_path="stockula.db"
)

# Without caching
manager = DataManager(use_cache=False)
```

## Properties

### `fetcher`

Returns the DataFetcher instance managed by this DataManager.

**Returns:**

- `DataFetcher`: The DataFetcher instance

**Example:**

```python
manager = DataManager()
fetcher = manager.fetcher

# Use the fetcher to get stock data
data = fetcher.get_data("AAPL", start_date="2023-01-01")
```

### `registry`

Returns the Registry instance (global registry).

**Returns:**

- `Registry`: The Registry instance

**Example:**

```python
manager = DataManager()
registry = manager.registry

# Check available repositories
print(registry.list_repositories())
# ['strategies']
```

## Methods

### `get_repository(name: str) -> Repository[Any] | None`

Get a repository by name from the registry.

**Parameters:**

- `name` (str): The name of the repository

**Returns:**

- `Repository[Any] | None`: The repository instance or None if not found

**Example:**

```python
manager = DataManager()

# Get the strategy repository
strategy_repo = manager.get_repository("strategies")
if strategy_repo:
    strategies = strategy_repo.get_available_strategy_names()
```

### `strategies`

Property that returns the strategy repository, auto-registering it if not exists.

**Type:** `StrategyRepository`

**Example:**

```python
manager = DataManager()

# Access strategy repository via property
strategy_repo = manager.strategies

# Use the repository
available = strategy_repo.get_available_strategy_names()
strategy_class = strategy_repo.get_strategy_class("smacross")
```

### `register_repository(name: str, repository: Repository[Any]) -> None`

Register a new repository with the registry.

**Parameters:**

- `name` (str): The name to register the repository under
- `repository` (Repository[Any]): The repository instance to register

**Raises:**

- `ValueError`: If a repository with the same name already exists

**Example:**

```python
from stockula.data import DataManager, DataRepository

manager = DataManager()

# Create and register a custom repository
custom_repo = DataRepository[dict]()
manager.register_repository("custom", custom_repo)

# Now accessible via get_repository
repo = manager.get_repository("custom")
```

### `update_database_manager(db_manager: DatabaseManager | None) -> None`

Update the database manager and sync all repositories.

**Parameters:**

- `db_manager` (DatabaseManager | None): The new database manager or None

**Example:**

```python
from stockula.database import DatabaseManager

manager = DataManager(db_manager=None)

# Later, add database support
db_manager = DatabaseManager("stockula.db")
manager.update_database_manager(db_manager)

# All components now use the new database manager
```

## Integration with Dependency Injection

The DataManager integrates seamlessly with Stockula's dependency injection container:

```python
# In container.py
from dependency_injector import containers, providers
from stockula.data import DataManager

class Container(containers.DeclarativeContainer):
    # Data manager - thread-safe singleton
    data_manager = providers.ThreadSafeSingleton(
        DataManager,
        db_manager=database_manager,
        logging_manager=logging_manager,
        use_cache=providers.Callable(lambda config: config.data.use_cache, stockula_config),
        db_path=providers.Callable(lambda config: config.data.db_path, stockula_config),
    )

    # Data fetcher extracted from data manager
    data_fetcher = providers.ThreadSafeSingleton(
        lambda data_mgr: data_mgr.fetcher,
        data_mgr=data_manager,
    )
```

## Usage Examples

### Basic Usage

```python
from stockula.data import DataManager

# Create data manager
manager = DataManager(use_cache=True, db_path="stockula.db")

# Access components
fetcher = manager.fetcher
registry = manager.registry
strategy_repo = manager.strategies

# Fetch stock data
data = fetcher.get_data("AAPL", start_date="2023-01-01")

# Get available strategies
strategies = strategy_repo.get_available_strategy_names()
print(f"Available strategies: {strategies}")

# Get strategy class
strategy_class = strategy_repo.get_strategy_class("smacross")
```

### Advanced Usage with Custom Repositories

```python
from stockula.data import DataManager, Repository
from typing import Dict

class IndicatorRepository(Repository[Dict[str, float]]):
    """Custom repository for technical indicators."""

    def add(self, key: str, item: Dict[str, float]) -> None:
        self._items[key] = item

    def get(self, key: str, default=None) -> Dict[str, float] | None:
        return self._items.get(key, default)

    def remove(self, key: str) -> None:
        if key not in self._items:
            raise KeyError(f"Indicator '{key}' not found")
        del self._items[key]

# Create and register custom repository
manager = DataManager()
indicator_repo = IndicatorRepository()
manager.register_repository("indicators", indicator_repo)

# Use the custom repository
indicator_repo.add("rsi_settings", {"period": 14, "overbought": 70, "oversold": 30})
settings = manager.get_repository("indicators").get("rsi_settings")
```

### Database Management

```python
from stockula.data import DataManager
from stockula.database import DatabaseManager

# Start without database
manager = DataManager(use_cache=False)

# Perform operations without persistence
data = manager.fetcher.get_data("AAPL")

# Later, enable database persistence
db_manager = DatabaseManager("stockula.db")
manager.update_database_manager(db_manager)

# Now all operations are persisted
```

## Best Practices

1. **Use dependency injection**: Let the container manage DataManager lifecycle
1. **Single instance**: Use one DataManager instance per application
1. **Lazy database**: Start without database if not needed, add later
1. **Custom repositories**: Register domain-specific repositories as needed
1. **Thread safety**: DataManager itself is not thread-safe, use ThreadSafeSingleton in DI container

## Migration from Direct Access

If you're migrating from direct access to DataFetcher:

```python
# Old approach
from stockula.data import DataFetcher

fetcher = DataFetcher(use_cache=True)

# New approach with DataManager
from stockula.data import DataManager

manager = DataManager(use_cache=True)
fetcher = manager.fetcher
strategy_repo = manager.strategies
```

The DataManager provides a cleaner, more maintainable approach to managing data components through a centralized
interface.
