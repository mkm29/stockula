# Registry Pattern Documentation

## Overview

The Stockula application uses a Registry pattern to manage various repositories of domain objects. This pattern provides a centralized location for accessing different types of repositories while maintaining loose coupling and testability.

## Architecture

### Core Components

1. **Registry** (`src/stockula/data/registry.py`)

   - Central registry for managing multiple repositories
   - Provides a single point of access to all repositories
   - Supports registration, retrieval, and removal of repositories

1. **Repository** (`src/stockula/data/repository.py`)

   - Abstract base class for all repositories
   - Defines common interface for managing collections of items
   - Provides generic type support for type safety

1. **DataRepository** (`src/stockula/data/repository.py`)

   - Simple in-memory implementation of Repository
   - Can be used for any type of data storage needs

1. **StrategyRepository** (`src/stockula/data/strategy_repository.py`)

   - Specialized repository for trading strategies
   - Supports database persistence
   - Manages strategy presets and groups

### Key Features

- **Type Safety**: Uses Python generics for type-safe repositories
- **Database Support**: Optional database persistence for repositories
- **Immutable Defaults**: Strategy presets use deep copies to prevent accidental modification
- **Case-Insensitive**: Strategy names are normalized to lowercase for consistency
- **Singleton Pattern**: Global registry instance ensures single point of access

## Usage Examples

### Basic Registry Usage

```python
from stockula.data.registry import Registry
from stockula.data.repository import DataRepository

# Create a registry
registry = Registry()

# Create and register a repository
user_repo = DataRepository[User]()
registry.register_repository("users", user_repo)

# Access repository
users = registry.get_repository("users")
# or use subscript notation
users = registry["users"]

# Check if repository exists
if "users" in registry:
    # Do something with users repository
    pass
```

### Strategy Repository Usage

```python
from stockula.data.strategy_repository import strategy_repository

# Get available strategies
strategies = strategy_repository.get_available_strategy_names()
# Returns: ['smacross', 'rsi', 'macd', ...]

# Get a strategy class
strategy_class = strategy_repository.get('smacross')
# or case-insensitive
strategy_class = strategy_repository.get('SMACross')

# Get strategy presets
presets = strategy_repository.get_strategy_preset('smacross')
# Returns: {'fast_period': 10, 'slow_period': 20}

# Update strategy preset
strategy_repository.update_strategy_preset('smacross', {
    'fast_period': 15,
    'slow_period': 30
})

# Work with strategy groups
groups = strategy_repository.get_strategy_groups()
basic_strategies = strategy_repository.get_strategies_in_group('basic')
```

### Database Integration

```python
from stockula.database.manager import DatabaseManager
from stockula.data.registry_setup import initialize_registry

# Initialize registry with database support
db_manager = DatabaseManager('path/to/db.sqlite')
initialize_registry(db_manager)

# Strategies are automatically synced to database
# and loaded on initialization
```

### Creating Custom Repositories

```python
from stockula.data.repository import Repository
from typing import TypeVar

T = TypeVar('T')

class CustomRepository(Repository[T]):
    """Custom repository implementation."""
    
    def add(self, key: str, item: T) -> None:
        """Add item with validation."""
        # Custom validation logic
        if self.validate_item(item):
            self._items[key] = item
        else:
            raise ValueError("Invalid item")
    
    def get(self, key: str, default: T | None = None) -> T | None:
        """Get item with custom logic."""
        # Custom retrieval logic
        return self._items.get(key, default)
    
    def remove(self, key: str) -> None:
        """Remove item with cleanup."""
        if key not in self._items:
            raise KeyError(f"Item {key} not found")
        # Cleanup logic
        del self._items[key]
    
    def validate_item(self, item: T) -> bool:
        """Validate item before adding."""
        # Implementation specific validation
        return True
```

## Design Decisions

### Why Registry Pattern?

1. **Centralized Access**: Single point of access for all repositories
1. **Loose Coupling**: Components depend on registry interface, not concrete implementations
1. **Testability**: Easy to mock repositories for testing
1. **Extensibility**: New repositories can be added without modifying existing code

### Why Lowercase Strategy Names?

- **Consistency**: Ensures consistent naming across the application
- **User-Friendly**: Users don't need to remember exact casing
- **Simplicity**: Reduces complexity by avoiding alias management

### Database Persistence

- **Optional**: Repositories work without database
- **Automatic Sync**: Changes are automatically persisted when database is available
- **Migration Support**: Uses SQLModel for easy schema migrations

## Testing

The registry pattern implementation has comprehensive test coverage:

- `test_registry.py`: Tests for Registry class (100% coverage)
- `test_registry_setup.py`: Tests for registry initialization (100% coverage)
- `test_repository.py`: Tests for base Repository classes (100% coverage)
- `test_strategy_repository.py`: Tests for StrategyRepository (93% coverage)

### Example Test

```python
def test_registry_operations():
    """Test basic registry operations."""
    registry = Registry()
    repo = DataRepository[str]()
    
    # Register repository
    registry.register_repository("test", repo)
    
    # Verify registration
    assert "test" in registry
    assert registry.get_repository("test") is repo
    
    # Remove repository
    registry.remove_repository("test")
    assert "test" not in registry
```

## Best Practices

1. **Use Type Hints**: Always specify generic types for repositories
1. **Handle Missing Repositories**: Check existence before accessing
1. **Immutable Defaults**: Use deep copies for default configurations
1. **Consistent Naming**: Use lowercase names for strategies
1. **Error Handling**: Handle database errors gracefully

## Future Enhancements

1. **Additional Repositories**:

   - IndicatorRepository for technical indicators
   - PortfolioRepository for portfolio configurations
   - BacktestRepository for backtest results

1. **Caching**: Add caching layer for frequently accessed items

1. **Event System**: Emit events when repositories are modified

1. **Validation Framework**: Standardized validation for all repositories

1. **Migration Tools**: Automated migration for repository schema changes
