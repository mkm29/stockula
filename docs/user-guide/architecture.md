# Architecture Overview

Stockula is built with a modular, domain-driven architecture that separates concerns and provides clean interfaces between components.

## High-Level Architecture

```mermaid
graph TB
    subgraph "User Interface"
        CLI[CLI Entry Point<br/>cli.py]
        Main[Legacy Entry Point<br/>main.py]
        Manager[Business Logic<br/>StockulaManager]
        Display[Results Display<br/>ResultsDisplay]
        Config[Configuration<br/>.config.yaml]
    end

    subgraph "Core Domain"
        Factory[Domain Factory]
        Portfolio[Portfolio]
        Asset[Asset]
        Ticker[Ticker]
        Category[Category]
        AllocatorManager[Allocator Manager<br/>Manages All Strategies]
        Allocator[Standard Allocator<br/>Basic Strategies]
        BacktestAllocator[Backtest Allocator<br/>Optimization]
    end

    subgraph "Data Layer"
        DataMgr[Data Manager<br/>Centralized Access]
        Fetcher[Data Fetcher<br/>yfinance wrapper]
        Registry[Registry<br/>Repository Pattern]
        StrategyRepo[Strategy Repository]
        DB[(SQLite Database<br/>stockula.db)]
        Cache[Cache Manager]
    end

    subgraph "Analysis Modules"
        TA[Technical Analysis<br/>finta]
        TAM[Technical Analysis Manager<br/>Coordinates Indicators]
        BT[Backtesting<br/>strategies]
        BTM[Backtesting Manager<br/>Coordinates Strategies]
        FC[Forecasting<br/>AutoTS]
        FCM[Forecasting Manager<br/>Coordinates Strategies]
    end

    subgraph "Configuration & Utilities"
        Models[Config Models<br/>Pydantic]
        LogManager[Logging Manager]
        Container[DI Container]
    end

    CLI --> Manager
    Main --> CLI
    CLI --> Config
    Manager --> Display
    Manager --> Container
    Config --> Models
    Models --> Factory
    Factory --> Portfolio
    Portfolio --> Asset
    Asset --> Ticker
    Asset --> Category
    Manager --> AllocatorManager
    AllocatorManager --> Allocator
    AllocatorManager --> BacktestAllocator
    Factory --> AllocatorManager

    Manager --> TAM
    Manager --> BTM
    Manager --> FCM
    TAM --> TA
    BTM --> BT
    FCM --> FC

    Container --> DataMgr
    DataMgr --> Fetcher
    DataMgr --> Registry
    Registry --> StrategyRepo
    
    TA --> Fetcher
    BT --> Fetcher
    FC --> Fetcher
    BTM --> StrategyRepo

    Fetcher --> DB
    Fetcher --> Cache
    Cache --> DB

    Container --> LogManager
    LogManager --> Models

    style CLI fill:#2196F3,stroke:#1976D2,color:#fff
    style Main fill:#90CAF9,stroke:#64B5F6,color:#333
    style Manager fill:#FF5722,stroke:#D84315,color:#fff
    style Display fill:#607D8B,stroke:#455A64,color:#fff
    style Config fill:#4CAF50,stroke:#388E3C,color:#fff
    style DB fill:#FF9800,stroke:#F57C00,color:#fff
    style Portfolio fill:#9C27B0,stroke:#7B1FA2,color:#fff
    style Factory fill:#9C27B0,stroke:#7B1FA2,color:#fff
    style LogManager fill:#607D8B,stroke:#455A64,color:#fff
    style Container fill:#795548,stroke:#5D4037,color:#fff
    style AllocatorManager fill:#E91E63,stroke:#C2185B,color:#fff
    style Allocator fill:#E91E63,stroke:#C2185B,color:#fff
    style BacktestAllocator fill:#E91E63,stroke:#C2185B,color:#fff
    style TAM fill:#00BCD4,stroke:#0097A7,color:#fff
    style BTM fill:#00BCD4,stroke:#0097A7,color:#fff
    style FCM fill:#00BCD4,stroke:#0097A7,color:#fff
    style DataMgr fill:#3F51B5,stroke:#303F9F,color:#fff
    style Registry fill:#3F51B5,stroke:#303F9F,color:#fff
    style StrategyRepo fill:#3F51B5,stroke:#303F9F,color:#fff
```

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Manager
    participant Display
    participant Config
    participant Factory
    participant Portfolio
    participant Fetcher
    participant DB
    participant Analysis

    User->>CLI: Run command
    CLI->>Config: Load .config.yaml
    CLI->>Manager: Create manager instance
    Manager->>Factory: Create domain objects
    Factory->>Portfolio: Build portfolio
    Portfolio->>Fetcher: Request prices
    Fetcher->>DB: Check cache
    alt Cache hit
        DB-->>Fetcher: Return cached data
    else Cache miss
        Fetcher->>yfinance: Fetch from API
        yfinance-->>Fetcher: Return data
        Fetcher->>DB: Store in cache
    end
    Fetcher-->>Portfolio: Return prices
    Manager->>Analysis: Run analysis
    Analysis-->>Manager: Return results
    Manager->>Display: Format results
    Display-->>CLI: Formatted output
    CLI-->>User: Display output
```

## Module Structure

```
src/stockula/
├── __init__.py           # Main package exports
├── main.py               # Legacy entry point (for backward compatibility)
├── cli.py               # CLI entry point with Typer
├── manager.py            # Business logic manager (StockulaManager)
├── display.py            # Results display and formatting (ResultsDisplay)
├── container.py          # Dependency injection container
├── interfaces.py         # Abstract interfaces
├── config/               # Configuration management
│   ├── __init__.py
│   ├── models.py        # Pydantic models
│   └── settings.py      # Config loading
├── domain/               # Domain models
│   ├── __init__.py
│   ├── portfolio.py     # Portfolio management
│   ├── asset.py         # Asset representation
│   ├── ticker.py        # Ticker & registry
│   ├── category.py      # Category enum
│   └── factory.py       # Domain object factory
├── allocation/           # Allocation strategies
│   ├── __init__.py
│   ├── manager.py       # AllocatorManager - coordinates strategies
│   ├── base_allocator.py # Base allocator interface
│   ├── allocator.py     # Standard allocation strategies
│   └── backtest_allocator.py  # Backtest-optimized allocation
├── data/                 # Data fetching module
│   ├── __init__.py
│   └── fetcher.py       # yfinance wrapper with SQLite caching
├── database/             # SQLite database management
│   ├── __init__.py
│   ├── manager.py       # Database operations
│   └── cli.py           # Command-line interface
├── technical_analysis/   # Technical indicators
│   ├── __init__.py
│   ├── manager.py       # TechnicalAnalysisManager - coordinates strategies
│   └── indicators.py    # finta wrapper
├── backtesting/         # Strategy backtesting
│   ├── __init__.py
│   ├── manager.py       # BacktestingManager - coordinates strategies
│   ├── strategies.py    # Pre-built strategies
│   └── runner.py        # Backtest execution
├── forecasting/         # Price prediction
│   ├── __init__.py
│   ├── manager.py       # ForecastingManager - coordinates strategies
│   └── forecaster.py    # AutoTS wrapper
└── utils/               # Utilities
    ├── __init__.py
    └── logging_manager.py # Centralized logging management
```

## Core Components

### Configuration Layer

**Purpose**: Manages all application settings using Pydantic for validation

**Key Files**:

- `config/models.py` - Pydantic models for type-safe configuration
- `config/settings.py` - Configuration loading and validation

**Features**:

- YAML file support with environment variable overrides
- Strong typing with Pydantic validation
- Hierarchical configuration structure
- Default value management

### Domain Layer

**Purpose**: Represents business concepts and rules

**Key Components**:

- **StockulaManager**: Orchestrates all business logic and operations
- **ResultsDisplay**: Handles formatting and presentation of results
- **Portfolio**: Manages collections of assets with allocation strategies
- **Asset**: Represents individual holdings with market data
- **Ticker**: Maps symbols to detailed information
- **Factory**: Creates and configures domain objects
- **Category**: Categorizes assets (INDEX, MOMENTUM, etc.)
- **AllocatorManager**: Coordinates all allocation strategies and provides unified interface
- **Allocator**: Standard allocator for basic strategies (equal weight, market cap, custom, dynamic, auto)
- **BacktestOptimizedAllocator**: Advanced allocation using backtest performance data
- **ForecastingManager**: Coordinates forecasting strategies and provides unified interface for different forecasting models

**Patterns**:

- Domain-driven design principles
- Factory pattern for object creation
- Value objects for immutable data

### Data Management Layer

**Purpose**: Provides centralized access to data fetching, registry, and repository components

**Key Components**:

- **DataManager**: Central coordinator for all data-related operations
- **DataFetcher**: Wrapper around yfinance for market data retrieval
- **Registry**: Global registry for managing repositories
- **StrategyRepository**: Repository for trading strategy classes and metadata

**Features**:

- Single point of access for data operations
- Automatic initialization of components
- Database connection management across all components
- Support for custom repository registration

**Benefits**:

- Consistent initialization and configuration
- Simplified dependency management
- Easy database connection updates
- Extensible repository pattern

### Strategy Registry

The `StrategyRegistry` is a centralized static class that manages all trading strategies, their mappings, parameters, and groups. It provides a single source of truth for strategy-related operations across the entire application.

#### Key Features

**Strategy Management:**

- **Centralized Class Registry**: Maps strategy names to strategy classes
- **Name Normalization**: Automatically converts between PascalCase and snake_case formats
- **Parameter Presets**: Default parameters for each strategy
- **Strategy Groups**: Predefined collections of strategies for different trading approaches

**API Methods:**

```python
# Strategy class resolution
StrategyRegistry.get_strategy_class("SMACross")  # Returns SMACrossStrategy
StrategyRegistry.get_strategy_class("double_ema_cross")  # Same class

# Name normalization
StrategyRegistry.normalize_strategy_name("DoubleEMACross")  # Returns "double_ema_cross"

# Validation and discovery
StrategyRegistry.is_valid_strategy("SMACross")  # Returns True
StrategyRegistry.get_available_strategy_names()  # List all strategies
StrategyRegistry.get_strategy_groups()  # Get predefined groups

# Parameter management
StrategyRegistry.get_strategy_preset("smacross")  # Get default parameters
StrategyRegistry.get_strategy_presets()  # Get all presets
```

#### Strategy Groups

Predefined strategy collections for different trading approaches:

- **basic**: `["smacross", "rsi"]` - Simple strategies for beginners
- **momentum**: `["rsi", "macd", "double_ema_cross"]` - Momentum-based strategies
- **trend**: `["smacross", "triple_ema_cross", "trima_cross"]` - Trend following strategies
- **advanced**: `["kama", "frama", "vama", "vidya"]` - Adaptive and sophisticated strategies
- **comprehensive**: All available strategies combined

#### Name Normalization

The registry handles multiple naming formats automatically:

| Input Format          | Normalized Output      | Strategy Class              |
| --------------------- | ---------------------- | --------------------------- |
| `"SMACross"`          | `"smacross"`           | `SMACrossStrategy`          |
| `"DoubleEMACross"`    | `"double_ema_cross"`   | `DoubleEMACrossStrategy`    |
| `"KaufmanEfficiency"` | `"kaufman_efficiency"` | `KaufmanEfficiencyStrategy` |
| `"ER"`                | `"kaufman_efficiency"` | `KaufmanEfficiencyStrategy` |

This ensures configuration files can use any naming convention while maintaining consistency internally.

### Data Layer

**Purpose**: Handles all data fetching and caching

**Key Components**:

- **DataFetcher**: yfinance wrapper with intelligent caching
- **DatabaseManager**: SQLite operations with schema management
- **Cache Strategy**: Automatic data freshness management

**Features**:

- Automatic SQLite caching for all market data
- Database migrations with Alembic
- Multiple data sources (yfinance, Treasury rates)
- Offline capability with cached data

### Analysis Modules

**Purpose**: Implements financial analysis algorithms

**Modules**:

1. **Technical Analysis** (`technical_analysis/`)

   - TechnicalAnalysisManager for coordinating analysis strategies
   - Wraps finta library for indicators
   - Pre-configured analysis groups (basic, momentum, trend, volatility)
   - Vectorized calculations for performance
   - Configurable indicator parameters
   - Automatic signal generation and summary

1. **Backtesting** (`backtesting/`)

   - BacktestingManager for coordinating backtesting strategies
   - Strategy implementation framework
   - Integration with backtesting.py library
   - Risk management features
   - Support for multiple strategies per asset
   - Strategy name mapping for compatibility

1. **Forecasting** (`forecasting/`)

   - ForecastingManager for coordinating different forecasting strategies
   - AutoTS integration for time series prediction
   - Multiple model ensembles (standard, fast, financial)
   - Confidence intervals and validation
   - Quick forecast for rapid predictions
   - Financial-specific forecasting optimizations
   - Sequential and parallel processing support

### Utilities

**Purpose**: Cross-cutting concerns and shared functionality

**Components**:

- **LoggingManager**: Centralized logging with Rich integration
- **Container**: Dependency injection container
- **Interfaces**: Abstract base classes for modularity

## Design Principles

### Separation of Concerns

Each module has a single responsibility:

- **cli.py**: Modern CLI entry point with Typer
- **main.py**: Legacy entry point for backward compatibility
- **manager.py**: Business logic orchestration and coordination
- **display.py**: Results formatting and presentation
- **Config**: Settings and validation
- **Domain**: Business logic and rules
- **Allocation**: Portfolio allocation strategies with AllocatorManager
- **Data**: External data access
- **Analysis**: Financial computations with dedicated managers
- **Utils**: Shared utilities

### Dependency Injection

The application uses a container-based dependency injection system:

```python
from stockula.container import Container

# Container manages all dependencies
container = Container()

# Automatic injection of configured services
data_fetcher = container.data_fetcher()
portfolio_factory = container.domain_factory()

# Manager-based access to strategies
allocator_manager = container.allocator_manager()
backtesting_manager = container.backtesting_manager()
forecasting_manager = container.forecasting_manager()
technical_analysis_manager = container.technical_analysis_manager()

# The managers provide access to all strategies
quantities = allocator_manager.calculate_quantities(config, tickers)
backtest_results = backtesting_manager.run_backtests(symbols, config)
forecast = forecasting_manager.forecast_symbol('AAPL', config)
ta_results = technical_analysis_manager.analyze(symbol, config)
```

### Interface-Based Design

Abstract interfaces enable modularity and testing:

```python
from stockula.interfaces import IDataFetcher, IBacktestRunner

class CustomDataFetcher(IDataFetcher):
    def get_stock_data(self, symbol: str) -> pd.DataFrame:
        # Custom implementation
        pass
```

### Rich CLI Integration

The Rich library provides enhanced user experience:

- **Progress Bars**: Real-time operation tracking
- **Tables**: Formatted result displays
- **Panels**: Grouped information display
- **Colors**: Status indication and highlighting

## Data Management

### SQLite Database Schema

The database stores all market data for fast access:

```sql
-- Core tables
stocks              -- Stock metadata
price_history       -- OHLCV data
dividends          -- Dividend history
splits             -- Stock splits
stock_info         -- Complete yfinance data as JSON

-- Options data
options_calls      -- Call options chains
options_puts       -- Put options chains
```

### Caching Strategy

Intelligent caching reduces API calls:

1. **Cache First**: Check database before API calls
1. **Freshness Check**: Validate data age
1. **Selective Updates**: Fetch only missing data
1. **Background Updates**: Refresh stale data automatically

### Migration Management

Database schema evolution with Alembic:

```bash
# Automatic migrations on startup
# Manual migration commands available
uv run python -m stockula.database.cli migrate upgrade head
```

## Performance Considerations

### Data Access Patterns

- **Bulk Operations**: Minimize database round trips
- **Vectorized Calculations**: Use pandas/numpy for speed
- **Lazy Loading**: Load data only when needed
- **Connection Pooling**: Reuse database connections

### Memory Management

- **Streaming**: Process large datasets incrementally
- **Chunking**: Break large operations into smaller pieces
- **Cleanup**: Automatic resource cleanup
- **Caching Limits**: Configurable cache size limits

### Async Operations

While not fully async, the architecture supports async patterns:

- **Progress Tracking**: Non-blocking progress updates
- **Background Tasks**: Concurrent data fetching
- **Graceful Shutdown**: Clean process termination

## Testing Architecture

### Test Organization

```
tests/
├── unit/               # Fast, isolated tests
│   ├── test_config.py
│   ├── test_domain.py
│   ├── test_strategies.py
│   └── ...
├── integration/        # Component integration tests
│   ├── test_data.py
│   ├── test_database.py
│   └── ...
└── conftest.py        # Shared fixtures
```

### Testing Patterns

- **Dependency Injection**: Easy mock injection
- **Interface Mocking**: Mock at interface boundaries
- **Fixture Reuse**: Shared test data and mocks
- **Isolated Tests**: No cross-test dependencies

## Extensibility

### Adding New Strategies

1. Create strategy class inheriting from `BaseStrategy`
1. Implement required methods (`init`, `next`)
1. Add strategy to `StrategyRegistry.STRATEGIES` mapping
1. Add normalized name to `StrategyRegistry.STRATEGY_NAME_MAPPING`
1. Add default parameters to `StrategyRegistry.STRATEGY_PRESETS`
1. Optionally add to strategy groups in `StrategyRegistry.STRATEGY_GROUPS`

**Example:**

```python
# 1. Create the strategy class
class MyCustomStrategy(BaseStrategy):
    period = 20  # Required class variable
    
    def init(self):
        # Strategy initialization
        pass
    
    def next(self):
        # Strategy logic
        pass

# 2. Add to StrategyRegistry
StrategyRegistry.STRATEGIES["my_custom"] = MyCustomStrategy
StrategyRegistry.STRATEGY_NAME_MAPPING["MyCustom"] = "my_custom"
StrategyRegistry.STRATEGY_NAME_MAPPING["my_custom"] = "my_custom"
StrategyRegistry.STRATEGY_PRESETS["my_custom"] = {"period": 20}
```

### Custom Data Sources

1. Implement `IDataFetcher` interface
1. Register in dependency injection container
1. Configure via settings if needed

### New Analysis Modules

1. Create module in appropriate directory
1. Define interface for contract
1. Implement concrete classes
1. Wire into main application flow

This architecture provides a solid foundation for financial analysis while remaining flexible and maintainable.
