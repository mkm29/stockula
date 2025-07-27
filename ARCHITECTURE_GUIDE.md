# Stockula Architecture Guide

This document consolidates all architectural decisions, refactoring changes, and implementation patterns that have shaped Stockula's current design.

## Table of Contents

1. [Domain Model Evolution](#domain-model-evolution)
2. [Read-Only Properties Implementation](#read-only-properties-implementation)
3. [Private Methods and Encapsulation](#private-methods-and-encapsulation)
4. [Hold-Only Assets Strategy](#hold-only-assets-strategy)
5. [Configuration Architecture](#configuration-architecture)
6. [Performance Optimizations](#performance-optimizations)

---

## Domain Model Evolution

### Major Architectural Shift: Allocation-Based to Quantity-Based

**Previous Design**: Portfolio allocations were based on fixed dollar amounts or percentages
**Current Design**: Portfolio tracking is based on actual share quantities with dynamic valuations

#### Key Changes Made

1. **Asset Class Transformation**
   - **Removed**: `allocation_amount` (fixed dollar allocation)
   - **Added**: `quantity` (number of shares)
   - **New method**: `get_value(current_price)` - calculates position value
   - Assets now represent actual share holdings

2. **Portfolio Class Redesign**
   - **Removed**: Complex inheritance hierarchy (AssetContainer → CompositeAssetContainer → Portfolio)
   - **Removed**: `_calculate_allocations()` method
   - **Added**: 
     - `get_portfolio_value(prices)` - calculates total portfolio value
     - `get_asset_allocations(prices)` - calculates current allocations with caching
     - `get_portfolio_summary(prices)` - comprehensive portfolio analysis
   - Portfolio is now a standalone class with no inheritance

3. **Category System Simplification**
   - **Before**: Category was a container class that held assets
   - **After**: Category is now a simple Enum for classification
   - **Removed**: `allocation_pct` and `allocation_amount` properties from categories
   - **Added**: Dynamic category allocation calculation methods
   - Categories include: TECHNOLOGY, HEALTHCARE, FINANCIAL, GROWTH, VALUE, LARGE_CAP, etc.

#### Benefits of This Architecture

1. **Realistic Modeling**: Portfolios now model actual share ownership instead of theoretical allocations
2. **Dynamic Valuations**: Portfolio values change with market prices in real-time
3. **Performance**: Allocation calculations are cached using `@lru_cache` for efficiency
4. **Flexibility**: Easy to track gains/losses and rebalancing needs
5. **Simplicity**: Removed unnecessary abstraction layers and inheritance complexity

#### Migration Example

```python
# OLD: Allocation-based configuration
portfolio:
  buckets:
    - name: tech_stocks
      allocation_amount: 60000
      tickers:
        - symbol: AAPL
          allocation_amount: 1000

# NEW: Quantity-based configuration
portfolio:
  tickers:
    - symbol: AAPL
      quantity: 10
      category: TECHNOLOGY

# OLD: Static allocation access
portfolio.categories[0].allocation_amount

# NEW: Dynamic allocation calculation
current_prices = {"AAPL": 195.50, "GOOGL": 140.25}
allocations = portfolio.get_allocation_by_category(current_prices)
```

---

## Read-Only Properties Implementation

### Complete Immutability Strategy

All domain model attributes that represent immutable characteristics have been converted to read-only properties using Python's `@property` decorator pattern.

#### Implementation Pattern

All read-only properties follow this consistent pattern:

```python
@dataclass
class Example:
    # InitVar for initialization
    attribute: InitVar[Type]
    
    # Private storage
    _attribute: Type = field(init=False, repr=False)
    
    # Post-init to set value
    def __post_init__(self, attribute: Type):
        self._attribute = attribute
    
    # Read-only property
    @property
    def attribute(self) -> Type:
        """Get attribute value (read-only)."""
        return self._attribute
```

#### Properties Made Read-Only by Class

##### 1. Ticker Class
- **Read-only**: `symbol` (ticker symbols should never change)
- **Mutable**: `sector`, `market_cap`, `category`, `price_range`, `metadata` (can be updated as company information changes)
- **Special**: Symbol automatically converted to uppercase during initialization

##### 2. Asset Class  
- **Read-only**: `ticker`, `quantity`, `category`, `allocation_amount`
- **Derived property**: `symbol` (convenience accessor for `ticker.symbol`)

##### 3. Portfolio Class
- **Read-only**: `name`, `initial_capital`, `allocation_method`
- **Mutable**: `rebalance_frequency`, `max_position_size`, `stop_loss_pct`, `assets`

##### 4. Category Class (Legacy)
- **Read-only**: `description`, `allocation_pct`, `allocation_amount`
- **Mutable**: `name`, `assets` list

#### Benefits of Read-Only Properties

1. **Data Integrity**: Core attributes cannot be accidentally modified
2. **Clear Intent**: API clearly shows which values are immutable
3. **Type Safety**: Properties maintain type hints and validation
4. **Validation**: Values validated once during creation
5. **Flexibility**: Collections and strategy parameters remain mutable where appropriate

#### Usage Example

```python
# Creating objects with read-only properties
asset = Asset(ticker=ticker, quantity=50, category=Category.TECHNOLOGY)
portfolio = Portfolio(name="Growth Portfolio", initial_capital=100000)

# Reading values (works)
print(asset.quantity)        # 50
print(portfolio.name)        # "Growth Portfolio"

# Setting values (raises AttributeError)
asset.quantity = 100         # Error!
portfolio.initial_capital = 200000  # Error!

# Mutable attributes still work
portfolio.max_position_size = 15.0  # OK
```

---

## Private Methods and Encapsulation

### Comprehensive Privacy Implementation

Methods starting with `_` are considered internal/private by Python convention. This refactoring improved encapsulation by hiding implementation details.

#### Methods Made Private

1. **Portfolio._calculate_allocations()** 
   - Internal allocation calculation method
   - Only used by `get_asset_percentage()`, `get_all_asset_percentages()`, and `validate_allocations()`

2. **DomainFactory private methods**:
   - `_create_ticker()` - Internal factory helper
   - `_create_asset()` - Internal factory helper
   - `_create_portfolio_bucket()` - Internal factory helper
   - Only called by public `create_portfolio()` method

3. **TickerRegistry._clear()**
   - Dangerous method for clearing the singleton registry
   - Primarily useful for testing, not production use

4. **DataConfig._get_ticker_symbols()**
   - Unused internal helper method
   - Not part of public API

#### Methods Kept Public

1. **Pydantic Validators**: All `@field_validator` and `@model_validator` methods must remain public
2. **Asset calculation methods**: `calculate_allocation()` and `calculate_percentage()` are part of public API
3. **TickerRegistry core methods**: `get_or_create()`, `get()`, `all()`, `__len__()`, `__contains__()`

#### Benefits of Encapsulation

1. **Better Encapsulation**: Internal implementation details are hidden
2. **Clearer API**: Public methods represent the intended interface
3. **Reduced Coupling**: External code can't depend on internal details
4. **Easier Refactoring**: Internal methods can change without breaking external code

---

## Hold-Only Assets Strategy

### Category-Based Trading Exclusion

Hold-only assets are securities that should be bought and held without active trading, protecting core portfolio positions from algorithmic strategies.

#### Implementation Details

1. **Category-Based Exclusion**: Assets marked as hold-only based on their `Category` enum value
2. **Default Hold-Only Categories**:
   - `INDEX` - Index funds and ETFs
   - `BOND` - Fixed income securities

#### Configuration

```yaml
backtest:
  hold_only_categories:
    - INDEX
    - BOND
    - ETF        # Add more categories as needed
    - DEFENSIVE  # Add defensive stocks
    - DIVIDEND   # Add dividend stocks

portfolio:
  tickers:
    - symbol: SPY
      quantity: 100
      category: INDEX      # This will be hold-only
    - symbol: AAPL
      quantity: 50
      category: TECHNOLOGY # This will be traded
```

#### How It Works

1. System identifies all assets in hold-only categories
2. Assets are listed at the start of trading operations
3. Backtesting strategies skip these assets entirely
4. Technical analysis and forecasting still run on these assets

#### Benefits

1. **Protects Core Holdings**: Prevents trading algorithms from churning index fund positions
2. **Reduces Transaction Costs**: Avoids unnecessary trades on buy-and-hold positions
3. **Maintains Asset Allocation**: Keeps strategic asset allocation intact
4. **Flexible Configuration**: Easy to customize which categories to hold

---

## Configuration Architecture

### Separation of Concerns: Configuration vs Domain

The architecture maintains a clear separation between mutable configuration models and immutable domain objects.

#### Configuration Models (Mutable)
- Located in `config/models.py`
- Use regular Pydantic attributes for flexibility during parsing
- Support validation and transformation during loading
- Can be modified before creating domain objects

#### Domain Models (Immutable)
- Located in `domain/` directory
- Use read-only properties to ensure immutability after creation
- Represent the actual business objects used during execution
- Cannot be modified once created

#### Bridge Pattern: DomainFactory
- Converts mutable configuration objects to immutable domain objects
- Handles validation and business logic during conversion
- Provides clean separation between configuration and runtime models

#### Configuration Evolution

**Previous Structure** (Complex hierarchical):
```yaml
portfolio:
  buckets:
    - name: tech_stocks
      allocation_amount: 60000
      tickers:
        - symbol: AAPL
          allocation_amount: 1000
```

**Current Structure** (Flat and simple):
```yaml
portfolio:
  initial_capital: 100000
  tickers:
    - symbol: AAPL
      quantity: 10
      category: TECHNOLOGY
```

---

## Performance Optimizations

### Caching Strategy for Dynamic Calculations

Since portfolios now calculate allocations dynamically based on current prices, caching is crucial for performance.

#### Allocation Calculation Caching

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def _calculate_allocations_cached(
    assets_tuple: Tuple[Tuple[str, float, float], ...],  # (symbol, quantity, price)
    total_value: float,
) -> Dict[str, Dict[str, float]]:
    """Cached calculation of asset allocations."""
    # Expensive allocation calculations cached here
```

#### Cache Strategy Details

1. **Module-level caching**: Uses `@lru_cache` decorator for automatic memoization
2. **Cache key composition**: Includes asset quantities and current prices
3. **Automatic invalidation**: Cache invalidates when prices or quantities change
4. **Performance benefit**: Significant improvement for repeated calculations
5. **Memory management**: LRU eviction prevents unbounded memory growth

#### Database Caching

The SQLite database provides another layer of caching:

1. **Automatic data storage**: All yfinance API calls are cached automatically
2. **Offline access**: Data available without internet connectivity
3. **Fast lookups**: Indexed database queries much faster than API calls
4. **Transparent integration**: Existing code works unchanged with caching enabled

---

## Summary of Architectural Principles

### 1. Immutability by Default
- Domain objects use read-only properties for core attributes
- Configuration objects remain mutable for flexibility
- Clear separation between configuration and runtime phases

### 2. Composition Over Inheritance
- Removed complex inheritance hierarchies
- Portfolio as standalone class rather than inherited container
- Simple aggregation relationships between objects

### 3. Performance Through Caching
- Multiple layers of caching (memory, database)
- Transparent caching that doesn't change APIs
- Automatic invalidation strategies

### 4. Clear API Boundaries
- Private methods hide implementation details
- Public methods represent intended interfaces
- Configuration clearly separated from domain logic

### 5. Realistic Financial Modeling
- Quantity-based portfolio tracking
- Dynamic valuations based on current market prices
- Hold-only asset protection for core positions

This architecture provides a solid foundation for quantitative trading applications while maintaining flexibility, performance, and data integrity.