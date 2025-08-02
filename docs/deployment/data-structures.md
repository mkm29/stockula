# Stockula Data Structures and Implementation Guide

This document provides comprehensive examples and implementation details for using Stockula's domain model structure, showcasing private methods, read-only properties, and best practices.

## Table of Contents

1. [Domain Model Usage Example](#domain-model-usage-example)
1. [Private Methods Implementation](#private-methods-implementation)
1. [Read-Only Properties Guide](#read-only-properties-guide)
1. [Best Practices](#best-practices)

______________________________________________________________________

## Domain Model Usage Example

### Creating and Using Portfolio Domain Objects

```python
"""Example demonstrating the new domain model structure."""

from stockula.config import StockulaConfig, TickerConfig, CategoryConfig
from stockula.domain import DomainFactory, TickerRegistry

# Create configuration using the new structure
config = StockulaConfig()

# Define some tickers
tech_tickers = [
    TickerConfig(
        symbol="AAPL",
        sector="Technology",
        market_cap=3000.0,
        category="large_cap",
        allocation_amount=9000  # $9k allocation
    ),
    TickerConfig(
        symbol="NVDA",
        sector="Technology",
        market_cap=1100.0,
        category="momentum",
        allocation_amount=15000  # $15k allocation
    ),
    TickerConfig(
        symbol="GOOGL",
        sector="Technology",
        market_cap=1800.0,
        category="large_cap"
        # No allocation specified - will be distributed equally with other unallocated
    )
]

# Create portfolio configuration
config.portfolio.initial_capital = 100000
config.portfolio.allocation_method = "custom"

# Create a tech category with $60k allocation
tech_category = CategoryConfig(
    name="technology",
    description="High-growth technology stocks",
    allocation_amount=60000,  # $60,000 allocation
    tickers=tech_tickers
)

# Add category to portfolio
config.portfolio.categories = [tech_category]

# Add some direct assets (not in categories)
config.data.tickers = [
    TickerConfig(
        symbol="SPY",
        category="index",
        allocation_amount=20000  # Fixed $20k allocation
    ),
    TickerConfig(
        symbol="BRK.B",
        sector="Financial",
        category="value",
        allocation_amount=20000  # $20k allocation
    )
]

# Create domain objects
factory = DomainFactory()
portfolio = factory.create_portfolio(config)

# The ticker registry ensures singleton pattern
registry = TickerRegistry()
print(f"Total registered tickers: {len(registry)}")

# Access ticker information
aapl = registry.get("AAPL")
if aapl:
    print(f"\n{aapl}")
    print(f"  Sector: {aapl.sector}")
    print(f"  Market Cap: ${aapl.market_cap}B")
    print(f"  Category: {aapl.category}")

# Validate and display allocations
validation = portfolio.validate_allocations()
print("\nPortfolio Validation:")
print(f"  Valid: {validation['valid']}")
print(f"  Total Allocated: ${validation['total_allocated']:,.2f}")
print(f"  Unallocated: ${validation['unallocated']:,.2f}")

print("\nAllocations:")
for symbol, amount in sorted(validation['allocations'].items()):
    pct = (amount / portfolio.initial_capital) * 100
    print(f"  {symbol}: ${amount:,.2f} ({pct:.1f}%)")

# Demonstrate new percentage methods
print("\nUsing Portfolio percentage methods:")
aapl_pct = portfolio.get_asset_percentage("AAPL")
if aapl_pct is not None:
    print(f"  AAPL percentage: {aapl_pct:.1f}%")

# Get all percentages
all_percentages = portfolio.get_all_asset_percentages()
print("\nAll asset percentages:")
for symbol, pct in sorted(all_percentages.items()):
    print(f"  {symbol}: {pct:.1f}%")

# Access assets through portfolio
print("\nAsset Details:")
for asset in portfolio.get_all_assets():
    print(f"  {asset}")
    # Demonstrate Asset's calculate_percentage method
    if asset.allocation_amount:
        asset_pct = asset.calculate_percentage(portfolio.initial_capital)
        print(f"    -> {asset_pct:.1f}% of portfolio")

# Access categories
print("\nCategory Summary:")
for category in portfolio.categories:
    allocation = category.calculate_allocation(portfolio.initial_capital)
    print(f"  {category.name}: ${allocation:,.2f}")
    for asset in category.assets:
        print(f"    - {asset}")
```

______________________________________________________________________

## Private Methods Implementation

### Overview

Methods starting with `_` are considered internal/private by Python convention. This analysis identifies methods that should be private to improve encapsulation and API clarity.

### Private Methods Demo

```python
"""Demonstrate the use of private methods in Portfolio class."""

from stockula.config import StockulaConfig
from stockula.domain import DomainFactory

# Create a simple configuration
config = StockulaConfig()
config.portfolio.initial_capital = 100000

# Create portfolio from configuration
factory = DomainFactory()
portfolio = factory.create_portfolio(config)

# Public methods work as expected
print("Using public methods:")
print(f"Portfolio: {portfolio}")
print(f"Initial Capital: ${portfolio.initial_capital:,.2f}")

# Get asset percentages using public method
percentages = portfolio.get_all_asset_percentages()
print(f"\nAsset Percentages: {percentages}")

# Validate allocations using public method
validation = portfolio.validate_allocations()
print("\nValidation Results:")
print(f"  Valid: {validation['valid']}")
print(f"  Total Allocated: ${validation['total_allocated']:,.2f}")

# The following would raise an AttributeError if uncommented:
# allocations = portfolio.calculate_allocations()  # This method is now private

# Internal methods (starting with _) should not be called directly
print("\nNote: _calculate_allocations() is now a private method")
print("It should only be used internally by the Portfolio class")

# However, Python doesn't enforce privacy, so it's still accessible if needed
# (but this is discouraged in production code)
try:
    # This will work but violates encapsulation principles
    allocations = portfolio._calculate_allocations()
    print(f"\nDirect access to private method (discouraged): {allocations}")
except AttributeError as e:
    print(f"\nCannot access private method: {e}")
```

### Methods Made Private

#### 1. Portfolio Class (`src/stockula/domain/portfolio.py`)

- **`_calculate_allocations()`** - Internal method for calculating dollar allocations
  - Only used internally by `get_asset_percentage()`, `get_all_asset_percentages()`, and `validate_allocations()`
  - Not intended for direct external use

#### 2. DomainFactory Class (`src/stockula/domain/factory.py`)

- **`_create_ticker()`** - Internal helper for creating ticker instances
- **`_create_asset()`** - Internal helper for creating asset instances
- **`_create_portfolio_bucket()`** - Internal helper for creating bucket instances
  - These are all internal implementation details only called by `create_portfolio()`
  - External code should only use the public `create_portfolio()` method

#### 3. TickerRegistry Class (`src/stockula/domain/ticker.py`)

- **`_clear()`** - Internal method for clearing the registry
  - Primarily useful for testing, not for production use
  - Making it private prevents accidental clearing of the singleton registry

#### 4. DataConfig Class (`src/stockula/config/models.py`)

- **`_get_ticker_symbols()`** - Internal helper method
  - Not used anywhere in the codebase currently
  - If needed in the future, it should remain an internal detail

### Methods That Should Remain Public

#### 1. PortfolioBucket.calculate_allocation()

- Called by Portfolio class, so it needs to be public
- Part of the public API between Portfolio and PortfolioBucket

#### 2. Pydantic Validators

- All `@field_validator` and `@model_validator` methods must remain public
- Pydantic framework requires access to these methods
- Examples: `validate_allocations()`, `validate_periods()`, `validate_thresholds()`

#### 3. Asset.calculate_allocation() and calculate_percentage()

- These are part of the public API
- Used by Portfolio and potentially by external code

#### 4. TickerRegistry public methods

- `get_or_create()`, `get()`, `all()` - Core public API
- `__len__()`, `__contains__()` - Python special methods must be public

### Benefits of These Changes

1. **Better Encapsulation**: Internal implementation details are hidden
1. **Clearer API**: Public methods represent the intended interface
1. **Reduced Coupling**: External code can't depend on internal details
1. **Easier Refactoring**: Internal methods can be changed without breaking external code

______________________________________________________________________

## Read-Only Properties Guide

### Overview

Converted allocation attributes in domain models to read-only properties using Python's `@property` decorator. This ensures that allocation values can only be set during object initialization and cannot be modified afterwards.

### Implementation Details

#### Using dataclasses with InitVar

```python
@dataclass
class PortfolioBucket:
    name: str
    allocation_pct: InitVar[Optional[float]] = None  # Init-only variable
    _allocation_pct: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self, allocation_pct: Optional[float]):
        self._allocation_pct = allocation_pct

    @property
    def allocation_pct(self) -> Optional[float]:
        """Get bucket-level percentage allocation (read-only)."""
        return self._allocation_pct
```

### Changes Made

#### 1. PortfolioBucket Class (`src/stockula/domain/portfolio.py`)

- Converted `allocation_pct` and `allocation_amount` to read-only properties
- Used `InitVar` for initialization parameters
- Private attributes: `_allocation_pct` and `_allocation_amount`
- Added `@property` decorators for read-only access

#### 2. Asset Class (`src/stockula/domain/asset.py`)

- Converted `allocation_amount` to read-only property
- Used `InitVar` for initialization parameter
- Private attribute: `_allocation_amount`
- Added `@property` decorator for read-only access

### Benefits

1. **Immutability**: Allocation values cannot be changed after object creation
1. **Data Integrity**: Prevents accidental modification of critical financial data
1. **Clear API**: Properties clearly indicate read-only intent
1. **Validation**: Values are validated once during initialization

### Testing Read-Only Properties

```python
"""Test that properties are truly read-only."""

from stockula.domain import PortfolioBucket, Asset, Ticker

# Test PortfolioBucket read-only properties
bucket = PortfolioBucket(name="Tech", allocation_pct=25.0)
print(f"Bucket allocation: {bucket.allocation_pct}%")

try:
    bucket.allocation_pct = 30.0  # This should fail
except AttributeError as e:
    print(f"✓ allocation_pct is read-only: {e}")

# Test Asset read-only properties
ticker = Ticker("AAPL")
asset = Asset(ticker_init=ticker, quantity_init=1.0, category_init=None)
print(f"Asset: {asset}")

# Note: Asset now uses InitVar fields for initialization
# ticker_init, quantity_init, and category_init parameters are used during construction
# The actual ticker, quantity, and category are accessible as read-only properties
```

Expected output:

```
✓ allocation_pct is read-only: property 'allocation_pct' of 'PortfolioBucket' object has no setter
✓ allocation_amount is read-only: property 'allocation_amount' of 'Asset' object has no setter
```

### Configuration vs Domain Models

- **Configuration models** (in `config/models.py`) retain regular attributes for flexibility during configuration parsing
- **Domain models** (in `domain/`) use read-only properties to ensure immutability after creation
- The `DomainFactory` bridges between mutable configs and immutable domain objects

This separation allows configuration to be easily modified before creating domain objects, while domain objects remain immutable during use.

______________________________________________________________________

## Best Practices

### 1. Use Private Methods for Internal Logic

- Prefix internal methods with `_` to indicate they're not part of the public API
- Keep public methods focused on the main use cases
- Document the intended use of private methods

### 2. Implement Read-Only Properties for Critical Data

- Use `@property` decorators for allocation amounts and percentages
- Combine with `InitVar` in dataclasses for clean initialization
- Store actual values in private attributes (prefixed with `_`)

### 3. Leverage the Singleton Pattern for Registries

- Use `TickerRegistry` to ensure ticker objects are unique across the application
- Access tickers through the registry rather than creating new instances

### 4. Separate Configuration from Domain Logic

- Keep configuration models mutable for flexibility
- Convert to immutable domain objects after validation
- Use factory pattern to bridge between the two

### 5. Validate Early and Often

- Implement validation in configuration models using Pydantic
- Add business logic validation in domain models
- Provide clear error messages for invalid states

### Python Private Method Convention

In Python, methods starting with a single underscore (`_`) are considered internal/private by convention. While Python doesn't enforce true privacy (the methods are still accessible), this convention signals to other developers that these methods:

- Are implementation details
- May change without notice
- Should not be used directly by external code
- Are not part of the public API

Double underscore methods (`__method`) trigger name mangling and provide stronger privacy, but single underscore is sufficient for most cases and is the preferred convention in the Python community.
