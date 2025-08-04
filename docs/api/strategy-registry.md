# StrategyRegistry API Reference

The `StrategyRegistry` is a centralized static class that manages all trading strategies in Stockula. It provides a single source of truth for strategy classes, naming conventions, parameter presets, and strategy groups.

## Overview

The registry eliminates code duplication and ensures consistency across the application by centralizing all strategy-related operations. It handles automatic name normalization, allowing configuration files to use either PascalCase or snake_case naming formats.

## Import

```python
from stockula.backtesting import StrategyRegistry
```

## Class Methods

### Strategy Class Resolution

#### `get_strategy_class(strategy_name: str) -> Type[BaseStrategy] | None`

Returns the strategy class for a given strategy name (with automatic normalization).

**Parameters:**
- `strategy_name` (str): Strategy name in any format

**Returns:**
- `Type[BaseStrategy] | None`: Strategy class or None if not found

**Example:**
```python
# Both formats work
cls1 = StrategyRegistry.get_strategy_class("SMACross")
cls2 = StrategyRegistry.get_strategy_class("smacross")
assert cls1 == cls2  # SMACrossStrategy

cls3 = StrategyRegistry.get_strategy_class("DoubleEMACross")
assert cls3.__name__ == "DoubleEMACrossStrategy"
```

#### `get_all_strategies() -> dict[str, Type[BaseStrategy]]`

Returns all available strategies as a dictionary mapping names to classes.

**Returns:**
- `dict[str, Type[BaseStrategy]]`: Dictionary of strategy names to classes

**Example:**
```python
all_strategies = StrategyRegistry.get_all_strategies()
print(f"Available: {list(all_strategies.keys())}")
# Available: ['smacross', 'sma_cross', 'rsi', 'macd', ...]
```

### Name Normalization

#### `normalize_strategy_name(strategy_name: str) -> str`

Normalizes strategy name to canonical snake_case format.

**Parameters:**
- `strategy_name` (str): Strategy name in any format

**Returns:**
- `str`: Normalized strategy name in snake_case

**Example:**
```python
# PascalCase to snake_case
assert StrategyRegistry.normalize_strategy_name("SMACross") == "smacross"
assert StrategyRegistry.normalize_strategy_name("DoubleEMACross") == "double_ema_cross"
assert StrategyRegistry.normalize_strategy_name("KaufmanEfficiency") == "kaufman_efficiency"

# Alternative names
assert StrategyRegistry.normalize_strategy_name("ER") == "kaufman_efficiency"

# Already normalized names pass through
assert StrategyRegistry.normalize_strategy_name("smacross") == "smacross"
assert StrategyRegistry.normalize_strategy_name("sma_cross") == "smacross"
```

### Strategy Discovery

#### `get_available_strategy_names() -> list[str]`

Returns list of all available strategy names (canonical snake_case).

**Returns:**
- `list[str]`: List of strategy names

**Example:**
```python
strategies = StrategyRegistry.get_available_strategy_names()
print(strategies)
# ['smacross', 'sma_cross', 'rsi', 'macd', 'double_ema_cross', ...]
```

#### `is_valid_strategy(strategy_name: str) -> bool`

Checks if a strategy name is valid.

**Parameters:**
- `strategy_name` (str): Strategy name to check

**Returns:**
- `bool`: True if the strategy exists, False otherwise

**Example:**
```python
assert StrategyRegistry.is_valid_strategy("SMACross") == True
assert StrategyRegistry.is_valid_strategy("smacross") == True
assert StrategyRegistry.is_valid_strategy("InvalidStrategy") == False
```

### Strategy Groups

#### `get_strategy_groups() -> dict[str, list[str]]`

Returns all available strategy groups.

**Returns:**
- `dict[str, list[str]]`: Dictionary of group names to strategy lists

**Example:**
```python
groups = StrategyRegistry.get_strategy_groups()
print(groups)
# {
#   'basic': ['smacross', 'rsi'],
#   'momentum': ['rsi', 'macd', 'double_ema_cross'],
#   'trend': ['smacross', 'triple_ema_cross', 'trima_cross'],
#   'advanced': ['kama', 'frama', 'vama', 'vidya'],
#   'comprehensive': [...]
# }
```

#### `is_valid_strategy_group(group_name: str) -> bool`

Checks if a strategy group name is valid.

**Parameters:**
- `group_name` (str): Strategy group name to check

**Returns:**
- `bool`: True if the group exists, False otherwise

**Example:**
```python
assert StrategyRegistry.is_valid_strategy_group("momentum") == True
assert StrategyRegistry.is_valid_strategy_group("invalid_group") == False
```

#### `get_strategies_in_group(group_name: str) -> list[str]`

Returns strategies in a specific group.

**Parameters:**
- `group_name` (str): Name of the strategy group

**Returns:**
- `list[str]`: List of strategy names in the group

**Raises:**
- `ValueError`: If the group name is not valid

**Example:**
```python
momentum_strategies = StrategyRegistry.get_strategies_in_group("momentum")
print(momentum_strategies)
# ['rsi', 'macd', 'double_ema_cross']
```

### Parameter Management

#### `get_strategy_presets() -> dict[str, dict[str, Any]]`

Returns default parameter presets for all strategies.

**Returns:**
- `dict[str, dict[str, Any]]`: Dictionary of strategy names to their default parameters

**Example:**
```python
presets = StrategyRegistry.get_strategy_presets()
print(presets["smacross"])
# {'fast_period': 10, 'slow_period': 20}
```

#### `get_strategy_preset(strategy_name: str) -> dict[str, Any]`

Returns parameter preset for a specific strategy.

**Parameters:**
- `strategy_name` (str): Strategy name (will be normalized)

**Returns:**
- `dict[str, Any]`: Dictionary of default parameters for the strategy

**Example:**
```python
# Both naming formats work
params1 = StrategyRegistry.get_strategy_preset("SMACross")
params2 = StrategyRegistry.get_strategy_preset("smacross")
assert params1 == params2
# {'fast_period': 10, 'slow_period': 20}

rsi_params = StrategyRegistry.get_strategy_preset("RSI")
print(rsi_params)
# {'period': 14, 'oversold_threshold': 30.0, 'overbought_threshold': 70.0}
```

### Validation

#### `validate_strategies(strategy_names: list[str]) -> tuple[list[str], list[str]]`

Validates a list of strategy names.

**Parameters:**
- `strategy_names` (list[str]): List of strategy names to validate

**Returns:**
- `tuple[list[str], list[str]]`: Tuple of (valid_strategies, invalid_strategies)

**Example:**
```python
names = ["SMACross", "InvalidStrategy", "double_ema_cross", "BadName"]
valid, invalid = StrategyRegistry.validate_strategies(names)

print(valid)    # ['smacross', 'double_ema_cross']
print(invalid)  # ['InvalidStrategy', 'BadName']
```

## Strategy Groups Reference

The registry provides predefined strategy groups for different trading approaches:

### Basic Group
**Strategies**: `["smacross", "rsi"]`
**Purpose**: Simple strategies suitable for beginners
**Characteristics**: Easy to understand, minimal parameters

### Momentum Group
**Strategies**: `["rsi", "macd", "double_ema_cross"]`
**Purpose**: Strategies that capitalize on price momentum
**Characteristics**: Focus on trend strength and momentum indicators

### Trend Group
**Strategies**: `["smacross", "triple_ema_cross", "trima_cross"]`
**Purpose**: Trend-following strategies
**Characteristics**: Moving average based, good for trending markets

### Advanced Group
**Strategies**: `["kama", "frama", "vama", "vidya"]`
**Purpose**: Sophisticated adaptive strategies
**Characteristics**: Dynamic parameters, adapt to market conditions

### Comprehensive Group
**Strategies**: All available strategies combined
**Purpose**: Complete strategy testing
**Characteristics**: Includes all 11 strategies for comprehensive backtesting

## Name Mapping Reference

The registry handles automatic conversion between different naming formats:

| PascalCase Input | Alternative Input | Normalized Output | Strategy Class |
|-----------------|------------------|------------------|----------------|
| `SMACross` | `sma_cross` | `smacross` | `SMACrossStrategy` |
| `RSI` | - | `rsi` | `RSIStrategy` |
| `MACD` | - | `macd` | `MACDStrategy` |
| `DoubleEMACross` | - | `double_ema_cross` | `DoubleEMACrossStrategy` |
| `TripleEMACross` | - | `triple_ema_cross` | `TripleEMACrossStrategy` |
| `TRIMACross` | - | `trima_cross` | `TRIMACrossStrategy` |
| `KAMA` | - | `kama` | `KAMAStrategy` |
| `FRAMA` | - | `frama` | `FRAMAStrategy` |
| `VAMA` | - | `vama` | `VAMAStrategy` |
| `VIDYA` | - | `vidya` | `VIDYAStrategy` |
| `KaufmanEfficiency` | `ER` | `kaufman_efficiency` | `KaufmanEfficiencyStrategy` |

## Usage Examples

### Configuration File Compatibility

The registry ensures that configuration files can use any naming format:

```yaml
# PascalCase format (legacy)
backtest:
  strategies:
    - name: SMACross
    - name: DoubleEMACross
    - name: KaufmanEfficiency

# snake_case format (recommended)
backtest:
  strategies:
    - name: smacross
    - name: double_ema_cross
    - name: kaufman_efficiency

# Mixed formats (also works)
backtest:
  strategies:
    - name: SMACross      # PascalCase
    - name: double_ema_cross  # snake_case
    - name: ER            # Alternative name
```

All formats are automatically normalized internally to ensure consistency.

### Adding Custom Strategies

To add a new strategy to the registry:

```python
from stockula.backtesting import StrategyRegistry
from stockula.backtesting.strategies import BaseStrategy

# 1. Create your strategy class
class MyCustomStrategy(BaseStrategy):
    period = 20  # Required class variable
    
    def init(self):
        # Strategy initialization
        pass
    
    def next(self):
        # Strategy logic
        pass

# 2. Add to registry
StrategyRegistry.STRATEGIES["my_custom"] = MyCustomStrategy
StrategyRegistry.STRATEGY_NAME_MAPPING["MyCustom"] = "my_custom"
StrategyRegistry.STRATEGY_NAME_MAPPING["my_custom"] = "my_custom"
StrategyRegistry.STRATEGY_PRESETS["my_custom"] = {"period": 20}

# 3. Optionally add to a strategy group
StrategyRegistry.STRATEGY_GROUPS["advanced"].append("my_custom")

# Now you can use it
strategy_class = StrategyRegistry.get_strategy_class("MyCustom")
assert strategy_class == MyCustomStrategy
```

### Integration with BacktestingManager

The registry is seamlessly integrated with the BacktestingManager:

```python
from stockula.backtesting import BacktestingManager

manager = BacktestingManager(data_fetcher, logging_manager)

# All these work due to automatic name normalization
result1 = manager.run_single_strategy("AAPL", "SMACross")
result2 = manager.run_single_strategy("AAPL", "smacross")
result3 = manager.run_single_strategy("AAPL", "sma_cross")

# Strategy groups work seamlessly
results = manager.run_multiple_strategies("AAPL", "momentum")

# Get available strategies
available = manager.get_available_strategies()  # Uses registry internally
```

## Best Practices

1. **Use snake_case in new configurations** for consistency
2. **Leverage strategy groups** for organized backtesting
3. **Validate strategy names** before running backtests
4. **Use the registry API** instead of hardcoding strategy names
5. **Check available strategies** programmatically for dynamic interfaces

The StrategyRegistry provides a robust, flexible foundation for strategy management that grows with your application needs while maintaining backward compatibility.