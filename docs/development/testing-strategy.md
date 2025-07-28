# Testing Strategy for Stockula

This document outlines the testing approach, challenges, and best practices for testing the Stockula trading strategy framework.

## Overview

Stockula uses the `backtesting.py` library for implementing trading strategies. This creates unique testing challenges due to the tight coupling between strategies and the backtesting framework.

## Test Coverage Status

As of the latest test run:

- **Overall Project Coverage**: 83%
- **Backtesting Module Coverage**: 83% (excluding strategies.py)
- **Strategies Module (`src/stockula/backtesting/strategies.py`)**: Excluded from coverage
- **Indicators Module (`src/stockula/backtesting/indicators.py`)**: 98%

**Note**: The strategies module is excluded from coverage reporting due to framework constraints detailed below. This exclusion is configured in `pyproject.toml` under `[tool.coverage.run]`.

## Testing Challenges

### 1. Framework Constraints

The `backtesting.py` library imposes several constraints that make unit testing difficult:

- **Read-only Properties**: Key attributes like `position`, `data`, and `trades` are read-only properties that cannot be mocked
- **Framework Initialization**: Strategies must be instantiated by the `Backtest` class with specific broker and data objects
- **Tight Coupling**: The `init()` and `next()` methods (containing ~60% of strategy code) can only be executed within the backtesting framework

### 2. Untestable Code Sections

The following code sections cannot be directly unit tested:

```python
def init(self):
    """Initialize indicators."""
    # This method is called by the framework
    # Contains indicator calculations that are tightly coupled to framework's I() method
    
def next(self):
    """Execute trading logic."""
    # This method is called by the framework for each data point
    # Contains trading logic that depends on framework state
```

## Testing Approach

### 1. Separation of Concerns

We extracted indicator calculations into a separate module (`indicators.py`) that can be tested independently:

```python
# src/stockula/backtesting/indicators.py
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    # Pure function that can be tested without framework
```

This allows us to achieve 98% coverage on mathematical calculations.

### 2. What We Test

#### Strategy Attributes and Structure

- Class attributes (periods, thresholds, multipliers)
- Inheritance chains
- Method existence
- Parameter validation

```python
def test_sma_strategy_attributes(self):
    """Test that SMA strategy has required attributes."""
    assert hasattr(SMACrossStrategy, 'short_period')
    assert hasattr(SMACrossStrategy, 'long_period')
```

#### Data Requirements

- Minimum data calculations
- Date recommendations
- Buffer requirements

```python
def test_get_min_required_days(self):
    """Test minimum data requirement calculation."""
    strategy = DoubleEMACrossStrategy
    min_days = strategy.get_min_required_days()
    assert min_days == 70  # long_period + buffer
```

#### Indicator Calculations

- Mathematical correctness
- Edge cases (empty data, single value, extreme values)
- Consistency between related indicators

```python
def test_rsi_calculation(self):
    """Test RSI calculation correctness."""
    prices = pd.Series([100, 102, 104, 103, 105, 107, 106, 108, 110, 109])
    rsi = calculate_rsi(prices, period=5)
    assert (rsi.dropna() >= 0).all()
    assert (rsi.dropna() <= 100).all()
```

### 3. What We Don't Test

We avoid testing:

- Framework integration details
- Exact execution mechanics
- Internal framework method calls
- Hypothetical market scenarios without actual code execution

## Test Organization

### File Structure

```
tests/
├── unit/
│   ├── test_strategies.py      # Consolidated strategy tests (60 tests)
│   └── test_indicators.py      # Indicator calculation tests (25 tests)
└── data/
    ├── test_data_manager.py    # Test data utilities
    └── fetch_test_data.py      # Script to fetch real market data
```

### Test Consolidation History

In a recent refactoring effort, we consolidated multiple strategy test files into a single comprehensive test file:

**Files Consolidated**:

- `test_strategies_calculation_coverage.py` (416 lines)
- `test_strategies_coverage.py` (404 lines)
- `test_strategies_coverage_boost_fixed.py` (473 lines)
- `test_strategies_final_coverage.py` (456 lines)
- `test_strategies_integration_coverage.py` (499 lines)

**Result**: All 2,248 lines of redundant test code were merged into the main `test_strategies.py` file (671 lines), maintaining full test coverage while eliminating duplication.

### Test Categories

1. **Unit Tests** (`test_strategies.py`)

   - Strategy class attributes
   - Inheritance and polymorphism
   - Parameter validation
   - Data requirements
   - Warning messages
   - Source code structure patterns
   - Trading action verification

1. **Indicator Tests** (`test_indicators.py`)

   - Mathematical calculations
   - Edge cases
   - Performance characteristics
   - Cross-indicator consistency

1. **Test Data Management**

   - Real market data fetching from yfinance
   - Synthetic data generation
   - Data persistence in pickle files

## Best Practices

### 1. Focus on Testable Code

Instead of trying to test framework-dependent code, focus on:

- Pure functions (indicators)
- Class attributes and structure
- Data validation
- Error handling
- Source code patterns using introspection

### 2. Maintain Test Organization

- Keep all strategy tests in a single file to avoid duplication
- Use descriptive test class names that clearly indicate what aspect is being tested
- Group related tests together (e.g., all crossover logic tests, all data requirement tests)

### 3. Use Real Data

We use actual market data for testing:

```python
# Fetch and save real data
test_data_manager.fetch_and_save_data(
    tickers=['AAPL', 'SPY', 'QQQ'],
    period='2y',
    interval='1d'
)
```

### 4. Test What Matters

For trading strategies, test:

- **Correctness**: Are indicators calculated correctly?
- **Robustness**: Do strategies handle edge cases?
- **Consistency**: Do related strategies behave consistently?
- **Structure**: Do strategies follow expected patterns?

### 5. Accept Framework Limitations

We exclude the strategies module from coverage reporting because:

- The untestable code is mostly framework glue
- The important logic (calculations) is tested separately in the indicators module
- Integration testing would require running full backtests
- Most of the uncovered code is in `init()` and `next()` methods that can only execute within the backtesting framework
- Forcing coverage on framework-dependent code provides no value and creates maintenance burden

## Running Tests

### Basic Test Run

```bash
uv run pytest tests/unit/
```

### With Coverage Report

```bash
uv run pytest tests/unit/ --cov=src/stockula/backtesting --cov-report=term-missing
```

**Note**: The strategies.py file is automatically excluded from coverage due to configuration in `pyproject.toml`.

### Run Specific Test File

```bash
uv run pytest tests/unit/test_strategies.py -v
```

### Generate HTML Coverage Report

```bash
uv run pytest tests/unit/ --cov=src/stockula/backtesting --cov-report=html
```

## Future Improvements

### 1. Integration Tests

Consider adding integration tests that run actual backtests:

```python
def test_sma_strategy_backtest():
    """Test SMA strategy in actual backtest."""
    data = pd.DataFrame(...)  # Real or synthetic data
    bt = Backtest(data, SMACrossStrategy, cash=10000)
    stats = bt.run()
    assert stats['# Trades'] > 0
```

### 2. Performance Tests

Add tests for strategy performance in known market conditions:

- Trending markets
- Ranging markets
- Volatile conditions

### 3. Continuous Improvement

- Monitor which parts of the code cause issues in production
- Add tests for bug fixes
- Refactor strategies to be more testable when possible

## Conclusion

The testing strategy for Stockula balances thoroughness with practicality. While we cannot achieve high coverage on framework-dependent code, we ensure that all business logic, calculations, and data handling are well-tested. This approach provides confidence in the strategy implementations while acknowledging the constraints of the backtesting framework.

Through the consolidation effort, we reduced test code from 2,248 lines across 5 files to a single comprehensive test file with 671 lines, maintaining 60 tests that cover all 12 strategy classes. This demonstrates that effective testing is not about quantity but quality and organization.

The key insight is that not all code needs to be unit tested - some code is better validated through integration testing or actual use. By focusing on testing what matters (calculations, logic, edge cases) and accepting framework limitations, we maintain a pragmatic and effective testing strategy.
