# Testing Guide for Stockula

This document provides comprehensive guidance for testing the Stockula project, including testing strategy, best practices, and recent improvements.

## Quick Start

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/stockula --cov-report=term-missing

# Run specific module tests
uv run pytest tests/unit/test_strategies.py -v

# Generate HTML coverage report
uv run pytest --cov=src/stockula --cov-report=html
open htmlcov/index.html
```

### Coverage Status

| Module          | Coverage | Notes                   |
| --------------- | -------- | ----------------------- |
| Overall Project | 83%      | Excluding strategies.py |
| main.py         | 83%      | Refactored to entry point |
| forecaster.py   | 83%      | +33% improvement        |
| factory.py      | 88%      | +9% improvement         |
| runner.py       | 89%      | +19% improvement        |
| indicators.py   | 98%      | Comprehensive coverage  |
| strategies.py   | Excluded | Framework constraints   |

## Testing Strategy

### Framework Constraints

The `backtesting.py` library imposes several constraints that make unit testing difficult:

- **Read-only Properties**: Key attributes like `position`, `data`, and `trades` are read-only
- **Framework Initialization**: Strategies must be instantiated by the `Backtest` class
- **Tight Coupling**: The `init()` and `next()` methods can only execute within the framework

### Our Approach

1. **Separation of Concerns**: Extract calculations to testable functions
1. **Test What Matters**: Focus on business logic, calculations, and edge cases
1. **Accept Limitations**: Exclude framework-dependent code from coverage

### What We Test

#### Unit Tests

- Pure functions (indicator calculations)
- Class attributes and structure
- Data validation and requirements
- Error handling and edge cases
- Source code patterns using introspection

#### Integration Tests

- Database operations with auto-creation
- Dependency injection wiring
- CLI functionality
- Full workflow scenarios

## Writing Tests

### Testing a New Strategy

```python
class TestMyStrategy:
    def test_attributes(self):
        """Test strategy has required attributes."""
        assert hasattr(MyStrategy, 'period')
        assert MyStrategy.period == 20

    def test_data_requirements(self):
        """Test minimum data requirements."""
        assert MyStrategy.get_min_required_days() == 40  # period + buffer

    def test_inheritance(self):
        """Test strategy inherits from BaseStrategy."""
        assert issubclass(MyStrategy, BaseStrategy)
```

### Testing a New Indicator

1. Extract calculation to `indicators.py`:

```python
def calculate_my_indicator(prices: pd.Series, period: int = 20) -> pd.Series:
    """Calculate my custom indicator."""
    return prices.rolling(window=period).mean()
```

2. Write comprehensive tests:

```python
class TestMyIndicator:
    def test_calculation(self):
        """Test basic calculation."""
        prices = pd.Series([100, 102, 101, 103, 105])
        result = calculate_my_indicator(prices, period=3)
        assert len(result) == len(prices)
        assert not pd.isna(result.iloc[-1])

    def test_edge_cases(self):
        """Test edge cases."""
        # Empty series
        assert len(calculate_my_indicator(pd.Series([]))) == 0

        # Single value
        single = pd.Series([100])
        assert len(calculate_my_indicator(single)) == 1
```

## Test Data Management

### Using Real Market Data

```python
from test_data_manager import test_data_manager

# Load saved test data
data = test_data_manager.load_data('AAPL', period='2y', interval='1d')

# Get subset for testing
subset = test_data_manager.get_test_data_subset(
    ticker='SPY',
    days=100,
    offset=0
)
```

### Creating Synthetic Data

```python
data = test_data_manager.create_synthetic_data(
    days=200,
    start_price=100.0,
    volatility=0.02,
    trend=0.001,
    seed=42  # For reproducibility
)
```

## Common Test Patterns

### Testing with Warnings

```python
def test_insufficient_data_warning(self):
    """Test warning for insufficient data."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Code that should trigger warning
        strategy = create_strategy_with_insufficient_data()

        # Verify warning
        assert len(w) > 0
        assert "Insufficient data" in str(w[0].message)
```

### Dependency Injection Testing

```python
def test_manager_with_mocks(self):
    # Create container and override dependencies
    container = Container()
    mock_data_fetcher = Mock()
    container.data_fetcher.override(mock_data_fetcher)

    # IMPORTANT: Wire the container to enable injection
    container.wire(modules=["stockula.manager"])

    # Create manager with mocked dependencies
    manager = StockulaManager(config, container, console)
    result = manager.run_technical_analysis("AAPL")
```

### Mocking Technical Indicators

```python
def create_iloc_mock(value):
    iloc_mock = Mock()
    iloc_mock.__getitem__ = Mock(return_value=value)
    return iloc_mock

# Setup indicator mock
sma_mock = Mock()
sma_mock.iloc = create_iloc_mock(150.0)
mock_ta_instance.sma.return_value = sma_mock
```

## Test Organization

```bash
tests/
├── unit/                           # Fast, isolated tests
│   ├── test_strategies.py          # Strategy tests (60 tests)
│   ├── test_indicators.py          # Indicator tests (25 tests)
│   ├── test_main.py                # CLI entry point tests
│   ├── test_manager.py             # Business logic tests
│   ├── test_display.py             # Results display tests
│   ├── test_forecaster.py          # Forecaster tests (83% coverage)
│   ├── test_factory.py             # Factory tests (27 tests)
│   ├── test_database_manager.py    # Database tests
│   └── test_domain.py              # Domain model tests
└── integration/                    # Tests with external dependencies
    └── test_data_fetching.py
```

## Recent Improvements (2025)

### Test Consolidation

We consolidated multiple redundant test files to improve maintainability:

**Strategy Tests**:

- Before: 2,248 lines across 5 files
- After: 671 lines in single `test_strategies.py`
- Result: 60 tests covering all 12 strategy classes

**Architecture Refactoring**:

- **main.py**: Refactored to minimal CLI entry point (~350 lines from ~1942)
- **manager.py**: New business logic orchestration module (StockulaManager)
- **display.py**: New results formatting module (ResultsDisplay)
- **allocation/**: New module for allocation strategies
- **forecaster.py**: Achieved 83% coverage (was ~50%)
- **factory.py**: Achieved 88% coverage (was 79%)
- **runner.py**: Achieved 89% coverage (was 70%)

### Benefits

- Eliminated thousands of lines of duplicate code
- Resolved 48 failing tests
- Improved coverage while reducing complexity
- Single source of truth for each module

## Debugging Failed Tests

### Common Issues

1. **NaN Comparisons**

```python
# Wrong
assert value > other_value  # Fails if either is NaN

# Correct
assert not pd.isna(value)
assert value > other_value
```

2. **Floating Point Comparisons**

```python
# Wrong
assert calculated == 0.1

# Correct
assert abs(calculated - 0.1) < 0.0001
# Or use pytest.approx
assert calculated == pytest.approx(0.1, rel=1e-3)
```

3. **Rich Console Output**

```python
# Instead of exact string matches
assert "SMA_20" in output
assert "150.00" in output
```

## Continuous Integration

### GitHub Actions Workflow

Tests run automatically via `.github/workflows/test.yml`:

- **Triggers**: Pull requests and pushes to main
- **Jobs**: Linting, Unit Tests, Integration Tests
- **Coverage**: Reports to Codecov

### Running CI Tests Locally

```bash
# Run linting
uv run ruff check src tests
uv run ruff format --check src tests

# Run unit tests with coverage
uv run pytest tests/unit -v --cov=stockula --cov-report=xml

# Run integration tests
DATABASE_URL=sqlite:///./test_stockula.db STOCKULA_ENV=test uv run pytest tests/integration -v
```

## Best Practices

### Test Checklist

When adding new functionality:

- [ ] Extract calculations to testable functions
- [ ] Write unit tests for calculations
- [ ] Test edge cases (empty data, single value, extremes)
- [ ] Test error conditions and warnings
- [ ] Verify inheritance and attributes
- [ ] Document untestable code
- [ ] Run tests with coverage
- [ ] Ensure all tests pass before committing

### Performance

```python
# Use small datasets for unit tests
def test_fast_calculation(self):
    data = create_test_data(days=100)  # Not 10,000

# Mark slow tests
@pytest.mark.slow
def test_comprehensive_backtest(self):
    pass
```

### Use Fixtures

```python
@pytest.fixture
def market_data():
    """Shared market data for tests."""
    return test_data_manager.load_data('SPY')

def test_with_fixture(market_data):
    assert len(market_data) > 0
```

## Coverage Details

For detailed information about test coverage status, improvements, and excluded modules, see:

- [**Test Coverage Status**](test-coverage-status.md) - Current coverage metrics and untested modules

## Getting Help

- Check existing tests for examples
- Use `pytest --markers` to see test markers
- Run `pytest --help` for options
- Open issues for testing challenges
