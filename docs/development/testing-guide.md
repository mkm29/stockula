# Testing Guide for Stockula

A quick reference guide for writing and running tests in the Stockula project.

## Quick Start

### Running All Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/stockula --cov-report=term-missing

# Run specific module tests
uv run pytest tests/unit/test_strategies.py -v
```

### Writing New Tests

#### Testing a New Strategy

1. **Add strategy attributes test**:

```python
def test_my_strategy_attributes(self):
    """Test strategy has required attributes."""
    assert hasattr(MyStrategy, 'period')
    assert MyStrategy.period == 20
```

2. **Add data requirements test**:

```python
def test_my_strategy_data_requirements(self):
    """Test minimum data requirements."""
    assert MyStrategy.get_min_required_days() == 40  # period + buffer
```

3. **Add inheritance test**:

```python
def test_my_strategy_inheritance(self):
    """Test strategy inherits from BaseStrategy."""
    assert issubclass(MyStrategy, BaseStrategy)
```

#### Testing a New Indicator

1. **Extract the calculation to `indicators.py`**:

```python
def calculate_my_indicator(prices: pd.Series, period: int = 20) -> pd.Series:
    """Calculate my custom indicator."""
    return prices.rolling(window=period).mean()  # Example
```

2. **Write comprehensive tests**:

```python
class TestMyIndicator:
    def test_calculation(self):
        """Test basic calculation."""
        prices = pd.Series([100, 102, 101, 103, 105])
        result = calculate_my_indicator(prices, period=3)
        assert len(result) == len(prices)
        assert not result.iloc[-1] != result.iloc[-1]  # Not NaN

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
# Create synthetic data for testing
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

### Mocking Framework Components

```python
def test_strategy_logic(self):
    """Test strategy logic with mocks."""
    # Create mock strategy attributes
    mock_strategy = Mock()
    mock_strategy.position = None
    mock_strategy.sma_short = Mock()
    mock_strategy.sma_long = Mock()

    # Set up indicator values
    mock_strategy.sma_short.__getitem__ = Mock(return_value=55)
    mock_strategy.sma_long.__getitem__ = Mock(return_value=50)

    # Test logic
    # Note: Direct testing of init() and next() is not possible
```

### Testing Indicator Calculations

```python
def test_indicator_properties(self):
    """Test indicator mathematical properties."""
    prices = create_test_prices()
    indicator = calculate_my_indicator(prices)

    # Test properties
    assert indicator.min() >= prices.min() * 0.9  # Within bounds
    assert indicator.max() <= prices.max() * 1.1
    assert not indicator.isna().all()  # Has valid values
```

## Debugging Failed Tests

### Common Issues and Solutions

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

3. **Data-Dependent Tests**

```python
# Make tests deterministic
np.random.seed(42)
data = create_synthetic_data(seed=42)
```

### Debugging Coverage Issues

```bash
# Generate detailed HTML coverage report
uv run pytest --cov=src/stockula --cov-report=html

# Open coverage report
open htmlcov/index.html

# Check specific file coverage
uv run pytest --cov=src/stockula/backtesting/strategies --cov-report=term-missing
```

## Test Checklist

When adding new functionality:

- [ ] Extract calculations to testable functions
- [ ] Write unit tests for calculations
- [ ] Test edge cases (empty data, single value, extreme values)
- [ ] Test error conditions and warnings
- [ ] Verify inheritance and attributes
- [ ] Document why certain code isn't tested
- [ ] Run tests with coverage to verify
- [ ] Ensure all tests pass before committing

## Performance Considerations

### Keep Tests Fast

```python
# Use small datasets for unit tests
def test_fast_calculation(self):
    # Use 100 data points, not 10,000
    data = create_test_data(days=100)

# Mark slow tests
@pytest.mark.slow
def test_comprehensive_backtest(self):
    # This test runs full backtest
    pass
```

### Use Fixtures for Shared Data

```python
@pytest.fixture
def market_data():
    """Shared market data for tests."""
    return test_data_manager.load_data('SPY')

def test_with_fixture(market_data):
    """Test using shared data."""
    assert len(market_data) > 0
```

## Continuous Integration

### GitHub Actions Workflow

Tests are automatically run through the `test.yml` workflow:

- **Triggers**: All pull requests and pushes to main
- **Jobs**:
  1. **Linting**: Code style checks with `ruff`
  1. **Unit Tests**: Fast, isolated tests with coverage
  1. **Integration Tests**: Currently disabled, will test with SQLite

### Running Tests Locally

```bash
# Run linting (same as CI)
uv run ruff check src tests
uv run ruff format --check src tests

# Run unit tests with coverage (same as CI)
uv run pytest tests/unit -v --cov=stockula --cov-report=xml --cov-report=term-missing

# Run integration tests (when enabled)
DATABASE_URL=sqlite:///./test_stockula.db STOCKULA_ENV=test uv run pytest tests/integration -v
```

### Test Organization

```
tests/
├── unit/           # Fast, isolated tests
│   ├── test_strategies.py
│   ├── test_indicators.py
│   └── ...
└── integration/    # Tests with database/external services
    ├── test_data_fetching.py
    └── ...
```

### Coverage Requirements

- Unit tests report to Codecov with `unit` flag
- Integration tests report with `integration` flag
- Coverage reports are optional (won't fail CI)

Ensure your tests:

- Are deterministic (same result every time)
- Don't depend on external services (for unit tests)
- Complete within reasonable time (\<5 minutes total)
- Follow the existing test structure

## Getting Help

- See `docs/development/testing-strategy.md` for detailed testing philosophy
- Check existing tests for examples
- Use `pytest --markers` to see available test markers
- Run `pytest --help` for pytest options
