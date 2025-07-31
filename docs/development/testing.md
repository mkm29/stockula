Unit testing a Python backtesting system is crucial for ensuring the correctness and reliability of your trading strategies and the backtesting framework itself.
Here's how you can approach unit testing in the context of Python backtesting:

1. Isolate Components for Testing:
   Strategy Logic:

- Test individual components of your trading strategy in isolation. For example, if your strategy calculates a moving average crossover, write a unit test to ensure the moving average calculation is correct for various data inputs.
  Order Management:
- Test functions responsible for placing, modifying, and canceling orders. Verify that orders are created with the correct parameters and that the order book is updated as expected.
  Position Management:
- Test functions that manage positions (opening, closing, adjusting size). Ensure that profit and loss calculations are accurate and that positions are correctly tracked.
  Data Handling:
- Test functions that load, process, and clean historical data. Verify that data is loaded correctly and that any transformations (e.g., resampling, indicator calculations) produce the expected output.

2. Use a Testing Framework:
   unittest (Built-in):
   Python's built-in unittest module provides a solid foundation for creating unit tests. You can define test classes that inherit from unittest.TestCase and write test methods to assert expected outcomes.
   pytest (Third-Party):
   pytest is a popular and powerful testing framework known for its simplicity and extensibility. It offers features like fixtures, parameterization, and clear test reporting, making it efficient for testing complex backtesting systems.
1. Create Mock Objects (if necessary):
   Mock External Dependencies: If your backtesting system interacts with external services (e.g., data providers, broker APIs), use mocking libraries (e.g., unittest.mock or pytest-mock) to simulate these interactions during unit tests. This allows you to test your code in isolation without relying on actual external connections.
1. Write Assertions:
   Verify Expected Outcomes: Use assertion methods provided by your chosen testing framework (e.g., assertEqual, assertTrue, assertFalse, assertAlmostEqual) to check if the actual output of your code matches the expected output for various test cases.

Example (using unittest for a simple indicator calculation):

```python
import unittest
import pandas as pd

class MyStrategy:
    def calculate_sma(self, data, window):
        return data['Close'].rolling(window=window).mean()

class TestMyStrategy(unittest.TestCase):
    def test_calculate_sma(self):
        data = pd.DataFrame({
            'Close': [10, 11, 12, 13, 14, 15]
        })
        expected_sma_3 = pd.Series([None, None, 11.0, 12.0, 13.0, 14.0])

        strategy = MyStrategy()
        calculated_sma = strategy.calculate_sma(data, 3)

        pd.testing.assert_series_equal(calculated_sma, expected_sma_3, check_dtype=False)

if __name__ == '__main__':
    unittest.main()
```

By implementing comprehensive unit tests, you can identify and fix bugs early in the development cycle, improve code quality, and gain confidence in the accuracy of your backtesting results.

## Integration Testing with Dependency Injection

When testing components that use dependency injection (like those using the `@inject` decorator from `dependency-injector`), you need to properly wire the container in your tests:

```python
from stockula.container import Container
from unittest.mock import Mock

def test_injected_function(self):
    # Create container and override dependencies
    container = Container()
    mock_data_fetcher = Mock()
    container.data_fetcher.override(mock_data_fetcher)

    # IMPORTANT: Wire the container to enable injection
    container.wire(modules=["stockula.main"])

    # Now you can call the injected function
    result = run_technical_analysis("AAPL", config)
```

## Testing Best Practices for Stockula

### Mock Setup for Technical Indicators

When mocking technical indicators that return pandas-like objects with `iloc` indexing:

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

### Database Testing

The DatabaseManager automatically creates stocks if they don't exist when storing price data. This means foreign key constraint tests should verify auto-creation rather than expecting exceptions:

```python
def test_foreign_key_constraint(self, test_database):
    # Price data can be added for non-existent stock (auto-creation)
    test_database.add_price_data(
        "NEWSTOCK", datetime.now(), 100.0, 101.0, 99.0, 100.5, 1000000, "1d"
    )

    # Verify the stock was auto-created
    with test_database.get_session() as session:
        stock = session.query(Stock).filter_by(symbol="NEWSTOCK").first()
        assert stock is not None
```

### Configuration Testing

When testing functions that expect specific configuration structures, ensure your test configs include all required fields:

```python
# For backtest tests, include strategies
sample_stockula_config.backtest.strategies = [
    StrategyConfig(name="smacross", parameters={"fast_period": 10, "slow_period": 20})
]
```

### Rich Console Output Testing

When testing functions that use Rich for console output, assertions should check for the formatted output components rather than exact string matches:

```python
# Instead of:
assert output == "SMA_20: 150.00"

# Use:
assert "SMA_20" in output
assert "150.00" in output
```
