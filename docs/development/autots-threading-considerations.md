# AutoTS Threading Considerations

## Overview

AutoTS has inherent threading limitations that require careful consideration when implementing parallel forecasting. This document outlines the threading constraints and recommended approaches for working with AutoTS in Stockula.

## Threading Limitations

AutoTS experiences issues with multi-threaded execution:

- Thread safety problems with internal model fitting
- Potential deadlocks when running multiple forecasts simultaneously
- Forced to use `max_workers=1` to avoid reliability issues

## Current Implementation

Stockula currently implements a workaround that effectively serializes AutoTS operations while maintaining a parallel interface:

```python
# Forced sequential execution
max_workers = 1  # AutoTS threading issues require this
```

## Recommended Approach

### Sequential Forecasting

Given AutoTS's threading constraints, sequential forecasting is often more reliable and can be equally performant:

**Benefits:**

- **Simpler code**: No thread synchronization complexity
- **More reliable**: Avoids AutoTS threading issues
- **Easier debugging**: Predictable execution flow
- **Lower overhead**: No thread pool management
- **Equal performance**: When limited to 1 worker anyway

### Implementation Pattern

```python
# Sequential approach with progress tracking
from stockula.forecasting.forecaster import StockForecaster

def forecast_portfolio_sequential(symbols, config, data_fetcher):
    results = {}

    for symbol in symbols:
        forecaster = StockForecaster(
            forecast_length=config.forecast.forecast_length,
            model_list=config.forecast.model_list,
            data_fetcher=data_fetcher
        )

        try:
            forecast = forecaster.forecast_from_symbol(symbol)
            results[symbol] = forecast
        except Exception as e:
            results[symbol] = {"error": str(e)}

    return results
```

## Performance Considerations

| Approach                 | Pros                          | Cons                               |
| ------------------------ | ----------------------------- | ---------------------------------- |
| Parallel (max_workers=1) | Maintains parallel interface  | Threading overhead with no benefit |
| Sequential               | Simple, reliable, no overhead | Cannot leverage multiple cores     |

## Future Improvements

### Alternative Parallelism Strategies

1. **Process-based parallelism**: Use multiprocessing instead of threading
1. **Batch processing**: Group symbols and process in separate processes
1. **Alternative libraries**: Consider libraries with better parallel support

### Migration Path

If migrating away from threading:

1. Update configuration to remove `max_workers` parameter
1. Replace parallel forecasting calls with sequential implementation
1. Update progress tracking for sequential execution
1. Remove threading-related imports and utilities

## Best Practices

1. **Always test with production data volumes**: Threading issues may only appear with larger datasets
1. **Monitor memory usage**: Sequential processing may have different memory patterns
1. **Implement proper error handling**: Ensure failures in one forecast don't affect others
1. **Consider user experience**: Provide clear progress feedback during long-running operations

## Configuration

Current configuration supports both approaches:

```yaml
forecast:
  # Threading configuration (currently limited)
  max_workers: 1  # Set to 1 due to AutoTS limitations

  # Alternative: Use sequential processing
  # sequential: true  # Future option
```

## Conclusion

While parallel processing is desirable for performance, AutoTS's threading limitations make sequential processing a more reliable choice. The codebase should be structured to easily switch between parallel and sequential implementations as the underlying libraries evolve.
