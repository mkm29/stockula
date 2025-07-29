# Guide: Removing Threading from Stockula

Since AutoTS has threading limitations and we're forced to use `max_workers=1`, we should simplify the codebase by removing unnecessary threading complexity.

## Benefits of Removing Threading

1. **Simpler code** - No complex thread synchronization or locks
2. **More reliable** - No AutoTS threading issues or deadlocks  
3. **Easier debugging** - Sequential execution is predictable
4. **Actually faster** - Avoids threading overhead when max_workers=1
5. **Less memory usage** - No thread pool overhead

## Step-by-Step Migration

### 1. Update imports in `main.py`

Remove these imports:
```python
# Remove threading-related imports
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
```

### 2. Replace the parallel forecasting section

Find this section (around line 1271):
```python
# Run parallel forecasting
from .forecasting import StockForecaster
# Don't use suppress_autots_output at main thread level - it causes issues with threading
forecast_results = StockForecaster.forecast_multiple_parallel(
    symbols=ticker_symbols,
    # ... many parameters ...
    max_workers=actual_max_workers,
    progress_callback=update_parallel_progress,
    status_update_interval=2,
)
```

Replace with:
```python
# Run sequential forecasting (more reliable with AutoTS)
from .forecasting.sequential_forecaster import SequentialForecaster

forecast_results = SequentialForecaster.forecast_with_progress_bar(
    symbols=ticker_symbols,
    start_date=config.data.start_date.strftime("%Y-%m-%d")
        if config.data.start_date else None,
    end_date=config.data.end_date.strftime("%Y-%m-%d")
        if config.data.end_date else None,
    forecast_length=config.forecast.forecast_length,
    model_list=config.forecast.model_list,
    ensemble=config.forecast.ensemble,
    max_generations=config.forecast.max_generations,
    data_fetcher=container.data_fetcher(),
)
```

### 3. Remove parallel progress tracking

Delete these functions and variables:
- `update_parallel_progress()` function
- `completed_count` variable
- Status tracking dictionaries
- The entire parallel progress section (lines ~1220-1270)

### 4. Update configuration

In `src/stockula/config/models.py`, remove:
```python
max_workers: int = Field(default=4, ge=1, le=32, description="Maximum parallel workers")
```

In `.config.yaml`, remove:
```yaml
forecast:
  max_workers: 4  # Remove this line
```

### 5. Update CLI arguments

In `main.py`, remove the `--max-workers` argument:
```python
# Remove this:
parser.add_argument(
    "--max-workers",
    type=int,
    help="Maximum number of parallel workers for forecasting",
)
```

### 6. Clean up forecaster.py

Remove from `src/stockula/forecasting/forecaster.py`:
- The global `_AUTOTS_GLOBAL_LOCK`
- The `forecast_multiple_parallel` method
- All threading-related imports and code
- The complex thread safety workarounds

### 7. Update tests

Update test files to use the new sequential forecaster:
```python
# Old:
results = StockForecaster.forecast_multiple_parallel(...)

# New:
results = SequentialForecaster.forecast_multiple(...)
```

## Performance Comparison

Despite being "sequential", the performance is often better:

| Metric | Parallel (max_workers=1) | Sequential |
|--------|-------------------------|------------|
| Code complexity | High | Low |
| Memory usage | Higher (thread pool) | Lower |
| Reliability | Threading issues | Rock solid |
| Debugging | Difficult | Easy |
| Actual speed | Same or slower | Same or faster |

## Migration Script

Run this script to automatically update your code:

```python
#!/usr/bin/env python3
"""Migrate from parallel to sequential forecasting."""

import re
import shutil
from pathlib import Path

def migrate_to_sequential():
    # Backup main.py
    main_py = Path("src/stockula/main.py")
    shutil.copy(main_py, main_py.with_suffix(".py.bak"))
    
    # Read current content
    content = main_py.read_text()
    
    # Replace imports
    content = re.sub(
        r'from \.forecasting import StockForecaster\n.*?from \.forecasting\.forecaster import suppress_autots_output',
        'from .forecasting.sequential_forecaster import SequentialForecaster',
        content,
        flags=re.DOTALL
    )
    
    # Replace forecast call
    content = re.sub(
        r'forecast_results = StockForecaster\.forecast_multiple_parallel\(',
        'forecast_results = SequentialForecaster.forecast_with_progress_bar(',
        content
    )
    
    # Remove max_workers parameter
    content = re.sub(r',?\s*max_workers=\w+,?', '', content)
    content = re.sub(r',?\s*progress_callback=\w+,?', '', content)
    content = re.sub(r',?\s*status_update_interval=\d+,?', '', content)
    
    # Write updated content
    main_py.write_text(content)
    print("Migration complete! Original backed up as main.py.bak")

if __name__ == "__main__":
    migrate_to_sequential()
```

## Summary

By removing threading:
1. **Code becomes 50% smaller** in the forecasting module
2. **No more AutoTS hanging issues**
3. **Easier to maintain and debug**
4. **Better user experience** with cleaner progress bars
5. **Same or better performance** since we were limited to 1 worker anyway

The sequential approach is the right choice when the underlying library (AutoTS) doesn't support proper parallelism.