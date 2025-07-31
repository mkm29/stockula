# Data Fetching

Stockula uses an intelligent data fetching system that combines yfinance for market data with SQLite caching for optimal performance.

## Overview

The data fetching module provides:

- **Automatic Caching**: All market data is cached in SQLite for fast access
- **Smart Updates**: Only fetches missing or stale data
- **Multiple Data Sources**: Stock prices, dividends, splits, options
- **Offline Support**: Works with cached data when API is unavailable
- **Rich Progress**: Visual progress tracking for data operations

## Data Sources

### Stock Price Data

Fetches OHLCV (Open, High, Low, Close, Volume) data:

```python
from stockula.data.fetcher import DataFetcher

fetcher = DataFetcher()

# Get stock data for a single ticker
data = fetcher.get_stock_data("AAPL", start_date="2023-01-01")

# Get data for multiple tickers
symbols = ["AAPL", "GOOGL", "MSFT"]
data = fetcher.get_stock_data(symbols, start_date="2023-01-01")
```

### Supported Intervals

| Interval           | Description  | Example Use Case    |
| ------------------ | ------------ | ------------------- |
| `1m`, `2m`, `5m`   | Minute data  | Intraday trading    |
| `15m`, `30m`, `1h` | Hourly data  | Short-term analysis |
| `1d`               | Daily data   | Standard analysis   |
| `1wk`              | Weekly data  | Medium-term trends  |
| `1mo`              | Monthly data | Long-term analysis  |

### Additional Data Types

#### Dividend History

```python
dividends = fetcher.get_dividends("AAPL")
```

#### Stock Splits

```python
splits = fetcher.get_splits("AAPL")
```

#### Company Information

```python
info = fetcher.get_stock_info("AAPL")
# Returns: sector, industry, market_cap, pe_ratio, etc.
```

#### Options Data

```python
# Get options chains
calls = fetcher.get_options_calls("AAPL", "2024-01-19")
puts = fetcher.get_options_puts("AAPL", "2024-01-19")
```

## Caching System

### SQLite Database Schema

```sql
-- Core market data tables
stocks              -- Stock metadata and info
price_history       -- OHLCV data with timestamps
dividends          -- Dividend payment history
splits             -- Stock split history
stock_info         -- Complete yfinance info as JSON

-- Options data
options_calls      -- Call options chains
options_puts       -- Put options chains
```

### Cache Strategy

The caching system follows these principles:

1. **Cache First**: Always check database before API calls
1. **Freshness Validation**: Ensure data is recent enough
1. **Selective Updates**: Only fetch missing date ranges
1. **Background Refresh**: Update stale data automatically

### Data Freshness Rules

| Data Type        | Freshness Window | Update Frequency              |
| ---------------- | ---------------- | ----------------------------- |
| Intraday (1m-1h) | 1 hour           | Real-time during market hours |
| Daily prices     | 6 hours          | End of trading day            |
| Weekly/Monthly   | 1 day            | Weekly/monthly updates        |
| Company info     | 7 days           | Weekly refresh                |
| Options          | 1 hour           | Real-time during market hours |

## Configuration

### Basic Configuration

```yaml
data:
  start_date: "2023-01-01"
  end_date: null              # defaults to today
  interval: "1d"
  use_cache: true
  db_path: "stockula.db"
```

### Advanced Settings

```yaml
data:
  # Date range settings
  start_date: "2020-01-01"
  end_date: "2024-01-01"
  interval: "1d"

  # Caching settings
  use_cache: true
  db_path: "stockula.db"
  cache_expiry_hours: 6
  max_cache_size_mb: 500

  # API settings
  request_delay: 0.1          # Delay between API requests
  max_retries: 3
  timeout_seconds: 30

  # Data validation
  validate_data: true
  drop_invalid_rows: true
  fill_missing_values: false
```

## Database Management

### Command Line Interface

```bash
# View database statistics
uv run python -m stockula.database.cli stats

# Manually fetch data
uv run python -m stockula.database.cli fetch AAPL MSFT GOOGL

# Query cached data
uv run python -m stockula.database.cli query AAPL --start 2023-01-01

# Clear cache for specific symbols
uv run python -m stockula.database.cli clear AAPL

# Vacuum database (reclaim space)
uv run python -m stockula.database.cli vacuum
```

### Programmatic Access

```python
from stockula.database.manager import DatabaseManager

db = DatabaseManager("stockula.db")

# Check what data is cached
symbols = db.get_cached_symbols()
print(f"Cached symbols: {symbols}")

# Get cache statistics
stats = db.get_cache_stats()
print(f"Database size: {stats['size_mb']:.2f} MB")
print(f"Total records: {stats['total_records']}")

# Manual cache management
db.clear_cache_for_symbol("AAPL")
db.vacuum_database()
```

## Performance Optimization

### Bulk Data Fetching

When fetching data for multiple symbols, use bulk operations:

```python
# Efficient: Single bulk request
symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
data = fetcher.get_stock_data(symbols, start_date="2023-01-01")

# Inefficient: Multiple individual requests
data = {}
for symbol in symbols:
    data[symbol] = fetcher.get_stock_data(symbol, start_date="2023-01-01")
```

### Date Range Optimization

Request only the data you need:

```python
# Get only recent data for real-time analysis
recent_data = fetcher.get_stock_data("AAPL", start_date="2024-01-01")

# Use appropriate intervals
intraday = fetcher.get_stock_data("AAPL", interval="5m", start_date="2024-01-20")
```

### Memory Management

For large datasets, consider chunking:

```python
from datetime import datetime, timedelta

def fetch_data_in_chunks(symbol, start_date, end_date, chunk_months=6):
    """Fetch data in smaller chunks to manage memory."""
    current_date = start_date
    all_data = []

    while current_date < end_date:
        chunk_end = min(current_date + timedelta(days=30*chunk_months), end_date)
        chunk_data = fetcher.get_stock_data(symbol,
                                          start_date=current_date,
                                          end_date=chunk_end)
        all_data.append(chunk_data)
        current_date = chunk_end

    return pd.concat(all_data)
```

## Error Handling

### Network Issues

The fetcher handles common network problems gracefully:

```python
try:
    data = fetcher.get_stock_data("AAPL")
except ConnectionError:
    # Falls back to cached data if available
    data = fetcher.get_cached_data("AAPL")
except TimeoutError:
    # Retry with exponential backoff
    data = fetcher.get_stock_data("AAPL", retries=3)
```

### Data Validation

Built-in validation catches common data issues:

```python
# Automatically handles:
# - Missing dates (weekends, holidays)
# - Invalid price data (negative prices, etc.)
# - Duplicate records
# - Timezone conversion issues

data = fetcher.get_stock_data("AAPL", validate=True)
```

### Missing Data

```python
# Check for missing data
missing_dates = fetcher.find_missing_dates("AAPL", "2023-01-01", "2023-12-31")

# Fill missing data
complete_data = fetcher.get_stock_data("AAPL",
                                     start_date="2023-01-01",
                                     fill_missing=True)
```

## Integration with Analysis Modules

### Technical Analysis Integration

```python
from stockula.technical_analysis.indicators import TechnicalAnalysis

# Data fetcher automatically integrates with technical analysis
ta = TechnicalAnalysis()
data = fetcher.get_stock_data("AAPL")
indicators = ta.calculate_indicators(data, indicators=["sma", "rsi", "macd"])
```

### Backtesting Integration

```python
from stockula.backtesting.runner import BacktestRunner

# Backtest runner uses cached data automatically
runner = BacktestRunner()
results = runner.run_backtest("AAPL", strategy="smacross",
                            start_date="2023-01-01")
```

### Forecasting Integration

```python
from stockula.forecasting.forecaster import Forecaster

# Forecaster gets historical data for training
forecaster = Forecaster()
forecast = forecaster.forecast_price("AAPL", forecast_length=30)
```

## Rich CLI Progress

When using the CLI, data fetching shows beautiful progress indicators:

```
⠋ Fetching price data for AAPL... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 85% 0:00:02
⠋ Caching data to database... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:01
✓ Successfully fetched and cached data for AAPL
```

Multi-symbol progress tracking:

```
⠋ Fetching data for 5 symbols...
✓ AAPL: 252 days cached
⠋ GOOGL: Fetching from API... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 60% 0:00:03
⠋ MSFT: Queued
⠋ AMZN: Queued
⠋ TSLA: Queued
```

## Best Practices

### Data Management

1. **Use appropriate date ranges**: Don't fetch more data than needed
1. **Leverage caching**: Let the system cache data automatically
1. **Batch operations**: Fetch multiple symbols together when possible
1. **Monitor cache size**: Vacuum database periodically for large datasets

### Configuration

1. **Set realistic cache expiry**: Balance freshness vs. API usage
1. **Use appropriate intervals**: Match interval to analysis timeframe
1. **Configure request delays**: Respect API rate limits

### Development

1. **Handle errors gracefully**: Network issues are common with financial APIs
1. **Validate data**: Always check for data quality issues
1. **Use offline mode**: Develop with cached data when possible
1. **Monitor API usage**: Track requests to avoid limits

The data fetching system provides a robust foundation for all Stockula analysis modules while maintaining high performance through intelligent caching.
