# Data Fetching

Stockula uses an intelligent data fetching system that combines yfinance for market data with a simplified TimescaleDB
architecture for optimal performance and time-series optimizations.

## Overview

The simplified data fetching module provides:

- **Consolidated DatabaseManager**: Single manager (1,766 lines) handling all database operations
- **Integrated Analytics**: 8 built-in analytics methods eliminating separate query overhead
- **TimescaleDB Caching**: High-performance time-series database caching with automatic partitioning
- **Hypertable Optimization**: Time-based partitioning for optimal query performance
- **Compression & Retention**: Intelligent data lifecycle management
- **Smart Updates**: Only fetches missing or stale data with time-series aware queries
- **Multiple Data Sources**: Stock prices, dividends, splits, options stored in optimized hypertables
- **Offline Support**: Works with cached data when API is unavailable
- **Rich Progress**: Visual progress tracking for data operations
- **Single Interface**: IDatabaseManager provides unified access to all database operations

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

### TimescaleDB Schema

```sql
-- Core market data tables (TimescaleDB hypertables)
stocks              -- Stock metadata and info
price_history       -- OHLCV data (hypertable, partitioned by time)
dividends          -- Dividend payment history (hypertable)
splits             -- Stock split history (hypertable)
stock_info         -- Complete yfinance info as JSON

-- Options data (hypertables for time-series performance)
options_calls      -- Call options chains (hypertable)
options_puts       -- Put options chains (hypertable)

-- TimescaleDB optimizations
-- Automatic time-based partitioning (default: 7-day chunks)
-- Compression policies for data older than 30 days
-- Retention policies for data lifecycle management
-- Continuous aggregates for real-time analytics
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

# TimescaleDB connection
database:
  host: "localhost"
  port: 5432
  name: "stockula"
  user: "stockula_user"
  password: "${STOCKULA_DB_PASSWORD}"
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
  cache_expiry_hours: 6

  # API settings
  request_delay: 0.1          # Delay between API requests
  max_retries: 3
  timeout_seconds: 30

  # Data validation
  validate_data: true
  drop_invalid_rows: true
  fill_missing_values: false

# TimescaleDB-specific settings
timescale:
  chunk_time_interval: "7 days"     # Hypertable chunk size
  compression_after: "30 days"      # Compress data older than 30 days
  retention_policy: "2 years"       # Retain data for 2 years
  connection_pool_size: 10          # Database connection pool size
  query_timeout: 30                 # Query timeout in seconds
```

## Database Management

### Command Line Interface

```bash
# View TimescaleDB statistics and hypertable status
uv run python -m stockula.database.manager status

# Manually fetch data
uv run python -m stockula.database.manager fetch AAPL MSFT GOOGL

# Query cached data with time-series optimizations
uv run python -m stockula.database.manager query AAPL --start 2023-01-01

# Clear cache for specific symbols
uv run python -m stockula.database.manager clear AAPL

# Check compression status
uv run python -m stockula.database.manager compression-status

# Monitor chunk and partition health
uv run python -m stockula.database.manager chunk-status

# Optimize hypertables and update policies
uv run python -m stockula.database.manager optimize
```

### Programmatic Access

```python
from stockula.database.manager import DatabaseManager
from stockula.database.interfaces import IDatabaseManager

# Single manager handles all operations
db: IDatabaseManager = DatabaseManager()

# All database operations through unified interface
symbols = db.get_cached_symbols()
stock_data = db.get_stock_data("AAPL")

# Integrated analytics methods
moving_avgs = db.calculate_moving_averages("AAPL", windows=[20, 50])
rsi_data = db.calculate_rsi("AAPL", period=14)
bollinger = db.calculate_bollinger_bands("AAPL", period=20)
correlation = db.calculate_correlation_matrix(["AAPL", "GOOGL", "MSFT"])

# TimescaleDB statistics and management
stats = db.get_cache_stats()
print(f"Database size: {stats['size_mb']:.2f} MB")
print(f"Total records: {stats['total_records']}")

# Cache management
db.clear_cache_for_symbol("AAPL")

# TimescaleDB-specific operations (all integrated)
db.compress_chunks(older_than="30 days")
db.update_retention_policies()
db.refresh_continuous_aggregates()
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

The data fetching system provides a robust foundation for all Stockula analysis modules while maintaining high
performance through TimescaleDB's time-series optimizations, automatic partitioning, and intelligent compression.
