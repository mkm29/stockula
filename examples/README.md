# Stockula Configuration Examples

This directory contains various configuration examples for different use cases.

## Configuration Files

### config.simple.yaml

Minimal configuration with basic settings for quick testing.

- Single strategy (SMACross)
- Small portfolio (AAPL, GOOGL, MSFT)
- Basic backtesting parameters

### config.example.yaml

Basic configuration example with common settings.

- Multiple tickers with sector information
- Portfolio allocation tracking
- Default backtesting strategies

### config.full.yaml

Comprehensive configuration showing all available options.

- Complete data, portfolio, and backtest settings
- Multiple strategies with parameters
- Technical analysis indicators
- Output configuration

### config.dynamic.yaml

Dynamic allocation example using percentages and amounts.

- Portfolio buckets with allocation amounts
- Percentage-based allocations
- Mixed allocation methods

### config.optimized.yaml

Optimized portfolio configuration with efficient settings.

- Market cap weighted allocation
- Rebalancing frequency
- Risk management parameters

### config.double-ema.yaml

Double EMA (34/55) portfolio strategy configuration.

- Specific EMA strategy parameters
- Category-based holdings (INDEX, MOMENTUM, SPECULATIVE)
- Risk management with ATR multipliers

### config.test-strategies.yaml

Minimal configuration for testing different strategies.

- Focus on strategy backtesting
- Single ticker (AAPL)
- Multiple strategy configurations

### config.test-vidya.yaml

Configuration for testing VIDYA strategy specifically.

- VIDYA strategy parameters
- Small test portfolio
- Quick testing setup

## Usage

To use any of these example configurations:

```bash
# Copy to root directory as .config.yaml
cp examples/config.example.yaml .config.yaml

# Or run directly with --config flag
uv run python -m stockula.main --config examples/config.full.yaml
```

## Key Configuration Sections

### Data Configuration

- `start_date` / `end_date`: Date range for historical data
- `use_cache`: Enable/disable SQLite caching
- `interval`: Data interval (1d, 1h, etc.)

### Portfolio Configuration

- `initial_capital`: Starting capital amount
- `allocation_method`: auto, equal_weight, market_cap, custom, dynamic
- `auto_allocate`: Automatic allocation based on category ratios
- `category_ratios`: Target allocation by category (must sum to 1.0)
- `tickers`: List of assets with properties

### Backtest Configuration

- `broker_config`: Broker-specific fee structure
  - Presets: robinhood, interactive_brokers, td_ameritrade, etrade, fidelity, schwab
- `strategies`: List of strategies to test
- `hold_only_categories`: Categories to exclude from trading

### Forecast Configuration

- `forecast_length`: Number of periods to forecast
- `model_list`: fast, default, slow, parallel
- `ensemble`: auto, simple, distance, horizontal

### Logging Configuration

- `enabled`: Turn logging on/off
- `level`: DEBUG, INFO, WARNING, ERROR
- `log_to_file`: Save logs to file
- `show_allocation_details`: Show detailed allocation calculations
