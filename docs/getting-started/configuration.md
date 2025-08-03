# Configuration

Stockula uses Pydantic for configuration validation and supports YAML files for easy settings management. By default, Stockula looks for `.config.yaml` in the current directory.

## Configuration Structure

All configuration is organized into logical sections:

```yaml
# Data fetching settings
data:
  start_date: "2023-01-01"
  end_date: null  # defaults to today
  interval: "1d"

# Portfolio management settings
portfolio:
  initial_capital: 100000
  allocation_method: equal_weight
  tickers:
    - symbol: AAPL
      quantity: 10
    - symbol: GOOGL
      quantity: 5

# Backtesting settings
backtest:
  initial_cash: 10000.0
  strategies:
    - name: smacross
      parameters:
        fast_period: 10
        slow_period: 20

# Forecasting settings (choose one mode)
# Option 1: Future prediction mode
forecast:
  forecast_length: 30        # Days to forecast from today
  model_list: "fast"

# Option 2: Historical evaluation mode
# forecast:
#   train_start_date: "2025-01-01"
#   train_end_date: "2025-03-31"
#   test_start_date: "2025-04-01"
#   test_end_date: "2025-06-30"
#   model_list: "fast"

# Technical analysis settings
technical_analysis:
  indicators: [sma, ema, rsi, macd, bbands, atr]
  sma_periods: [20, 50, 200]

# Output settings
output:
  format: "console"  # or "json"
  save_results: true
  results_dir: "./results"

# Logging settings
logging:
  enabled: true
  level: "INFO"
  log_to_file: false
```

## Data Configuration

Configure data fetching behavior:

```yaml
data:
  start_date: "2023-01-01"      # YYYY-MM-DD format
  end_date: "2024-01-01"        # null for today
  interval: "1d"                # 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo
  use_cache: true               # Use SQLite database caching
  db_path: "stockula.db"        # Database file path
```

### Supported Intervals

- Intraday: `1m`, `2m`, `5m`, `15m`, `30m`, `60m`, `90m`, `1h`
- Daily: `1d`, `5d`
- Weekly/Monthly: `1wk`, `1mo`, `3mo`

## Portfolio Configuration

### Basic Portfolio Setup

```yaml
portfolio:
  name: "My Portfolio"
  initial_capital: 100000
  allocation_method: equal_weight
  tickers:
    - symbol: AAPL
      quantity: 10
    - symbol: GOOGL
      quantity: 5
    - symbol: MSFT
      quantity: 8
```

### Advanced Portfolio Features

```yaml
portfolio:
  initial_capital: 100000
  allocation_method: custom
  dynamic_allocation: true
  allow_fractional_shares: true
  capital_utilization_target: 0.95
  rebalance_frequency: monthly
  max_position_size: 20.0      # Max 20% per position
  stop_loss_pct: 10.0          # Global stop loss

  tickers:
    - symbol: AAPL
      sector: Technology
      market_cap: 3000.0
      category: large_cap
      allocation_amount: 25000   # $25k allocation
    - symbol: NVDA
      category: momentum
      allocation_pct: 15.0       # 15% of portfolio
```

### Allocation Methods

| Method               | Description                                      |
| -------------------- | ------------------------------------------------ |
| `equal_weight`       | Equal allocation across all tickers              |
| `market_cap`         | Weight by market capitalization                  |
| `custom`             | Use specified allocation amounts/percentages     |
| `dynamic`            | Calculate quantities based on allocation targets |
| `auto`               | Automatic allocation based on categories         |
| `backtest_optimized` | Use backtest results to optimize allocation      |

### Portfolio Buckets

Organize assets into logical groups:

````yaml
portfolio:
  initial_capital: 100000
  buckets:
    - name: core_holdings
      description: "Long-term core positions"
      allocation_amount: 50000
      tickers:
        - symbol: SPY
          allocation_amount: 20000
        - symbol: QQQ
          allocation_amount: 15000
        - symbol: VTI
          allocation_amount: 15000

    - name: growth_stocks
      description: "High growth technology"
      allocation_amount: 30000
      tickers:
        - symbol: NVDA
        - symbol: AMD
        - symbol: GOOGL
        - symbol: META

### Backtest-Optimized Allocation

Use historical backtesting to determine optimal allocations:

```yaml
portfolio:
  initial_capital: 100000
  allocation_method: backtest_optimized
  tickers:
    - symbol: AAPL
      category: TECH
    - symbol: SPY
      category: INDEX
    - symbol: GLD
      category: COMMODITY

# Configure backtest optimization
backtest_optimization:
  train_start_date: "2023-01-01"
  train_end_date: "2023-12-31"
  test_start_date: "2024-01-01"
  test_end_date: "2024-06-30"
  ranking_metric: "Return [%]"      # Default: "Return [%]", can also use "Sharpe Ratio", etc.
  min_allocation_pct: 2.0
  max_allocation_pct: 25.0
  initial_allocation_pct: 2.0
````

See [Allocation Strategies](../user-guide/allocation-strategies.md) for detailed information.

````

## Backtest Configuration

### Basic Backtesting

```yaml
backtest:
  initial_cash: 10000.0
  commission: 0.002            # 0.2% commission (legacy)
  margin: 1.0                  # Margin requirement
  strategies:
    - name: smacross
      parameters:
        fast_period: 10
        slow_period: 20
````

### Broker Configuration

Use realistic broker fee structures:

```yaml
backtest:
  # Use a preset broker
  broker_config:
    name: "robinhood"  # Zero commission with TAF

  # Or custom configuration
  broker_config:
    name: "custom"
    commission_type: "percentage"
    commission_value: 0.001      # 0.1%
    min_commission: 1.0          # $1 minimum
    max_commission: 10.0         # $10 maximum
    regulatory_fees: 0.0000229   # SEC fee
    exchange_fees: 0.003         # Exchange fees
```

### Available Broker Presets

| Broker                | Commission   | Notes                            |
| --------------------- | ------------ | -------------------------------- |
| `robinhood`           | $0           | TAF on sells (waived ≤50 shares) |
| `interactive_brokers` | $0.005/share | $1 minimum, tiered pricing       |
| `td_ameritrade`       | $0           | SEC fees only                    |
| `etrade`              | $0           | SEC fees only                    |
| `fidelity`            | $0           | SEC fees only                    |
| `schwab`              | $0           | SEC fees only                    |

### Strategy Configuration

Multiple strategies can be tested simultaneously:

```yaml
backtest:
  strategies:
    - name: smacross
      parameters:
        fast_period: 10
        slow_period: 20

    - name: rsi
      parameters:
        period: 14
        oversold_threshold: 30
        overbought_threshold: 70

    - name: doubleemacross
      parameters:
        fast_period: 34
        slow_period: 55
        momentum_atr_multiple: 1.25
        speculative_atr_multiple: 1.0
```

See the [Strategies API Reference](../api/strategies.md) for all available strategies and parameters.

## Technical Analysis Configuration

Configure which indicators to calculate:

```yaml
technical_analysis:
  indicators: [sma, ema, rsi, macd, bbands, atr]
  sma_periods: [20, 50, 200]
  ema_periods: [12, 26]
  rsi_period: 14
  macd_params:
    period_fast: 12
    period_slow: 26
    signal: 9
  bbands_params:
    period: 20
    std: 2
  atr_period: 14
```

## Forecast Configuration

Stockula supports two mutually exclusive forecasting modes:

### Future Prediction Mode

Forecast N days into the future from today:

```yaml
forecast:
  forecast_length: 30           # Days to forecast from today
  frequency: "infer"            # Data frequency: 'D' (daily), 'W' (weekly), 'M' (monthly), or 'infer' to auto-detect
  prediction_interval: 0.95     # Confidence interval
  model_list: "fast"            # fast, default, slow, parallel
  ensemble: "auto"              # auto, simple, distance, horizontal
  max_generations: 5            # Model search iterations
  num_validations: 2            # Validation splits
  validation_method: "backwards" # backwards, seasonal, similarity
```

### Historical Evaluation Mode

Train on historical data and evaluate accuracy on a test period:

```yaml
forecast:
  # Train/test split for evaluation
  train_start_date: "2025-01-01"   # Training period start
  train_end_date: "2025-03-31"     # Training period end
  test_start_date: "2025-04-01"    # Test period start
  test_end_date: "2025-06-30"      # Test period end

  # Model configuration
  frequency: "infer"
  prediction_interval: 0.95
  model_list: "fast"
  ensemble: "auto"
  max_generations: 5
  num_validations: 2
  validation_method: "backwards"
```

**Important**: Do not specify both `forecast_length` and test dates. Choose one mode or the other. The system will validate this and raise an error if both are configured.

### Performance Optimization

For faster forecasting in either mode:

```yaml
forecast:
  # Choose your mode (forecast_length OR test dates)
  forecast_length: 14           # Shorter forecasts
  # OR use shorter test periods for evaluation

  model_list: "fast"            # Only fast models
  max_generations: 2            # Fewer iterations
  num_validations: 1            # Single validation
  ensemble: "simple"            # Simpler ensemble
```

## Logging Configuration

Control logging behavior:

```yaml
logging:
  enabled: true
  level: "INFO"                 # DEBUG, INFO, WARNING, ERROR
  show_allocation_details: true
  show_price_fetching: true
  log_to_file: false
  log_file: "stockula.log"
  max_log_size: 10485760        # 10MB
  backup_count: 3
```

## Output Configuration

Configure result output:

```yaml
output:
  format: "console"             # console or json
  save_results: true
  results_dir: "./results"
```

## Environment Variables

Override configuration with environment variables:

```bash
export STOCKULA_CONFIG_FILE=my_config.yaml
export STOCKULA_DEBUG=true
export STOCKULA_LOG_LEVEL=DEBUG
```

## Validation

Stockula uses Pydantic for configuration validation. Invalid configurations will show helpful error messages:

```bash
# Example validation error
ValidationError: 2 validation errors for StockulaConfig
portfolio.initial_capital
  ensure this value is greater than 0 (type=value_error.number.not_gt; limit_value=0)
backtest.commission
  ensure this value is less than or equal to 1 (type=value_error.number.not_le; limit_value=1)
```

## Example Configurations

Complete examples are available in the `examples/` directory:

- `config.simple.yaml` - Basic portfolio setup
- `config.full.yaml` - All features demonstrated
- `config.strategies.yaml` - Multiple strategy comparison
- `config.technical.yaml` - Advanced technical analysis
- `config.forecast.yaml` - Optimized forecasting

## Programmatic Configuration

Create configurations in Python:

```python
from stockula.config.models import StockulaConfig, PortfolioConfig, BacktestConfig
from stockula.config.settings import load_config, save_config

# Load from file
config = load_config("my_config.yaml")

# Create programmatically
config = StockulaConfig(
    portfolio=PortfolioConfig(
        initial_capital=50000,
        tickers=[
            {"symbol": "AAPL", "quantity": 10},
            {"symbol": "GOOGL", "quantity": 5}
        ]
    ),
    backtest=BacktestConfig(
        initial_cash=10000,
        strategies=[
            {"name": "smacross", "parameters": {"fast_period": 10, "slow_period": 20}}
        ]
    )
)

# Save to file
save_config(config, "output.yaml")
```
