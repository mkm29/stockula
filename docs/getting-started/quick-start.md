# Quick Start

Get up and running with Stockula in minutes. This guide covers basic usage and common workflows.

## Basic Usage

### Command Line Mode

The simplest way to use Stockula is with single ticker analysis:

```bash
# Analyze a single stock with default settings
uv run python -m stockula.main --ticker AAPL

# Run specific analysis modes
uv run python -m stockula.main --ticker GOOGL --mode ta        # Technical analysis only
uv run python -m stockula.main --ticker MSFT --mode backtest  # Backtesting only
uv run python -m stockula.main --ticker AMZN --mode forecast  # Forecasting only
uv run python -m stockula.main --ticker NVDA --mode all       # All analyses
```

### Configuration File Mode

For more complex scenarios, use configuration files:

```bash
# Use default configuration file (.config.yaml)
uv run python -m stockula.main

# Use a specific configuration file
uv run python -m stockula.main --config examples/config.full.yaml

# Copy an example to use as default
cp examples/config.simple.yaml .config.yaml
uv run python -m stockula.main
```

### Output Formats

```bash
# Default console output with Rich formatting
uv run python -m stockula.main --ticker AAPL

# JSON output for programmatic use
uv run python -m stockula.main --ticker AAPL --output json
```

## Common Workflows

### Portfolio Analysis

Create a `.config.yaml` file for portfolio analysis:

```yaml
data:
  start_date: "2023-01-01"
  end_date: null  # defaults to today

portfolio:
  initial_capital: 100000
  allocation_method: equal_weight
  tickers:
    - symbol: AAPL
      quantity: 10
    - symbol: GOOGL
      quantity: 5
    - symbol: MSFT
      quantity: 8

backtest:
  initial_cash: 10000.0
  strategies:
    - name: smacross
      parameters:
        fast_period: 10
        slow_period: 20
```

Run the analysis:

```bash
uv run python -m stockula.main --mode all
```

### Strategy Backtesting

Test different trading strategies:

```bash
# Test with default SMA crossover strategy
uv run python -m stockula.main --ticker AAPL --mode backtest

# Use custom configuration with multiple strategies
uv run python -m stockula.main --config examples/config.strategies.yaml --mode backtest
```

### Technical Analysis

Compute technical indicators:

```bash
# Run technical analysis on a stock
uv run python -m stockula.main --ticker TSLA --mode ta

# With custom indicator settings
uv run python -m stockula.main --config examples/config.technical.yaml --mode ta
```

### Price Forecasting

Generate price predictions:

```bash
# Basic forecasting
uv run python -m stockula.main --ticker NVDA --mode forecast

# With custom forecast settings (faster execution)
uv run python -m stockula.main --config examples/config.forecast.yaml --mode forecast
```

## Understanding the Output

### Rich CLI Features

Stockula provides beautiful, colored output using the Rich library:

#### Progress Bars

Real-time progress tracking for long-running operations:

```
⠋ Backtesting SMACROSS on AAPL... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 85% 0:00:02
```

#### Results Tables

Clean, formatted tables for results:

##### Technical Analysis

```
                    Technical Analysis Results
┏━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Ticker ┃ SMA_20        ┃ RSI_14         ┃ MACD             ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ AAPL   │ $150.25       │ 65.4           │ 2.15             │
└────────┴───────────────┴────────────────┴──────────────────┘
```

##### Backtesting Results

Backtesting output starts with portfolio information:

```
=== Backtesting Results ===

Portfolio Information:
  Initial Capital: $10,000
  Start Date: 2023-01-01
  End Date: 2023-12-31
  Trading Days: 252
  Calendar Days: 365

                         Backtesting Results
┏━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Ticker ┃ Strategy  ┃ Return     ┃ Sharpe Ratio   ┃ Max Drawdown   ┃
┡━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ AAPL   │ SMACROSS  │ +15.50%    │ 1.25           │ -8.30%         │
└────────┴───────────┴────────────┴────────────────┴────────────────┘
```

#### Strategy Summaries

Detailed panels for backtest results:

```
╭───────────────────────────── STRATEGY: SMACROSS ─────────────────────────────╮
│  Parameters: {'fast_period': 10, 'slow_period': 20}                          │
│  Portfolio Value at Start Date: $10,000.00                                   │
│  Portfolio Value at End (Backtest): $12,345.67                               │
│  Strategy Performance:                                                       │
│    Average Return: +23.46%                                                   │
│    Winning Stocks: 8                                                         │
│    Losing Stocks: 2                                                          │
│    Total Trades: 45                                                          │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Example Configurations

Stockula includes several example configurations in the `examples/` directory:

- `config.simple.yaml` - Basic portfolio with a few stocks
- `config.full.yaml` - Complete configuration with all features
- `config.strategies.yaml` - Multiple trading strategies comparison
- `config.technical.yaml` - Advanced technical analysis settings
- `config.forecast.yaml` - Optimized forecasting configuration

Copy any example to `.config.yaml` to use as your default:

```bash
cp examples/config.simple.yaml .config.yaml
uv run python -m stockula.main
```

## Database Caching

Stockula automatically caches all market data in a SQLite database:

```bash
# View database statistics
uv run python -m stockula.database.cli stats

# Manually fetch and cache data
uv run python -m stockula.database.cli fetch AAPL MSFT GOOGL

# Query cached data
uv run python -m stockula.database.cli query AAPL
```

## Next Steps

Now that you're familiar with basic usage:

- [Configuration Guide](configuration.md) - Learn about all configuration options
- [Architecture Overview](../user-guide/architecture.md) - Understand how Stockula works
- [Backtesting Guide](../user-guide/backtesting.md) - Deep dive into strategy testing
- [API Reference](../api/strategies.md) - Explore programmatic usage
