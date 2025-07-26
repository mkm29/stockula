# Stockula

Stockula is a comprehensive Python trading platform that provides tools for technical analysis, backtesting, data fetching, and price forecasting. Built with modern Python practices, it integrates popular financial libraries to offer a complete solution for quantitative trading strategy development.

## Features

- **📊 Technical Analysis**: Calculate popular indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.) using the finta library
- **🔄 Backtesting**: Test trading strategies on historical data with detailed performance metrics
- **📈 Data Fetching**: Retrieve real-time and historical market data via yfinance
- **🔮 Price Forecasting**: Automated time series forecasting using AutoTS
- **🚀 Fast Package Management**: Uses uv for lightning-fast dependency management

## Installation

1. Install `uv` (if not already installed):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

1. Clone the repository:

   ```bash
   git clone https://github.com/mkm29/stockula.git
   cd stockula
   ```

1. Install dependencies:

   ```bash
   uv sync
   ```

## Quick Start

Run the demo to see all features in action:

```bash
uv run python -m stockula.main --symbol AAPL --demo all
```

Or run specific demos:

```bash
# Data fetching demo
uv run python -m stockula.main --symbol TSLA --demo data

# Technical analysis demo
uv run python -m stockula.main --symbol MSFT --demo ta

# Backtesting demo
uv run python -m stockula.main --symbol GOOGL --demo backtest

# Forecasting demo
uv run python -m stockula.main --symbol AMZN --demo forecast
```

## Usage Examples

### Data Fetching

```python
from stockula import DataFetcher

# Initialize fetcher
fetcher = DataFetcher()

# Get historical data
data = fetcher.get_stock_data("AAPL", start="2023-01-01", end="2024-01-01")

# Get real-time price
current_price = fetcher.get_realtime_price("AAPL")

# Get company info
info = fetcher.get_info("AAPL")
```

### Technical Analysis

```python
from stockula import DataFetcher, TechnicalIndicators

# Fetch data
fetcher = DataFetcher()
data = fetcher.get_stock_data("AAPL")

# Calculate indicators
ta = TechnicalIndicators(data)

# Get various indicators
sma_20 = ta.sma(period=20)
ema_20 = ta.ema(period=20)
rsi_14 = ta.rsi(period=14)
macd = ta.macd()
bbands = ta.bbands()
```

### Backtesting

```python
from stockula import BacktestRunner, SMACrossStrategy, RSIStrategy

# Initialize backtest runner
runner = BacktestRunner(cash=10000, commission=0.002)

# Run backtest with SMA crossover strategy
results = runner.run_from_symbol("AAPL", SMACrossStrategy)

# Optimize strategy parameters
optimal_params = runner.optimize(
    data,
    SMACrossStrategy,
    fast_period=range(5, 20),
    slow_period=range(20, 50)
)
```

### Price Forecasting

```python
from stockula import StockForecaster

# Initialize forecaster
forecaster = StockForecaster(forecast_length=30)

# Forecast stock prices
predictions = forecaster.forecast_from_symbol("AAPL")

# Get forecast with confidence intervals
print(f"30-day forecast: ${predictions['forecast'].iloc[-1]:.2f}")
print(f"Confidence interval: ${predictions['lower_bound'].iloc[-1]:.2f} - ${predictions['upper_bound'].iloc[-1]:.2f}")
```

## Module Structure

```
src/stockula/
├── __init__.py           # Main package exports
├── main.py               # CLI demo application
├── data/                 # Data fetching module
│   ├── __init__.py
│   └── fetcher.py       # yfinance wrapper
├── technical_analysis/   # Technical indicators
│   ├── __init__.py
│   └── indicators.py    # finta wrapper
├── backtesting/         # Strategy backtesting
│   ├── __init__.py
│   ├── strategies.py    # Pre-built strategies
│   └── runner.py        # Backtest execution
└── forecasting/         # Price prediction
    ├── __init__.py
    └── forecaster.py    # AutoTS wrapper
```

## Pre-built Strategies

Stockula includes several ready-to-use trading strategies:

- **SMACrossStrategy**: Simple Moving Average crossover strategy
- **RSIStrategy**: Relative Strength Index based strategy
- **MACDStrategy**: MACD (Moving Average Convergence Divergence) strategy

## Development

### Running Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run ruff format
```

### Linting

```bash
uv run ruff check
```

## Dependencies

- **pandas**: Data manipulation and analysis
- **yfinance**: Yahoo Finance data fetching
- **finta**: Financial technical analysis indicators
- **backtesting**: Strategy backtesting framework
- **autots**: Automated time series forecasting
- **matplotlib**: Plotting and visualization

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
