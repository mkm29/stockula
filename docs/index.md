# Stockula

A comprehensive Python trading platform that provides tools for technical analysis, backtesting, data fetching, and price forecasting. Built with modern Python practices, it integrates popular financial libraries to offer a complete solution for quantitative trading strategy development.

## Features

- **📊 Technical Analysis**: Calculate popular indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.) using the finta library
- **🔄 Backtesting**: Test trading strategies on historical data with detailed performance metrics
- **📈 Data Fetching**: Retrieve real-time and historical market data via yfinance
- **🗄️ SQLite Database**: Automatic caching of all yfinance data with robust lookup capabilities
- **🔮 Price Forecasting**: Automated time series forecasting using AutoTS
- **📝 Centralized Logging**: Professional logging system with configurable levels and file rotation
- **🚀 Fast Package Management**: Uses uv for lightning-fast dependency management
- **🎨 Rich CLI Interface**: Beautiful progress bars, tables, and colored output using Rich library

## Requirements

### System Requirements

- **Python**: 3.13 or higher
- **Operating System**: macOS, Linux, or Windows
- **Memory**: Minimum 8GB RAM recommended
- **Storage**: 100MB free space (more if caching extensive historical data)

### Key Dependencies

- **pandas**: Data manipulation and analysis
- **yfinance**: Yahoo Finance data fetching
- **finta**: Financial technical analysis indicators
- **backtesting**: Strategy backtesting framework
- **autots**: Automated time series forecasting
- **rich**: Enhanced CLI formatting with progress bars, tables, and colors

## Quick Start

1. **Install uv** (if not already installed):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

1. **Clone and install**:

   ```bash
   git clone https://github.com/mkm29/stockula.git
   cd stockula
   uv sync
   ```

1. **Run analysis**:

   ```bash
   # Simple ticker analysis
   uv run python -m stockula.main --ticker AAPL

   # Use configuration file
   uv run python -m stockula.main --config examples/config.simple.yaml

   # Run specific modes
   uv run python -m stockula.main --ticker GOOGL --mode backtest
   uv run python -m stockula.main --ticker MSFT --mode forecast
   ```

## Next Steps

- [Installation Guide](getting-started/installation.md) - Detailed setup instructions
- [Configuration](getting-started/configuration.md) - Learn about configuration options
- [Architecture Overview](user-guide/architecture.md) - Understand how Stockula works
- [Backtesting Guide](user-guide/backtesting.md) - Start testing trading strategies
- [API Reference](api/strategies.md) - Explore available strategies and APIs

## License

MIT License - see LICENSE file for details
