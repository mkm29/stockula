# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Stockula is a Python library for trading strategies and backtesting. It provides a framework for implementing and testing trading strategies using historical data. The project uses Python 3.13+ and is managed with the `uv` package manager.

## Essential Commands

### Setup and Dependencies

- **Install uv**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Initialize project**: `uv init`
- **Install dependencies**: `uv sync`
- **Add new dependency**: `uv add <package>`
- **Add dev dependency**: `uv add --dev <package>`

### Development Commands

- **Run tests**: `uv run pytest`
- **Run linter**: `uv run ruff check`
- **Fix linting issues**: `uv run ruff check --fix`
- **Format code**: `uv run ruff format`

### Running the Application

- **Run main module**: `uv run python -m stockula.main`
- **Start Jupyter Lab**: `uv run jupyter lab`

## Architecture and Key Patterns

### Project Structure

```
stockula/
├── src/stockula/        # Main package directory
│   ├── __init__.py      # Package initialization
│   └── main.py          # Entry point (currently empty)
├── pyproject.toml       # Project configuration and dependencies
├── uv.lock              # Locked dependency versions
└── README.md            # Basic setup instructions
```

### Key Dependencies

The project includes several specialized financial and ML libraries:

- **Trading & Backtesting**: `alpaca-py` (trading API), `backtesting` (strategy testing)
- **Time Series Forecasting**: `prophet` (Facebook's forecasting), `autots` (automated TS), `darts` (deep learning TS)
- **Technical Analysis**: `finta` (financial indicators), `mplfinance` (financial plotting)
- **Data Analysis**: `pandas`, `matplotlib`, `pyql` (quantitative finance)
- **Development**: `jupyter`/`jupyterlab` (interactive development)

### Development Workflow

1. The project uses `uv` for fast, reliable Python package management
1. Development dependencies (`ruff`, `pytest`) are specified in `[project.optional-dependencies.dev]`
1. The package is built using `hatchling` as specified in `[build-system]`
1. Source code is packaged from `src/stockula` as defined in `[tool.hatch.build.targets.wheel]`

### Testing Strategies

When implementing trading strategies:

- Use `backtesting` library for historical data testing
- Leverage Jupyter notebooks for interactive strategy development
- Test with various market conditions using the forecasting libraries

### Code Quality

- **Linting**: The project uses `ruff` for fast Python linting
- **Testing**: `pytest` is available for unit and integration tests
- Run both before committing: `uv run ruff check && uv run pytest`

______________________________________________________________________

## AI Development Team Configuration

*Updated by team-configurator on 2025-07-29*

Your project uses: Python 3.13+, SQLAlchemy, Prophet/AutoTS/Darts (ML), Alpaca API, Rich CLI

### Specialist Assignments

#### Financial & ML Development

- **Trading Strategy Development** → @backend-developer

  - Implement trading algorithms and strategies
  - Work with backtesting framework and indicators
  - Optimize strategy performance and metrics

- **ML/Forecasting Models** → @backend-developer + @performance-optimizer

  - Time series forecasting with Prophet, AutoTS, Darts
  - Model training, validation, and optimization
  - Feature engineering for financial data

#### API & Data Management

- **API Architecture** → @api-architect

  - Design RESTful endpoints for trading operations
  - Alpaca API integration and data fetching
  - Real-time data streaming architecture

- **Database Operations** → @backend-developer

  - SQLAlchemy/SQLModel schema design
  - Alembic migrations and database optimization
  - Efficient time series data storage

#### Core Development

- **Backend Logic** → @backend-developer

  - Domain-driven design implementation
  - Dependency injection with container pattern
  - Service layer architecture and business logic

- **Data Processing** → @backend-developer + @performance-optimizer

  - Pandas data manipulation and analysis
  - Technical indicators calculation (FinTA)
  - Efficient data pipelines for backtesting

#### Quality & Documentation

- **Code Reviews** → @code-reviewer

  - Python best practices and patterns
  - Type hints and Pydantic validation
  - Performance and security considerations

- **Performance Optimization** → @performance-optimizer

  - Backtest execution speed optimization
  - ML model inference optimization
  - Database query performance tuning

- **Documentation** → @documentation-specialist

  - API documentation and examples
  - Strategy development guides
  - ML model documentation

### How to Use Your Team

**For Trading Strategies:**

```
"Implement a mean reversion strategy using Bollinger Bands"
"Optimize the backtesting runner for parallel execution"
```

**For ML/Forecasting:**

```
"Add LSTM forecasting using Darts library"
"Implement feature engineering for Prophet models"
```

**For API Development:**

```
"Design REST endpoints for portfolio management"
"Integrate real-time data streaming from Alpaca"
```

**For Data Management:**

```
"Optimize database schema for time series data"
"Implement efficient data caching for backtests"
```

**For Code Quality:**

```
"Review the forecasting module for best practices"
"Optimize the technical indicators calculations"
```

### Project-Specific Recommendations

1. **Testing Strategy**: Use pytest with fixtures for market data mocking
1. **Performance**: Profile backtesting loops and pandas operations regularly
1. **Architecture**: Maintain clean separation between domain, data, and presentation layers
1. **ML Models**: Version control model artifacts and track experiment metrics
1. **Financial Data**: Implement proper handling of market holidays and trading hours

Your specialized AI team is configured for quantitative finance and ML development!
