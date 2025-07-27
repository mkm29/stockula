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
