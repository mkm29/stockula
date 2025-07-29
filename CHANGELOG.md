# Changelog

## [Unreleased]

### Features

- **Forecast Evaluation**: Enhanced forecast mode with train/test split evaluation

  - Automatic accuracy calculation using RMSE, MAE, and MAPE metrics
  - Portfolio-level accuracy display showing weighted average performance
  - Train/test period configuration in data section of config file
  - Detailed evaluation metrics table showing model performance per ticker

- **Backtest Train/Test Split**: Added train/test split functionality for backtesting

  - Split historical data into training and testing periods
  - Parameter optimization on training data
  - Out-of-sample performance validation on test data
  - Performance degradation metrics between train and test periods
  - Enhanced output display showing both train and test results

### Bug Fixes

- **Backtest Progress Bars**: Suppressed unwanted "Backtest.run: 0%|..." progress output by redirecting stderr during backtest execution

## [0.2.0](https://github.com/mkm29/stockula/compare/v0.1.0...v0.2.0) (2025-01-29)

### Features

- **Logging Management**: New module-specific logging level configuration via `set_module_log_level()` method
- **SQLModel Integration**: Refactored database models to use SQLModel for improved type safety and validation
- **Detailed Reporting**: Enhanced backtest reporting with automatic JSON report saving
- **Strategy Enhancements**:
  - VAMA (Volume Adjusted Moving Average) strategy
  - Kaufman Efficiency strategy
  - Improved strategy configuration with date-based allocation
- **Configuration Improvements**:
  - Structured data models for backtest results
  - Enhanced broker configuration documentation
  - Better organization of example configurations in `examples/` directory
- **Forecast Mode Enhancement**: Made `forecast_length` and test dates mutually exclusive
  - `forecast_length` now defaults to None instead of 14
  - Added validation to ensure only one forecast mode is active at a time
  - Improved CLI display to show appropriate mode (future prediction vs historical evaluation)
  - Updated documentation to clarify the two distinct forecast modes

### Bug Fixes

- **Test Suite**: Resolved test isolation issues where tests passed individually but failed in full suite
- **Type Hints**: Fixed type hint issues in logging manager and database models
- **Strategy Display**: Corrected display of strategy-specific results in portfolio summary
- **Alembic Migrations**: Fixed multiple head revisions warning by properly linking migration chain
- **Forecast Display**:
  - Fixed portfolio value table to show actual dates instead of "Initial"
  - Updated labels to "Observed Value" and "Predicted Value" for clarity
  - Fixed predicted value calculation for future prediction mode
- **Test Suite**: Fixed failing test in `test_main_forecast_mode_with_warning` by adding missing quantity attribute to mock asset
- **Forecast Configuration**:
  - Fixed missing frequency parameter in container's StockForecaster factory
  - Removed Motif models from FINANCIAL_MODEL_LIST to avoid "k too large" warnings with small datasets
  - Improved frequency handling by defaulting to 'D' and auto-inferring from data to reduce AutoTS warnings

### Code Refactoring

- **Database Models**: Migrated from SQLAlchemy declarative base to SQLModel for better type hints
- **Test Infrastructure**: Improved test database fixtures and isolation
- **Code Organization**: Improved module structure and consolidated redundant strategy validation checks

### Miscellaneous Chores

- Removed obsolete test files and configurations
- Removed outdated unit tests for strategy calculations
- Removed test database file from version control
- Removed duplicate strategy implementations

## [0.1.0](https://github.com/mkm29/stockula/releases/tag/v0.1.0) (2025-01-28)

### Features

- Initial release of Stockula trading strategy library
- Core backtesting framework with 12 trading strategies:
  - Simple Moving Average (SMA) Crossover
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Double Exponential Moving Average (EMA) Crossover
  - Triple Exponential Moving Average (TEMA) Crossover
  - Triangular Moving Average (TRIMA) Crossover
  - Variable Index Dynamic Average (VIDYA)
  - Kaufman's Adaptive Moving Average (KAMA)
  - Fractal Adaptive Moving Average (FRAMA)
  - Volume Adjusted Moving Average (VAMA)
  - Kaufman Efficiency Strategy
- Comprehensive technical indicators library with 98% test coverage
- Data fetching capabilities from multiple sources (Yahoo Finance, Alpaca)
- Database support for storing historical data and backtest results
- Configuration management system with YAML support
- Portfolio management and analysis tools
- Time series forecasting integration (Prophet, AutoTS, Darts)
- CLI interface for running backtests and managing data
- Comprehensive test suite with 493 tests
- Documentation for testing strategy and development practices

### Build System

- Python 3.13+ support
- Package management with uv
- Test coverage configuration excluding framework-dependent code
- GitHub Actions ready structure
- Comprehensive .gitignore for Python projects
