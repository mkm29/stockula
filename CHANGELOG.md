# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Forecast Evaluation**: Enhanced forecast mode with train/test split evaluation
  - Automatic accuracy calculation using RMSE, MAE, and MAPE metrics
  - Portfolio-level accuracy display showing weighted average performance
  - Train/test period configuration in data section of config file
  - Detailed evaluation metrics table showing model performance per ticker

### Changed

- **Forecast Output**: Improved forecast mode display to show:
  - Portfolio value at test start and end dates
  - Overall portfolio accuracy percentage
  - Individual ticker evaluation metrics when train/test dates are configured

## [0.2.0] - 2025-01-29

### Added

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

### Changed

- **Database Models**: Migrated from SQLAlchemy declarative base to SQLModel for better type hints
- **Test Infrastructure**: 
  - Improved test database fixtures and isolation
  - Enhanced test coverage for trading strategies
  - Fixed test pollution issues in main test suite
- **Portfolio Configuration**: Updated default capital utilization to 100%
- **Logging Output**: Improved formatting and clarity of log messages

### Fixed

- **Test Suite**: Resolved test isolation issues where tests passed individually but failed in full suite
- **Type Hints**: Fixed type hint issues in logging manager and database models
- **Strategy Display**: Corrected display of strategy-specific results in portfolio summary

### Removed

- Obsolete test files and configurations
- Outdated unit tests for strategy calculations
- Test database file from version control
- Duplicate strategy implementations (cleaned up in analysis)

### Internal

- Improved code organization and module structure
- Enhanced test coverage details in documentation
- Consolidated redundant strategy validation checks

## [0.1.0] - 2025-01-28

### Added

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

### Infrastructure

- Python 3.13+ support
- Package management with uv
- Test coverage configuration excluding framework-dependent code
- GitHub Actions ready structure
- Comprehensive .gitignore for Python projects

[0.2.0]: https://github.com/mkm29/stockula/releases/tag/v0.2.0
[0.1.0]: https://github.com/mkm29/stockula/releases/tag/v0.1.0
[unreleased]: https://github.com/mkm29/stockula/compare/v0.2.0...HEAD
