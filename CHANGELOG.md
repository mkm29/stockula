# Changelog

## [Unreleased]

## [v0.1.0] - 2025-07-30

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
- **Logging Management**: New module-specific logging level configuration via `set_module_log_level()` method
- **SQLModel Integration**: Refactored database models to use SQLModel for improved type safety and validation
- **Detailed Reporting**: Enhanced backtest reporting with automatic JSON report saving
- **Configuration Improvements**:
  - Structured data models for backtest results
  - Enhanced broker configuration documentation
  - Better organization of example configurations in `examples/` directory
- **Forecast Evaluation**: Enhanced forecast mode with train/test split evaluation
  - Automatic accuracy calculation using RMSE, MAE, and MAPE metrics
  - Portfolio-level accuracy display showing weighted average performance
  - Train/test period configuration in data section of config file
  - Detailed evaluation metrics table showing model performance per ticker
  - Made `forecast_length` and test dates mutually exclusive
  - `forecast_length` now defaults to None instead of 14
  - Added validation to ensure only one forecast mode is active at a time
  - Improved CLI display to show appropriate mode (future prediction vs historical evaluation)
  - Updated documentation to clarify the two distinct forecast modes
- **Backtest Train/Test Split**: Added train/test split functionality for backtesting
  - Split historical data into training and testing periods
  - Parameter optimization on training data
  - Out-of-sample performance validation on test data
  - Performance degradation metrics between train and test periods
  - Enhanced output display showing both train and test results
- **Portfolio Holdings Display**: Added detailed portfolio holdings table showing tickers, types, and quantities
- **Auto-Allocation Algorithm**: Improved portfolio allocation strategy for balanced share distribution
  - Target-based allocation replacing greedy algorithm
  - Better capital utilization (near 100%)
  - Balanced position sizes across all holdings

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
- **Backtest Progress Bars**: Suppressed unwanted "Backtest.run: 0%|..." progress output by redirecting stderr during backtest execution
- **Mock Object Handling**: Fixed Mock object formatting errors in portfolio display
- **Zero Allocation Handling**: Fixed issue with tickers having 0% category allocation

### Build System

- Python 3.13+ support
- Package management with uv
- Test coverage configuration excluding framework-dependent code
- GitHub Actions ready structure
- Comprehensive .gitignore for Python projects
