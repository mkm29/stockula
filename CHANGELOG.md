# Changelog

## [0.2.0](https://github.com/mkm29/stockula/compare/stockula-v0.1.0...stockula-v0.2.0) (2025-07-31)


### Features

* Add initial versioning configuration and changelog documentation ([b134945](https://github.com/mkm29/stockula/commit/b134945a74a48d472e7859ef9a5879ff332a89c8))
* add pytest-xdist for parallel test execution and improve test d… ([e07846a](https://github.com/mkm29/stockula/commit/e07846a4a5ece26ad576fa5d81e2119bf4d059c7))
* add pytest-xdist for parallel test execution and improve test database handling ([006a288](https://github.com/mkm29/stockula/commit/006a288ebe50754418395f720a6cec4c7a1e1cbb))
* **config:** Add release configuration and manifest for version 0.2.… ([08486a4](https://github.com/mkm29/stockula/commit/08486a40bc9cbf5e593a56aa6f8bee8e89afcd7f))
* **config:** Add release configuration and manifest for version 0.2.0; update .gitignore to exclude JSON files ([7a21aff](https://github.com/mkm29/stockula/commit/7a21affad0efb0e4347452821f09c494265c709a))
* **docs:** Add development documentation for AutoTS threading considerations ([785597f](https://github.com/mkm29/stockula/commit/785597ffab151fd92d5ffb212424685b0c9151c4))
* **docs:** Update changelog and user guide for train/test evaluation enhancements ([785597f](https://github.com/mkm29/stockula/commit/785597ffab151fd92d5ffb212424685b0c9151c4))
* **docs:** Update changelog and user guide for train/test evaluation… ([82c2fbb](https://github.com/mkm29/stockula/commit/82c2fbbce7eb0ae10863ed3348e5125cd2c3f60e))
* **docs:** Update changelog for version 0.2.0; enhance forecast evaluation and backtest functionality ([e061515](https://github.com/mkm29/stockula/commit/e061515b033024d0ee2c3ac9e609a035b00b09eb))
* **domain:** Exclude tickers with 0% allocation from portfolio creation and tests ([ac987dc](https://github.com/mkm29/stockula/commit/ac987dc3dccd2ae83c5b9f6552dbca0e92a4af8e))
* Enhance main.py with portfolio summary and initial value calculations ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
* **forecasting:** Implement train/test split evaluation for stock price forecasting ([ce74bd6](https://github.com/mkm29/stockula/commit/ce74bd64b610ac2e6b8dc1407f2dcc8864a77e24))
* Implement Ticker domain model with singleton registry ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
* Initialize SQLite database for stock data storage ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
* **logging:** Add method to set module-specific logging levels ([8b95356](https://github.com/mkm29/stockula/commit/8b953561b30c73f58bd0d270e7b887f688a762e0))
* **logging:** Add method to set module-specific logging levels ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
* **logging:** Update LoggingManager to handle additional third-party libraries ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
* **main:** Add detailed portfolio holdings display in console output ([5bbf0da](https://github.com/mkm29/stockula/commit/5bbf0da294022211d783a2bb8c5113374f15a47b))
* **main:** Enhance forecasting output with return percentage and improve logging ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
* **main:** Enhance portfolio value display with forecasted values and accuracy metrics ([785597f](https://github.com/mkm29/stockula/commit/785597ffab151fd92d5ffb212424685b0c9151c4))


### Bug Fixes

* **backtesting:** Suppress progress output in backtest runner ([785597f](https://github.com/mkm29/stockula/commit/785597ffab151fd92d5ffb212424685b0c9151c4))
* Correct import paths in main.py ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
* Display correct strategy-specific backtest results in portfolio summary ([6bd5a97](https://github.com/mkm29/stockula/commit/6bd5a9786e6662448d73229c387fe2c38ed1b788))
* **docs:** Update documentation links to use relative paths ([a5079ba](https://github.com/mkm29/stockula/commit/a5079bad4b6aaec35a8b19b3d8918ae0dd5e3686))
* update .gitignore and CHANGELOG for CLAUDE file renaming ([324c7e6](https://github.com/mkm29/stockula/commit/324c7e615841c48c713a03085515b7da490d79d6))
* update .gitignore and CHANGELOG for CLAUDE file renaming ([4fd14b1](https://github.com/mkm29/stockula/commit/4fd14b1cf4dde2500690d22ddb91fd8af0edc907))


### Documentation

* Add broker configuration documentation to README ([e474458](https://github.com/mkm29/stockula/commit/e47445876c2437c3fe5cea99054d587e308ea36b))
* Update README to include system requirements, dependencies, architecture overview, and data flow ([505b797](https://github.com/mkm29/stockula/commit/505b7973878f7e6fd541e12c79f02b9b61074aea))
* Update stockula.yml and stockula.yml.example with new structure ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
* Update storage requirement in README for data caching ([da020fa](https://github.com/mkm29/stockula/commit/da020fab7edd3e1de4a962f2620f85b4cf26f2a5))
* Update user guide and README to reflect sorting by return in analysis modes ([5bbf0da](https://github.com/mkm29/stockula/commit/5bbf0da294022211d783a2bb8c5113374f15a47b))


### Code Refactoring

* Create structured data models for backtest results ([d8b5fe2](https://github.com/mkm29/stockula/commit/d8b5fe2ff3a1ce63265fdf8afb5132e83a16adcf))
* Organize configuration files into examples directory ([ecc4c0b](https://github.com/mkm29/stockula/commit/ecc4c0bd62b57f47cbbe4fc5887f341cd12006ae))
* Update forecaster.py to streamline imports ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))


### Tests

* **autots:** Create direct test for AutoTS to isolate hanging issues ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
* **config:** Add minimal test configuration for two tickers ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
* **config:** Add quick test configuration with a single ticker ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
* **config:** Add simple test configuration for forecast testing ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
* **config:** Add test configuration for parallel progress tracking ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
* **parallel:** Implement tests for parallel forecasting to debug hanging issues ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
* **progress:** Create test script to demonstrate parallel forecasting progress tracking ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
* **sequential:** Implement sequential forecasting test to verify functionality ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
* **single:** Create test for single ticker forecasting to debug hanging issues ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
* **unit:** Refactor main tests to use parallel forecasting and improve logging checks ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))
* **unit:** Update unit tests for forecaster to check output suppression ([bb7d4f7](https://github.com/mkm29/stockula/commit/bb7d4f75a93ccf009b484a1cba2da907c87e9cfe))


### Miscellaneous Chores

* Add CLAUDE.market_data and .claude to .gitignore ([0cf0065](https://github.com/mkm29/stockula/commit/0cf006507c51b49bbd8b66b165c33e28434ccb81))
* Add example configuration files for different allocation strategies ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
* Add SQLite database initialization and testing script ([0699081](https://github.com/mkm29/stockula/commit/069908175957c2e944a898dd2edc27129f84d892))
* comment out pull_request branch triggers in workflow files ([bc9ee6f](https://github.com/mkm29/stockula/commit/bc9ee6f54cc4c1ccdcd6826345b73cafdfa6a77f))
* copy in README.md in Dockerfile ([f6a8db0](https://github.com/mkm29/stockula/commit/f6a8db0f69652d56e8e562db6c44d8ae266a93cf))
* fixed pip install command in  Dockerfile ([4765432](https://github.com/mkm29/stockula/commit/4765432a8cde9a789cbb16b8ea72fd2a42d408fc))
* **main:** release stockula 0.3.0 ([45922f8](https://github.com/mkm29/stockula/commit/45922f84f7ca86183f8c6b1cb60d21913f3135f3))
* remove branch triggers for main in Docker and Release workflows ([c258b4e](https://github.com/mkm29/stockula/commit/c258b4ef1a8e41eaa8fb95d4fc7f40f8613be188))
* remove branch triggers for main in Docker and Release workflows ([7074d82](https://github.com/mkm29/stockula/commit/7074d82544f45156970e02dcc6dc7d5cf460283e))
* reset version to v0.1.0 for initial release ([036fa12](https://github.com/mkm29/stockula/commit/036fa12646f68a386c82b38dff7ec28766d344cd))
* Update .gitignore to include AutoTS cache and sqlite database files ([6117bd3](https://github.com/mkm29/stockula/commit/6117bd30f6d55fcf62b798eb2cf9ded29300cfb3))
* update changelog with test suite fixes and improvements; modif… ([cf734ea](https://github.com/mkm29/stockula/commit/cf734ea913c9819b78da6b5b2dcb6e1b4bedf44a))
* update changelog with test suite fixes and improvements; modify Dockerfile version; enhance testing documentation; add backward compatibility methods in DatabaseManager ([8555fc6](https://github.com/mkm29/stockula/commit/8555fc60e54affe6956cf18654b2b0f41352f315))
* updated .dockerignore in order to build uv project ([a4eb5da](https://github.com/mkm29/stockula/commit/a4eb5daf64c9bdcae3529f9864ac7adf35e96214))
* updated docker-build job to add build time labels ([71ecc8b](https://github.com/mkm29/stockula/commit/71ecc8b488e6b79b7550738894cf9826b0e4d90b))

## [Unreleased]

### Bug Fixes

- **Test Suite Fixes**: Fixed multiple integration test failures
  - Added backward compatibility methods to DatabaseManager (`add_stock_info`, `add_dividends`, `add_splits`, `add_options_data`)
  - Fixed dependency injection issues in tests by properly wiring the container with `container.wire(modules=["stockula.main"])`
  - Updated test assertions to match Rich table formatting output
  - Fixed `StockForecaster` test to expect correct default frequency value ('infer' instead of 'D')
  - Updated foreign key constraint test to reflect actual behavior where stocks are auto-created
  - Fixed mock object setup for technical indicators to properly support subscriptable operations
  - Added missing strategy configuration to backtest tests

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
