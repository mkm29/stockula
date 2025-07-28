# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[unreleased]: https://github.com/mkm29/stockula/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mkm29/stockula/releases/tag/v0.1.0