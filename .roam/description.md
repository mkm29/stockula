# Project Architecture

## Project Overview

- **Files:** 214
- **Symbols:** 2692
- **Edges:** 1221
- **Languages:** python (120), markdown (39), yaml (29), bash (7), json (3), toml (1)

## Directory Structure

| Directory    | Files | Primary Language |
| ------------ | ----- | ---------------- |
| `src/`       | 55    | python           |
| `tests/`     | 43    | python           |
| `docs/`      | 34    | markdown         |
| `./`         | 27    | yaml             |
| `examples/`  | 20    | yaml             |
| `utils/`     | 9     | python           |
| `.github/`   | 9     | yaml             |
| `scripts/`   | 7     | bash             |
| `alembic/`   | 7     | python           |
| `notebooks/` | 2     | markdown         |
| `config/`    | 1     | yaml             |

## Entry Points

- `src/stockula/__init__.py`
- `src/stockula/__main__.py`
- `src/stockula/allocation/__init__.py`
- `src/stockula/backtesting/__init__.py`
- `src/stockula/config/__init__.py`
- `src/stockula/data/__init__.py`
- `src/stockula/database/__init__.py`
- `src/stockula/domain/__init__.py`
- `src/stockula/forecasting/__init__.py`
- `src/stockula/forecasting/backends/__init__.py`
- `src/stockula/main.py`
- `src/stockula/technical_analysis/__init__.py`
- `src/stockula/utils/__init__.py`
- `tests/__init__.py`
- `tests/integration/__init__.py`
- `tests/unit/__init__.py`
- `utils/__init__.py`
- `examples/automatic_dynamic_rates_example.py`
- `examples/backtest_optimized_allocation_example.py`
- `examples/dynamic_sharpe_example.py`
- `examples/forecast_chronos.py`
- `examples/pipeline_example.py`
- `examples/treasury_rate_example.py`
- `scripts/export_to_gluonts_file_dataset.py`
- `tests/data/fetch_test_data.py`

## Key Abstractions

Top symbols by importance (PageRank):

| Symbol                                                                       | Kind     | Location                                           |
| ---------------------------------------------------------------------------- | -------- | -------------------------------------------------- |
| `print def print(self, *args, **kwargs)`                                     | method   | `src/stockula/cli_manager.py:83`                   |
| `StockulaConfig class StockulaConfig(BaseModel)`                             | class    | `src/stockula/config/models.py:666`                |
| `exception def exception(self, message: str) -> None`                        | method   | `src/stockula/utils/logging_manager.py:168`        |
| `BacktestOptimizedAllocator class BacktestOptimizedAllocator(BaseAllocator)` | class    | `src/stockula/allocation/backtest_allocator.py:29` |
| `TickerConfig class TickerConfig(BaseModel)`                                 | class    | `src/stockula/config/models.py:51`                 |
| `DatabaseManager class DatabaseManager`                                      | class    | `src/stockula/database/manager.py:21`              |
| `Container class Container(containers.DeclarativeContainer)`                 | class    | `src/stockula/container.py:19`                     |
| `StockulaManager class StockulaManager`                                      | class    | `src/stockula/manager.py:22`                       |
| `BaseStrategy class BaseStrategy(Strategy)`                                  | class    | `src/stockula/backtesting/strategies.py:22`        |
| `StrategyRepository class StrategyRepository(Repository[type[BaseSt...`      | class    | `src/stockula/data/strategy_repository.py:26`      |
| `Stock class Stock(SQLModel, table=True)`                                    | class    | `src/stockula/database/models.py:17`               |
| `ILoggingManager class ILoggingManager(ABC)`                                 | class    | `src/stockula/interfaces.py:75`                    |
| `BacktestRunner class BacktestRunner`                                        | class    | `src/stockula/backtesting/runner.py:15`            |
| `TestStrategyRegistry class TestStrategyRegistry`                            | class    | `tests/unit/test_backtesting_registry.py:16`       |
| \`mock_config @pytest.fixture                                                |          |                                                    |
| def mock_config()\`                                                          | function | `tests/unit/test_main.py:20`                       |

## Architecture

- **Dependency layers:** 13
- **Cycles (SCCs):** 12
- **Layer distribution:** L0: 2501 symbols, L1: 74 symbols, L2: 16 symbols, L3: 3 symbols, L4: 4 symbols

## Testing

**Test directories:** `tests/`

- **Test files:** 49
- **Source files:** 165
- **Test-to-source ratio:** 0.30

## Coding Conventions

Follow these conventions when writing code in this project:

- **Functions:** Use `snake_case` (99% of 180 functions)
- **Classes:** Use `PascalCase` (100% of 302 classes)
- **Methods:** Use `snake_case` (100% of 1620 methods)
- **Imports:** Prefer absolute imports (100% are cross-directory)
- **Test files:** test\_\*.py

## Complexity Hotspots

Average function complexity: 2.9 (2358 functions analyzed)

Functions with highest complexity (consider refactoring):

| Function                               | Complexity | Location                                        |
| -------------------------------------- | ---------- | ----------------------------------------------- |
| `get_stock_data_batch`                 | 171        | `src/stockula/data/fetcher.py:435`              |
| `calculate_auto_allocation_quantities` | 119        | `src/stockula/allocation/allocator.py:81`       |
| `run_main_processing`                  | 111        | `src/stockula/manager.py:1045`                  |
| `_get_calculation_prices`              | 99         | `src/stockula/allocation/base_allocator.py:53`  |
| `analyze_symbol`                       | 99         | `src/stockula/technical_analysis/manager.py:65` |
| `get_current_prices`                   | 92         | `src/stockula/data/fetcher.py:180`              |
| `run_technical_analysis`               | 88         | `src/stockula/manager.py:239`                   |
| `show_portfolio_forecast_value`        | 85         | `src/stockula/display.py:768`                   |
| `_display_forecast_results`            | 73         | `src/stockula/display.py:533`                   |
| `pytest_sessionfinish`                 | 65         | `tests/conftest.py:20`                          |

## Domain Keywords

- **Top domain terms:** strategy, backtest, calculate, portfolio, allocation, forecast, strategies, results, manager,
  quantities, initialization, calculation, dynamic, fetcher, date, logging, exception, ticker, category, symbol

## Core Modules

Most-imported modules (everything depends on these):

| Module                                   | Imported By | Symbols Used |
| ---------------------------------------- | ----------- | ------------ |
| `src/stockula/utils/logging_manager.py`  | 39 files    | 48           |
| `src/stockula/config/models.py`          | 36 files    | 211          |
| `src/stockula/cli_manager.py`            | 24 files    | 70           |
| `src/stockula/interfaces.py`             | 20 files    | 58           |
| `src/stockula/backtesting/strategies.py` | 18 files    | 98           |
| `src/stockula/database/models.py`        | 16 files    | 47           |
| `src/stockula/domain/ticker.py`          | 16 files    | 29           |
| `src/stockula/data/fetcher.py`           | 15 files    | 28           |
| `src/stockula/cli.py`                    | 14 files    | 24           |
| `src/stockula/backtesting/runner.py`     | 12 files    | 21           |
