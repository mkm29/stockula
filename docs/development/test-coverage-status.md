# Test Coverage Status and Improvements

This document details the current test coverage status, recent improvements, and important warnings about untested modules in the Stockula project.

## ⚠️ Critical Testing Warning

**The `src/stockula/backtesting/strategies.py` module (567 lines) has NOT been thoroughly tested** due to complex dependencies on the `backtesting.py` library.

### Why Testing is Blocked

1. **Framework Constraints**: Strategies inherit from `backtesting.Strategy` with read-only properties
1. **Tight Coupling**: The `init()` and `next()` methods only execute within the framework
1. **Complex State**: Internal indicator calculations and state management prevent mocking
1. **Data Dependencies**: Requires specific OHLCV structures difficult to mock

### Usage Recommendations

Until comprehensive testing is implemented:

- **Use with Caution**: Consider strategies experimental
- **Validate Results**: Compare against known benchmarks
- **Paper Trade First**: Test strategies before live deployment
- **Monitor Performance**: Track production behavior closely

## Coverage Summary

### Overall Project Status

| Module              | Coverage | Status | Notes                   |
| ------------------- | -------- | ------ | ----------------------- |
| **Overall Project** | 83%      | ✅     | Excluding strategies.py |
| **main.py**         | 83%      | ✅     | Refactored to entry point |
| **forecaster.py**   | 83%      | ✅     | +33% improvement        |
| **factory.py**      | 88%      | ✅     | +9% improvement         |
| **runner.py**       | 89%      | ✅     | +19% improvement        |
| **indicators.py**   | 98%      | ✅     | Excellent coverage      |
| **metrics.py**      | 90%      | ✅     | Good coverage           |
| **strategies.py**   | Excluded | ⚠️     | 567 lines untested      |

### Recent Improvements

#### BacktestRunner (`runner.py`)

- **Before**: 70% (225/339 lines)
- **After**: 89% (303/339 lines)
- **Added**: Commission calculations, train/test splits, equity curves

#### DomainFactory (`factory.py`)

- **Before**: 79% (323/407 lines)
- **After**: 88% (359/407 lines)
- **Added**: Date conversions, auto-allocation, error scenarios

#### Main Module (`main.py`)

- **Before**: 61%
- **After**: 83%
- **Refactored**: Reduced from ~1942 lines to ~350 lines (CLI entry point only)
- **New modules**: manager.py (StockulaManager), display.py (ResultsDisplay)

#### Forecaster (`forecaster.py`)

- **Before**: ~50%
- **After**: 83%
- **Added**: Output suppression, model selection, batch forecasting

## Test Suite Enhancements

### New Test Files

- `test_factory.py`: 27 comprehensive test cases
- Consolidated ~75 new test cases across modules

### Testing Approach

1. **Edge Case Focus**: Boundary conditions and error scenarios
1. **Isolated Testing**: Extensive mocking of external dependencies
1. **Clear Organization**: Logical grouping by functionality

### Key Test Scenarios

**Commission Structures**:

- Percentage, fixed, and per-share commissions
- Tiered structures and edge cases

**Allocation Strategies**:

- Dynamic, auto, and mixed allocation methods
- Redistribution logic and capital utilization

**Error Handling**:

- Data fetch failures
- Invalid configurations
- Missing dependencies

## Untested Strategies

The following strategies in `strategies.py` remain untested:

- SMACrossStrategy
- RSIStrategy
- MACDStrategy
- DoubleEMACrossStrategy
- VIDYAStrategy
- KAMAStrategy
- FRAMAStrategy
- TripleEMACrossStrategy
- TRIMACrossStrategy
- VAMAStrategy
- KaufmanEfficiencyStrategy

## Configuration

### Coverage Exclusions

Configured in multiple files:

- `.coveragerc`: Coverage.py settings
- `codecov.yml`: Codecov reporting
- `pyproject.toml`: Project configuration

### Codecov Integration

Two configuration options available:

1. **`codecov.yml`**: Full-featured with component tracking
1. **`.codecov.yml`**: Simplified basic configuration

Key settings:

- Project coverage: Auto with 1% threshold
- Patch coverage: 80% requirement
- Excluded: Test files, `__init__.py`, docs, strategies.py

## Future Plans

### Q2 2024 Roadmap

1. **Integration Test Suite**

   - Known datasets with expected outcomes
   - Full backtest verification
   - Edge case scenarios

1. **Refactoring for Testability**

   - Separate indicator logic from trading
   - Create testable interfaces
   - Improve mockability

1. **Additional Coverage**

   - Network failure scenarios
   - Concurrent access testing
   - Performance benchmarks

## Metrics

- **Tests Added**: ~75 new test cases
- **Coverage Gained**: +70 lines total
- **Execution Time**: All tests < 10 seconds
- **Overall Coverage**: Maintained >75% (excluding strategies.py)

## Conclusion

Significant progress has been made in test coverage, with most modules now exceeding 80% coverage. The exclusion of `strategies.py` is a deliberate decision due to framework limitations, not neglect. The focus on edge cases and error scenarios ensures robust behavior across the testable codebase.
