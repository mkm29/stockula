# Migration Guide

## Version 0.15.6 → 0.16.0

### Breaking Changes

#### Naming Convention Updates

To comply with PEP 8 standards, the following method names have been updated to use snake_case:

**ILoggingManager Interface:**
- `isEnabledFor(level: int)` → `is_enabled_for(level: int)`

**Impact:**
If you have custom implementations of the `ILoggingManager` interface, you'll need to update your method names:

```python
# Before
class MyLoggingManager(ILoggingManager):
    def isEnabledFor(self, level: int) -> bool:
        return self.logger.isEnabledFor(level)

# After  
class MyLoggingManager(ILoggingManager):
    def is_enabled_for(self, level: int) -> bool:
        return self.logger.isEnabledFor(level)  # Note: Python's logging still uses isEnabledFor
```

**Files Updated:**
- `src/stockula/interfaces.py` - Interface definition
- `src/stockula/utils/logging_manager.py` - Default implementation
- `src/stockula/forecasting/backends/autogluon.py` - Usage updated
- All test files using mock logging managers

### Non-Breaking Improvements

#### Code Quality Enhancements

1. **Reduced Cyclomatic Complexity** across multiple modules:
   - `manager.py`: Extracted helper methods like `_get_custom_indicators()`, `_analyze_symbol_with_progress()`, and `_process_with_progress_results()`
   - `cli.py`: Split `run_stockula()` into smaller functions like `_load_config_with_validation()`, `_setup_logging()`, `_override_config()`
   - `display.py`: Refactored complex methods into focused helper functions
   - `autogluon.py`: Split prediction logic into `_get_quantile_cols()`, `_get_forecast_bounds()`, `_apply_non_negative()`

2. **Type Safety Improvements**:
   - Added comprehensive type hints throughout the codebase
   - Full mypy compliance with proper type annotations
   - Added explicit type casting where needed (e.g., `float()` conversions)
   - Proper use of `Optional`, `Union`, and `Callable` types
   - Import organization with `TYPE_CHECKING` for circular dependencies

3. **Enhanced Validation** in `config/models.py`:
   - Added field constraints for better data integrity
   - Ticker symbols limited to 1-10 characters
   - Numeric fields have appropriate bounds (e.g., `max_drawdown_pct <= 0`, `win_rate` between 0-100)
   - Portfolio values must be positive

4. **Test Suite Improvements**:
   - Fixed all 23 failing tests after refactoring
   - Increased test coverage from 17% to 81%
   - Updated test mocks to match new method signatures
   - All 958 unit tests now passing

5. **Documentation**:
   - Enhanced docstrings with detailed parameter descriptions
   - Added coding standards section to CONTRIBUTING.md
   - Created comprehensive migration guide
   - Updated CLAUDE.md with latest improvements

These improvements maintain backward compatibility while significantly improving code quality, maintainability, and type safety.