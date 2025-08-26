# Migration Guide

## Version 0.15.6 → Next Version

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

1. **Reduced Code Duplication** in `manager.py`:
   - Extracted `_update_config_quantities()` method
   - Created `_convert_to_python_float()` utility method

2. **Enhanced Validation** in `config/models.py`:
   - Added field constraints for better data integrity
   - Ticker symbols limited to 1-10 characters
   - Numeric fields have appropriate bounds

3. **Improved Type Hints**:
   - Better type annotations in `display.py` and `cli.py`
   - More specific optional parameter types

4. **Documentation**:
   - Enhanced docstrings with detailed parameter descriptions
   - Added coding standards section to CONTRIBUTING.md

These improvements maintain backward compatibility while improving code quality and maintainability.