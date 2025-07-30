# Troubleshooting

Common issues and solutions when using Stockula.

## Installation Issues

### uv Installation Problems

**Issue**: `uv` command not found

```bash
bash: uv: command not found
```

**Solutions**:

1. Install uv using the official installer:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

1. Add uv to your PATH (add to ~/.bashrc or ~/.zshrc):

   ```bash
   export PATH="$HOME/.cargo/bin:$PATH"
   ```

1. Restart your terminal or source your profile:

   ```bash
   source ~/.bashrc  # or ~/.zshrc
   ```

### Dependency Installation Failures

**Issue**: Package installation fails with build errors

```
error: Microsoft Visual C++ 14.0 is required
```

**Solutions**:

1. **Windows**: Install Microsoft C++ Build Tools
1. **macOS**: Install Xcode command line tools:
   ```bash
   xcode-select --install
   ```
1. **Linux**: Install build essentials:
   ```bash
   sudo apt-get install build-essential python3-dev
   ```

### Python Version Issues

**Issue**: Incompatible Python version

```
python 3.13+ is required
```

**Solutions**:

1. Check your Python version:

   ```bash
   python --version
   ```

1. Install Python 3.13+ using pyenv:

   ```bash
   pyenv install 3.13.0
   pyenv global 3.13.0
   ```

## Configuration Issues

### Configuration File Not Found

**Issue**:

```
FileNotFoundError: .config.yaml not found
```

**Solutions**:

1. Create a basic configuration file:

   ```bash
   cp examples/config.simple.yaml .config.yaml
   ```

1. Specify configuration explicitly:

   ```bash
   uv run python -m stockula.main --config examples/config.full.yaml
   ```

1. Use command line mode:

   ```bash
   uv run python -m stockula.main --ticker AAPL
   ```

### Invalid Configuration Format

**Issue**: YAML parsing errors

```
yaml.scanner.ScannerError: mapping values are not allowed here
```

**Solutions**:

1. Check YAML indentation (use spaces, not tabs)
1. Validate YAML syntax online
1. Check for special characters that need quoting
1. Use the provided examples as templates

### Pydantic Validation Errors

**Issue**: Configuration validation fails

```
ValidationError: 2 validation errors for StockulaConfig
portfolio.initial_capital
  ensure this value is greater than 0
```

**Solutions**:

1. Check required fields are present
1. Verify data types match expectations
1. Ensure numeric values are within valid ranges
1. Review the [Configuration Guide](getting-started/configuration.md)

## Data Fetching Issues

### Network Connection Problems

**Issue**: Unable to fetch data from yfinance

```
ConnectionError: Failed to fetch data for AAPL
```

**Solutions**:

1. Check internet connection
1. Try with cached data:
   ```bash
   uv run python -m stockula.database.cli query AAPL
   ```
1. Use different ticker symbols
1. Wait and retry (API rate limiting)

### Invalid Ticker Symbols

**Issue**:

```
ValueError: No data found for symbol INVALID
```

**Solutions**:

1. Verify ticker symbol is correct
1. Check if symbol exists on Yahoo Finance
1. Use common symbols for testing (AAPL, GOOGL, MSFT)
1. Try different exchanges (.TO for Toronto, .L for London)

### Date Range Issues

**Issue**:

```
ValueError: Start date cannot be after end date
```

**Solutions**:

1. Verify date format (YYYY-MM-DD)
1. Check start_date < end_date
1. Ensure dates are not in the future
1. Use null for end_date to default to today

## Database Issues

### SQLite Permission Errors

**Issue**:

```
sqlite3.OperationalError: database is locked
```

**Solutions**:

1. Close other applications using the database
1. Delete the database file and let it recreate:
   ```bash
   rm stockula.db
   ```
1. Check file permissions:
   ```bash
   ls -la stockula.db
   chmod 644 stockula.db
   ```

### Database Corruption

**Issue**:

```
sqlite3.DatabaseError: database disk image is malformed
```

**Solutions**:

1. Backup and recreate database:

   ```bash
   mv stockula.db stockula.db.backup
   uv run python -m stockula.database.cli init
   ```

1. Try database repair:

   ```bash
   uv run python -m stockula.database.cli vacuum
   ```

## Analysis Issues

### Technical Analysis Errors

**Issue**: Insufficient data for indicators

```
ValueError: Not enough data points for SMA(50)
```

**Solutions**:

1. Increase date range:

   ```yaml
   data:
     start_date: "2022-01-01"  # Use earlier date
   ```

1. Reduce indicator periods:

   ```yaml
   technical_analysis:
     sma_periods: [10, 20]  # Instead of [20, 50, 200]
   ```

1. Check data availability for the symbol

### Backtesting Failures

**Issue**: Strategy execution errors

```
RuntimeError: Strategy failed to initialize
```

**Solutions**:

1. Check strategy parameters are valid
1. Ensure sufficient historical data
1. Verify broker configuration
1. Test with simpler strategies first

### Forecasting Issues

**Issue**: AutoTS model training fails

```
AutoTSException: No models succeeded
```

**Solutions**:

1. Use faster model list:

   ```yaml
   forecast:
     model_list: "fast"
     max_generations: 2
   ```

1. Reduce forecast length:

   ```yaml
   forecast:
     forecast_length: 7  # Instead of 30
   ```

1. Check for sufficient historical data (at least 100 data points)

## Performance Issues

### Slow Data Fetching

**Issue**: Long wait times for data retrieval

**Solutions**:

1. Use cached data when possible
1. Reduce date ranges
1. Fetch fewer symbols at once
1. Check network connection speed

### High Memory Usage

**Issue**: Out of memory errors during analysis

**Solutions**:

1. Process smaller portfolios
1. Reduce historical data range
1. Use faster forecast models
1. Close unnecessary applications

### Long Forecasting Times

**Issue**: Forecasting takes too long

**Solutions**:

1. Use fast model configuration:

   ```yaml
   forecast:
     model_list: "fast"
     max_generations: 2
     num_validations: 1
   ```

1. Reduce forecast length

1. Process fewer symbols simultaneously

1. Consider using cloud computing for large forecasts

## Rich CLI Issues

### Terminal Display Problems

**Issue**: Garbled output or missing formatting

**Solutions**:

1. Update terminal to support Unicode

1. Set terminal encoding:

   ```bash
   export LANG=en_US.UTF-8
   ```

1. Disable Rich formatting:

   ```bash
   export NO_COLOR=1
   ```

1. Use plain output format:

   ```yaml
   output:
     format: "json"
   ```

### Progress Bar Issues

**Issue**: Progress bars not displaying correctly

**Solutions**:

1. Ensure terminal supports ANSI codes
1. Run in interactive mode
1. Update terminal application
1. Check terminal width is sufficient

## Common Error Messages

### "ModuleNotFoundError"

```
ModuleNotFoundError: No module named 'stockula'
```

**Solution**: Ensure you're using `uv run` prefix:

```bash
uv run python -m stockula.main --ticker AAPL
```

### "Permission denied"

```
PermissionError: [Errno 13] Permission denied
```

**Solutions**:

1. Check file permissions
1. Run without sudo/admin privileges
1. Ensure proper directory ownership

### "KeyError" in Results

```
KeyError: 'Return [%]'
```

**Solution**: Check that backtesting completed successfully and returned valid results.

## Debug Mode

Enable debug logging for more detailed error information:

```bash
# Set environment variable
export STOCKULA_DEBUG=true
export STOCKULA_LOG_LEVEL=DEBUG

# Or in configuration
logging:
  enabled: true
  level: "DEBUG"
```

## Getting Help

### Check Logs

Enable detailed logging:

```yaml
logging:
  enabled: true
  level: "DEBUG"
  log_to_file: true
  log_file: "debug.log"
```

### Minimal Reproduction

Create a minimal example that reproduces the issue:

```python
from stockula.data.fetcher import DataFetcher

# Test basic functionality
fetcher = DataFetcher()
data = fetcher.get_stock_data("AAPL", start_date="2024-01-01")
print(data.head())
```

### System Information

Collect system information for bug reports:

```bash
python --version
uv --version
pip list | grep -E "(pandas|yfinance|rich)"
```

### Test with Examples

Try the provided examples to isolate issues:

```bash
# Test with simple configuration
uv run python -m stockula.main --config examples/config.simple.yaml

# Test single ticker mode
uv run python -m stockula.main --ticker AAPL --mode ta
```

### Database Diagnostics

Check database health:

```bash
# Database statistics
uv run python -m stockula.database.cli stats

# Test database connection
uv run python -m stockula.database.cli query AAPL --limit 5
```

## Testing Issues

### Dependency Injection Test Failures

**Issue**: Tests fail with "AttributeError: 'Provide' object has no attribute 'get_stock_data'"

```
AttributeError: 'Provide' object has no attribute 'get_stock_data'
```

**Solution**: Wire the container in your test setup:

```python
from stockula.container import Container

container = Container()
container.data_fetcher.override(mock_data_fetcher)
container.wire(modules=["stockula.main"])  # This is required!
```

### Mock Object Errors

**Issue**: "TypeError: 'Mock' object is not subscriptable"

**Solution**: Create proper mock objects that support pandas-like operations:

```python
def create_iloc_mock(value):
    iloc_mock = Mock()
    iloc_mock.__getitem__ = Mock(return_value=value)
    return iloc_mock

sma_mock = Mock()
sma_mock.iloc = create_iloc_mock(150.0)
```

### Rich Console Output in Tests

**Issue**: Test assertions fail due to Rich table formatting

**Solution**: Check for content rather than exact format:

```python
# Instead of: assert output == "SMA_20: 150.00"
assert "SMA_20" in output
assert "150.00" in output
```

### Database Test Issues

**Issue**: Foreign key constraint tests fail because stocks are auto-created

**Solution**: Update tests to verify auto-creation behavior:

```python
# Stocks are auto-created when storing price data
test_database.add_price_data("NEWSTOCK", ...)
# Verify the stock was created
stock = session.query(Stock).filter_by(symbol="NEWSTOCK").first()
assert stock is not None
```

## FAQ

**Q: Why is forecasting so slow?**
A: Forecasting uses machine learning models that can be computationally intensive. Use the "fast" model list and reduce forecast length for better performance.

**Q: Can I use data from sources other than Yahoo Finance?**
A: Currently, Stockula uses yfinance as the primary data source. Custom data sources can be implemented by extending the DataFetcher class.

**Q: Why do I get different results between runs?**
A: Some forecasting models include randomness. For reproducible results, set random seeds or use deterministic models only.

**Q: How much historical data do I need?**
A: Minimum 100 data points for most analysis. Recommended: 2-3 years for backtesting, 1+ years for forecasting.

**Q: Can I run Stockula on a server without a display?**
A: Yes, set `NO_COLOR=1` environment variable or use JSON output format for headless operation.

For additional help, check the [GitHub Issues](https://github.com/mkm29/stockula/issues) page or create a new issue with your specific problem.
