# Pipeline Orchestration

The Stockula Pipeline provides a streamlined workflow for portfolio optimization and backtesting. It automates the
process of finding optimal asset allocations and validating them through historical backtesting.

## Overview

The pipeline orchestrates the following workflow:

1. **Load Configuration**: Load and validate your base portfolio configuration
1. **Run Optimization**: Find optimal asset allocations based on historical data and forecasts
1. **Save Optimized Config**: Persist the optimized allocations to a new configuration file
1. **Run Backtesting**: Test the optimized portfolio using historical data
1. **Compare Results**: Analyze performance differences between original and optimized allocations

## Quick Start

### Command Line Usage

The simplest way to use the pipeline is through the CLI:

```bash
# Run the complete pipeline
uv run python -m stockula pipeline \
    --base-config .stockula.yaml \
    --optimized-config .stockula-optimized.yaml \
    --output results.json
```

### Python API

For programmatic control:

```python
from stockula.pipeline import StockulaPipeline

# Create and run pipeline
pipeline = StockulaPipeline(base_config_path=".stockula.yaml")
results = pipeline.run_full_pipeline(
    optimized_config_path=".stockula-optimized.yaml"
)
```

## CLI Command Reference

The `stockula pipeline` command provides comprehensive options for controlling the workflow:

### Basic Usage

```bash
uv run python -m stockula pipeline [OPTIONS]
```

### Options

| Option                | Short | Description                                   | Default          |
| --------------------- | ----- | --------------------------------------------- | ---------------- |
| `--base-config`       | `-b`  | Path to base configuration file               | `.stockula.yaml` |
| `--optimized-config`  | `-o`  | Path to save optimized configuration          | None             |
| `--output`            |       | Path to save pipeline results (json/yaml/csv) | None             |
| `--skip-optimization` |       | Skip optimization and use existing config     | False            |
| `--skip-backtest`     |       | Skip backtesting after optimization           | False            |
| `--verbose`           | `-v`  | Enable verbose output                         | False            |

### Examples

#### Full Pipeline Execution

Run optimization followed by backtesting:

```bash
uv run python -m stockula pipeline \
    --base-config portfolio.yaml \
    --optimized-config portfolio-opt.yaml \
    --output results.json \
    --verbose
```

#### Optimization Only

Run just the optimization step:

```bash
uv run python -m stockula pipeline \
    --base-config portfolio.yaml \
    --optimized-config optimized.yaml \
    --skip-backtest
```

#### Backtesting with Existing Config

Use a previously optimized configuration:

```bash
uv run python -m stockula pipeline \
    --base-config optimized.yaml \
    --skip-optimization \
    --output backtest-results.json
```

## Python API

The `StockulaPipeline` class provides fine-grained control over the optimization and backtesting workflow.

### Basic Usage

```python
from stockula.pipeline import StockulaPipeline

# Initialize pipeline
pipeline = StockulaPipeline(
    base_config_path="portfolio.yaml",
    verbose=True
)

# Run complete workflow
results = pipeline.run_full_pipeline(
    optimized_config_path="optimized.yaml"
)
```

### Class Reference

#### `StockulaPipeline`

Main pipeline orchestration class.

##### Constructor

```python
StockulaPipeline(
    base_config_path: str | Path | None = None,
    verbose: bool = False,
    console: Console | None = None
)
```

**Parameters:**

- `base_config_path`: Path to the base configuration file
- `verbose`: Enable verbose output
- `console`: Rich console for output (creates one if not provided)

##### Methods

###### `load_configuration(config_path=None)`

Load and validate a configuration file.

```python
config = pipeline.load_configuration("portfolio.yaml")
```

**Returns:** `StockulaConfig` instance

###### `run_optimization(config=None, save_config_path=None, **params)`

Run portfolio optimization.

```python
optimized_config, results = pipeline.run_optimization(
    config=config,
    save_config_path="optimized.yaml",
    time_limit=60  # Additional optimization parameters
)
```

**Returns:** Tuple of `(optimized_config, optimization_results)`

###### `run_backtest(config=None, use_optimized=True, **params)`

Run backtesting with specified configuration.

```python
results = pipeline.run_backtest(
    config=optimized_config,
    use_optimized=True,
    commission=0.001  # Additional backtest parameters
)
```

**Returns:** Dictionary of backtest results

###### `run_full_pipeline(base_config_path=None, optimized_config_path=None, **params)`

Execute the complete pipeline workflow.

```python
results = pipeline.run_full_pipeline(
    optimized_config_path="optimized.yaml",
    optimization={"time_limit": 60},
    backtest={"commission": 0.001}
)
```

**Returns:** Combined results dictionary

###### `save_results(path, format='json')`

Save pipeline results to file.

```python
pipeline.save_results("results.json", format="json")
pipeline.save_results("results.yaml", format="yaml")
pipeline.save_results("results.csv", format="csv")
```

**Supported formats:** `json`, `yaml`, `csv`

###### `save_optimized_config(path)`

Save the optimized configuration to a YAML file.

```python
pipeline.save_optimized_config("optimized-portfolio.yaml")
```

### Advanced Examples

#### Custom Optimization Parameters

```python
from stockula.pipeline import StockulaPipeline

pipeline = StockulaPipeline(verbose=True)

# Load configuration
config = pipeline.load_configuration("portfolio.yaml")

# Run optimization with custom parameters
optimized_config, opt_results = pipeline.run_optimization(
    config=config,
    save_config_path="optimized.yaml",
    time_limit=120,  # 2 minutes time limit
    min_weight=0.05,  # Minimum 5% allocation
    max_weight=0.40   # Maximum 40% allocation
)

# Display optimization metrics
if "metrics" in opt_results:
    print("Optimization Metrics:")
    for key, value in opt_results["metrics"].items():
        print(f"  {key}: {value}")
```

#### Comparing Strategies

Compare performance before and after optimization:

```python
from stockula.pipeline import StockulaPipeline

pipeline = StockulaPipeline(base_config_path="portfolio.yaml")

# Load base configuration
base_config = pipeline.load_configuration()

# Run backtest with original allocation
print("Testing original allocation...")
original_results = pipeline.run_backtest(
    config=base_config,
    use_optimized=False
)

# Run optimization
print("Optimizing portfolio...")
optimized_config, _ = pipeline.run_optimization(config=base_config)

# Run backtest with optimized allocation
print("Testing optimized allocation...")
optimized_results = pipeline.run_backtest(
    config=optimized_config,
    use_optimized=True
)

# Compare results
orig_return = original_results["summary"]["total_return"]
opt_return = optimized_results["summary"]["total_return"]
improvement = opt_return - orig_return

print(f"\nPerformance Comparison:")
print(f"Original Return: {orig_return:.2%}")
print(f"Optimized Return: {opt_return:.2%}")
print(f"Improvement: {improvement:+.2%}")
```

#### Batch Processing Multiple Portfolios

Process multiple portfolio configurations:

```python
from pathlib import Path
from stockula.pipeline import StockulaPipeline

# List of portfolio configurations
portfolios = [
    "conservative.yaml",
    "moderate.yaml",
    "aggressive.yaml"
]

results = {}

for portfolio_file in portfolios:
    print(f"\nProcessing {portfolio_file}...")

    pipeline = StockulaPipeline(
        base_config_path=portfolio_file,
        verbose=False
    )

    # Run optimization and backtesting
    result = pipeline.run_full_pipeline(
        optimized_config_path=f"optimized-{portfolio_file}"
    )

    # Store results
    results[portfolio_file] = result

    # Save individual results
    pipeline.save_results(
        f"results-{portfolio_file.replace('.yaml', '.json')}",
        format="json"
    )

# Summary report
print("\n=== Portfolio Optimization Summary ===")
for portfolio, result in results.items():
    if "backtest" in result and "summary" in result["backtest"]:
        summary = result["backtest"]["summary"]
        print(f"\n{portfolio}:")
        print(f"  Return: {summary.get('total_return', 'N/A')}")
        print(f"  Sharpe: {summary.get('sharpe_ratio', 'N/A')}")
```

#### Error Handling

Properly handle errors in pipeline execution:

```python
from stockula.pipeline import StockulaPipeline
from pydantic import ValidationError

try:
    pipeline = StockulaPipeline(
        base_config_path="portfolio.yaml",
        verbose=True
    )

    # Run pipeline with error handling
    results = pipeline.run_full_pipeline(
        optimized_config_path="optimized.yaml"
    )

    # Save results
    pipeline.save_results("results.json")

except FileNotFoundError as e:
    print(f"Configuration file not found: {e}")

except ValidationError as e:
    print(f"Configuration validation error: {e}")

except KeyboardInterrupt:
    print("\nPipeline interrupted by user")

except Exception as e:
    print(f"Unexpected error: {e}")
    raise
```

## Integration with Existing Modes

The pipeline integrates seamlessly with Stockula's existing modes:

### Using Optimized Config with Other Modes

Once you've created an optimized configuration, use it with any Stockula mode:

```bash
# Technical analysis with optimized portfolio
uv run python -m stockula \
    --config .stockula-optimized.yaml \
    --mode ta

# Forecasting with optimized allocation
uv run python -m stockula \
    --config .stockula-optimized.yaml \
    --mode forecast \
    --days 30

# Run specific backtest strategy
uv run python -m stockula \
    --config .stockula-optimized.yaml \
    --mode backtest
```

### Workflow Integration

Typical workflow combining pipeline with other features:

1. **Create base configuration** with initial allocations
1. **Run pipeline** to optimize allocations
1. **Use optimized config** for daily operations
1. **Periodically re-optimize** as market conditions change

```bash
# Step 1: Initial setup
cat > portfolio.yaml << EOF
portfolio:
  initial_capital: 100000
  tickers:
    - symbol: AAPL
      quantity: 100
    - symbol: GOOGL
      quantity: 50
    - symbol: MSFT
      quantity: 75
EOF

# Step 2: Optimize
uv run python -m stockula pipeline \
    -b portfolio.yaml \
    -o portfolio-optimized.yaml

# Step 3: Daily analysis
uv run python -m stockula \
    --config portfolio-optimized.yaml \
    --mode forecast

# Step 4: Re-optimize monthly
uv run python -m stockula pipeline \
    -b portfolio-optimized.yaml \
    -o portfolio-reoptimized.yaml
```

## Performance Considerations

### Time Complexity

- **Optimization**: O(n × m × t) where n=assets, m=strategies, t=time periods
- **Backtesting**: O(n × s × t) where s=number of strategies

### Memory Usage

- Pipeline caches results in memory
- Large portfolios may require increased memory
- Use `--skip-backtest` for optimization-only workflows

### Optimization Tips

1. **Use time limits** for large portfolios:

   ```python
   pipeline.run_optimization(time_limit=300)  # 5 minutes
   ```

1. **Limit strategy combinations** in backtesting:

   ```yaml
   backtest:
     strategies:
       - name: rsi
         enabled: true
       - name: macd
         enabled: false  # Disable unused strategies
   ```

1. **Process portfolios in batches** for multiple configurations

## Troubleshooting

### Common Issues

#### Configuration Not Found

```
Error: Configuration file not found: portfolio.yaml
```

**Solution:** Ensure the configuration file exists and the path is correct.

#### Validation Errors

```
Configuration validation error: portfolio.initial_capital must be positive
```

**Solution:** Check your configuration against the schema requirements.

#### Insufficient Data

```
Warning: Not enough historical data for optimization
```

**Solution:** Ensure your date ranges include sufficient historical data (typically 1-2 years).

### Debug Mode

Enable verbose output for debugging:

```python
import logging

# Set logging level
logging.basicConfig(level=logging.DEBUG)

# Create pipeline with verbose mode
pipeline = StockulaPipeline(verbose=True)

# Run with detailed output
results = pipeline.run_full_pipeline()
```

## Best Practices

1. **Start with conservative allocations** in your base configuration
1. **Use appropriate time ranges** (1-2 years of historical data)
1. **Validate optimized allocations** before production use
1. **Save all configurations** for audit trail
1. **Monitor performance** and re-optimize periodically
1. **Test strategies** on out-of-sample data
1. **Document configuration changes** for compliance

## See Also

- [Configuration Guide](../configuration.md) - Setting up portfolio configurations
- [Backtesting](backtesting.md) - Detailed backtesting documentation
- [Optimization](optimization.md) - Portfolio optimization strategies
- [API Reference](../api/pipeline.md) - Complete API documentation
