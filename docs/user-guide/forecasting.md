# Forecasting

Stockula provides time series forecasting capabilities using AutoGluon TimeSeries or a simple fallback implementation, enabling you to predict future stock prices with confidence intervals.

## Overview

The forecasting module offers:

- **ForecastingManager**: Centralized coordinator for all forecasting operations
- **AutoGluon Integration**: Advanced automated machine learning for time series (Python < 3.13)
- **Simple Fallback**: Linear regression-based forecasting for environments without AutoGluon
- **Confidence Intervals**: Statistical uncertainty quantification
- **Flexible Configuration**: Support for both future predictions and historical evaluation
- **Rich Visualization**: Progress tracking and result display

## Backend Support

### AutoGluon TimeSeries (Recommended)

When available (requires Python < 3.13), Stockula uses AutoGluon TimeSeries, which provides:

- **Automated Model Selection**: Automatically selects the best model from multiple candidates
- **Deep Learning Models**: Access to state-of-the-art neural network architectures
- **Ensemble Methods**: Combines multiple models for better accuracy
- **Robust Evaluation**: Built-in cross-validation and metrics

To install AutoGluon:

```bash
# For Python < 3.13
pip install "stockula[forecasting]"
# or
pip install autogluon-timeseries>=1.2.0
```

### Simple Fallback (Python 3.13+)

For Python 3.13+ or environments without AutoGluon, Stockula provides a simple linear regression-based forecasting backend that:

- **Linear Trend Projection**: Uses linear regression to identify and project trends
- **Confidence Intervals**: Calculates statistical bounds based on historical variance
- **Fast Execution**: Minimal computational requirements
- **No Additional Dependencies**: Uses only scikit-learn

## ForecastingManager

The ForecastingManager provides a unified interface for all forecasting operations.

### Basic Usage

```python
from stockula.container import Container

container = Container()
forecasting_manager = container.forecasting_manager()

# Forecast a single symbol
result = forecasting_manager.forecast_symbol('AAPL', config)

# Forecast multiple symbols with progress tracking
results = forecasting_manager.forecast_multiple_symbols_with_progress(
    ['AAPL', 'GOOGL', 'MSFT'], 
    config
)

# Quick forecast with fast settings
quick_result = forecasting_manager.quick_forecast('AAPL', forecast_days=7)
```

### Manager Methods

- `forecast_symbol(symbol, config)`: Forecast a single stock symbol
- `forecast_multiple_symbols(symbols, config)`: Forecast multiple symbols
- `forecast_multiple_symbols_with_progress(symbols, config, console)`: Forecast with progress bar
- `quick_forecast(symbol, forecast_days, historical_days)`: Fast forecast with minimal configuration
- `validate_forecast_config(config)`: Validate forecast configuration

## Configuration

Forecast configuration is done through the `ForecastConfig` class:

```python
from stockula.config import ForecastConfig

# Future prediction mode
forecast_config = ForecastConfig(
    forecast_length=7,  # Predict 7 days into the future
    frequency="infer",  # Auto-detect data frequency
    prediction_interval=0.9,  # 90% confidence interval
    preset="medium_quality",  # AutoGluon preset (if available)
    time_limit=120,  # Max seconds for training
    eval_metric="MASE",  # Evaluation metric
    no_negatives=True  # Prevent negative price predictions
)

# Historical evaluation mode
forecast_config = ForecastConfig(
    train_start_date="2024-01-01",
    train_end_date="2024-12-31",
    test_start_date="2025-01-01",
    test_end_date="2025-01-31",
    frequency="D",
    prediction_interval=0.95
)
```

### AutoGluon Presets (when available)

- **`fast_training`**: Quick results with reduced accuracy
- **`medium_quality`**: Balanced speed and accuracy (default)
- **`high_quality`**: Better accuracy, longer training time
- **`best_quality`**: Maximum accuracy, longest training time

### Evaluation Metrics

- **`MASE`**: Mean Absolute Scaled Error (default, scale-independent)
- **`MAPE`**: Mean Absolute Percentage Error
- **`MAE`**: Mean Absolute Error
- **`RMSE`**: Root Mean Squared Error
- **`SMAPE`**: Symmetric Mean Absolute Percentage Error
- **`WAPE`**: Weighted Average Percentage Error

## Example: Complete Forecasting Workflow

```python
from stockula.config import StockulaConfig
from stockula.container import create_container

# Load configuration
config = StockulaConfig.from_file("config.yaml")

# Create container
container = create_container()

# Get forecasting manager
forecasting_manager = container.forecasting_manager()

# Forecast multiple symbols
symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]
results = forecasting_manager.forecast_multiple_symbols_with_progress(
    symbols, 
    config
)

# Display results
for symbol, result in results.items():
    if "error" not in result:
        print(f"{symbol}:")
        print(f"  Current Price: ${result['current_price']:.2f}")
        print(f"  Forecast Price: ${result['forecast_price']:.2f}")
        print(f"  Confidence Range: ${result['lower_bound']:.2f} - ${result['upper_bound']:.2f}")
        print(f"  Best Model: {result['best_model']}")
    else:
        print(f"{symbol}: Error - {result['error']}")
```

## CLI Usage

Forecast using the command line:

```bash
# Single ticker forecast
uv run python -m stockula --ticker AAPL --mode forecast

# Portfolio forecast
uv run python -m stockula --config portfolio.yaml --mode forecast

# Forecast with custom dates for evaluation
uv run python -m stockula --ticker AAPL --mode forecast \
    --train-start 2024-01-01 --train-end 2024-12-31 \
    --test-start 2025-01-01 --test-end 2025-01-31
```

## Understanding Forecast Results

The forecast results include:

- **`current_price`**: Latest known price
- **`forecast_price`**: Predicted future price
- **`lower_bound`**: Lower confidence interval
- **`upper_bound`**: Upper confidence interval
- **`forecast_length`**: Number of days forecasted
- **`best_model`**: Model used for prediction
- **`metrics`**: Evaluation metrics (if using test dates)

## Performance Considerations

### With AutoGluon

- **Initial Run**: 30-120 seconds per symbol (model training)
- **Memory Usage**: 2-4 GB for typical datasets
- **GPU Support**: Can utilize GPU if available

### With Simple Fallback

- **Speed**: < 1 second per symbol
- **Memory Usage**: Minimal (< 100 MB)
- **Accuracy**: Lower than AutoGluon but suitable for quick estimates

## Troubleshooting

### AutoGluon Not Available

If you see "AutoGluon not available" warning:

1. Check Python version: `python --version` (must be < 3.13)
1. Install AutoGluon: `pip install autogluon-timeseries`
1. For Python 3.13+, the simple fallback will be used automatically

### Forecast Errors

Common issues and solutions:

- **"No data available"**: Ensure the ticker symbol is valid and data dates are correct
- **"forecast_length must be positive"**: Set a valid forecast_length in configuration
- **Memory errors**: Reduce the time_limit or use fast_training preset

## Best Practices

1. **Data Requirements**: Provide at least 30 days of historical data for meaningful predictions
1. **Forecast Horizon**: Shorter forecasts (7-30 days) are generally more accurate
1. **Confidence Intervals**: Use wider intervals (0.95) for risk-sensitive decisions
1. **Model Selection**: Let AutoGluon automatically select models unless you have specific requirements
1. **Evaluation**: Use train/test split to assess forecast accuracy on historical data

## Migration from AutoTS

Stockula has migrated from AutoTS to AutoGluon TimeSeries for improved performance and accuracy. Key changes:

- **Simpler Configuration**: No need to specify model lists or ensemble methods
- **Better Performance**: AutoGluon generally provides better accuracy
- **Automatic Fallback**: Simple linear regression for environments without AutoGluon
- **Python 3.13 Support**: Works with latest Python via fallback implementation

All existing configurations will work with minimal changes. The system automatically handles the backend selection based on availability.
