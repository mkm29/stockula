# Forecasting Backends

Stockula now supports multiple time series forecasting backends, allowing you to choose the best tool for your needs.
Currently supported backends:

- **AutoTS** (default): Feature-rich with many statistical and ML models
- **AutoGluon**: Modern AutoML with deep learning models and GPU acceleration

## Backend Comparison

| Feature             | AutoTS                                 | AutoGluon                            |
| ------------------- | -------------------------------------- | ------------------------------------ |
| Statistical Models  | ✅ Extensive (ARIMA, ETS, Theta, etc.) | ✅ Basic (ARIMA, ETS)                |
| Tree-based Models   | ✅ Via regression wrappers             | ✅ Native (LightGBM)                 |
| Deep Learning       | ⚠️ Limited                             | ✅ Extensive (DeepAR, TFT, PatchTST) |
| Zero-shot Models    | ❌                                     | ✅ Chronos                           |
| GPU Acceleration    | ⚠️ Limited                             | ✅ Full support                      |
| Ensemble Methods    | ✅ Multiple options                    | ✅ Automatic                         |
| Model Selection     | ✅ Genetic algorithm                   | ✅ AutoML                            |
| Python 3.13 Support | ✅ Full                                | ✅ Full                              |
| Memory Usage        | 🟡 Moderate                            | 🔴 Higher                            |
| Training Speed      | 🟢 Fast                                | 🟡 Depends on preset                 |

## Configuration

Select the backend in your configuration file (`.stockula.yaml`):

```yaml
forecast:
  # Choose backend: 'autots' or 'autogluon'
  backend: autogluon

  forecast_length: 30
  prediction_interval: 0.95

  # AutoTS-specific settings (used when backend: autots)
  model_list: clean  # Options: ultra_fast, fast, clean, financial
  ensemble: auto
  max_generations: 2
  num_validations: 2
  validation_method: backwards

  # AutoGluon-specific settings (used when backend: autogluon)
  preset: medium_quality  # Options: fast_training, medium_quality, high_quality, best_quality
  time_limit: 300  # Optional: time limit in seconds
  eval_metric: MASE  # Evaluation metric for model selection (MASE recommended for stocks)
```

## Installation

### Basic Installation

AutoTS is included by default. To add AutoGluon:

```bash
# IMPORTANT: AutoGluon currently requires pandas<2.3.0, which conflicts with
# stockula's pandas>=2.3.1 requirement. To use AutoGluon, you need to install
# it in a separate environment or downgrade pandas:

# Option 1: Create a separate environment for AutoGluon
python -m venv autogluon-env
source autogluon-env/bin/activate  # On Windows: autogluon-env\Scripts\activate
pip install pandas==2.2.2
pip install autogluon.timeseries

# Option 2: Downgrade pandas (may affect other features)
pip install pandas==2.2.2
pip install autogluon.timeseries

# Note: This compatibility issue will be resolved when AutoGluon updates to support pandas>=2.3
```

### GPU Installation

For GPU acceleration (recommended for AutoGluon):

```bash
# Install GPU dependencies
pip install stockula[gpu,autogluon]

# Or use the requirements file
pip install -r requirements-gpu.txt
```

## Usage Examples

### Using AutoTS Backend (Default)

```python
from stockula.config import StockulaConfig
from stockula.forecasting import ForecastingManager
from stockula.data import DataFetcher

# Configure with AutoTS
config = StockulaConfig(
    forecast={
        "backend": "autots",
        "forecast_length": 30,
        "model_list": "financial",  # Use financial models
        "max_generations": 3,
    }
)

# Create manager and forecast
manager = ForecastingManager(data_fetcher=DataFetcher())
result = manager.forecast_symbol("AAPL", config)
```

### Using AutoGluon Backend

```python
# Configure with AutoGluon
config = StockulaConfig(
    forecast={
        "backend": "autogluon",
        "forecast_length": 30,
        "preset": "high_quality",  # Use more models for better accuracy
        "time_limit": 600,  # 10 minute time limit
    }
)

# Create manager and forecast
manager = ForecastingManager(data_fetcher=DataFetcher())
result = manager.forecast_symbol("AAPL", config)
```

### Comparing Backends

```python
# Compare forecasts from both backends
manager = ForecastingManager(data_fetcher=DataFetcher())
comparison = manager.compare_backends("AAPL", config)

# Results from both backends
autots_forecast = comparison["autots"]["forecast"]
autogluon_forecast = comparison["autogluon"]["forecast"]
```

### Quick Forecasting

```python
# Quick forecast with minimal configuration
manager = ForecastingManager(data_fetcher=DataFetcher())

# Ultra-fast AutoTS forecast
quick_result = manager.quick_forecast(
    "AAPL",
    forecast_days=7,
    backend="autots"
)

# Fast AutoGluon forecast
quick_result = manager.quick_forecast(
    "AAPL",
    forecast_days=7,
    backend="autogluon"
)
```

## CLI Usage

```bash
# Use AutoTS backend (default)
python -m stockula --ticker AAPL --mode forecast --days 30

# Use AutoGluon backend
python -m stockula --ticker AAPL --mode forecast --days 30 --backend autogluon

# Compare backends
python -m stockula --ticker AAPL --mode forecast --days 30 --compare-backends
```

## Backend Selection Guide

### When to Use AutoTS

- **Statistical analysis**: Need traditional time series models (ARIMA, ETS)
- **Fast iteration**: Rapid prototyping and experimentation
- **Limited resources**: Lower memory requirements
- **Ensemble flexibility**: Want control over ensemble methods
- **Stable patterns**: Data with clear trends and seasonality

### When to Use AutoGluon

- **Deep learning**: Want to leverage neural networks
- **GPU available**: Can utilize GPU acceleration
- **Complex patterns**: Non-linear relationships in data
- **Best accuracy**: Willing to trade speed for accuracy
- **AutoML preference**: Want automatic model selection

## Model Details

### AutoTS Models

AutoTS provides several preset model lists:

- **ultra_fast**: 3 models (Naive methods only)
- **fast**: 6 models (Basic statistical models)
- **clean**: 8 models (No warnings, production-ready)
- **financial**: 12 models (Optimized for stock data)

### AutoGluon Models

AutoGluon automatically selects from:

- **Statistical**: ETS, ARIMA, Theta, Naive, SeasonalNaive
- **Tree-based**: LightGBM with time series features
- **Deep Learning**: DeepAR, TemporalFusionTransformer, PatchTST
- **Zero-shot**: Chronos (pre-trained foundation model)

## Evaluation Metrics (AutoGluon)

AutoGluon uses evaluation metrics to select the best model during training. For stock price forecasting, the choice of
metric is important:

### Recommended: MASE (Mean Absolute Scaled Error)

- **Scale-independent**: Works equally well for stocks at $10 or $1000
- **Robust**: Not affected by outliers as much as squared metrics
- **Interpretable**: Values < 1 mean better than naive forecast
- **Default in Stockula**: Best for comparing performance across different stocks

### Alternative Metrics

| Metric    | When to Use                      | Considerations                              |
| --------- | -------------------------------- | ------------------------------------------- |
| **MASE**  | Different price scales (default) | Best for portfolio with varied stock prices |
| **MAPE**  | Percentage errors matter         | Undefined for zero values, asymmetric       |
| **MAE**   | Absolute dollar errors matter    | Scale-dependent, good for single stock      |
| **RMSE**  | Large errors are critical        | Penalizes outliers heavily                  |
| **SMAPE** | Balanced percentage errors       | More symmetric than MAPE                    |
| **WAPE**  | Volume-weighted accuracy         | Good for high-volume stocks                 |

### Configuration Example

```yaml
forecast:
  backend: autogluon
  eval_metric: MASE  # Default, recommended for stocks
  # eval_metric: MAE  # Use if dollar accuracy matters more
  # eval_metric: RMSE  # Use if large errors must be minimized
```

## Performance Considerations

### Memory Usage

- **AutoTS**: ~1-2GB for typical datasets
- **AutoGluon**: ~2-4GB, more with deep learning models

### Training Time

Typical training times for 1000 data points:

| Backend   | Preset/Config  | Time (CPU) | Time (GPU) |
| --------- | -------------- | ---------- | ---------- |
| AutoTS    | ultra_fast     | 2-5s       | N/A        |
| AutoTS    | clean          | 10-30s     | N/A        |
| AutoTS    | financial      | 30-60s     | N/A        |
| AutoGluon | fast_training  | 30-60s     | 20-40s     |
| AutoGluon | medium_quality | 2-5min     | 1-3min     |
| AutoGluon | high_quality   | 5-15min    | 3-8min     |

### Accuracy

Based on internal benchmarks on stock price data:

- **AutoTS clean**: MASE ~0.8-1.2 (better than naive when < 1.0)
- **AutoTS financial**: MASE ~0.7-1.0
- **AutoGluon medium**: MASE ~0.6-0.9
- **AutoGluon high**: MASE ~0.5-0.8

*Note: MASE (Mean Absolute Scaled Error) is scale-independent and more appropriate for comparing performance across
different stocks. Values < 1.0 indicate the model beats a naive forecast. Actual performance varies significantly based
on the specific stock and market conditions.*

## Troubleshooting

### AutoGluon Import Error

```bash
ImportError: AutoGluon TimeSeries not installed
```

Solution:

```bash
pip install autogluon.timeseries
```

### GPU Not Detected (AutoGluon)

Ensure CUDA is properly installed:

```python
import torch
print(torch.cuda.is_available())  # Should be True
```

### Memory Issues

For large datasets or limited memory:

1. Use AutoTS with fast presets
1. Reduce AutoGluon time_limit
1. Use fast_training preset for AutoGluon

### Slow Training

To speed up training:

1. **AutoTS**: Use `model_list="ultra_fast"` or reduce `max_generations`
1. **AutoGluon**: Use `preset="fast_training"` or set a `time_limit`

## Advanced Usage

### Custom Model Lists

```python
# AutoTS with custom models
config = StockulaConfig(
    forecast={
        "backend": "autots",
        "model_list": ["ARIMA", "ETS", "Theta"],  # Specific models
    }
)

# AutoGluon with specific models
from stockula.forecasting.backends import AutoGluonBackend
backend = AutoGluonBackend(
    models=["DeepAR", "TemporalFusionTransformer"],  # Only deep learning
)
```

### Backend Abstraction

Create custom backends by implementing the `ForecastBackend` interface:

```python
from stockula.forecasting.backends import ForecastBackend, ForecastResult

class MyCustomBackend(ForecastBackend):
    def fit(self, data, target_column="Close", **kwargs):
        # Implement fitting logic
        pass

    def predict(self, **kwargs):
        # Implement prediction logic
        return ForecastResult(...)

    def get_model_info(self):
        # Return model information
        pass
```

## Future Backends

Planned backend support:

- **Prophet**: Facebook's Prophet for business time series
- **NeuralProphet**: Neural network version of Prophet
- **Darts**: Comprehensive time series library
- **StatsForecast**: High-performance statistical methods
- **GluonTS**: Advanced deep learning models

## Contributing

To add a new forecasting backend:

1. Create a new class inheriting from `ForecastBackend`
1. Implement required methods: `fit()`, `predict()`, `get_model_info()`
1. Add backend to `factory.py`
1. Update configuration models
1. Add tests and documentation

See `backends/autots.py` and `backends/autogluon.py` for examples.
