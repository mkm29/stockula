# Stockula Forecasting Models Guide

## Overview

Stockula uses AutoTS for time series forecasting with carefully curated model lists optimized for financial data. The system automatically selects appropriate models based on the data type and configuration.

## Model Lists

### Fast Financial Models

The `FAST_FINANCIAL_MODEL_LIST` contains 6 carefully selected models that provide the best balance of speed and accuracy for stock price forecasting:

| Model                 | Description                                       | Speed             | Best For                                     |
| --------------------- | ------------------------------------------------- | ----------------- | -------------------------------------------- |
| **LastValueNaive**    | Uses the last observed value as the forecast      | Very Fast (< 1s)  | Baseline predictions, volatile stocks        |
| **AverageValueNaive** | Uses a moving average of recent values            | Very Fast (< 1s)  | Smoothing short-term volatility              |
| **SeasonalNaive**     | Captures and projects seasonal patterns           | Fast (< 5s)       | Stocks with clear weekly/monthly patterns    |
| **ETS**               | Exponential Smoothing (Error, Trend, Seasonality) | Fast (5-10s)      | Trending stocks with seasonal components     |
| **ARIMA**             | AutoRegressive Integrated Moving Average          | Moderate (10-20s) | Well-behaved time series with clear patterns |
| **Theta**             | Statistical decomposition method                  | Fast (5-10s)      | Robust general-purpose forecasting           |

**Total execution time**: 15-30 seconds per symbol with default settings

### Full Financial Models

The `FINANCIAL_MODEL_LIST` includes 16 models for more comprehensive analysis:

- All models from the fast list, plus:
- **GLS** - Generalized Least Squares
- **RollingRegression** - Adaptive regression over time windows
- **WindowRegression** - Fixed window regression analysis
- **VAR** - Vector Autoregression for multivariate series
- **VECM** - Vector Error Correction for cointegrated series
- **DynamicFactor** - Captures underlying market factors
- **MotifSimulation** - Pattern-based forecasting
- **SectionalMotif** - Cross-sectional pattern analysis
- **NVAR** - Neural Vector Autoregression

**Total execution time**: 2-5 minutes per symbol with default settings

## Configuration

### Using Fast Models (Default)

```yaml
forecast:
  model_list: "fast"  # Automatically uses FAST_FINANCIAL_MODEL_LIST for stock data
  max_generations: 2  # Reduced for speed
  num_validations: 1  # Minimal validation
```

### Using Full Models

```yaml
forecast:
  model_list: "financial"  # Uses complete FINANCIAL_MODEL_LIST
  max_generations: 3      # More thorough search
  num_validations: 2      # Better validation
```

### Custom Model Selection

```yaml
forecast:
  model_list: "slow"  # WARNING: May include models unsuitable for financial data
```

## Why These Models?

### Models We Avoid

AutoTS includes many models that are problematic for stock data:

- **GLM with Gamma/InversePower** - Causes domain errors with financial data
- **FBProphet** - Permission issues with cmdstanpy, slow execution
- **ARDL** - Numerical stability issues with stock prices
- **Neural Network Models** - Often overfit on financial time series
- **Models using binary metrics** - Cause DataConversionWarning with continuous data

### Selection Criteria

Our financial models were selected based on:

1. **Numerical Stability** - No domain errors or convergence issues
1. **Speed** - Complete forecasts in reasonable time
1. **Accuracy** - Proven track record with financial time series
1. **Robustness** - Handle missing data and outliers gracefully

## Performance Optimization

### Speed Tips

1. **Use Fast Models**: Default `model_list="fast"` for quick results
1. **Reduce Generations**: Set `max_generations=1` for fastest execution
1. **Increase Workers**: Set `max_workers=8` if you have 8+ CPU cores
1. **Limit Data**: Use only necessary historical data (e.g., 1 year)

### Accuracy Tips

1. **Use Full Models**: Set `model_list="financial"` for best results
1. **Increase Generations**: Set `max_generations=5` for thorough search
1. **Add Validations**: Set `num_validations=3` for better model selection
1. **More Data**: Use 2-3 years of historical data

## Example Usage

### Quick Forecast (15-30 seconds)

```bash
# Using defaults optimized for speed
uv run python -m stockula.main --config config.yaml --mode forecast
```

### Thorough Forecast (2-5 minutes)

```yaml
# config.yaml
forecast:
  model_list: "financial"
  max_generations: 5
  num_validations: 3
  max_workers: 8
```

### Understanding the Output

When forecasting starts, you'll see:

```
Starting parallel forecasting...
Configuration: max_workers=4, max_generations=2, num_validations=1
Using fast financial model list (6 models) for Close
```

This confirms:

- Number of parallel workers processing symbols
- Model evolution generations
- Validation splits for model selection
- Which model list is being used

## Troubleshooting

### Forecasts Taking Too Long

- Check if `model_list="fast"` is set
- Reduce `max_generations` to 1
- Ensure you're using financial models (check logs)

### Getting Warnings

- Financial models automatically suppress most warnings
- If warnings persist, check you're forecasting "Close" or "Price" columns
- Custom model lists may include problematic models

### Poor Forecast Quality

- Try `model_list="financial"` for more models
- Increase `max_generations` to 3-5
- Ensure sufficient historical data (200+ days)
