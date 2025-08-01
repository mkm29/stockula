# Forecasting

Stockula provides advanced time series forecasting capabilities using AutoTS, enabling you to predict future stock prices with confidence intervals and multiple model validation.

## Overview

The forecasting module offers:

- **AutoTS Integration**: Automated model selection and optimization
- **Multiple Models**: Ensemble of forecasting algorithms optimized for financial data
- **Confidence Intervals**: Statistical uncertainty quantification
- **Model Validation**: Cross-validation and backtesting
- **Train/Test Evaluation**: Historical accuracy assessment with RMSE, MAE, and MAPE metrics
- **Performance Optimization**: Configurable speed vs. accuracy trade-offs
- **Rich Visualization**: Progress tracking and result display

## AutoTS Models

Stockula uses AutoTS for time series forecasting with carefully curated model lists optimized for financial data. The system automatically selects appropriate models based on the data type and configuration.

### Model Lists

#### Fast Financial Models (Default)

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

#### Full Financial Models

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

### Why These Models?

#### Models We Avoid

AutoTS includes many models that are problematic for stock data:

- **GLM with Gamma/InversePower** - Causes domain errors with financial data
- **FBProphet** - Permission issues with cmdstanpy, slow execution
- **ARDL** - Numerical stability issues with stock prices
- **Neural Network Models** - Often overfit on financial time series
- **Models using binary metrics** - Cause DataConversionWarning with continuous data

#### Selection Criteria

Our financial models were selected based on:

1. **Numerical Stability** - No domain errors or convergence issues
1. **Speed** - Complete forecasts in reasonable time
1. **Accuracy** - Proven track record with financial time series
1. **Robustness** - Handle missing data and outliers gracefully

## Configuration

### Forecast Modes

Stockula supports two mutually exclusive forecast modes:

1. **Future Prediction Mode**: Forecast N days into the future from today
1. **Historical Evaluation Mode**: Train on historical data and evaluate accuracy on a test period

### Future Prediction Mode

```yaml
forecast:
  forecast_length: 30           # Days to forecast from today
  model_list: "fast"            # fast, financial, or slow
  prediction_interval: 0.95     # 95% confidence interval
  # Note: Do NOT specify test dates when using forecast_length
```

### Historical Evaluation Mode

```yaml
forecast:
  # Train/test split for historical evaluation
  train_start_date: "2025-01-01"   # Training data start
  train_end_date: "2025-03-31"     # Training data end
  test_start_date: "2025-04-01"    # Test data start (for evaluation)
  test_end_date: "2025-06-30"      # Test data end

  model_list: "fast"            # fast, financial, or slow
  prediction_interval: 0.95     # 95% confidence interval
  # Note: Do NOT specify forecast_length when using train/test dates
```

**Important**: `forecast_length` and test dates (`test_start_date`/`test_end_date`) are mutually exclusive. You must choose one mode or the other.

### Advanced Configuration

```yaml
forecast:
  # Mode 1: Future prediction (choose one)
  forecast_length: 30           # Days ahead to forecast from today

  # Mode 2: Historical evaluation (choose one)
  # train_start_date: "2025-01-01"
  # train_end_date: "2025-03-31"
  # test_start_date: "2025-04-01"
  # test_end_date: "2025-06-30"

  # Common parameters
  frequency: "infer"            # D, W, M, or infer from data
  prediction_interval: 0.95     # Confidence interval (0.8, 0.9, 0.95, 0.99)

  # Model selection
  model_list: "fast"            # fast (default), financial, slow, or custom list
  ensemble: "auto"              # auto, simple, distance, horizontal
  max_generations: 5            # Genetic algorithm iterations
  num_validations: 2            # Cross-validation folds
  validation_method: "backwards" # backwards, seasonal, similarity

  # Performance optimization
  max_workers: 4                # Parallel workers for forecasting
  drop_most_recent: 0           # Drop recent data points
  drop_data_older_than_periods: 1000  # Limit historical data
  constraint: null              # Constraints on forecasts
  holiday_country: "US"         # Holiday effects
  subset: null                  # Subset of models to try

  # Advanced settings
  transformer_list: "auto"      # Data transformations
  transformer_max_depth: 5      # Transformation complexity
  models_mode: "default"        # Model generation mode
  models_to_validate: 0.15      # Fraction of models to validate
```

### Using Model Lists

```yaml
forecast:
  model_list: "fast"          # Automatically uses FAST_FINANCIAL_MODEL_LIST for stock data
  max_generations: 2          # Reduced for speed
  num_validations: 1          # Minimal validation
```

```yaml
forecast:
  model_list: "financial"     # Uses complete FINANCIAL_MODEL_LIST
  max_generations: 3          # More thorough search
  num_validations: 2          # Better validation
```

```yaml
forecast:
  model_list: "slow"          # WARNING: May include models unsuitable for financial data
```

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

## Usage Examples

### Command Line Usage

#### Quick Forecast (15-30 seconds)

```bash
# Using defaults optimized for speed
uv run python -m stockula.main --config config.yaml --mode forecast
```

#### Thorough Forecast (2-5 minutes)

```yaml
# config.yaml
forecast:
  model_list: "financial"
  max_generations: 5
  num_validations: 3
  max_workers: 8
```

```bash
uv run python -m stockula.main --config config.yaml --mode forecast
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

### Programmatic Usage

```python
from stockula.forecasting.forecaster import StockForecaster
from stockula.data.fetcher import DataFetcher
from stockula.config.settings import load_config

# Get historical data
fetcher = DataFetcher()
data = fetcher.get_stock_data("AAPL", start_date="2023-01-01")

# Create forecaster
config = load_config("myconfig.yaml")
forecaster = StockForecaster(
    forecast_length=30,
    model_list="fast",
    max_generations=2,
    data_fetcher=fetcher
)

# Generate forecast
forecast_result = forecaster.fit_predict(data, target_column="Close")

print(f"Forecast: {forecast_result['forecast'].iloc[-1]:.2f}")
print(f"Lower Bound: {forecast_result['lower_bound'].iloc[-1]:.2f}")
print(f"Upper Bound: {forecast_result['upper_bound'].iloc[-1]:.2f}")
```

### Batch Forecasting

```python
from stockula.domain.factory import DomainFactory

# Load portfolio configuration
config = load_config("myconfig.yaml")
factory = DomainFactory(config)
portfolio = factory.create_portfolio()

# Forecast entire portfolio
forecaster = StockForecaster(
    forecast_length=30,
    model_list="fast",
    data_fetcher=factory.data_fetcher
)

portfolio_forecasts = {}

for asset in portfolio.assets:
    try:
        data = factory.data_fetcher.get_stock_data(asset.ticker)
        forecast = forecaster.fit_predict(data)
        portfolio_forecasts[asset.ticker] = {
            'current_price': data['Close'].iloc[-1],
            'forecast_price': forecast['forecast'].iloc[-1],
            'lower_bound': forecast['lower_bound'].iloc[-1],
            'upper_bound': forecast['upper_bound'].iloc[-1]
        }
        print(f"{asset.ticker}: ${forecast['forecast'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"Error forecasting {asset.ticker}: {e}")
```

## Rich CLI Output

### Mode Detection

The CLI automatically detects which mode you're using and displays appropriate information:

#### Future Prediction Mode

```
╭────────────────────────────────────────────────────────────────╮
│ FORECAST MODE - IMPORTANT NOTES:                               │
│ • Forecasting 14 days into the future                          │
│ • AutoTS will try multiple models to find the best fit         │
│ • This process may take several minutes per ticker             │
│ • Press Ctrl+C at any time to cancel                           │
│ • Enable logging for more detailed progress information        │
╰────────────────────────────────────────────────────────────────╯
```

#### Historical Evaluation Mode

```
╭────────────────────────────────────────────────────────────────╮
│ FORECAST MODE - IMPORTANT NOTES:                               │
│ • Evaluating forecast on test period: 2025-04-01 to 2025-06-30 │
│ • AutoTS will try multiple models to find the best fit         │
│ • This process may take several minutes per ticker             │
│ • Press Ctrl+C at any time to cancel                           │
│ • Enable logging for more detailed progress information        │
╰────────────────────────────────────────────────────────────────╯
```

### Portfolio Value Summary

The portfolio value table shows different information based on the forecast mode:

#### Future Prediction Mode

```
               Portfolio Value
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Metric          ┃ Date       ┃ Value      ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ Observed Value  │ 2025-07-29 │ $20,000.00 │
│ Predicted Value │ 2025-08-13 │ $20,456.32 │
└─────────────────┴────────────┴────────────┘
```

#### Historical Evaluation Mode

```
               Portfolio Value
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Metric          ┃ Date       ┃ Value      ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ Observed Value  │ 2025-04-01 │ $20,000.00 │
│ Predicted Value │ 2025-06-30 │ $19,934.32 │
│ Accuracy        │ 2025-06-30 │ 90.8621%   │
└─────────────────┴────────────┴────────────┘
```

- **Observed Value**: The current portfolio value at the start date (today for future mode, test start for evaluation mode)
- **Predicted Value**: The forecasted portfolio value at the end date based on individual stock predictions
- **Accuracy**: (Evaluation mode only) The average forecast accuracy across all stocks, calculated as 100% - MAPE

### Forecast Results Table

Forecast results are displayed in descending order by expected return percentage (highest returns first):

```
                    Price Forecasts
┏━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Ticker ┃ Current Price ┃ Forecast Price ┃ Confidence Range ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ NVDA   │ $875.40       │ $920.15 ↑     │ $850.30 - $995.20│
│ AAPL   │ $150.25       │ $155.80 ↑     │ $145.20 - $165.40│
│ GOOGL  │ $2,750.80     │ $2,825.45 ↑   │ $2,650.30 - $3,010.60│
│ MSFT   │ $405.60       │ $412.25 ↑     │ $385.40 - $445.80│
│ TSLA   │ $248.50       │ $235.30 ↓     │ $215.10 - $265.50│
└────────┴───────────────┴────────────────┴──────────────────┘
```

Note: In this example, NVDA shows the highest expected return (+5.1%), followed by AAPL (+3.7%), GOOGL (+2.7%), MSFT (+1.6%), and TSLA (-5.3%).

### Forecast Evaluation Metrics

When train/test dates are configured, you'll also see accuracy metrics:

```
                         Model Performance on Test Data
┏━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Ticker ┃   RMSE ┃    MAE ┃ MAPE % ┃ Train Period       ┃ Test Period         ┃
┡━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ AAPL   │ $27.26 │ $24.12 │ 12.69% │ 2025-01-02 to      │ 2025-04-01 to       │
│        │        │        │        │ 2025-03-31         │ 2025-06-30          │
│ NVDA   │  $8.49 │  $6.75 │  6.68% │ 2025-01-02 to      │ 2025-04-01 to       │
│        │        │        │        │ 2025-03-31         │ 2025-06-30          │
│ TSLA   │ $58.38 │ $56.51 │ 22.20% │ 2025-01-02 to      │ 2025-04-01 to       │
│        │        │        │        │ 2025-03-31         │ 2025-06-30          │
└────────┴────────┴────────┴────────┴────────────────────┴─────────────────────┘
```

### Progress Tracking

AutoTS provides detailed progress information:

```
⠋ Training models on historical data...
```

## Train/Test Evaluation

When using Historical Evaluation Mode (train/test dates configured without forecast_length), Stockula automatically:

1. **Trains models** on historical data from `train_start_date` to `train_end_date`
1. **Makes predictions** for the period from `test_start_date` to `test_end_date`
1. **Compares predictions** to actual prices during the test period
1. **Calculates accuracy metrics**:
   - **RMSE (Root Mean Square Error)**: Average prediction error in dollars
   - **MAE (Mean Absolute Error)**: Average absolute error in dollars
   - **MAPE (Mean Absolute Percentage Error)**: Average percentage error
   - **Accuracy**: Calculated as 100% - MAPE

### Example Configuration

```yaml
forecast:
  # Historical evaluation mode - NO forecast_length specified
  train_start_date: "2025-01-01"   # 3 months of training data
  train_end_date: "2025-03-31"
  test_start_date: "2025-04-01"    # 3 months of test data
  test_end_date: "2025-06-30"

  # Model configuration
  model_list: "fast"
  prediction_interval: 0.95
```

### Interpreting Results

- **Accuracy > 95%**: Excellent model performance
- **Accuracy 90-95%**: Good model performance
- **Accuracy 85-90%**: Acceptable performance
- **Accuracy < 85%**: Consider adjusting model parameters

The portfolio-level accuracy shown in the summary is the average of individual stock accuracies, weighted by their portfolio allocation.

## Forecast Interpretation

### Direction Indicators

- **↗ Bullish**: Forecast price > current price
- **↘ Bearish**: Forecast price < current price
- **→ Neutral**: Forecast price ≈ current price (±1%)

### Confidence Levels

| Interval | Interpretation               |
| -------- | ---------------------------- |
| 80%      | High confidence range        |
| 90%      | Standard confidence range    |
| 95%      | Conservative range (default) |
| 99%      | Very conservative range      |

### Model Quality Indicators

- **Best Model**: The model selected by AutoTS based on validation performance
- **Ensemble Score**: 0.0-1.0, higher is better
- **Validation MAE**: Mean Absolute Error on validation data
- **Cross-validation Score**: Performance across multiple periods

## Advanced Features

### Custom Model Lists

```yaml
forecast:
  model_list:
    - "LastValueNaive"
    - "SeasonalNaive"
    - "ARIMA"
    - "ETS"
    - "Theta"
    - "WindowRegression"
```

### Ensemble Methods

```yaml
forecast:
  ensemble: "distance"          # Weight by model accuracy distance
  # ensemble: "simple"          # Simple average of predictions
  # ensemble: "horizontal"      # Horizontal ensemble (advanced)
  # ensemble: "auto"            # AutoTS selects best method
```

### Transformations

AutoTS applies data transformations to improve forecast accuracy:

```yaml
forecast:
  transformer_list: "all"       # All available transformations
  # transformer_list: "fast"    # Basic transformations only
  # transformer_list: "superfast" # Minimal transformations
  transformer_max_depth: 3      # Limit transformation complexity
```

### Validation Methods

```yaml
forecast:
  validation_method: "backwards"  # Standard time series validation
  # validation_method: "seasonal" # Seasonal cross-validation
  # validation_method: "similarity" # Similar period validation
  num_validations: 3             # Number of validation folds
```

## Model Performance Analysis

### Model Comparison

```python
def compare_forecast_models(symbol, forecast_length=30):
    """Compare different model configurations."""
    configs = {
        'fast': {'model_list': 'fast', 'max_generations': 2},
        'financial': {'model_list': 'financial', 'max_generations': 3},
        'slow': {'model_list': 'slow', 'max_generations': 10}
    }

    results = {}

    for name, config_params in configs.items():
        # Create forecaster with specific config
        forecaster = StockForecaster(
            forecast_length=forecast_length,
            model_list=config_params['model_list'],
            max_generations=config_params['max_generations'],
            data_fetcher=fetcher
        )

        data = fetcher.get_stock_data(symbol)
        forecast = forecaster.fit_predict(data)

        results[name] = {
            'forecast_price': forecast['forecast'].iloc[-1],
            'confidence_width': forecast['upper_bound'].iloc[-1] - forecast['lower_bound'].iloc[-1],
            'best_model': forecaster.get_best_model()['model_name']
        }

    return results
```

### Accuracy Tracking

```python
def track_forecast_accuracy(symbol, days_back=90, forecast_length=7):
    """Track historical forecast accuracy."""
    data = fetcher.get_stock_data(symbol, start_date="2023-01-01")

    accuracies = []

    for i in range(days_back, len(data) - forecast_length):
        # Historical data up to point i
        historical_data = data.iloc[:i]

        # Make forecast
        forecaster = StockForecaster(
            forecast_length=forecast_length,
            model_list="fast",
            data_fetcher=fetcher
        )
        forecast = forecaster.fit_predict(historical_data)

        # Actual prices
        actual_prices = data.iloc[i:i+forecast_length]['Close']
        forecast_prices = forecast['forecast']

        # Calculate accuracy
        mae = np.mean(np.abs(forecast_prices - actual_prices))
        mape = np.mean(np.abs((forecast_prices - actual_prices) / actual_prices)) * 100

        accuracies.append({
            'date': data.index[i],
            'mae': mae,
            'mape': mape,
            'direction_correct': np.sign(forecast_prices.iloc[-1] - forecast_prices.iloc[0]) ==
                               np.sign(actual_prices.iloc[-1] - actual_prices.iloc[0])
        })

    return pd.DataFrame(accuracies)
```

## Integration with Trading Strategies

### Forecast-Based Strategy

```python
from stockula.backtesting.strategies import BaseStrategy

class ForecastStrategy(BaseStrategy):
    """Trade based on price forecasts."""

    forecast_days = 7
    confidence_threshold = 0.8

    def init(self):
        # Initialize forecaster
        from stockula.forecasting.forecaster import StockForecaster
        self.forecaster = StockForecaster(
            forecast_length=self.forecast_days,
            model_list="fast",
            max_generations=2
        )

    def next(self):
        # Get recent data for forecasting
        recent_data = self.data.df.iloc[-252:]  # Last year of data

        try:
            # Generate forecast
            forecast = self.forecaster.fit_predict(recent_data)

            current_price = self.data.Close[-1]
            forecast_price = forecast['forecast'].iloc[-1]

            # Trading logic
            price_change = (forecast_price - current_price) / current_price

            if price_change > 0.03 and not self.position:  # 3% upside
                self.buy()
            elif price_change < -0.03 and self.position:   # 3% downside
                self.sell()

        except Exception as e:
            # Handle forecasting errors gracefully
            pass
```

### Portfolio Rebalancing

```python
def forecast_based_rebalancing(portfolio, forecaster, rebalance_threshold=0.05):
    """Rebalance portfolio based on forecasts."""
    forecasts = {}

    # Generate forecasts for all assets
    for asset in portfolio.assets:
        data = forecaster.data_fetcher.get_stock_data(asset.ticker)
        forecast = forecaster.fit_predict(data)
        forecasts[asset.ticker] = {
            'current_price': data['Close'].iloc[-1],
            'forecast_price': forecast['forecast'].iloc[-1]
        }

    # Calculate expected returns
    expected_returns = {}
    for ticker, forecast in forecasts.items():
        current_price = forecast['current_price']
        forecast_price = forecast['forecast_price']
        expected_return = (forecast_price - current_price) / current_price
        expected_returns[ticker] = expected_return

    # Rank assets by expected return
    ranked_assets = sorted(expected_returns.items(), key=lambda x: x[1], reverse=True)

    # Rebalance if significant differences
    top_return = ranked_assets[0][1]
    bottom_return = ranked_assets[-1][1]

    if top_return - bottom_return > rebalance_threshold:
        # Overweight top performers, underweight bottom performers
        print(f"Rebalancing recommended: {top_return:.2%} spread")
        return ranked_assets

    return None  # No rebalancing needed
```

## Best Practices

### Data Quality

1. **Sufficient History**: Use at least 2-3 years of data for training
1. **Data Frequency**: Match forecast frequency to your use case
1. **Handle Gaps**: Clean missing data before forecasting
1. **Outlier Treatment**: Consider removing extreme outliers

### Model Selection

1. **Start with Fast Models**: Test feasibility before using slow models
1. **Use Financial Models**: Default `model_list="fast"` uses optimized financial models
1. **Cross-Validate**: Always validate on out-of-sample data
1. **Ensemble Benefits**: Use ensemble methods for better robustness
1. **Regular Retraining**: Update models with new data periodically

### Forecast Interpretation

1. **Confidence Intervals**: Always consider uncertainty ranges
1. **Direction vs. Magnitude**: Focus on direction for trading decisions
1. **Validation Scores**: Trust forecasts with better validation performance
1. **Market Context**: Consider current market conditions

### Production Usage

1. **Error Handling**: Gracefully handle forecast failures
1. **Performance Monitoring**: Track forecast accuracy over time
1. **Model Decay**: Retrain models when accuracy degrades
1. **Computational Resources**: Balance accuracy vs. computation time

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

### Common Issues and Solutions

1. **"Frequency is 'None'! Data frequency not recognized."**

   - This warning appears when AutoTS cannot automatically detect the data frequency
   - Solution: Stockula now defaults to 'D' (daily) frequency and attempts to infer the actual frequency automatically

1. **"k too large for size of data in motif"**

   - This warning occurred with Motif pattern recognition models when pattern length exceeded data size
   - Solution: Already fixed - Motif models have been removed from the default financial model list

1. **Alembic migration warnings**

   - These warnings indicate database schema updates
   - Solution: The migrations now check for existing indexes before creating them

### Performance Tips

1. **Reduce model search time**: Use `model_list: "fast"` and lower `max_generations`
1. **Handle small datasets**: Ensure at least 1-2 years of historical data for best results
1. **Memory usage**: For large portfolios, consider forecasting in batches
1. **Parallel processing**: Set `max_workers` > 1 for concurrent forecasting

The forecasting module provides a powerful foundation for predictive analysis while maintaining ease of use through AutoTS automation and Rich CLI integration.
