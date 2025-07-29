# Forecasting

Stockula provides advanced time series forecasting capabilities using AutoTS, enabling you to predict future stock prices with confidence intervals and multiple model validation.

## Overview

The forecasting module offers:

- **AutoTS Integration**: Automated model selection and optimization
- **Multiple Models**: Ensemble of forecasting algorithms
- **Confidence Intervals**: Statistical uncertainty quantification
- **Model Validation**: Cross-validation and backtesting
- **Train/Test Evaluation**: Historical accuracy assessment with RMSE, MAE, and MAPE metrics
- **Performance Optimization**: Configurable speed vs. accuracy trade-offs
- **Rich Visualization**: Progress tracking and result display

## AutoTS Models

### Fast Models (Default)

Optimized for speed while maintaining reasonable accuracy:

- **Naive**: Simple baseline forecasts
- **SeasonalNaive**: Seasonal pattern repetition
- **Linear Regression**: Trend-based linear models
- **Exponential Smoothing**: Simple ETS models

### Default Models

Balanced performance and accuracy:

- **ARIMA**: Auto-regressive integrated moving average
- **ETS**: Error, trend, seasonality models
- **Theta**: Theta forecasting method
- **TBATS**: Complex seasonality handling

### Slow Models

Maximum accuracy with longer computation time:

- **Prophet**: Facebook's forecasting algorithm
- **LSTM**: Long short-term memory neural networks
- **VAR**: Vector autoregression
- **VECM**: Vector error correction models

### Financial Models

When `use_financial_models` is enabled (default), Stockula uses a curated list of models optimized for stock price data:

- **LastValueNaive**: Simple but effective for financial data
- **AverageValueNaive**: Moving average approach
- **SeasonalNaive**: Captures seasonal patterns
- **ARIMA**: Classic time series model
- **ETS**: Exponential smoothing
- **DynamicFactor**: Good for capturing trends
- **VAR**: Vector autoregression
- **Regression Models**: Univariate, Multivariate, Window, and Datepart regression
- **NVAR**: Neural VAR for complex patterns
- **Theta**: Theta method
- **ARDL**: Autoregressive distributed lag

Note: Motif models (pattern recognition) have been excluded to avoid warnings with small datasets.

## Configuration

### Forecast Modes

Stockula supports two mutually exclusive forecast modes:

1. **Future Prediction Mode**: Forecast N days into the future from today
1. **Historical Evaluation Mode**: Train on historical data and evaluate accuracy on a test period

### Future Prediction Mode

```yaml
forecast:
  forecast_length: 30           # Days to forecast from today
  model_list: "fast"            # fast, default, slow
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
  
  model_list: "fast"            # fast, default, slow
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
  model_list: "fast"            # fast, default, slow, or custom list
  ensemble: "auto"              # auto, simple, distance, horizontal
  max_generations: 5            # Genetic algorithm iterations
  num_validations: 2            # Cross-validation folds
  validation_method: "backwards" # backwards, seasonal, similarity
  
  # Performance optimization
  drop_most_recent: 0           # Drop recent data points
  drop_data_older_than_periods: 1000  # Limit historical data
  constraint: null              # Constraints on forecasts
  holiday_country: "US"         # Holiday effects
  subset: null                  # Subset of models to try
  
  # Advanced settings
  transformer_list: "auto"      # Data transformations
  transformer_max_depth: 5      # Transformation complexity
  models_mode: "default"        # Model generation mode
  num_validations: 2            # Number of validation rounds
  models_to_validate: 0.15      # Fraction of models to validate
```

### Performance Optimization

For faster forecasting:

```yaml
forecast:
  forecast_length: 14           # Shorter forecasts
  model_list: "fast"            # Only fast models
  max_generations: 2            # Fewer iterations
  num_validations: 1            # Single validation
  ensemble: "simple"            # Simpler ensemble
  transformer_list: "fast"      # Basic transformations
```

For maximum accuracy:

```yaml
forecast:
  forecast_length: 30
  model_list: "slow"            # All available models
  max_generations: 10           # More genetic iterations
  num_validations: 5            # Extensive validation
  ensemble: "auto"              # Best ensemble method
  transformer_list: "all"       # All transformations
```

## Choosing Between Forecast Modes

### When to Use Future Prediction Mode

Use `forecast_length` when you want to:

- Predict future stock prices from today
- Make trading decisions based on expected future prices
- Generate forecasts for portfolio planning
- Create alerts for expected price movements

Example:

```yaml
forecast:
  forecast_length: 14  # Predict 14 days into the future
```

### When to Use Historical Evaluation Mode

Use train/test dates when you want to:

- Evaluate model accuracy on historical data
- Compare different forecasting models
- Backtest forecast-based trading strategies
- Understand model performance before using it for real predictions

Example:

```yaml
forecast:
  train_start_date: "2024-01-01"
  train_end_date: "2024-12-31"
  test_start_date: "2025-01-01"
  test_end_date: "2025-01-31"
```

## Usage Examples

### Command Line Usage

```bash
# Basic forecasting
uv run python -m stockula.main --ticker AAPL --mode forecast

# With custom configuration
uv run python -m stockula.main --config examples/config.forecast.yaml --mode forecast

# Portfolio forecasting
uv run python -m stockula.main --config myconfig.yaml --mode forecast
```

### Programmatic Usage

```python
from stockula.forecasting.forecaster import Forecaster
from stockula.data.fetcher import DataFetcher
from stockula.config.settings import load_config

# Get historical data
fetcher = DataFetcher()
data = fetcher.get_stock_data("AAPL", start_date="2023-01-01")

# Create forecaster
config = load_config("myconfig.yaml")
forecaster = Forecaster(config)

# Generate forecast
forecast_result = forecaster.forecast_price("AAPL", forecast_length=30)

print(f"Current Price: ${forecast_result['current_price']:.2f}")
print(f"Forecast Price: ${forecast_result['forecast_price']:.2f}")
print(f"Confidence Range: ${forecast_result['lower_bound']:.2f} - ${forecast_result['upper_bound']:.2f}")
```

### Batch Forecasting

```python
from stockula.domain.factory import DomainFactory

# Load portfolio configuration
config = load_config("myconfig.yaml")
factory = DomainFactory(config)
portfolio = factory.create_portfolio()

# Forecast entire portfolio
forecaster = Forecaster(config)
portfolio_forecasts = {}

for asset in portfolio.assets:
    try:
        forecast = forecaster.forecast_price(asset.ticker, forecast_length=30)
        portfolio_forecasts[asset.ticker] = forecast
        print(f"{asset.ticker}: {forecast['forecast_price']:.2f} ({forecast['direction']})")
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

```
                    Price Forecasts                     
┏━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Ticker ┃ Current Price ┃ Forecast Price ┃ Confidence Range ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ AAPL   │ $150.25       │ $155.80 ↑     │ $145.20 - $165.40│
│ NVDA   │ $875.40       │ $920.15 ↑     │ $850.30 - $995.20│
│ TSLA   │ $248.50       │ $235.30 ↓     │ $215.10 - $265.50│
│ GOOGL  │ $2,750.80     │ $2,825.45 ↑   │ $2,650.30 - $3,010.60│
│ MSFT   │ $405.60       │ $412.25 ↑     │ $385.40 - $445.80│
└────────┴───────────────┴────────────────┴──────────────────┘
```

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
⠋ Training fast models with 5 generations... (AutoTS is working...)
⠋ Generation 1/5: Testing 45 models... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 60% 0:01:23
⠋ Generation 2/5: Testing 32 models... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 80% 0:00:45
⠋ Validating best models... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 95% 0:00:15
⠋ Building ensemble forecast... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:05
✓ Forecasting completed for AAPL
```

### Detailed Forecast Information

```
╭─────────────────────── FORECAST: AAPL ───────────────────────╮
│                                                              │
│  Current Price: $150.25                                      │
│  30-day Forecast: $155.80 (+3.7%)                            │
│                                                              │
│  Confidence Interval (95%): $145.20 - $165.40               │
│  Expected Range: ±6.8%                                       │
│                                                              │
│  Model Performance:                                          │
│    Best Model: ETS (AAA)                                     │
│    Ensemble Score: 0.84                                      │
│    Validation MAE: $2.15                                     │
│                                                              │
│  Forecast Details:                                           │
│    Trend: Bullish ↗                                         │
│    Volatility: Moderate                                      │
│    Seasonality: Weak                                         │
│                                                              │
╰──────────────────────────────────────────────────────────────╯
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

- **Ensemble Score**: 0.0-1.0, higher is better
- **Validation MAE**: Mean Absolute Error on validation data
- **Cross-validation Score**: Performance across multiple periods

## Advanced Features

### Custom Model Lists

```yaml
forecast:
  model_list:
    - "Naive"
    - "SeasonalNaive"
    - "LinearRegression"
    - "ETS"
    - "ARIMA"
    - "Theta"
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
        'default': {'model_list': 'default', 'max_generations': 5},
        'slow': {'model_list': 'slow', 'max_generations': 10}
    }
    
    results = {}
    
    for name, config_params in configs.items():
        config = load_config()
        config.forecast.update(config_params)
        
        forecaster = Forecaster(config)
        forecast = forecaster.forecast_price(symbol, forecast_length)
        
        results[name] = {
            'forecast_price': forecast['forecast_price'],
            'confidence_width': forecast['upper_bound'] - forecast['lower_bound'],
            'ensemble_score': forecast.get('ensemble_score', 0),
            'computation_time': forecast.get('computation_time', 0)
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
        forecast = forecaster.forecast_from_data(historical_data, forecast_length)
        
        # Actual prices
        actual_prices = data.iloc[i:i+forecast_length]['Close']
        
        # Calculate accuracy
        mae = np.mean(np.abs(forecast['forecast'] - actual_prices))
        mape = np.mean(np.abs((forecast['forecast'] - actual_prices) / actual_prices)) * 100
        
        accuracies.append({
            'date': data.index[i],
            'mae': mae,
            'mape': mape,
            'direction_correct': np.sign(forecast['forecast'][-1] - forecast['forecast'][0]) == 
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
    
    def init(self):
        self.forecaster = Forecaster(self.config)
        self.forecast_length = 7
        self.confidence_threshold = 0.8
    
    def next(self):
        # Get recent data for forecasting
        recent_data = self.data.df.iloc[-252:]  # Last year of data
        
        try:
            # Generate forecast
            forecast = self.forecaster.forecast_from_data(recent_data, self.forecast_length)
            
            current_price = self.data.Close[-1]
            forecast_price = forecast['forecast_price']
            confidence = forecast['ensemble_score']
            
            # Trading logic
            if confidence > self.confidence_threshold:
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
        forecast = forecaster.forecast_price(asset.ticker, forecast_length=30)
        forecasts[asset.ticker] = forecast
    
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
        new_allocations = calculate_new_allocations(ranked_assets)
        return new_allocations
    
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

### Common Warnings

1. **"Frequency is 'None'! Data frequency not recognized."**

   - This warning appears when AutoTS cannot automatically detect the data frequency
   - Solution: Ensure your data has consistent date intervals or explicitly set `frequency` in config
   - Default behavior: Stockula now defaults to 'D' (daily) frequency and attempts to infer the actual frequency from your data automatically

1. **"k too large for size of data in motif"**

   - This warning occurred with Motif pattern recognition models when pattern length exceeded data size
   - Solution: Already fixed - Motif models have been removed from the default financial model list
   - If using custom model lists, avoid UnivariateMotif and MultivariateMotif with small datasets

1. **Alembic migration warnings**

   - These warnings indicate database schema updates
   - Solution: The migrations now check for existing indexes before creating them
   - The warnings should no longer appear after the fix

### Performance Tips

1. **Reduce model search time**: Use `model_list: "fast"` and lower `max_generations`
1. **Handle small datasets**: Ensure at least 2-3 years of historical data for best results
1. **Memory usage**: For large portfolios, consider forecasting in batches
1. **Parallel processing**: Set `max_workers` > 1 for concurrent forecasting (use cautiously)

The forecasting module provides a powerful foundation for predictive analysis while maintaining ease of use through AutoTS automation and Rich CLI integration.
