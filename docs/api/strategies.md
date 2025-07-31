# Strategies API Reference

This reference covers all available trading strategies, their parameters, and how to create custom strategies in Stockula.

## Built-in Strategies

### Simple Moving Average Crossover (SMACROSS)

A classic trend-following strategy using two simple moving averages.

**Class**: `SMAStrategy`

**Parameters**:

- `fast_period` (int, default: 10): Period for fast moving average
- `slow_period` (int, default: 20): Period for slow moving average

**Strategy Logic**:

- **Buy Signal**: Fast MA crosses above slow MA
- **Sell Signal**: Fast MA crosses below slow MA

**Example Configuration**:

```yaml
backtest:
  strategies:
    - name: smacross
      parameters:
        fast_period: 10
        slow_period: 20
```

**Code Example**:

```python
from stockula.backtesting.runner import BacktestRunner

config = {
    "backtest": {
        "strategies": [
            {
                "name": "smacross",
                "parameters": {
                    "fast_period": 12,
                    "slow_period": 26
                }
            }
        ]
    }
}

runner = BacktestRunner(config)
results = runner.run_from_symbol("AAPL")
```

### RSI Strategy

Momentum-based strategy using the Relative Strength Index oscillator.

**Class**: `RSIStrategy`

**Parameters**:

- `period` (int, default: 14): RSI calculation period
- `oversold_threshold` (float, default: 30): Oversold level for buy signals
- `overbought_threshold` (float, default: 70): Overbought level for sell signals

**Strategy Logic**:

- **Buy Signal**: RSI falls below oversold threshold
- **Sell Signal**: RSI rises above overbought threshold

**Example Configuration**:

```yaml
backtest:
  strategies:
    - name: rsi
      parameters:
        period: 14
        oversold_threshold: 25
        overbought_threshold: 75
```

### Double EMA Crossover (DOUBLEEMACROSS)

Advanced trend strategy using exponential moving averages with volatility-based position sizing.

**Class**: `DoubleEMAStrategy`

**Parameters**:

- `fast_period` (int, default: 34): Fast EMA period (Fibonacci number)
- `slow_period` (int, default: 55): Slow EMA period (Fibonacci number)
- `momentum_atr_multiple` (float, default: 1.25): ATR multiplier for momentum trades
- `speculative_atr_multiple` (float, default: 1.0): ATR multiplier for speculative trades

**Strategy Logic**:

- Uses Fibonacci-based EMA periods for natural market harmonics
- Incorporates Average True Range (ATR) for volatility adjustment
- Differentiates between momentum and speculative trade types

**Example Configuration**:

```yaml
backtest:
  strategies:
    - name: doubleemacross
      parameters:
        fast_period: 21
        slow_period: 55
        momentum_atr_multiple: 1.5
        speculative_atr_multiple: 0.8
```

## Strategy Base Classes

### BaseStrategy

All custom strategies inherit from `BaseStrategy`.

```python
from stockula.backtesting.strategies import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    def init(self):
        """Initialize indicators and parameters."""
        pass

    def next(self):
        """Execute trading logic for each bar."""
        pass
```

**Available Data**:

- `self.data.Open`: Open prices
- `self.data.High`: High prices
- `self.data.Low`: Low prices
- `self.data.Close`: Close prices
- `self.data.Volume`: Volume data
- `self.data.df`: Complete DataFrame

**Available Methods**:

- `self.buy(size=None, limit=None, stop=None, sl=None, tp=None)`: Place buy order
- `self.sell(size=None, limit=None, stop=None, sl=None, tp=None)`: Place sell order
- `self.position`: Current position (0 if flat, >0 if long, \<0 if short)
- `self.equity`: Current account equity
- `self.I(func, *args, **kwargs)`: Create indicators

### TechnicalStrategy

Base class for strategies using technical indicators.

```python
from stockula.backtesting.strategies import TechnicalStrategy
from stockula.technical_analysis.indicators import TechnicalAnalysis

class MACDStrategy(TechnicalStrategy):
    def init(self):
        ta = TechnicalAnalysis()

        # MACD indicator
        macd_data = ta.calculate_macd(self.data.df, fast=12, slow=26, signal=9)
        self.macd = self.I(lambda: macd_data['macd'])
        self.macd_signal = self.I(lambda: macd_data['signal'])

        # RSI filter
        self.rsi = self.I(ta.calculate_rsi, self.data.Close, period=14)

    def next(self):
        # MACD crossover with RSI filter
        if (self.macd[-1] > self.macd_signal[-1] and
            self.macd[-2] <= self.macd_signal[-2] and
            self.rsi[-1] < 70 and
            not self.position):
            self.buy()
        elif (self.macd[-1] < self.macd_signal[-1] and
              self.macd[-2] >= self.macd_signal[-2] and
              self.position):
            self.sell()
```

## Custom Strategy Development

### Creating a Custom Strategy

1. **Inherit from BaseStrategy or TechnicalStrategy**
1. **Implement the `init()` method** to set up indicators
1. **Implement the `next()` method** for trading logic
1. **Register the strategy** with the StrategyRegistry

```python
from stockula.backtesting.strategies import BaseStrategy, StrategyRegistry
from stockula.technical_analysis.indicators import TechnicalAnalysis

class BollingerBandStrategy(BaseStrategy):
    """Bollinger Band mean reversion strategy."""

    def __init__(self, period=20, std_dev=2, rsi_period=14):
        self.period = period
        self.std_dev = std_dev
        self.rsi_period = rsi_period
        super().__init__()

    def init(self):
        ta = TechnicalAnalysis()

        # Bollinger Bands
        bbands = ta.calculate_bbands(
            self.data.df,
            period=self.period,
            std=self.std_dev
        )
        self.bb_upper = self.I(lambda: bbands['upper'])
        self.bb_middle = self.I(lambda: bbands['middle'])
        self.bb_lower = self.I(lambda: bbands['lower'])

        # RSI for additional confirmation
        self.rsi = self.I(ta.calculate_rsi, self.data.Close, period=self.rsi_period)

    def next(self):
        price = self.data.Close[-1]

        # Mean reversion logic
        if (price <= self.bb_lower[-1] and
            self.rsi[-1] < 30 and
            not self.position):
            self.buy()
        elif (price >= self.bb_upper[-1] and
              self.rsi[-1] > 70 and
              self.position):
            self.sell()
        elif (price >= self.bb_middle[-1] and
              self.position and
              self.position.pl_pct > 0.02):  # 2% profit
            self.sell()

# Register the strategy
StrategyRegistry.register("bollinger_bands", BollingerBandStrategy)
```

### Advanced Strategy Features

#### Position Sizing

```python
class VolatilityAdjustedStrategy(BaseStrategy):
    def init(self):
        ta = TechnicalAnalysis()
        self.atr = self.I(ta.calculate_atr, self.data.High, self.data.Low,
                         self.data.Close, period=14)
        self.sma = self.I(ta.calculate_sma, self.data.Close, period=20)

    def next(self):
        if self.data.Close[-1] > self.sma[-1] and not self.position:
            # Risk-based position sizing
            risk_per_trade = 0.02  # 2% risk per trade
            stop_distance = 2 * self.atr[-1]  # 2 ATR stop

            if stop_distance > 0:
                position_size = (self.equity * risk_per_trade) / stop_distance
                self.buy(size=position_size)
```

#### Stop Loss and Take Profit

```python
class StopLossStrategy(BaseStrategy):
    def init(self):
        ta = TechnicalAnalysis()
        self.sma = self.I(ta.calculate_sma, self.data.Close, period=20)
        self.atr = self.I(ta.calculate_atr, self.data.High, self.data.Low,
                         self.data.Close, period=14)

    def next(self):
        if self.data.Close[-1] > self.sma[-1] and not self.position:
            entry_price = self.data.Close[-1]

            # Dynamic stop loss based on ATR
            stop_loss = entry_price - (2 * self.atr[-1])
            take_profit = entry_price + (3 * self.atr[-1])

            self.buy(sl=stop_loss, tp=take_profit)
```

#### Multiple Timeframe Analysis

```python
class MultiTimeframeStrategy(BaseStrategy):
    def init(self):
        ta = TechnicalAnalysis()

        # Get daily data for trend filter
        daily_data = self.get_daily_data()
        self.daily_sma = self.I(ta.calculate_sma, daily_data.Close, period=50)

        # Hourly signals
        self.hourly_sma = self.I(ta.calculate_sma, self.data.Close, period=20)
        self.rsi = self.I(ta.calculate_rsi, self.data.Close, period=14)

    def next(self):
        # Only trade in direction of daily trend
        daily_bullish = self.data.Close[-1] > self.daily_sma[-1]
        hourly_signal = self.data.Close[-1] > self.hourly_sma[-1]
        momentum_ok = self.rsi[-1] > 50

        if daily_bullish and hourly_signal and momentum_ok and not self.position:
            self.buy()
        elif (not daily_bullish or not hourly_signal) and self.position:
            self.sell()
```

## Strategy Registry

### Registering Strategies

```python
from stockula.backtesting.strategies import StrategyRegistry

# Register a custom strategy
StrategyRegistry.register("my_strategy", MyCustomStrategy)

# Check available strategies
available = StrategyRegistry.list_strategies()
print(available)  # ['smacross', 'rsi', 'doubleemacross', 'my_strategy']

# Get strategy class
strategy_class = StrategyRegistry.get_strategy("my_strategy")
```

### Dynamic Strategy Loading

```python
def load_custom_strategies(strategy_dir):
    """Load strategies from Python files."""
    import importlib.util
    import os

    for filename in os.listdir(strategy_dir):
        if filename.endswith('.py'):
            module_name = filename[:-3]
            file_path = os.path.join(strategy_dir, filename)

            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Auto-register strategies found in module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and
                    issubclass(attr, BaseStrategy) and
                    attr != BaseStrategy):
                    StrategyRegistry.register(module_name, attr)
```

## Strategy Optimization

### Parameter Optimization

```python
from itertools import product

def optimize_strategy_parameters(symbol, strategy_class, param_ranges,
                               start_date, end_date):
    """Optimize strategy parameters using grid search."""

    best_return = -float('inf')
    best_params = None
    results = []

    # Generate all parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())

    for combination in product(*param_values):
        params = dict(zip(param_names, combination))

        try:
            # Create strategy with parameters
            strategy = strategy_class(**params)

            # Run backtest
            runner = BacktestRunner()
            result = runner.run_strategy(symbol, strategy, start_date, end_date)

            # Track results
            results.append({
                'params': params,
                'return': result['Return [%]'],
                'sharpe': result['Sharpe Ratio'],
                'max_drawdown': result['Max. Drawdown [%]'],
                'trades': result['# Trades']
            })

            # Update best
            if result['Return [%]'] > best_return:
                best_return = result['Return [%]']
                best_params = params

        except Exception as e:
            print(f"Error with params {params}: {e}")

    return best_params, best_return, results

# Example usage
param_ranges = {
    'fast_period': range(5, 21),
    'slow_period': range(20, 51),
}

best_params, best_return, all_results = optimize_strategy_parameters(
    'AAPL', SMAStrategy, param_ranges, '2020-01-01', '2023-01-01'
)

print(f"Best parameters: {best_params}")
print(f"Best return: {best_return:.2f}%")
```

### Walk-Forward Optimization

```python
def walk_forward_optimization(symbol, strategy_class, param_ranges,
                            lookback_months=12, forward_months=3):
    """Perform walk-forward optimization."""

    data = fetcher.get_stock_data(symbol, start_date='2018-01-01')
    results = []

    # Define optimization and testing periods
    start_date = data.index[0]
    end_date = data.index[-1]

    current_date = start_date + pd.DateOffset(months=lookback_months)

    while current_date + pd.DateOffset(months=forward_months) <= end_date:
        # Optimization period
        opt_start = current_date - pd.DateOffset(months=lookback_months)
        opt_end = current_date

        # Testing period
        test_start = current_date
        test_end = current_date + pd.DateOffset(months=forward_months)

        # Optimize on historical data
        best_params, _, _ = optimize_strategy_parameters(
            symbol, strategy_class, param_ranges, opt_start, opt_end
        )

        # Test on forward period
        strategy = strategy_class(**best_params)
        runner = BacktestRunner()
        test_result = runner.run_strategy(symbol, strategy, test_start, test_end)

        results.append({
            'test_period': f"{test_start.date()} to {test_end.date()}",
            'optimized_params': best_params,
            'forward_return': test_result['Return [%]'],
            'forward_sharpe': test_result['Sharpe Ratio']
        })

        current_date += pd.DateOffset(months=forward_months)

    return results
```

## Performance Metrics

### Built-in Metrics

All strategies automatically calculate these metrics:

- **Return [%]**: Total return percentage
- **Sharpe Ratio**: Risk-adjusted return
- **Max. Drawdown [%]**: Largest peak-to-trough decline
- **Volatility [%]**: Annualized volatility
- **# Trades**: Number of round-trip trades
- **Win Rate [%]**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss

### Custom Metrics

```python
def calculate_custom_metrics(strategy_results):
    """Calculate additional performance metrics."""
    trades = strategy_results._trades

    # Calmar Ratio
    annual_return = strategy_results.Return
    max_drawdown = abs(strategy_results['Max. Drawdown [%]'])
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

    # Sortino Ratio
    returns = strategy_results._equity_curve.pct_change().dropna()
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino_ratio = annual_return / downside_std if downside_std > 0 else 0

    # Average Trade Duration
    if len(trades) > 0:
        durations = [(trade.ExitTime - trade.EntryTime).days for trade in trades]
        avg_duration = np.mean(durations)
    else:
        avg_duration = 0

    return {
        'Calmar Ratio': calmar_ratio,
        'Sortino Ratio': sortino_ratio,
        'Avg Trade Duration': avg_duration
    }
```

## Best Practices

### Strategy Development

1. **Start Simple**: Begin with basic logic before adding complexity
1. **Use Proper Indicators**: Leverage the technical analysis module
1. **Include Risk Management**: Always implement position sizing and stops
1. **Validate Logic**: Test edge cases and unusual market conditions

### Parameter Selection

1. **Avoid Overfitting**: Don't optimize on limited data
1. **Use Robust Ranges**: Test wide parameter ranges
1. **Cross-Validate**: Use walk-forward or cross-validation
1. **Consider Market Regimes**: Parameters may work differently in different markets

### Testing and Validation

1. **Out-of-Sample Testing**: Reserve data for final validation
1. **Multiple Markets**: Test across different instruments
1. **Multiple Time Periods**: Include various market conditions
1. **Transaction Costs**: Always include realistic costs

### Production Considerations

1. **Error Handling**: Handle data issues gracefully
1. **Performance Monitoring**: Track live performance vs. backtest
1. **Regular Reoptimization**: Update parameters periodically
1. **Risk Limits**: Implement portfolio-level risk controls

The strategies API provides a flexible framework for developing, testing, and optimizing trading strategies while maintaining clean separation between strategy logic and execution infrastructure.
