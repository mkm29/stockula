# Backtesting

Stockula provides a comprehensive backtesting framework that allows you to test trading strategies against historical data with realistic broker costs and constraints.

## Overview

The backtesting module offers:

- **Pre-built Strategies**: SMA Cross, RSI, Double EMA, and more
- **Custom Strategies**: Easy framework for implementing your own strategies
- **Realistic Costs**: Multiple broker fee structures and commission models
- **Rich Reporting**: Detailed performance metrics and visualizations
- **Portfolio Testing**: Test strategies across multiple assets
- **Risk Management**: Built-in position sizing and stop-loss features

## Available Strategies

### Simple Moving Average Crossover (SMACROSS)

Classic trend-following strategy using two moving averages.

**Parameters**:

- `fast_period` (default: 10): Short-term moving average period
- `slow_period` (default: 20): Long-term moving average period

**Logic**:

- **Buy**: When fast MA crosses above slow MA
- **Sell**: When fast MA crosses below slow MA

```yaml
backtest:
  strategies:
    - name: smacross
      parameters:
        fast_period: 10
        slow_period: 20
```

### RSI Strategy

Momentum-based strategy using Relative Strength Index.

**Parameters**:

- `period` (default: 14): RSI calculation period
- `oversold_threshold` (default: 30): Buy signal threshold
- `overbought_threshold` (default: 70): Sell signal threshold

**Logic**:

- **Buy**: When RSI falls below oversold threshold
- **Sell**: When RSI rises above overbought threshold

```yaml
backtest:
  strategies:
    - name: rsi
      parameters:
        period: 14
        oversold_threshold: 30
        overbought_threshold: 70
```

### Double EMA Crossover (DOUBLEEMACROSS)

Advanced trend strategy with momentum and volatility filtering.

**Parameters**:

- `fast_period` (default: 34): Fast EMA period
- `slow_period` (default: 55): Slow EMA period
- `momentum_atr_multiple` (default: 1.25): ATR multiplier for momentum trades
- `speculative_atr_multiple` (default: 1.0): ATR multiplier for speculative trades

**Logic**:

- Uses Fibonacci-based EMA periods (34, 55)
- Incorporates ATR for volatility-adjusted position sizing
- Differentiates between momentum and speculative trades

```yaml
backtest:
  strategies:
    - name: doubleemacross
      parameters:
        fast_period: 34
        slow_period: 55
        momentum_atr_multiple: 1.25
        speculative_atr_multiple: 1.0
```

## Configuration

### Basic Backtesting Setup

```yaml
data:
  start_date: "2023-01-01"
  end_date: "2024-01-01"

backtest:
  initial_cash: 10000.0
  strategies:
    - name: smacross
      parameters:
        fast_period: 10
        slow_period: 20
```

### Train/Test Split Configuration

Stockula supports train/test split for backtesting, allowing you to optimize strategies on training data and validate performance on out-of-sample test data:

```yaml
data:
  # Training period
  train_start_date: "2023-01-01"
  train_end_date: "2023-12-31"

  # Testing period (out-of-sample)
  test_start_date: "2024-01-01"
  test_end_date: "2024-06-30"

backtest:
  initial_cash: 20000.0
  optimize: true  # Enable parameter optimization on training data
  optimization_params:
    fast_period: [5, 10, 15]
    slow_period: [20, 30, 40]
  strategies:
    - name: smacross
      parameters:
        fast_period: 10
        slow_period: 20
```

When train/test split is configured:

- Strategy parameters are optimized on the training period
- Performance is evaluated on the test period
- Both train and test results are displayed for comparison
- Performance degradation between periods is calculated

### Advanced Configuration

```yaml
backtest:
  initial_cash: 100000.0
  margin: 1.0                    # No margin trading
  exclusive_orders: true         # One position at a time
  trade_on_close: false         # Trade on next open
  hedging: false                # No hedging allowed

  # Broker configuration
  broker_config:
    name: "robinhood"           # Use preset broker

  # Multiple strategies
  strategies:
    - name: smacross
      parameters:
        fast_period: 10
        slow_period: 20
    - name: rsi
      parameters:
        period: 14
        oversold_threshold: 25    # More aggressive entry
        overbought_threshold: 75  # More aggressive exit
```

## Broker Configuration

### Preset Brokers

Stockula includes realistic fee structures for major brokers:

```yaml
backtest:
  broker_config:
    name: "robinhood"           # Zero commission + TAF
    # name: "interactive_brokers" # $0.005/share, $1 minimum
    # name: "td_ameritrade"       # $0 commission + SEC fees
    # name: "etrade"              # $0 commission + SEC fees
    # name: "fidelity"            # $0 commission + SEC fees
    # name: "schwab"              # $0 commission + SEC fees
```

### Custom Broker Configuration

```yaml
backtest:
  broker_config:
    name: "custom"
    commission_type: "per_share"  # or "percentage", "fixed"
    commission_value: 0.005       # $0.005 per share
    min_commission: 1.0           # $1 minimum
    max_commission: 10.0          # $10 maximum
    regulatory_fees: 0.0000229    # SEC fee (0.00229%)
    exchange_fees: 0.003          # Exchange fees
    taf_fee: 0.000119            # Trade Activity Fee
```

### Commission Types

| Type         | Description               | Example         |
| ------------ | ------------------------- | --------------- |
| `per_share`  | Fixed amount per share    | $0.005/share    |
| `percentage` | Percentage of trade value | 0.1% of trade   |
| `fixed`      | Fixed amount per trade    | $1.00 per trade |

## Usage Examples

### Command Line Usage

```bash
# Basic backtesting with default strategy
uv run python -m stockula.main --ticker AAPL --mode backtest

# With custom configuration
uv run python -m stockula.main --config examples/config.strategies.yaml --mode backtest

# Portfolio backtesting
uv run python -m stockula.main --config myconfig.yaml --mode backtest
```

### Programmatic Usage

```python
from stockula.backtesting.runner import BacktestRunner
from stockula.config.settings import load_config
from stockula.data.fetcher import DataFetcher

# Load configuration
config = load_config("myconfig.yaml")

# Create backtest runner
runner = BacktestRunner(config)

# Run single ticker backtest
results = runner.run_from_symbol("AAPL")

# Run portfolio backtest
portfolio_results = runner.run_portfolio_backtest()

print(f"Return: {results['Return [%]']:.2f}%")
print(f"Sharpe Ratio: {results['Sharpe Ratio']:.2f}")
print(f"Max Drawdown: {results['Max. Drawdown [%]']:.2f}%")
```

## Rich CLI Output

### Backtesting Results Display

The backtesting output begins with general portfolio information, followed by detailed results for each strategy and asset:

#### Portfolio Information Section

```
=== Backtesting Results ===

Portfolio Information:
  Initial Capital: $10,000
  Start Date: 2023-01-01
  End Date: 2023-12-31
  Trading Days: 252
  Calendar Days: 365
```

#### Standard Backtesting Results Table

```
                         Backtesting Results
┏━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Ticker ┃ Strategy  ┃ Return     ┃ Sharpe Ratio   ┃ Max Drawdown   ┃
┡━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ AAPL   │ SMACROSS  │ +15.50%    │ 1.25           │ -8.30%         │
│ GOOGL  │ SMACROSS  │ +8.75%     │ 0.98           │ -12.45%        │
│ MSFT   │ SMACROSS  │ -2.15%     │ -0.15          │ -18.90%        │
│ NVDA   │ RSI       │ +22.40%    │ 1.85           │ -15.20%        │
│ TSLA   │ RSI       │ +5.30%     │ 0.65           │ -25.80%        │
└────────┴───────────┴────────────┴────────────────┴────────────────┘
```

#### Train/Test Split Results Table

When using train/test split configuration, the output shows both training and testing performance:

```
              Ticker-Level Backtest Results (Train/Test Split)
┏━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Ticker ┃ Strategy  ┃ Train Return ┃ Test Return ┃ Train Sharpe ┃ Test Sharpe ┃
┡━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ AAPL   │ SMACROSS  │ +18.50%      │ +12.30%     │ 1.45         │ 1.10        │
│ GOOGL  │ SMACROSS  │ +10.25%      │ +7.80%      │ 1.20         │ 0.95        │
│ MSFT   │ SMACROSS  │ +5.75%       │ -3.20%      │ 0.80         │ -0.25       │
│ NVDA   │ SMACROSS  │ +25.40%      │ +19.80%     │ 2.10         │ 1.75        │
│ TSLA   │ SMACROSS  │ +8.90%       │ +3.50%      │ 0.95         │ 0.55        │
└────────┴───────────┴──────────────┴─────────────┴──────────────┴─────────────┘

Data Periods:
  Training: 2023-01-01 to 2023-12-31 (252 days)
  Testing:  2024-01-01 to 2024-06-30 (126 days)
```

### Strategy Summary Panel

Strategy summaries are displayed in descending order by "Return During Period" (highest returns first):

```
╭───────────────────────────── STRATEGY: SMACROSS ─────────────────────────────╮
│                                                                              │
│  Parameters: {'fast_period': 10, 'slow_period': 20}                          │
│  Broker: robinhood (Zero commission + TAF)                                   │
│                                                                              │
│  Portfolio Value at Start Date: $10,000.00                                   │
│  Portfolio Value at End (Backtest): $11,550.00                               │
│                                                                              │
│  Strategy Performance:                                                       │
│    Average Return: +15.50%                                                   │
│    Winning Stocks: 3                                                         │
│    Losing Stocks: 1                                                          │
│    Total Trades: 45                                                          │
│                                                                              │
│  Return During Period: $1,550.00 (+15.50%)                                   │
│                                                                              │
│  Detailed report saved to:                                                   │
│  results/reports/strategy_report_smacross_20250127_143022.json               │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### Progress Tracking

```
⠋ Backtesting SMACROSS on AAPL... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 85% 0:00:02
⠋ Computing performance metrics... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:01
✓ Backtesting completed for AAPL
```

## Performance Metrics

### Core Metrics

- **Return [%]**: Total return percentage
- **Sharpe Ratio**: Risk-adjusted return measure
- **Max. Drawdown [%]**: Largest peak-to-trough decline
- **Volatility [%]**: Annualized volatility
- **# Trades**: Total number of round-trip trades

### Additional Metrics

- **Win Rate [%]**: Percentage of profitable trades
- **Best Trade [%]**: Best single trade return
- **Worst Trade [%]**: Worst single trade return
- **Avg. Trade Duration**: Average holding period
- **Profit Factor**: Gross profit / Gross loss ratio

### Risk Metrics

- **Calmar Ratio**: Annual return / Max drawdown
- **Sortino Ratio**: Downside risk-adjusted return
- **Beta**: Market correlation
- **Alpha**: Excess return vs. benchmark

## Custom Strategy Development

### Creating a Custom Strategy

```python
from stockula.backtesting.strategies import BaseStrategy
from stockula.technical_analysis.indicators import TechnicalAnalysis

class MACDStrategy(BaseStrategy):
    """MACD crossover strategy with RSI filter."""

    def init(self):
        """Initialize indicators."""
        ta = TechnicalAnalysis()

        # MACD indicator
        macd_data = ta.calculate_macd(self.data.df, fast=12, slow=26, signal=9)
        self.macd = self.I(lambda: macd_data['macd'])
        self.signal = self.I(lambda: macd_data['signal'])

        # RSI filter
        self.rsi = self.I(ta.calculate_rsi, self.data.Close, period=14)

    def next(self):
        """Execute trading logic."""
        # Entry conditions
        macd_bullish = self.macd[-1] > self.signal[-1]
        macd_cross = self.macd[-2] <= self.signal[-2]
        rsi_filter = self.rsi[-1] < 70  # Not overbought

        # Exit conditions
        macd_bearish = self.macd[-1] < self.signal[-1]
        macd_cross_down = self.macd[-2] >= self.signal[-2]

        # Trading logic
        if macd_bullish and macd_cross and rsi_filter and not self.position:
            self.buy()
        elif macd_bearish and macd_cross_down and self.position:
            self.sell()
```

### Registering Custom Strategies

```python
from stockula.backtesting.strategies import StrategyRegistry

# Register the custom strategy
StrategyRegistry.register("macd_rsi", MACDStrategy)

# Use in configuration
config = {
    "backtest": {
        "strategies": [
            {
                "name": "macd_rsi",
                "parameters": {}
            }
        ]
    }
}
```

## Advanced Features

### Position Sizing

```python
class PositionSizedStrategy(BaseStrategy):
    def init(self):
        self.atr = self.I(ta.calculate_atr, self.data.High, self.data.Low,
                         self.data.Close, period=14)

    def next(self):
        if self.should_buy():
            # Risk-based position sizing
            risk_per_trade = 0.02  # 2% risk per trade
            stop_distance = 2 * self.atr[-1]  # 2 ATR stop
            position_size = (self.equity * risk_per_trade) / stop_distance

            self.buy(size=position_size)
```

### Stop Loss and Take Profit

```python
class StopLossStrategy(BaseStrategy):
    def next(self):
        if self.should_buy():
            entry_price = self.data.Close[-1]
            stop_loss = entry_price * 0.95    # 5% stop loss
            take_profit = entry_price * 1.15  # 15% take profit

            self.buy(sl=stop_loss, tp=take_profit)
```

### Multiple Timeframe Analysis

```python
class MultiTimeframeStrategy(BaseStrategy):
    def init(self):
        # Daily trend filter
        daily_data = self.get_daily_data()
        self.daily_trend = self.I(ta.calculate_sma, daily_data.Close, period=50)

        # Hourly signals
        self.hourly_ma = self.I(ta.calculate_sma, self.data.Close, period=20)

    def next(self):
        # Only trade in direction of daily trend
        daily_bullish = self.data.Close[-1] > self.daily_trend[-1]
        hourly_signal = self.data.Close[-1] > self.hourly_ma[-1]

        if daily_bullish and hourly_signal:
            self.buy()
```

## Strategy Optimization

### Parameter Optimization

```python
from itertools import product

def optimize_strategy(symbol, strategy_class, param_ranges):
    """Optimize strategy parameters."""
    best_return = -float('inf')
    best_params = None

    # Generate parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())

    for combination in product(*param_values):
        params = dict(zip(param_names, combination))

        # Run backtest with these parameters
        strategy = strategy_class(**params)
        results = runner.run_strategy(symbol, strategy)

        if results['Return [%]'] > best_return:
            best_return = results['Return [%]']
            best_params = params

    return best_params, best_return

# Example usage
param_ranges = {
    'fast_period': range(5, 21),
    'slow_period': range(20, 51),
}

best_params, best_return = optimize_strategy('AAPL', SMAStrategy, param_ranges)
```

### Walk-Forward Analysis

```python
def walk_forward_analysis(symbol, strategy_class, lookback_days=252):
    """Perform walk-forward analysis."""
    data = fetcher.get_stock_data(symbol, start_date="2020-01-01")
    results = []

    for i in range(lookback_days, len(data), 63):  # Quarterly rebalancing
        # Optimization period
        opt_data = data.iloc[i-lookback_days:i]
        best_params = optimize_on_data(opt_data, strategy_class)

        # Testing period
        test_data = data.iloc[i:i+63]
        test_results = backtest_with_params(test_data, strategy_class, best_params)

        results.append({
            'period': data.index[i],
            'params': best_params,
            'return': test_results['Return [%]']
        })

    return results
```

## Integration with Portfolio Analysis

### Portfolio-Level Backtesting

```python
from stockula.domain.factory import DomainFactory

def backtest_portfolio(config):
    """Backtest entire portfolio."""
    factory = DomainFactory(config)
    portfolio = factory.create_portfolio()

    portfolio_results = {}
    total_value = 0

    for asset in portfolio.assets:
        # Run strategy on each asset
        results = runner.run_from_symbol(asset.ticker)
        portfolio_results[asset.ticker] = results

        # Calculate weighted contribution
        weight = asset.allocation_amount / portfolio.total_value
        contribution = results['Return [%]'] * weight
        total_value += contribution

    portfolio_results['portfolio_return'] = total_value
    return portfolio_results
```

### Correlation Analysis

```python
def analyze_strategy_correlation(tickers, strategy):
    """Analyze correlation between strategy returns."""
    returns = {}

    for ticker in tickers:
        results = runner.run_from_symbol(ticker, strategy=strategy)
        returns[ticker] = results['Return [%]']

    return_series = pd.Series(returns)
    correlation_matrix = return_series.corr()

    return correlation_matrix
```

## Best Practices

### Strategy Development

1. **Start Simple**: Begin with basic strategies before adding complexity
1. **Use Multiple Timeframes**: Combine different timeframes for better signals
1. **Include Filters**: Use additional indicators to filter false signals
1. **Risk Management**: Always include position sizing and stop losses

### Testing

1. **Out-of-Sample Testing**: Reserve data for final validation
1. **Walk-Forward Analysis**: Test strategy adaptability over time
1. **Multiple Markets**: Test across different market conditions
1. **Transaction Costs**: Always include realistic broker costs

### Optimization

1. **Avoid Overfitting**: Don't optimize on limited data
1. **Robust Parameters**: Choose parameter ranges that work across periods
1. **Cross-Validation**: Use multiple validation techniques
1. **Monitor Performance**: Track strategy performance in live conditions

The backtesting framework provides a comprehensive environment for developing, testing, and optimizing trading strategies with realistic market conditions and costs.

## See Also

- [Allocation Strategies](allocation-strategies.md) - Learn about BacktestOptimizedAllocator which uses backtest results to optimize portfolio allocation
- [Strategies API Reference](../api/strategies.md) - Detailed documentation of all available trading strategies
