# Technical Analysis

Stockula provides comprehensive technical analysis capabilities through the finta library wrapper, offering over 40 technical indicators with clean, consistent interfaces.

## Overview

The technical analysis module offers:

- **40+ Indicators**: Moving averages, oscillators, volatility, volume indicators
- **Batch Processing**: Calculate multiple indicators simultaneously
- **Rich Display**: Beautiful tables with formatted results
- **Flexible Configuration**: Customizable parameters for all indicators
- **Performance Optimized**: Vectorized calculations using pandas

## Available Indicators

### Moving Averages

| Indicator | Description                       | Default Period |
| --------- | --------------------------------- | -------------- |
| **SMA**   | Simple Moving Average             | 20             |
| **EMA**   | Exponential Moving Average        | 12, 26         |
| **WMA**   | Weighted Moving Average           | 20             |
| **DEMA**  | Double Exponential Moving Average | 21             |
| **TEMA**  | Triple Exponential Moving Average | 21             |
| **TRIMA** | Triangular Moving Average         | 20             |

### Oscillators

| Indicator    | Description             | Parameters              |
| ------------ | ----------------------- | ----------------------- |
| **RSI**      | Relative Strength Index | period=14               |
| **STOCH**    | Stochastic Oscillator   | k_period=14, d_period=3 |
| **WILLIAMS** | Williams %R             | period=14               |
| **CCI**      | Commodity Channel Index | period=20               |
| **MFI**      | Money Flow Index        | period=14               |

### Trend Indicators

| Indicator | Description                           | Parameters                 |
| --------- | ------------------------------------- | -------------------------- |
| **MACD**  | Moving Average Convergence Divergence | fast=12, slow=26, signal=9 |
| **ADX**   | Average Directional Index             | period=14                  |
| **AROON** | Aroon Oscillator                      | period=25                  |
| **PSAR**  | Parabolic SAR                         | af=0.02, max_af=0.2        |

### Volatility Indicators

| Indicator  | Description                   | Parameters       |
| ---------- | ----------------------------- | ---------------- |
| **BBANDS** | Bollinger Bands               | period=20, std=2 |
| **ATR**    | Average True Range            | period=14        |
| **NATR**   | Normalized Average True Range | period=14        |
| **TRANGE** | True Range                    | -                |

### Volume Indicators

| Indicator | Description               | Parameters |
| --------- | ------------------------- | ---------- |
| **OBV**   | On-Balance Volume         | -          |
| **AD**    | Accumulation/Distribution | -          |
| **CMF**   | Chaikin Money Flow        | period=20  |
| **EMV**   | Ease of Movement          | period=14  |

## Configuration

### Basic Configuration

```yaml
technical_analysis:
  indicators: [sma, ema, rsi, macd, bbands, atr]
  sma_periods: [20, 50, 200]
  ema_periods: [12, 26]
  rsi_period: 14
```

### Advanced Configuration

```yaml
technical_analysis:
  # Core indicators
  indicators: [sma, ema, rsi, macd, bbands, atr, adx, stoch, williams, cci]

  # Moving averages
  sma_periods: [10, 20, 50, 100, 200]
  ema_periods: [8, 12, 21, 26, 50]
  wma_periods: [20, 50]

  # Oscillators
  rsi_period: 14
  stoch_params:
    k_period: 14
    d_period: 3
    smooth_k: 3
  williams_period: 14
  cci_period: 20
  mfi_period: 14

  # Trend indicators
  macd_params:
    period_fast: 12
    period_slow: 26
    signal: 9
  adx_period: 14
  aroon_period: 25
  psar_params:
    af: 0.02
    max_af: 0.2

  # Volatility
  bbands_params:
    period: 20
    std: 2
  atr_period: 14

  # Volume
  cmf_period: 20
  emv_period: 14
```

## Usage Examples

### Command Line Usage

```bash
# Basic technical analysis
uv run python -m stockula.main --ticker AAPL --mode ta

# With custom configuration
uv run python -m stockula.main --config examples/config.technical.yaml --mode ta

# Multiple tickers
uv run python -m stockula.main --config myconfig.yaml --mode ta
```

### Programmatic Usage

```python
from stockula.technical_analysis.indicators import TechnicalAnalysis
from stockula.data.fetcher import DataFetcher

# Get stock data
fetcher = DataFetcher()
data = fetcher.get_stock_data("AAPL", start_date="2023-01-01")

# Create technical analysis instance
ta = TechnicalAnalysis()

# Calculate specific indicators
sma_20 = ta.calculate_sma(data, period=20)
rsi_14 = ta.calculate_rsi(data, period=14)
macd = ta.calculate_macd(data, fast=12, slow=26, signal=9)

# Calculate multiple indicators at once
indicators = ta.calculate_indicators(
    data,
    indicators=["sma", "ema", "rsi", "macd", "bbands"],
    sma_periods=[20, 50],
    ema_periods=[12, 26]
)

print(indicators)
```

### Batch Analysis

```python
from stockula.config.settings import load_config
from stockula.domain.factory import DomainFactory
from stockula.technical_analysis.indicators import TechnicalAnalysis

# Load configuration
config = load_config("myconfig.yaml")

# Create portfolio
factory = DomainFactory(config)
portfolio = factory.create_portfolio()

# Run technical analysis on entire portfolio
ta = TechnicalAnalysis()
results = {}

for asset in portfolio.assets:
    ticker_data = fetcher.get_stock_data(asset.ticker)
    results[asset.ticker] = ta.calculate_indicators(
        ticker_data,
        indicators=config.technical_analysis.indicators
    )
```

## Rich CLI Output

### Technical Analysis Results Table

```
                    Technical Analysis Results
┏━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Ticker ┃ SMA_20        ┃ RSI_14         ┃ MACD             ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ AAPL   │ $150.25       │ 65.4           │ 2.15             │
│ GOOGL  │ $2,750.80     │ 58.2           │ -1.25            │
│ MSFT   │ $405.60       │ 72.1           │ 3.40             │
│ NVDA   │ $875.40       │ 45.8           │ -5.20            │
│ TSLA   │ $248.50       │ 38.9           │ -8.75            │
└────────┴───────────────┴────────────────┴──────────────────┘
```

### Color-Coded Values

- **Green**: Bullish signals (RSI < 30, positive MACD)
- **Red**: Bearish signals (RSI > 70, negative MACD)
- **Yellow**: Neutral zones
- **Blue**: Price levels and moving averages

### Progress Tracking

```
⠋ Computing SMA(20) for AAPL... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 60% 0:00:01
⠋ Computing RSI(14) for AAPL... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 80% 0:00:01
⠋ Computing MACD for AAPL... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:01
✓ Technical analysis completed for AAPL
```

## Indicator Details

### Simple Moving Average (SMA)

Calculates the arithmetic mean of prices over a specified period.

```python
# Single period
sma_20 = ta.calculate_sma(data, period=20)

# Multiple periods
sma_multiple = ta.calculate_sma(data, periods=[10, 20, 50, 200])
```

**Interpretation**:

- Price above SMA: Bullish trend
- Price below SMA: Bearish trend
- SMA crossovers: Trend change signals

### Exponential Moving Average (EMA)

Gives more weight to recent prices, making it more responsive to price changes.

```python
ema_12 = ta.calculate_ema(data, period=12)
ema_26 = ta.calculate_ema(data, period=26)
```

**Interpretation**:

- More sensitive to recent price movements than SMA
- Faster signal generation
- Less lag in trend identification

### Relative Strength Index (RSI)

Momentum oscillator measuring speed and magnitude of price changes.

```python
rsi = ta.calculate_rsi(data, period=14)
```

**Interpretation**:

- **RSI > 70**: Potentially overbought (sell signal)
- **RSI < 30**: Potentially oversold (buy signal)
- **RSI 30-70**: Normal trading range

### MACD (Moving Average Convergence Divergence)

Shows relationship between two moving averages of a security's price.

```python
macd = ta.calculate_macd(data, fast=12, slow=26, signal=9)
# Returns: MACD line, Signal line, Histogram
```

**Interpretation**:

- **MACD > Signal**: Bullish momentum
- **MACD < Signal**: Bearish momentum
- **Zero crossover**: Trend change indication

### Bollinger Bands

Volatility indicator with upper and lower bands around a moving average.

```python
bbands = ta.calculate_bbands(data, period=20, std=2)
# Returns: Upper band, Middle band (SMA), Lower band
```

**Interpretation**:

- **Price near upper band**: Potentially overbought
- **Price near lower band**: Potentially oversold
- **Band squeeze**: Low volatility, potential breakout

### Average True Range (ATR)

Measures market volatility by calculating the average of true ranges.

```python
atr = ta.calculate_atr(data, period=14)
```

**Interpretation**:

- **High ATR**: High volatility
- **Low ATR**: Low volatility
- Useful for position sizing and stop-loss placement

## Advanced Features

### Custom Indicator Parameters

```python
# Customized RSI with different period
rsi_21 = ta.calculate_rsi(data, period=21)

# MACD with custom settings
macd_custom = ta.calculate_macd(data, fast=8, slow=21, signal=5)

# Bollinger Bands with wider bands
bbands_wide = ta.calculate_bbands(data, period=20, std=2.5)
```

### Signal Generation

```python
def generate_signals(data, indicators):
    """Generate buy/sell signals from indicators."""
    signals = pd.DataFrame(index=data.index)

    # RSI signals
    signals['rsi_buy'] = indicators['rsi'] < 30
    signals['rsi_sell'] = indicators['rsi'] > 70

    # MACD signals
    signals['macd_buy'] = (indicators['macd'] > indicators['macd_signal']) & \
                         (indicators['macd'].shift(1) <= indicators['macd_signal'].shift(1))
    signals['macd_sell'] = (indicators['macd'] < indicators['macd_signal']) & \
                          (indicators['macd'].shift(1) >= indicators['macd_signal'].shift(1))

    # Combined signals
    signals['buy'] = signals['rsi_buy'] & signals['macd_buy']
    signals['sell'] = signals['rsi_sell'] & signals['macd_sell']

    return signals
```

### Screening and Filtering

```python
def screen_stocks(tickers, criteria):
    """Screen stocks based on technical criteria."""
    results = []

    for ticker in tickers:
        data = fetcher.get_stock_data(ticker)
        indicators = ta.calculate_indicators(data, indicators=["rsi", "macd", "sma"])

        latest = indicators.iloc[-1]

        # Apply screening criteria
        if (latest['rsi'] < criteria['max_rsi'] and
            latest['macd'] > 0 and
            latest['close'] > latest['sma_50']):
            results.append({
                'ticker': ticker,
                'rsi': latest['rsi'],
                'macd': latest['macd'],
                'price_vs_sma': (latest['close'] - latest['sma_50']) / latest['sma_50']
            })

    return results
```

## Integration with Backtesting

Technical indicators integrate seamlessly with backtesting strategies:

```python
from stockula.backtesting.strategies import TechnicalStrategy

class RSIStrategy(TechnicalStrategy):
    def init(self):
        self.rsi = self.I(ta.calculate_rsi, self.data.Close, period=14)

    def next(self):
        if self.rsi[-1] < 30 and not self.position:
            self.buy()
        elif self.rsi[-1] > 70 and self.position:
            self.sell()
```

## Performance Considerations

### Vectorized Calculations

All indicators use vectorized pandas operations for optimal performance:

```python
# Efficient: Vectorized calculation
sma = data['close'].rolling(window=20).mean()

# Inefficient: Loop-based calculation
sma = []
for i in range(len(data)):
    if i >= 19:
        sma.append(data['close'].iloc[i-19:i+1].mean())
    else:
        sma.append(np.nan)
```

### Memory Management

For large datasets, consider chunking:

```python
def calculate_indicators_chunked(data, chunk_size=1000):
    """Calculate indicators in chunks for large datasets."""
    results = []

    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size]
        chunk_indicators = ta.calculate_indicators(chunk, indicators=["sma", "rsi"])
        results.append(chunk_indicators)

    return pd.concat(results)
```

## Best Practices

### Indicator Selection

1. **Choose complementary indicators**: Mix trend, momentum, and volatility indicators
1. **Avoid redundancy**: Don't use multiple similar indicators (e.g., multiple MAs)
1. **Consider timeframe**: Match indicator periods to your trading timeframe
1. **Test effectiveness**: Backtest indicator combinations

### Parameter Optimization

1. **Use standard periods initially**: 14 for RSI, 20 for SMA, etc.
1. **Optimize systematically**: Test parameter ranges methodically
1. **Avoid overfitting**: Don't optimize on limited data
1. **Consider market conditions**: Parameters may need adjustment for different markets

### Signal Interpretation

1. **Use multiple confirmations**: Don't rely on single indicator signals
1. **Consider market context**: Indicators work differently in trending vs. ranging markets
1. **Filter false signals**: Use additional criteria to reduce noise
1. **Monitor performance**: Track indicator effectiveness over time

The technical analysis module provides a comprehensive toolkit for market analysis while maintaining simplicity and performance through its clean interface and Rich integration.
