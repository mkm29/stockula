# Strategies API Reference

This reference covers all available trading strategies, their parameters, and how to configure them in Stockula.

## Strategy Registry

All strategies in Stockula are managed through the centralized `StrategyRegistry` class, which provides:

- **Automatic Name Normalization**: Accepts both PascalCase (`SMACross`) and snake_case (`smacross`) formats
- **Strategy Class Resolution**: Maps strategy names to their implementation classes
- **Parameter Management**: Provides default parameters for each strategy
- **Group Organization**: Organizes strategies into logical groups for different trading approaches

The registry ensures consistency across the application and handles configuration compatibility between different naming formats.

## Available Strategies

Stockula provides 11 built-in trading strategies that can be used for backtesting. All strategies are internally normalized to lowercase, snake_case names, but configuration files can use either PascalCase or snake_case formats.

### Simple Moving Average Crossover (`smacross`)

A classic trend-following strategy using two simple moving averages.

**Class**: `SMACrossStrategy`

**Parameters**:

- `fast_period` (int, default: 10): Period for fast moving average
- `slow_period` (int, default: 20): Period for slow moving average

**Strategy Logic**:

- **Buy Signal**: Fast SMA crosses above slow SMA
- **Sell Signal**: Fast SMA crosses below slow SMA

**Example Configuration**:

```yaml
backtest:
  strategies:
    - name: smacross
      parameters:
        fast_period: 10
        slow_period: 20
```

### RSI Strategy (`rsi`)

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
        oversold_threshold: 30.0
        overbought_threshold: 70.0
```

### MACD Strategy (`macd`)

Uses Moving Average Convergence Divergence for trend and momentum signals.

**Class**: `MACDStrategy`

**Parameters**:

- `fast_period` (int, default: 12): Fast EMA period
- `slow_period` (int, default: 26): Slow EMA period
- `signal_period` (int, default: 9): Signal line EMA period

**Strategy Logic**:

- **Buy Signal**: MACD line crosses above signal line
- **Sell Signal**: MACD line crosses below signal line

**Example Configuration**:

```yaml
backtest:
  strategies:
    - name: macd
      parameters:
        fast_period: 12
        slow_period: 26
        signal_period: 9
```

### Double EMA Crossover (`double_ema_cross`)

Advanced trend strategy using exponential moving averages with ATR-based stop losses.

**Class**: `DoubleEMACrossStrategy`

**Parameters**:

- `fast_period` (int, default: 34): Fast EMA period
- `slow_period` (int, default: 55): Slow EMA period
- `momentum_atr_multiple` (float, default: 1.25): ATR multiplier for momentum positions
- `speculative_atr_multiple` (float, default: 1.0): ATR multiplier for speculative positions
- `atr_period` (int, default: 14): ATR calculation period

**Strategy Logic**:

- **Buy Signal**: Fast EMA crosses above slow EMA
- **Sell Signal**: Fast EMA crosses below slow EMA
- **Stop Loss**: ATR-based stop loss with different multipliers for different asset classes

**Example Configuration**:

```yaml
backtest:
  strategies:
    - name: double_ema_cross
      parameters: {}  # Uses default parameters
```

### Triple EMA Crossover (`triple_ema_cross`)

Uses Triple Exponential Moving Average (TEMA) to reduce lag in trend following.

**Class**: `TripleEMACrossStrategy`

**Parameters**:

- `fast` (int, default: 5): Fast TEMA parameter
- `medium` (int, default: 10): Medium TEMA parameter
- `slow` (int, default: 20): Slow TEMA parameter
- `fast_period` (int, default: 9): Fast period
- `slow_period` (int, default: 21): Slow period

**Strategy Logic**:

- Uses TEMA formula: 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
- **Buy Signal**: Fast TEMA crosses above slow TEMA
- **Sell Signal**: Fast TEMA crosses below slow TEMA

**Example Configuration**:

```yaml
backtest:
  strategies:
    - name: triple_ema_cross
      parameters: {}
```

### Triangular Moving Average Crossover (`trima_cross`)

Uses Triangular Moving Averages for smoother trend following.

**Class**: `TRIMACrossStrategy`

**Parameters**:

- `fast_period` (int, default: 14): Fast TRIMA period
- `slow_period` (int, default: 28): Slow TRIMA period
- `atr_multiple` (float, default: 1.2): ATR multiplier for stop loss

**Strategy Logic**:

- TRIMA double-smooths data to filter noise
- **Buy Signal**: Fast TRIMA crosses above slow TRIMA
- **Sell Signal**: Fast TRIMA crosses below slow TRIMA

**Example Configuration**:

```yaml
backtest:
  strategies:
    - name: trima_cross
      parameters: {}
```

### KAMA Strategy (`kama`)

Kaufman's Adaptive Moving Average adapts to market volatility and trend strength.

**Class**: `KAMAStrategy`

**Parameters**:

- `period` (int, default: 14): Main calculation period
- `fast_sc` (int, default: 2): Fast smoothing constant
- `slow_sc` (int, default: 30): Slow smoothing constant
- `er_period` (int, default: 10): Efficiency Ratio period

**Strategy Logic**:

- Adapts smoothing based on Efficiency Ratio (ER)
- **Buy Signal**: Fast KAMA crosses above slow KAMA
- **Sell Signal**: Fast KAMA crosses below slow KAMA

**Example Configuration**:

```yaml
backtest:
  strategies:
    - name: kama
      parameters: {}
```

### FRAMA Strategy (`frama`)

Fractal Adaptive Moving Average uses fractal geometry to adjust smoothing.

**Class**: `FRAMAStrategy`

**Parameters**:

- `period` (int, default: 14): Main calculation period
- `frama_period` (int, default: 16): FRAMA-specific period (must be even)

**Strategy Logic**:

- Calculates fractal dimension to determine trend strength
- **Buy Signal**: Fast FRAMA crosses above slow FRAMA
- **Sell Signal**: Fast FRAMA crosses below slow FRAMA

**Example Configuration**:

```yaml
backtest:
  strategies:
    - name: frama
      parameters: {}
```

### VAMA Strategy (`vama`)

Volume Adjusted Moving Average weights prices by volume.

**Class**: `VAMAStrategy`

**Parameters**:

- `period` (int, default: 8): Main calculation period
- `vama_period` (int, default: 8): VAMA calculation period
- `slow_vama_period` (int, default: 21): Slow VAMA period

**Strategy Logic**:

- Weights moving average by volume to emphasize volume-driven moves
- **Buy Signal**: Fast VAMA crosses above slow VAMA
- **Sell Signal**: Fast VAMA crosses below slow VAMA

**Example Configuration**:

```yaml
backtest:
  strategies:
    - name: vama
      parameters: {}
```

### VIDYA Strategy (`vidya`)

Variable Index Dynamic Average adapts to market conditions using CMO.

**Class**: `VIDYAStrategy`

**Parameters**:

- `period` (int, default: 14): Main calculation period
- `alpha` (float, default: 0.2): Base alpha value
- `cmo_period` (int, default: 9): Chande Momentum Oscillator period
- `smoothing_period` (int, default: 12): Smoothing period

**Strategy Logic**:

- Uses Chande Momentum Oscillator to adapt smoothing
- **Buy Signal**: Fast VIDYA crosses above slow VIDYA
- **Sell Signal**: Fast VIDYA crosses below slow VIDYA

**Example Configuration**:

```yaml
backtest:
  strategies:
    - name: vidya
      parameters: {}
```

### Kaufman Efficiency Strategy (`kaufman_efficiency`)

Uses Efficiency Ratio to identify trending vs. ranging markets.

**Class**: `KaufmanEfficiencyStrategy`

**Parameters**:

- `period` (int, default: 10): Efficiency Ratio calculation period
- `fast_sc` (int, default: 2): Fast smoothing constant
- `slow_sc` (int, default: 30): Slow smoothing constant
- `er_upper_threshold` (float, default: 0.5): Upper threshold for buy signals
- `er_lower_threshold` (float, default: 0.3): Lower threshold for sell signals

**Strategy Logic**:

- **Buy Signal**: Efficiency Ratio above upper threshold (strong trend)
- **Sell Signal**: Efficiency Ratio below lower threshold (weak trend)

**Example Configuration**:

```yaml
backtest:
  strategies:
    - name: kaufman_efficiency
      parameters: {}
```

## Strategy Groups

The BacktestingManager provides predefined strategy groups for different trading approaches:

### Basic Group

- `smacross`
- `rsi`

### Momentum Group

- `rsi`
- `macd`
- `double_ema_cross`

### Trend Group

- `smacross`
- `triple_ema_cross`
- `trima_cross`

### Advanced Group

- `kama`
- `frama`
- `vama`
- `vidya`

### Comprehensive Group

All available strategies combined.

## Configuration Examples

### Single Strategy Configuration

```yaml
backtest:
  strategies:
    - name: smacross
      parameters:
        fast_period: 10
        slow_period: 20
```

### Multiple Strategies Configuration

```yaml
backtest:
  strategies:
    - name: smacross
      parameters:
        fast_period: 10
        slow_period: 20
    - name: rsi
      parameters:
        period: 14
        oversold_threshold: 30.0
        overbought_threshold: 70.0
    - name: double_ema_cross
      parameters: {}
    - name: kama
      parameters: {}
```

### Using Strategy Registry Programmatically

```python
from stockula.backtesting import StrategyRegistry, BacktestingManager

# Get strategy information
available_strategies = StrategyRegistry.get_available_strategy_names()
strategy_groups = StrategyRegistry.get_strategy_groups()
strategy_presets = StrategyRegistry.get_strategy_presets()

# Name normalization
normalized_name = StrategyRegistry.normalize_strategy_name("SMACross")  # Returns "smacross"

# Strategy class resolution
strategy_class = StrategyRegistry.get_strategy_class("DoubleEMACross")  # Returns DoubleEMACrossStrategy

# Validation
is_valid = StrategyRegistry.is_valid_strategy("RSI")  # Returns True

# Get default parameters
params = StrategyRegistry.get_strategy_preset("smacross")  # Returns {"fast_period": 10, "slow_period": 20}

# Using with BacktestingManager
manager = BacktestingManager(data_fetcher, logging_manager)

# Run a strategy group
results = manager.run_multiple_strategies(
    ticker="AAPL",
    strategy_group="momentum"
)

# Run a single strategy with any naming format
result = manager.run_single_strategy(
    ticker="AAPL",
    strategy_name="SMACross"  # PascalCase works
)

result2 = manager.run_single_strategy(
    ticker="AAPL", 
    strategy_name="double_ema_cross"  # snake_case works too
)
```

## Performance Considerations

### Data Requirements

Each strategy has different minimum data requirements:

- **Simple strategies** (SMA, RSI, MACD): ~30-50 days minimum
- **Adaptive strategies** (KAMA, FRAMA, VIDYA): ~60-100 days minimum
- **Complex strategies** (Triple EMA, TRIMA): ~75-120 days minimum

### Computational Complexity

Strategies ranked by computational complexity (fastest to slowest):

1. **SMA Cross** - Simple moving averages
1. **RSI** - Single oscillator calculation
1. **MACD** - Exponential moving averages
1. **Double EMA Cross** - EMAs with ATR
1. **TRIMA Cross** - Double-smoothed averages
1. **KAMA** - Efficiency ratio calculations
1. **VAMA** - Volume-weighted calculations
1. **VIDYA** - CMO-based adaptation
1. **FRAMA** - Fractal dimension calculations
1. **Triple EMA Cross** - Multiple EMA calculations
1. **Kaufman Efficiency** - Complex efficiency analysis

## Best Practices

### Strategy Selection

1. **Start with basic strategies** (SMA, RSI) to understand market behavior
1. **Use adaptive strategies** (KAMA, VIDYA) for volatile markets
1. **Combine trend and momentum** strategies for robust portfolios
1. **Test multiple timeframes** to find optimal parameters

### Parameter Optimization

1. **Avoid overfitting** - test on out-of-sample data
1. **Use robust parameter ranges** - avoid extreme values
1. **Consider transaction costs** in optimization
1. **Validate across different market regimes**

### Risk Management

- All advanced strategies include ATR-based stop losses
- Consider position sizing based on volatility
- Use appropriate stop loss multipliers for different asset classes
- Monitor maximum drawdown across strategies

The strategies in Stockula are designed to be robust, well-tested, and suitable for both research and practical trading applications.
