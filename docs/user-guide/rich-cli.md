# Rich CLI Features

Stockula uses the Rich library to provide an enhanced command-line experience with beautiful formatting, progress bars, and interactive elements.

## Overview

The Rich CLI integration provides:

- **Progress Bars**: Real-time progress tracking with time estimates
- **Formatted Tables**: Clean, colored tables for results display
- **Panels**: Bordered panels for strategy summaries and reports
- **Colors**: Contextual colors for positive/negative returns and status
- **Interactive Elements**: Graceful interrupt handling and status updates

## Progress Bars

Stockula shows progress for all time-intensive operations:

### Backtesting Progress

```
⠋ Backtesting SMACROSS on AAPL... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 85% 0:00:02
```

Features:

- Strategy name and ticker being processed
- Progress percentage and visual bar
- Time remaining estimate
- Spinner animation for active processing

### Technical Analysis Progress

```
⠋ Computing SMA(20) for AAPL... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 60% 0:00:01
```

Shows:

- Current indicator being calculated
- Detailed step information
- Progress through indicator list

### Data Fetching Progress

```
⠋ Fetching price for NVDA... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 75% 0:00:03
```

Displays:

- Current symbol being fetched
- Multi-symbol progress tracking
- Network operation status

### Forecasting Progress

```
⠋ Training fast models with 5 generations... (AutoTS is working...)
```

AutoTS forecasting shows:

- Model training status
- Generation count
- Descriptive status messages

## Result Tables

### Technical Analysis Results

```
                    Technical Analysis Results
┏━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Ticker ┃ SMA_20        ┃ RSI_14         ┃ MACD             ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ AAPL   │ $150.25       │ 65.4           │ 2.15             │
│ GOOGL  │ $2,750.80     │ 58.2           │ -1.25            │
│ MSFT   │ $405.60       │ 72.1           │ 3.40             │
└────────┴───────────────┴────────────────┴──────────────────┘
```

Features:

- Automatic column sizing
- Currency formatting for prices
- Decimal precision for ratios
- Color coding for positive/negative values

### Backtesting Results

The backtesting display starts with portfolio information followed by detailed results:

#### Portfolio Information

```
=== Backtesting Results ===

Portfolio Information:
  Initial Capital: $10,000
  Start Date: 2023-01-01
  End Date: 2023-12-31
  Trading Days: 252
  Calendar Days: 365
```

#### Results Table

```
                         Backtesting Results
┏━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Ticker ┃ Strategy  ┃ Return     ┃ Sharpe Ratio   ┃ Max Drawdown   ┃
┡━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ AAPL   │ SMACROSS  │ +15.50%    │ 1.25           │ -8.30%         │
│ GOOGL  │ SMACROSS  │ +8.75%     │ 0.98           │ -12.45%        │
│ MSFT   │ SMACROSS  │ -2.15%     │ -0.15          │ -18.90%        │
└────────┴───────────┴────────────┴────────────────┴────────────────┘
```

Color coding:

- **Green**: Positive returns, good ratios
- **Red**: Negative returns, poor performance
- **Yellow**: Warning levels
- **Gray**: Neutral values

### Forecast Results

#### Portfolio Value Summary

The portfolio value table adapts based on forecast mode:

**Future Prediction Mode:**

```
               Portfolio Value
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Metric          ┃ Date       ┃ Value      ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ Observed Value  │ 2025-07-29 │ $20,000.00 │
│ Predicted Value │ 2025-08-13 │ $20,456.32 │
└─────────────────┴────────────┴────────────┘
```

**Historical Evaluation Mode:**

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

#### Individual Stock Forecasts

```
                    Price Forecasts
┏━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Ticker ┃ Current Price ┃ Forecast Price ┃ Confidence Range ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ AAPL   │ $150.25       │ $155.80        │ $145.20 - $165.40│
│ NVDA   │ $875.40       │ $920.15        │ $850.30 - $995.20│
│ TSLA   │ $248.50       │ $265.30        │ $235.10 - $290.50│
└────────┴───────────────┴────────────────┴──────────────────┘
```

Features:

- Clear labeling of observed vs predicted values
- Date display for both current and forecast dates
- Currency formatting with $ prefix
- Confidence interval ranges
- Accuracy metrics for evaluation mode

## Strategy Summary Panels

Detailed panels show comprehensive strategy results:

```
╭───────────────────────────── STRATEGY: SMACROSS ─────────────────────────────╮
│                                                                              │
│  Parameters: {'fast_period': 10, 'slow_period': 20}                          │
│  Broker: robinhood (Zero commission + TAF)                                   │
│                                                                              │
│  Portfolio Value at Start Date: $100,000.00                                  │
│  Portfolio Value at End (Backtest): $112,345.67                              │
│                                                                              │
│  Strategy Performance:                                                       │
│    Average Return: +12.35%                                                   │
│    Winning Stocks: 8                                                         │
│    Losing Stocks: 2                                                          │
│    Total Trades: 45                                                          │
│                                                                              │
│  Return During Period: $12,345.67 (+12.35%)                                  │
│                                                                              │
│  Detailed report saved to:                                                   │
│  results/reports/strategy_report_smacross_20250127_143022.json               │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
```

Panel features:

- Bordered sections for visual organization
- Color-coded returns (green for positive, red for negative)
- Hierarchical information layout
- File path references for detailed reports

## Portfolio Summary

```
          Portfolio Summary
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Property          ┃ Value          ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ Name              │ Tech Portfolio │
│ Initial Capital   │ $100,000.00    │
│ Total Assets      │ 8              │
│ Allocation Method │ equal_weight   │
└───────────────────┴────────────────┘
```

## Current Portfolio Value

```
         Current Portfolio Value
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Metric                  ┃ Value       ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Current Portfolio Value │ $112,450.00 │
│ Total Return            │ +$12,450.00 │
│ Return Percentage       │ +12.45%     │
│ Best Performer          │ NVDA        │
│ Worst Performer         │ INTC        │
└─────────────────────────┴─────────────┘
```

## Status Messages and Alerts

### Information Panels

Important operational information:

```
╭─────────────────────────────────────────────────────────╮
│ FORECAST MODE - IMPORTANT NOTES:                        │
│ • AutoTS will try multiple models to find the best fit  │
│ • This process may take several minutes per ticker      │
│ • Press Ctrl+C at any time to cancel                    │
│ • Enable logging for more detailed progress information │
╰─────────────────────────────────────────────────────────╯
```

### Error Handling

Graceful error display:

```
[red]Error backtesting smacross on AAPL: Insufficient data for strategy[/red]
[yellow]Warning: NVDA requires at least 60 days of data, only 45 available[/yellow]
[green]✓ Successfully completed technical analysis for MSFT[/green]
```

## Interactive Features

### Keyboard Interrupts

Graceful handling of Ctrl+C:

```
⠋ Forecasting AAPL... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 45% 0:01:23

^C
[yellow]Forecast interrupted by user. Partial results will be displayed.[/yellow]

[green]✓ Forecasting completed for 3 of 5 tickers[/green]
```

### Status Updates

Real-time status during operations:

```
[blue]Loading configuration from .config.yaml...[/blue]
[green]✓ Configuration loaded successfully[/green]
[blue]Initializing data fetcher...[/blue]
[green]✓ Connected to database: stockula.db[/green]
[blue]Starting portfolio analysis...[/blue]
```

## Color Scheme

### Semantic Colors

| Color       | Usage                      | Context                       |
| ----------- | -------------------------- | ----------------------------- |
| **Green**   | Success, positive returns  | Profits, completed operations |
| **Red**     | Errors, negative returns   | Losses, failed operations     |
| **Yellow**  | Warnings, neutral          | Cautions, informational       |
| **Blue**    | Information, processing    | Status updates, headers       |
| **Cyan**    | Highlights, property names | Table headers, labels         |
| **Magenta** | Special values             | Strategy names, emphasis      |

### Return Color Coding

Returns are automatically color-coded:

- **Bright Green**: Returns > +10%
- **Green**: Returns > +2%
- **Yellow**: Returns between -2% and +2%
- **Red**: Returns < -2%
- **Bright Red**: Returns < -10%

## Terminal Compatibility

Rich automatically adapts to terminal capabilities:

### Full Color Terminals

- iTerm2, Terminal.app (macOS)
- Windows Terminal, ConEmu (Windows)
- GNOME Terminal, Konsole (Linux)

### Limited Color Terminals

- Basic 16-color fallback
- Monochrome mode for accessibility
- ASCII-only mode for basic terminals

### CI/CD Environments

- Automatic detection of CI environments
- Plain text output when appropriate
- Configurable via `NO_COLOR` environment variable

## Configuration

Rich CLI features can be controlled via configuration:

```yaml
output:
  format: "console"        # or "json" for plain output

logging:
  enabled: true           # Shows detailed Rich status messages
  level: "INFO"           # Controls verbosity of Rich output
```

Environment variables:

```bash
# Disable all Rich formatting
export NO_COLOR=1

# Force color output even in non-interactive mode
export FORCE_COLOR=1

# Set specific color mode
export TERM=xterm-256color
```

## Performance Considerations

Rich formatting is optimized for performance:

- **Lazy Rendering**: Tables built only when displayed
- **Efficient Updates**: Progress bars update efficiently
- **Memory Management**: Large datasets paginated automatically
- **Terminal Detection**: Skips formatting in non-interactive mode

The Rich CLI integration makes Stockula pleasant to use while maintaining high performance and broad terminal compatibility.
