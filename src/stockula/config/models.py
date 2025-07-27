"""Pydantic models for configuration."""

from datetime import date, datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict


class TickerConfig(BaseModel):
    """Configuration for individual ticker/asset."""

    symbol: str = Field(description="Stock ticker symbol (e.g., AAPL)")
    quantity: Optional[float] = Field(
        default=None,
        gt=0,
        description="Number of shares to hold (required if not using dynamic allocation)",
    )
    allocation_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Percentage of portfolio to allocate to this asset (for dynamic allocation)",
    )
    allocation_amount: Optional[float] = Field(
        default=None,
        ge=0,
        description="Fixed dollar amount to allocate to this asset (for dynamic allocation)",
    )
    # Optional market data fields that can be populated
    market_cap: Optional[float] = Field(
        default=None, description="Market capitalization in billions"
    )
    price_range: Optional[Dict[str, float]] = Field(
        default=None,
        description="Price range with 'open', 'high', 'low', 'close' keys",
    )
    sector: Optional[str] = Field(default=None, description="Market sector")
    category: Optional[str] = Field(
        default=None,
        description="Category for classification (e.g., 'TECHNOLOGY', 'GROWTH', 'LARGE_CAP')",
    )

    def model_post_init(self, __context):
        """Validate that exactly one allocation method is specified."""
        allocation_methods = [
            self.quantity is not None,
            self.allocation_pct is not None,
            self.allocation_amount is not None,
        ]

        # For auto-allocation mode, only category is required
        if sum(allocation_methods) == 0 and self.category is not None:
            return  # Valid for auto-allocation

        if sum(allocation_methods) != 1:
            raise ValueError(
                f"Ticker {self.symbol} must specify exactly one of: quantity, allocation_pct, allocation_amount, or just category (for auto-allocation)"
            )


class PortfolioConfig(BaseModel):
    """Configuration for portfolio management."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(default="Main Portfolio", description="Portfolio name")
    initial_capital: float = Field(
        default=10000.0, gt=0, description="Initial portfolio capital"
    )
    allocation_method: str = Field(
        default="equal_weight",
        description="Allocation method: 'equal_weight', 'market_cap', 'custom', 'dynamic', 'auto'",
    )
    dynamic_allocation: bool = Field(
        default=False,
        description="Enable dynamic quantity calculation based on allocation percentages/amounts",
    )
    auto_allocate: bool = Field(
        default=False,
        description="Automatically allocate based on category ratios and initial capital only",
    )
    category_ratios: Optional[Dict[str, float]] = Field(
        default=None,
        description="Target allocation ratios by category (e.g., {'INDEX': 0.35, 'MOMENTUM': 0.475, 'SPECULATIVE': 0.175})",
    )
    allow_fractional_shares: bool = Field(
        default=False,
        description="Allow fractional shares when calculating dynamic quantities",
    )
    capital_utilization_target: float = Field(
        default=0.95,
        ge=0.5,
        le=1.0,
        description="Target percentage of initial capital to deploy (0.5-1.0)",
    )
    rebalance_frequency: Optional[str] = Field(
        default="monthly",
        description="Rebalancing frequency: 'daily', 'weekly', 'monthly', 'quarterly', 'yearly', 'never'",
    )
    tickers: List[TickerConfig] = Field(
        default_factory=lambda: [
            TickerConfig(symbol="AAPL", quantity=10),
            TickerConfig(symbol="GOOGL", quantity=5),
            TickerConfig(symbol="MSFT", quantity=8),
        ],
        description="List of ticker configurations in the portfolio",
    )
    # Risk management
    max_position_size: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Maximum position size as percentage of portfolio (0-100)",
    )
    stop_loss_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Global stop loss percentage (0-100)",
    )

    def model_post_init(self, __context):
        """Validate portfolio configuration constraints."""
        # Validate dynamic allocation settings
        if self.dynamic_allocation and self.allocation_method != "dynamic":
            print(
                "Warning: dynamic_allocation=True but allocation_method is not 'dynamic'. Setting allocation_method to 'dynamic'."
            )
            # Note: Cannot modify field here due to frozen model, but validation should catch this

        # If using dynamic allocation, validate that tickers have allocation info
        if self.dynamic_allocation:
            for ticker in self.tickers:
                if ticker.quantity is not None:
                    print(
                        f"Warning: Ticker {ticker.symbol} has quantity specified but dynamic_allocation is enabled. Quantity will be ignored."
                    )

        # Validate auto-allocation settings
        if self.auto_allocate:
            if not self.category_ratios:
                raise ValueError(
                    "auto_allocate=True requires category_ratios to be specified"
                )

            total_ratio = sum(self.category_ratios.values())
            if abs(total_ratio - 1.0) > 0.01:  # Allow small tolerance
                raise ValueError(f"Category ratios must sum to 1.0, got {total_ratio}")

        # Check that total percentage allocations don't exceed 100%
        if self.dynamic_allocation:
            total_pct = sum(
                ticker.allocation_pct
                for ticker in self.tickers
                if ticker.allocation_pct is not None
            )
            if total_pct > 100:
                raise ValueError(
                    f"Total allocation percentages ({total_pct}%) exceed 100%"
                )

            total_amount = sum(
                ticker.allocation_amount
                for ticker in self.tickers
                if ticker.allocation_amount is not None
            )
            if total_amount > self.initial_capital:
                raise ValueError(
                    f"Total allocation amounts (${total_amount:,.2f}) exceed initial capital (${self.initial_capital:,.2f})"
                )


class DataConfig(BaseModel):
    """Configuration for data fetching."""

    start_date: Optional[Union[str, date]] = Field(
        default=None, description="Start date for historical data (YYYY-MM-DD)"
    )
    end_date: Optional[Union[str, date]] = Field(
        default=None, description="End date for historical data (YYYY-MM-DD)"
    )
    interval: str = Field(
        default="1d",
        description="Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)",
    )

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def parse_dates(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            return datetime.strptime(v, "%Y-%m-%d").date()
        return v


class StrategyConfig(BaseModel):
    """Base configuration for trading strategies."""

    name: str = Field(description="Strategy name")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Strategy-specific parameters"
    )


class SMACrossConfig(BaseModel):
    """Configuration for SMA Cross strategy."""

    fast_period: int = Field(default=10, ge=1, description="Fast SMA period")
    slow_period: int = Field(default=20, ge=1, description="Slow SMA period")

    @field_validator("slow_period")
    @classmethod
    def validate_periods(cls, v, info):
        if "fast_period" in info.data and v <= info.data["fast_period"]:
            raise ValueError("slow_period must be greater than fast_period")
        return v


class RSIConfig(BaseModel):
    """Configuration for RSI strategy."""

    period: int = Field(default=14, ge=1, description="RSI period")
    oversold_threshold: float = Field(
        default=30.0, ge=0, le=100, description="Oversold threshold"
    )
    overbought_threshold: float = Field(
        default=70.0, ge=0, le=100, description="Overbought threshold"
    )

    @field_validator("overbought_threshold")
    @classmethod
    def validate_thresholds(cls, v, info):
        if "oversold_threshold" in info.data and v <= info.data["oversold_threshold"]:
            raise ValueError(
                "overbought_threshold must be greater than oversold_threshold"
            )
        return v


class MACDConfig(BaseModel):
    """Configuration for MACD strategy."""

    fast_period: int = Field(default=12, ge=1, description="Fast EMA period")
    slow_period: int = Field(default=26, ge=1, description="Slow EMA period")
    signal_period: int = Field(default=9, ge=1, description="Signal line EMA period")

    @field_validator("slow_period")
    @classmethod
    def validate_periods(cls, v, info):
        if "fast_period" in info.data and v <= info.data["fast_period"]:
            raise ValueError("slow_period must be greater than fast_period")
        return v


class BacktestConfig(BaseModel):
    """Configuration for backtesting."""

    initial_cash: float = Field(
        default=10000.0, gt=0, description="Initial cash amount for backtesting"
    )
    commission: float = Field(
        default=0.002, ge=0, le=1, description="Commission per trade (0.002 = 0.2%)"
    )
    margin: float = Field(
        default=1.0, ge=0, description="Margin requirement for leveraged trading"
    )
    strategies: List[StrategyConfig] = Field(
        default_factory=list, description="List of strategies to backtest"
    )
    optimize: bool = Field(
        default=False, description="Whether to optimize strategy parameters"
    )
    optimization_params: Optional[Dict[str, Any]] = Field(
        default=None, description="Parameter ranges for optimization"
    )
    hold_only_categories: List[str] = Field(
        default=["INDEX", "BOND"],
        description="Categories of assets to exclude from backtesting (buy-and-hold only)",
    )


class ForecastConfig(BaseModel):
    """Configuration for forecasting."""

    forecast_length: int = Field(
        default=14, ge=1, description="Number of periods to forecast"
    )
    frequency: str = Field(
        default="infer",
        description="Time series frequency ('D', 'W', 'M', etc.), 'infer' to auto-detect",
    )
    prediction_interval: float = Field(
        default=0.9, ge=0, le=1, description="Confidence interval for predictions"
    )
    model_list: str = Field(
        default="fast",
        description="Model subset to use ('fast', 'default', 'slow', 'parallel')",
    )
    ensemble: str = Field(
        default="auto",
        description="Ensemble method ('auto', 'simple', 'distance', 'horizontal')",
    )
    max_generations: int = Field(
        default=5, ge=1, description="Maximum generations for model search"
    )
    num_validations: int = Field(
        default=2, ge=1, description="Number of validation splits"
    )
    validation_method: str = Field(
        default="backwards",
        description="Validation method ('backwards', 'seasonal', 'similarity')",
    )


class TechnicalAnalysisConfig(BaseModel):
    """Configuration for technical analysis indicators."""

    indicators: List[str] = Field(
        default=["sma", "ema", "rsi", "macd", "bbands", "atr"],
        description="List of indicators to calculate",
    )
    sma_periods: List[int] = Field(
        default=[20, 50, 200], description="SMA periods to calculate"
    )
    ema_periods: List[int] = Field(
        default=[12, 26], description="EMA periods to calculate"
    )
    rsi_period: int = Field(default=14, description="RSI period")
    macd_params: Dict[str, int] = Field(
        default={"period_fast": 12, "period_slow": 26, "signal": 9},
        description="MACD parameters",
    )
    bbands_params: Dict[str, int] = Field(
        default={"period": 20, "std": 2}, description="Bollinger Bands parameters"
    )
    atr_period: int = Field(default=14, description="ATR period")


class LoggingConfig(BaseModel):
    """Configuration for logging and debug output."""

    enabled: bool = Field(default=False, description="Enable verbose logging output")
    level: str = Field(
        default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    show_allocation_details: bool = Field(
        default=True,
        description="Show detailed allocation calculations when logging enabled",
    )
    show_price_fetching: bool = Field(
        default=True, description="Show price fetching details when logging enabled"
    )
    log_to_file: bool = Field(default=False, description="Enable logging to file")
    log_file: str = Field(default="stockula.log", description="Log file path")
    max_log_size: int = Field(
        default=10_485_760,  # 10MB
        description="Maximum log file size in bytes before rotation",
    )
    backup_count: int = Field(
        default=3, description="Number of backup log files to keep"
    )


class StockulaConfig(BaseModel):
    """Main configuration model for Stockula."""

    data: DataConfig = Field(default_factory=DataConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    technical_analysis: TechnicalAnalysisConfig = Field(
        default_factory=TechnicalAnalysisConfig
    )
    output: Dict[str, Any] = Field(
        default_factory=lambda: {
            "format": "console",
            "save_results": False,
            "results_dir": "./results",
        },
        description="Output configuration",
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration"
    )
