"""Pydantic models for configuration."""

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BacktestResult(BaseModel):
    """Individual backtest result for a single asset."""

    ticker: str = Field(description="Asset ticker symbol")
    strategy: str = Field(description="Strategy name")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Strategy parameters used")
    return_pct: float = Field(description="Return percentage")
    sharpe_ratio: float = Field(description="Sharpe ratio")
    max_drawdown_pct: float = Field(description="Maximum drawdown percentage")
    num_trades: int = Field(description="Number of trades executed")
    win_rate: float | None = Field(default=None, description="Win rate percentage")


class StrategyBacktestSummary(BaseModel):
    """Summary of backtest results for a single strategy across all assets."""

    strategy_name: str = Field(description="Strategy name")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    initial_portfolio_value: float = Field(description="Initial portfolio value")
    final_portfolio_value: float = Field(description="Final portfolio value after backtest")
    total_return_pct: float = Field(description="Total portfolio return percentage")
    total_trades: int = Field(description="Total trades across all assets")
    winning_stocks: int = Field(description="Number of stocks with positive returns")
    losing_stocks: int = Field(description="Number of stocks with negative returns")
    average_return_pct: float = Field(description="Average return across all assets")
    average_sharpe_ratio: float = Field(description="Average Sharpe ratio")
    detailed_results: list[BacktestResult] = Field(default_factory=list, description="Per-asset results")


class PortfolioBacktestResults(BaseModel):
    """Complete backtest results for all strategies."""

    initial_portfolio_value: float = Field(description="Initial portfolio value")
    initial_capital: float = Field(description="Initial capital")
    date_range: dict[str, str] = Field(description="Backtest date range")
    broker_config: dict[str, Any] = Field(description="Broker configuration used")
    strategy_summaries: list[StrategyBacktestSummary] = Field(
        default_factory=list, description="Summary results for each strategy"
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="When backtest was run")


class TickerConfig(BaseModel):
    """Configuration for individual ticker/asset."""

    symbol: str = Field(description="Stock ticker symbol (e.g., AAPL)")
    quantity: float | None = Field(
        default=None,
        gt=0,
        description="Number of shares to hold (required if not using dynamic allocation)",
    )
    allocation_pct: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Percentage of portfolio to allocate to this asset (for dynamic allocation)",
    )
    allocation_amount: float | None = Field(
        default=None,
        ge=0,
        description="Fixed dollar amount to allocate to this asset (for dynamic allocation)",
    )
    # Optional market data fields that can be populated
    market_cap: float | None = Field(default=None, description="Market capitalization in billions")
    price_range: dict[str, float] | None = Field(
        default=None,
        description="Price range with 'open', 'high', 'low', 'close' keys",
    )
    sector: str | None = Field(default=None, description="Market sector")
    category: str | None = Field(
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

        # For backtest_optimized allocation, no allocation fields are required
        # This will be validated at the portfolio level
        if sum(allocation_methods) == 0:
            return  # Valid for backtest_optimized or other dynamic methods

        if sum(allocation_methods) != 1:
            raise ValueError(
                f"Ticker {self.symbol} must specify exactly one of: quantity, allocation_pct, "
                f"allocation_amount, or none of them (for backtest_optimized or auto-allocation)"
            )


class PortfolioConfig(BaseModel):
    """Configuration for portfolio management."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(default="Main Portfolio", description="Portfolio name")
    initial_capital: float = Field(default=10000.0, gt=0, description="Initial portfolio capital")
    allocation_method: str = Field(
        default="equal_weight",
        description=(
            "Allocation method: 'equal_weight', 'market_cap', 'custom', 'dynamic', 'auto', 'backtest_optimized'"
        ),
    )
    dynamic_allocation: bool = Field(
        default=False,
        description="Enable dynamic quantity calculation based on allocation percentages/amounts",
    )
    auto_allocate: bool = Field(
        default=False,
        description="Automatically allocate based on category ratios and initial capital only",
    )
    category_ratios: dict[str, float] | None = Field(
        default=None,
        description="Target allocation ratios by category "
        "(e.g., {'INDEX': 0.35, 'MOMENTUM': 0.475, 'SPECULATIVE': 0.175})",
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
    rebalance_frequency: str | None = Field(
        default="monthly",
        description="Rebalancing frequency: 'daily', 'weekly', 'monthly', 'quarterly', 'yearly', 'never'",
    )
    tickers: list[TickerConfig] = Field(
        default_factory=lambda: [
            TickerConfig(symbol="AAPL", quantity=10),
            TickerConfig(symbol="GOOGL", quantity=5),
            TickerConfig(symbol="MSFT", quantity=8),
        ],
        description="List of ticker configurations in the portfolio",
    )
    # Risk management
    max_position_size: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Maximum position size as percentage of portfolio (0-100)",
    )
    stop_loss_pct: float | None = Field(
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
                "Warning: dynamic_allocation=True but allocation_method is not 'dynamic'. "
                "Setting allocation_method to 'dynamic'."
            )
            # Note: Cannot modify field here due to frozen model, but validation should catch this

        # If using dynamic allocation, validate that tickers have allocation info
        if self.dynamic_allocation:
            for ticker in self.tickers:
                if ticker.quantity is not None:
                    print(
                        f"Warning: Ticker {ticker.symbol} has quantity specified but dynamic_allocation is enabled. "
                        f"Quantity will be ignored."
                    )

        # Validate backtest_optimized allocation settings
        if self.allocation_method == "backtest_optimized":
            for ticker in self.tickers:
                if (
                    ticker.quantity is not None
                    or ticker.allocation_pct is not None
                    or ticker.allocation_amount is not None
                ):
                    print(
                        f"Warning: Ticker {ticker.symbol} has allocation specified but "
                        f"allocation_method='backtest_optimized'. Allocations will be calculated automatically."
                    )

        # Validate auto-allocation settings
        if self.auto_allocate:
            if not self.category_ratios:
                raise ValueError("auto_allocate=True requires category_ratios to be specified")

            total_ratio = sum(self.category_ratios.values())
            if abs(total_ratio - 1.0) > 0.01:  # Allow small tolerance
                raise ValueError(f"Category ratios must sum to 1.0, got {total_ratio}")

        # Check that total percentage allocations don't exceed 100%
        if self.dynamic_allocation:
            total_pct = sum(ticker.allocation_pct for ticker in self.tickers if ticker.allocation_pct is not None)
            if total_pct > 100:
                raise ValueError(f"Total allocation percentages ({total_pct}%) exceed 100%")

            total_amount = sum(
                ticker.allocation_amount for ticker in self.tickers if ticker.allocation_amount is not None
            )
            if total_amount > self.initial_capital:
                raise ValueError(
                    f"Total allocation amounts (${total_amount:,.2f}) exceed "
                    f"initial capital (${self.initial_capital:,.2f})"
                )


class DataConfig(BaseModel):
    """Configuration for data fetching."""

    # General date range for data fetching
    start_date: str | date | None = Field(default=None, description="Start date for historical data (YYYY-MM-DD)")
    end_date: str | date | None = Field(default=None, description="End date for historical data (YYYY-MM-DD)")
    interval: str = Field(
        default="1d",
        description="Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)",
    )
    use_cache: bool = Field(default=True, description="Use database caching for fetched data")
    db_path: str = Field(default="stockula.db", description="Path to SQLite database file for caching")

    @field_validator(
        "start_date",
        "end_date",
        mode="before",
    )
    @classmethod
    def parse_dates(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            return datetime.strptime(v, "%Y-%m-%d").date()
        return v

    @model_validator(mode="after")
    def validate_date_ranges(self):
        """Validate date ranges."""
        # Validate date range
        if self.start_date and self.end_date:
            if self.start_date >= self.end_date:
                raise ValueError("start_date must be before end_date")

        return self


class StrategyConfig(BaseModel):
    """Base configuration for trading strategies."""

    name: str = Field(description="Strategy name")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Strategy-specific parameters")


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
    oversold_threshold: float = Field(default=30.0, ge=0, le=100, description="Oversold threshold")
    overbought_threshold: float = Field(default=70.0, ge=0, le=100, description="Overbought threshold")

    @field_validator("overbought_threshold")
    @classmethod
    def validate_thresholds(cls, v, info):
        if "oversold_threshold" in info.data and v <= info.data["oversold_threshold"]:
            raise ValueError("overbought_threshold must be greater than oversold_threshold")
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


class BrokerConfig(BaseModel):
    """Configuration for broker-specific fees and commissions."""

    name: str = Field(default="custom", description="Broker name or 'custom' for custom fee structure")
    commission_type: str = Field(
        default="percentage",
        description="Commission type: 'percentage', 'fixed', 'tiered', 'per_share'",
    )
    commission_value: float | dict[str, float] = Field(
        default=0.002,
        description="Commission value (float for simple types, dict for tiered)",
    )
    min_commission: float | None = Field(default=None, description="Minimum commission per trade")
    max_commission: float | None = Field(default=None, description="Maximum commission per trade")
    per_share_commission: float | None = Field(default=None, description="Commission per share (for per_share type)")
    regulatory_fees: float = Field(default=0.0, description="Additional regulatory fees (SEC, FINRA) as percentage")
    exchange_fees: float = Field(default=0.0, description="Exchange fees per trade")

    @classmethod
    def from_broker_preset(cls, broker_name: str) -> "BrokerConfig":
        """Create BrokerConfig from preset broker configurations."""
        presets = {
            "interactive_brokers": {
                "name": "interactive_brokers",
                "commission_type": "tiered",
                "commission_value": {
                    "0": 0.005,  # $0.005 per share for first 500k shares/month
                    "500000": 0.004,  # $0.004 per share above 500k shares/month
                },
                "min_commission": 1.0,
                "max_commission": None,
                "per_share_commission": 0.005,
                "regulatory_fees": 0.0000229,  # SEC fee
                "exchange_fees": 0.003,  # Exchange fees
            },
            "td_ameritrade": {
                "name": "td_ameritrade",
                "commission_type": "fixed",
                "commission_value": 0.0,  # Zero commission
                "regulatory_fees": 0.0000229,
                "exchange_fees": 0.0,
            },
            "etrade": {
                "name": "etrade",
                "commission_type": "fixed",
                "commission_value": 0.0,  # Zero commission
                "regulatory_fees": 0.0000229,
                "exchange_fees": 0.0,
            },
            "robinhood": {
                "name": "robinhood",
                "commission_type": "fixed",
                "commission_value": 0.0,  # Zero commission
                "regulatory_fees": 0.0,  # SEC fee is $0 as of May 14, 2024
                "exchange_fees": 0.000166,  # TAF (Trading Activity Fee) per share for equity sells
            },
            "fidelity": {
                "name": "fidelity",
                "commission_type": "fixed",
                "commission_value": 0.0,  # Zero commission
                "regulatory_fees": 0.0000229,
                "exchange_fees": 0.0,
            },
            "schwab": {
                "name": "schwab",
                "commission_type": "fixed",
                "commission_value": 0.0,  # Zero commission
                "regulatory_fees": 0.0000229,
                "exchange_fees": 0.0,
            },
        }

        if broker_name.lower() in presets:
            return cls(**presets[broker_name.lower()])
        else:
            raise ValueError(f"Unknown broker preset: {broker_name}")


class BacktestConfig(BaseModel):
    """Configuration for backtesting."""

    initial_cash: float = Field(default=10000.0, gt=0, description="Initial cash amount for backtesting")

    # Backtest-specific date range
    start_date: str | date | None = Field(default=None, description="Start date for backtesting (YYYY-MM-DD)")
    end_date: str | date | None = Field(default=None, description="End date for backtesting (YYYY-MM-DD)")
    commission: float = Field(
        default=0.002,
        ge=0,
        le=1,
        description="Commission per trade (0.002 = 0.2%) - deprecated, use broker_config",
    )
    broker_config: BrokerConfig | None = Field(default=None, description="Broker-specific fee configuration")
    margin: float = Field(default=1.0, ge=0, description="Margin requirement for leveraged trading")
    strategies: list[StrategyConfig] = Field(default_factory=list, description="List of strategies to backtest")
    optimize: bool = Field(default=False, description="Whether to optimize strategy parameters")
    optimization_params: dict[str, Any] | None = Field(default=None, description="Parameter ranges for optimization")
    hold_only_categories: list[str] = Field(
        default=["INDEX", "BOND"],
        description="Categories of assets to exclude from backtesting (buy-and-hold only)",
    )

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def parse_dates(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            return datetime.strptime(v, "%Y-%m-%d").date()
        return v

    @model_validator(mode="after")
    def validate_date_ranges(self):
        """Validate date ranges."""
        if self.start_date and self.end_date:
            if self.start_date >= self.end_date:
                raise ValueError("start_date must be before end_date")
        return self

    def model_post_init(self, __context):
        """Initialize broker config from commission if not provided."""
        if self.broker_config is None:
            # Use legacy commission field for backward compatibility
            self.broker_config = BrokerConfig(
                name="legacy",
                commission_type="percentage",
                commission_value=self.commission,
            )


class ForecastConfig(BaseModel):
    """Configuration for forecasting."""

    forecast_length: int | None = Field(
        default=None,
        ge=1,
        description="Number of periods to forecast from today (mutually exclusive with test dates)",
    )

    # Train/test split for forecast evaluation
    train_start_date: str | date | None = Field(default=None, description="Start date for training data (YYYY-MM-DD)")
    train_end_date: str | date | None = Field(default=None, description="End date for training data (YYYY-MM-DD)")
    test_start_date: str | date | None = Field(default=None, description="Start date for testing data (YYYY-MM-DD)")
    test_end_date: str | date | None = Field(default=None, description="End date for testing data (YYYY-MM-DD)")
    frequency: str = Field(
        default="infer",
        description="Time series frequency ('D', 'W', 'M', etc.), 'infer' to auto-detect",
    )
    prediction_interval: float = Field(default=0.9, ge=0, le=1, description="Confidence interval for predictions")
    model_list: str = Field(
        default="fast",
        description="Model subset to use ('fast', 'default', 'slow', 'parallel', 'financial')",
    )
    ensemble: str = Field(
        default="auto",
        description="Ensemble method ('auto', 'simple', 'distance', 'horizontal')",
    )
    use_financial_models: bool = Field(
        default=True,
        description="Use financial-appropriate models to avoid statsmodels warnings",
    )
    max_generations: int = Field(default=2, ge=1, description="Maximum generations for model search")
    num_validations: int = Field(default=1, ge=1, description="Number of validation splits")
    validation_method: str = Field(
        default="backwards",
        description="Validation method ('backwards', 'seasonal', 'similarity')",
    )
    max_workers: int = Field(default=1, ge=1, description="Maximum parallel workers for forecasting")

    @field_validator(
        "train_start_date",
        "train_end_date",
        "test_start_date",
        "test_end_date",
        mode="before",
    )
    @classmethod
    def parse_dates(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            return datetime.strptime(v, "%Y-%m-%d").date()
        return v

    @model_validator(mode="after")
    def validate_date_ranges(self):
        """Validate date ranges and ensure mutual exclusivity."""
        # Check mutual exclusivity between forecast_length and test dates
        has_test_dates = self.test_start_date is not None and self.test_end_date is not None
        has_forecast_length = self.forecast_length is not None

        if has_test_dates and has_forecast_length:
            raise ValueError(
                "Cannot specify both forecast_length and test dates. "
                "Use forecast_length for future predictions OR test dates for historical evaluation."
            )

        # If using test dates, train dates are required
        if has_test_dates and (self.train_start_date is None or self.train_end_date is None):
            raise ValueError("When using test dates for evaluation, train dates must also be specified.")

        # Validate training date range
        if self.train_start_date and self.train_end_date:
            if self.train_start_date >= self.train_end_date:
                raise ValueError("train_start_date must be before train_end_date")

        # Validate testing date range
        if self.test_start_date and self.test_end_date:
            if self.test_start_date >= self.test_end_date:
                raise ValueError("test_start_date must be before test_end_date")

        # Validate that test period comes after training period
        if self.train_end_date and self.test_start_date:
            if self.test_start_date < self.train_end_date:
                raise ValueError("test_start_date must be after or equal to train_end_date")

        return self


class TechnicalAnalysisConfig(BaseModel):
    """Configuration for technical analysis indicators."""

    indicators: list[str] = Field(
        default=["sma", "ema", "rsi", "macd", "bbands", "atr"],
        description="List of indicators to calculate",
    )
    sma_periods: list[int] = Field(default=[20, 50, 200], description="SMA periods to calculate")
    ema_periods: list[int] = Field(default=[12, 26], description="EMA periods to calculate")
    rsi_period: int = Field(default=14, description="RSI period")
    macd_params: dict[str, int] = Field(
        default={"period_fast": 12, "period_slow": 26, "signal": 9},
        description="MACD parameters",
    )
    bbands_params: dict[str, int] = Field(default={"period": 20, "std": 2}, description="Bollinger Bands parameters")
    atr_period: int = Field(default=14, description="ATR period")


class LoggingConfig(BaseModel):
    """Configuration for logging and debug output."""

    enabled: bool = Field(default=False, description="Enable verbose logging output")
    level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)")
    show_allocation_details: bool = Field(
        default=True,
        description="Show detailed allocation calculations when logging enabled",
    )
    show_price_fetching: bool = Field(default=True, description="Show price fetching details when logging enabled")
    log_to_file: bool = Field(default=False, description="Enable logging to file")
    log_file: str = Field(default="stockula.log", description="Log file path")
    max_log_size: int = Field(
        default=10_485_760,  # 10MB
        description="Maximum log file size in bytes before rotation",
    )
    backup_count: int = Field(default=3, description="Number of backup log files to keep")


class BacktestOptimizationConfig(BaseModel):
    """Configuration for backtest-optimized allocation strategy."""

    train_start_date: str | date | None = Field(default=None, description="Start date for training period (YYYY-MM-DD)")
    train_end_date: str | date | None = Field(default=None, description="End date for training period (YYYY-MM-DD)")
    test_start_date: str | date | None = Field(default=None, description="Start date for testing period (YYYY-MM-DD)")
    test_end_date: str | date | None = Field(default=None, description="End date for testing period (YYYY-MM-DD)")
    ranking_metric: str = Field(
        default="Return [%]",
        description="Metric to use for ranking strategies: 'Return [%]', 'Sharpe Ratio', 'Sortino Ratio', etc.",
    )
    min_allocation_pct: float = Field(
        default=2.0, ge=0.0, le=100.0, description="Minimum allocation percentage per asset (0-100)"
    )
    max_allocation_pct: float = Field(
        default=25.0, ge=0.0, le=100.0, description="Maximum allocation percentage per asset (0-100)"
    )
    initial_allocation_pct: float = Field(
        default=2.0, ge=0.0, le=100.0, description="Initial allocation percentage for training period"
    )

    @field_validator("train_start_date", "train_end_date", "test_start_date", "test_end_date", mode="before")
    @classmethod
    def parse_date(cls, v):
        """Parse date strings to date objects."""
        if isinstance(v, str):
            return datetime.strptime(v, "%Y-%m-%d").date()
        return v

    @model_validator(mode="after")
    def validate_dates(self):
        """Validate date ranges."""
        if self.train_start_date and self.train_end_date:
            if self.train_start_date >= self.train_end_date:
                raise ValueError("train_start_date must be before train_end_date")

        if self.test_start_date and self.test_end_date:
            if self.test_start_date >= self.test_end_date:
                raise ValueError("test_start_date must be before test_end_date")

        if self.train_end_date and self.test_start_date:
            if self.train_end_date > self.test_start_date:
                raise ValueError("train_end_date must be before or equal to test_start_date")

        if self.min_allocation_pct > self.max_allocation_pct:
            raise ValueError("min_allocation_pct must be less than or equal to max_allocation_pct")

        return self


class StockulaConfig(BaseModel):
    """Main configuration model for Stockula."""

    data: DataConfig = Field(default_factory=DataConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    backtest_optimization: BacktestOptimizationConfig | None = Field(
        default=None, description="Configuration for backtest-optimized allocation"
    )
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    technical_analysis: TechnicalAnalysisConfig = Field(default_factory=TechnicalAnalysisConfig)
    output: dict[str, Any] = Field(
        default_factory=lambda: {
            "format": "console",
            "save_results": False,
            "results_dir": "./results",
        },
        description="Output configuration",
    )
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
