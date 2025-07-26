"""Pydantic models for configuration."""

from datetime import date, datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Configuration for data fetching."""

    tickers: List[str] = Field(
        default=["AAPL", "GOOGL", "MSFT"],
        description="List of stock ticker symbols to analyze",
    )
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


class StockulaConfig(BaseModel):
    """Main configuration model for Stockula."""

    data: DataConfig = Field(default_factory=DataConfig)
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
