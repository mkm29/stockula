"""Consolidated SQLModel models for TimescaleDB-optimized Stockula database.

This module defines database models using SQLModel, fully optimized for TimescaleDB
with hypertables, compression policies, continuous aggregates, and advanced time-series
features. It consolidates all database model functionality into a single module,
eliminating the need for separate timescale_models.py.

Key Features:
- TimescaleDB hypertables for time-series data (price history, dividends, splits, options)
- JSONB storage for flexible and efficient data structures
- Compression policies for optimal storage and query performance
- Continuous aggregates for fast analytical queries
- Full backward compatibility with existing code
- Type-safe models with comprehensive validation
- PostgreSQL-optimized indexes and constraints

Architecture:
- Time-series tables use TIMESTAMP WITH TIME ZONE for precise partitioning
- Configurable chunk intervals optimized for different data frequencies
- Automated compression after configurable time periods
- Retention policies for long-term data management
- Rich relationship mapping between all entities

Models:
- Stock: Core stock metadata with sector/industry classification
- PriceHistory: OHLCV time-series data with technical indicators
- Dividend: Corporate dividend payment history
- Split: Stock split and reverse split tracking
- OptionsCall/OptionsPut: Options chain data with Greeks
- StockInfo: Raw yfinance metadata stored as JSONB
- Strategy/StrategyPreset: Trading strategy definitions and parameters
- AutoTSModel/AutoTSPreset: Time-series forecasting model configurations
- PriceAggregates*: Continuous aggregates for real-time analytics
"""

import json
from datetime import UTC, datetime
from datetime import date as DateType
from typing import Any, ClassVar, Optional, cast

from sqlalchemy import TIMESTAMP, Index, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, Relationship, SQLModel, UniqueConstraint

# TimescaleDB optimization constants
PRICE_CHUNK_INTERVAL = "1 day"
DIVIDEND_CHUNK_INTERVAL = "1 month"
SPLIT_CHUNK_INTERVAL = "1 year"
OPTIONS_CHUNK_INTERVAL = "1 day"
COMPRESSION_INTERVAL = "7 days"
RETENTION_PERIOD = "5 years"


class Stock(SQLModel, table=True):  # type: ignore[call-arg]
    """Stock metadata table with PostgreSQL optimizations."""

    __tablename__ = "stocks"
    __table_args__ = (
        Index("idx_stocks_symbol", "symbol"),
        Index("idx_stocks_sector", "sector"),
        Index("idx_stocks_market_cap", "market_cap"),
        Index("idx_stocks_updated_at", "updated_at"),
    )

    symbol: str = Field(primary_key=True, description="Stock ticker symbol")
    name: str | None = Field(default=None, description="Company name")
    sector: str | None = Field(default=None, description="Business sector")
    industry: str | None = Field(default=None, description="Industry classification")
    market_cap: float | None = Field(default=None, description="Market capitalization")
    exchange: str | None = Field(default=None, description="Stock exchange")
    currency: str | None = Field(default="USD", description="Trading currency")

    # Use PostgreSQL TIMESTAMP WITH TIME ZONE
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(TIMESTAMP(timezone=True), server_default=func.current_timestamp()),
        description="Timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(
            TIMESTAMP(timezone=True),
            server_default=func.current_timestamp(),
            onupdate=func.current_timestamp(),
        ),
        description="Timestamp when the record was last updated",
    )

    # Relationships
    price_history: list["PriceHistory"] = Relationship(back_populates="stock", cascade_delete=True)
    dividends: list["Dividend"] = Relationship(back_populates="stock", cascade_delete=True)
    splits: list["Split"] = Relationship(back_populates="stock", cascade_delete=True)
    options_calls: list["OptionsCall"] = Relationship(back_populates="stock", cascade_delete=True)
    options_puts: list["OptionsPut"] = Relationship(back_populates="stock", cascade_delete=True)
    stock_info: Optional["StockInfo"] = Relationship(back_populates="stock", cascade_delete=True)


class PriceHistory(SQLModel, table=True):  # type: ignore[call-arg]
    """TimescaleDB hypertable for historical OHLCV price data.

    This table is designed as a TimescaleDB hypertable partitioned by time
    for optimal time-series performance.
    """

    __tablename__ = "price_history"
    __table_args__ = (
        # Unique constraint for data integrity
        UniqueConstraint("symbol", "timestamp", "interval", name="uq_price_history"),
        # Optimized indexes for TimescaleDB time-series queries
        Index("idx_price_history_symbol_timestamp", "symbol", "timestamp"),
        Index("idx_price_history_timestamp", "timestamp"),  # Time-based queries
        Index("idx_price_history_symbol_interval", "symbol", "interval"),
        Index("idx_price_history_volume", "volume"),  # Volume analysis
        Index("idx_price_history_close_price", "close_price"),  # Price analysis
    )

    id: int | None = Field(default=None, primary_key=True, description="Primary key")
    symbol: str = Field(foreign_key="stocks.symbol", description="Stock ticker symbol")

    # Use TIMESTAMP WITH TIME ZONE for TimescaleDB partitioning
    timestamp: datetime = Field(
        sa_column=Column(TIMESTAMP(timezone=True)), description="Timestamp with timezone for time-series partitioning"
    )

    # OHLCV data
    open_price: float | None = Field(default=None, description="Opening price")
    high_price: float | None = Field(default=None, description="Highest price")
    low_price: float | None = Field(default=None, description="Lowest price")
    close_price: float | None = Field(default=None, description="Closing price")
    volume: int | None = Field(default=None, description="Trading volume")
    interval: str = Field(default="1d", description="Data interval (1d, 1h, etc.)")

    # Additional computed fields for technical analysis
    adjusted_close: float | None = Field(default=None, description="Adjusted closing price")
    typical_price: float | None = Field(default=None, description="Typical price (H+L+C)/3")
    price_change: float | None = Field(default=None, description="Price change from previous close")
    price_change_pct: float | None = Field(default=None, description="Price change percentage")

    # Backward compatibility: support date-based queries
    date: DateType | None = Field(default=None, description="Trading date (computed from timestamp)")

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(TIMESTAMP(timezone=True), server_default=func.current_timestamp()),
        description="Timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(
            TIMESTAMP(timezone=True),
            server_default=func.current_timestamp(),
            onupdate=func.current_timestamp(),
        ),
        description="Timestamp when the record was last updated",
    )

    # Relationships
    stock: Stock = Relationship(back_populates="price_history")

    @classmethod
    def create_hypertable_sql(cls) -> str:
        """Generate SQL to create TimescaleDB hypertable for price data."""
        return f"""
        SELECT create_hypertable(
            '{cls.__tablename__}',
            'timestamp',
            chunk_time_interval => INTERVAL '{PRICE_CHUNK_INTERVAL}',
            if_not_exists => TRUE
        );
        """

    @classmethod
    def create_compression_policy_sql(cls) -> str:
        """Generate SQL to create compression policy for price data.

        Optimized for time-series queries with symbol and interval segmentation.
        """
        return f"""
        ALTER TABLE {cls.__tablename__} SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'symbol, interval',
            timescaledb.compress_orderby = 'timestamp DESC'
        );

        SELECT add_compression_policy('{cls.__tablename__}', INTERVAL '{COMPRESSION_INTERVAL}');
        """

    @classmethod
    def create_retention_policy_sql(cls, retention_period: str = RETENTION_PERIOD) -> str:
        """Generate SQL to create data retention policy for price data."""
        return f"""
        SELECT add_retention_policy('{cls.__tablename__}', INTERVAL '{retention_period}');
        """


class Dividend(SQLModel, table=True):  # type: ignore[call-arg]
    """TimescaleDB-optimized dividend payment history."""

    __tablename__ = "dividends"
    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", name="uq_dividends"),
        Index("idx_dividends_symbol_timestamp", "symbol", "timestamp"),
        Index("idx_dividends_timestamp", "timestamp"),
        Index("idx_dividends_amount", "amount"),
    )

    id: int | None = Field(default=None, primary_key=True, description="Primary key")
    symbol: str = Field(foreign_key="stocks.symbol", description="Stock ticker symbol")

    timestamp: datetime = Field(sa_column=Column(TIMESTAMP(timezone=True)), description="Dividend payment timestamp")
    amount: float = Field(description="Dividend amount per share")
    currency: str | None = Field(default="USD", description="Currency of dividend")
    dividend_type: str | None = Field(default="cash", description="Type of dividend (cash, stock, etc.)")

    # Backward compatibility
    date: DateType | None = Field(default=None, description="Dividend payment date (computed from timestamp)")

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(TIMESTAMP(timezone=True), server_default=func.current_timestamp()),
        description="Timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(
            TIMESTAMP(timezone=True),
            server_default=func.current_timestamp(),
            onupdate=func.current_timestamp(),
        ),
        description="Timestamp when the record was last updated",
    )

    # Relationships
    stock: Stock = Relationship(back_populates="dividends")

    @classmethod
    def create_hypertable_sql(cls) -> str:
        """Generate SQL to create TimescaleDB hypertable for dividend data."""
        return f"""
        SELECT create_hypertable(
            '{cls.__tablename__}',
            'timestamp',
            chunk_time_interval => INTERVAL '{DIVIDEND_CHUNK_INTERVAL}',
            if_not_exists => TRUE
        );
        """


class Split(SQLModel, table=True):  # type: ignore[call-arg]
    """TimescaleDB-optimized stock split history."""

    __tablename__ = "splits"
    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", name="uq_splits"),
        Index("idx_splits_symbol_timestamp", "symbol", "timestamp"),
        Index("idx_splits_timestamp", "timestamp"),
        Index("idx_splits_ratio", "ratio"),
    )

    id: int | None = Field(default=None, primary_key=True, description="Primary key")
    symbol: str = Field(foreign_key="stocks.symbol", description="Stock ticker symbol")

    timestamp: datetime = Field(sa_column=Column(TIMESTAMP(timezone=True)), description="Split timestamp")
    ratio: float = Field(description="Split ratio (e.g., 2.0 for 2:1 split)")
    split_type: str | None = Field(default="forward", description="Type of split (forward, reverse)")

    # Backward compatibility
    date: DateType | None = Field(default=None, description="Split date (computed from timestamp)")

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(TIMESTAMP(timezone=True), server_default=func.current_timestamp()),
        description="Timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(
            TIMESTAMP(timezone=True),
            server_default=func.current_timestamp(),
            onupdate=func.current_timestamp(),
        ),
        description="Timestamp when the record was last updated",
    )

    # Relationships
    stock: Stock = Relationship(back_populates="splits")

    @classmethod
    def create_hypertable_sql(cls) -> str:
        """Generate SQL to create TimescaleDB hypertable for stock split data."""
        return f"""
        SELECT create_hypertable(
            '{cls.__tablename__}',
            'timestamp',
            chunk_time_interval => INTERVAL '{SPLIT_CHUNK_INTERVAL}',
            if_not_exists => TRUE
        );
        """


class OptionsCall(SQLModel, table=True):  # type: ignore[call-arg]
    """TimescaleDB-optimized call options data."""

    __tablename__ = "options_calls"
    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "expiration_timestamp",
            "strike",
            "contract_symbol",
            "data_timestamp",
            name="uq_options_calls",
        ),
        Index("idx_options_calls_symbol_exp", "symbol", "expiration_timestamp"),
        Index("idx_options_calls_data_timestamp", "data_timestamp"),
        Index("idx_options_calls_strike", "strike"),
        Index("idx_options_calls_volume", "volume"),
    )

    id: int | None = Field(default=None, primary_key=True, description="Primary key")
    symbol: str = Field(foreign_key="stocks.symbol", description="Stock ticker symbol")

    data_timestamp: datetime = Field(
        sa_column=Column(TIMESTAMP(timezone=True)), description="When the options data was recorded"
    )
    expiration_timestamp: datetime = Field(
        sa_column=Column(TIMESTAMP(timezone=True)), description="Option expiration timestamp"
    )

    strike: float = Field(description="Strike price")
    last_price: float | None = Field(default=None, description="Last traded price")
    bid: float | None = Field(default=None, description="Bid price")
    ask: float | None = Field(default=None, description="Ask price")
    volume: int | None = Field(default=None, description="Trading volume")
    open_interest: int | None = Field(default=None, description="Open interest")
    implied_volatility: float | None = Field(default=None, description="Implied volatility")
    in_the_money: bool | None = Field(default=None, description="Whether option is in the money")
    contract_symbol: str | None = Field(default=None, description="Option contract symbol")

    # Additional calculated fields for advanced options analysis
    intrinsic_value: float | None = Field(default=None, description="Intrinsic value")
    time_value: float | None = Field(default=None, description="Time value")
    delta: float | None = Field(default=None, description="Delta Greek")
    gamma: float | None = Field(default=None, description="Gamma Greek")
    theta: float | None = Field(default=None, description="Theta Greek")
    vega: float | None = Field(default=None, description="Vega Greek")

    # Backward compatibility
    expiration_date: DateType | None = Field(default=None, description="Option expiration date (computed)")

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(TIMESTAMP(timezone=True), server_default=func.current_timestamp()),
        description="Timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(
            TIMESTAMP(timezone=True),
            server_default=func.current_timestamp(),
            onupdate=func.current_timestamp(),
        ),
        description="Timestamp when the record was last updated",
    )

    # Relationships
    stock: Stock = Relationship(back_populates="options_calls")

    @classmethod
    def create_hypertable_sql(cls) -> str:
        """Generate SQL to create TimescaleDB hypertable for call options data."""
        return f"""
        SELECT create_hypertable(
            '{cls.__tablename__}',
            'data_timestamp',
            chunk_time_interval => INTERVAL '{OPTIONS_CHUNK_INTERVAL}',
            if_not_exists => TRUE
        );
        """

    @classmethod
    def create_compression_policy_sql(cls) -> str:
        """Generate SQL to create compression policy for call options data."""
        return f"""
        ALTER TABLE {cls.__tablename__} SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'symbol, expiration_timestamp',
            timescaledb.compress_orderby = 'data_timestamp DESC'
        );

        SELECT add_compression_policy('{cls.__tablename__}', INTERVAL '{COMPRESSION_INTERVAL}');
        """


class OptionsPut(SQLModel, table=True):  # type: ignore[call-arg]
    """TimescaleDB-optimized put options data."""

    __tablename__ = "options_puts"
    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "expiration_timestamp",
            "strike",
            "contract_symbol",
            "data_timestamp",
            name="uq_options_puts",
        ),
        Index("idx_options_puts_symbol_exp", "symbol", "expiration_timestamp"),
        Index("idx_options_puts_data_timestamp", "data_timestamp"),
        Index("idx_options_puts_strike", "strike"),
        Index("idx_options_puts_volume", "volume"),
    )

    id: int | None = Field(default=None, primary_key=True, description="Primary key")
    symbol: str = Field(foreign_key="stocks.symbol", description="Stock ticker symbol")

    data_timestamp: datetime = Field(
        sa_column=Column(TIMESTAMP(timezone=True)), description="When the options data was recorded"
    )
    expiration_timestamp: datetime = Field(
        sa_column=Column(TIMESTAMP(timezone=True)), description="Option expiration timestamp"
    )

    strike: float = Field(description="Strike price")
    last_price: float | None = Field(default=None, description="Last traded price")
    bid: float | None = Field(default=None, description="Bid price")
    ask: float | None = Field(default=None, description="Ask price")
    volume: int | None = Field(default=None, description="Trading volume")
    open_interest: int | None = Field(default=None, description="Open interest")
    implied_volatility: float | None = Field(default=None, description="Implied volatility")
    in_the_money: bool | None = Field(default=None, description="Whether option is in the money")
    contract_symbol: str | None = Field(default=None, description="Option contract symbol")

    # Additional calculated fields for advanced options analysis
    intrinsic_value: float | None = Field(default=None, description="Intrinsic value")
    time_value: float | None = Field(default=None, description="Time value")
    delta: float | None = Field(default=None, description="Delta Greek")
    gamma: float | None = Field(default=None, description="Gamma Greek")
    theta: float | None = Field(default=None, description="Theta Greek")
    vega: float | None = Field(default=None, description="Vega Greek")

    # Backward compatibility
    expiration_date: DateType | None = Field(default=None, description="Option expiration date (computed)")

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(TIMESTAMP(timezone=True), server_default=func.current_timestamp()),
        description="Timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(
            TIMESTAMP(timezone=True),
            server_default=func.current_timestamp(),
            onupdate=func.current_timestamp(),
        ),
        description="Timestamp when the record was last updated",
    )

    # Relationships
    stock: Stock = Relationship(back_populates="options_puts")

    @classmethod
    def create_hypertable_sql(cls) -> str:
        """Generate SQL to create TimescaleDB hypertable for put options data."""
        return f"""
        SELECT create_hypertable(
            '{cls.__tablename__}',
            'data_timestamp',
            chunk_time_interval => INTERVAL '{OPTIONS_CHUNK_INTERVAL}',
            if_not_exists => TRUE
        );
        """

    @classmethod
    def create_compression_policy_sql(cls) -> str:
        """Generate SQL to create compression policy for put options data."""
        return f"""
        ALTER TABLE {cls.__tablename__} SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'symbol, expiration_timestamp',
            timescaledb.compress_orderby = 'data_timestamp DESC'
        );

        SELECT add_compression_policy('{cls.__tablename__}', INTERVAL '{COMPRESSION_INTERVAL}');
        """


class StockInfo(SQLModel, table=True):  # type: ignore[call-arg]
    """Raw yfinance info data stored as JSONB for efficient querying."""

    __tablename__ = "stock_info"

    symbol: str = Field(foreign_key="stocks.symbol", primary_key=True, description="Stock ticker symbol")

    # Use PostgreSQL JSONB for better performance and querying
    info_jsonb: dict[str, Any] | None = Field(
        default=None,
        sa_column=Column(JSONB, nullable=True),
        description="JSONB-encoded stock information from yfinance",
    )

    # Backward compatibility with text JSON
    info_json: str | None = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
        description="JSON-encoded stock information (legacy)",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(TIMESTAMP(timezone=True), server_default=func.current_timestamp()),
        description="Timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(
            TIMESTAMP(timezone=True),
            server_default=func.current_timestamp(),
            onupdate=func.current_timestamp(),
        ),
        description="Timestamp when the record was last updated",
    )

    # Relationships
    stock: Stock = Relationship(back_populates="stock_info")

    @property
    def info_dict(self) -> dict[str, Any]:
        """Get info as dictionary, preferring JSONB over JSON."""
        if self.info_jsonb is not None:
            return self.info_jsonb
        elif self.info_json is not None:
            return cast(dict[str, Any], json.loads(self.info_json))
        else:
            return {}

    def set_info(self, info: dict[str, Any]) -> None:
        """Set info from dictionary, storing in both formats for compatibility."""
        self.info_jsonb = info
        self.info_json = json.dumps(info, default=str)


class Strategy(SQLModel, table=True):  # type: ignore[call-arg]
    """Model for trading strategies with PostgreSQL optimizations."""

    __tablename__ = "strategies"
    __table_args__ = (
        UniqueConstraint("name", name="uq_strategy_name"),
        Index("idx_strategies_name", "name"),
        Index("idx_strategies_category", "category"),
        Index("idx_strategies_is_active", "is_active"),
    )

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(description="Strategy name")
    class_name: str = Field(description="Full class name (e.g., 'SMACrossStrategy')")
    module_path: str = Field(description="Module path for importing")
    description: str | None = Field(default=None, description="Strategy description")
    category: str | None = Field(default=None, description="Strategy category (e.g., 'momentum', 'trend')")
    is_active: bool = Field(default=True, description="Whether strategy is active")

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(TIMESTAMP(timezone=True), server_default=func.current_timestamp()),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(
            TIMESTAMP(timezone=True),
            server_default=func.current_timestamp(),
            onupdate=func.current_timestamp(),
        ),
    )

    # Relationships
    presets: list["StrategyPreset"] = Relationship(back_populates="strategy")


class StrategyPreset(SQLModel, table=True):  # type: ignore[call-arg]
    """Model for strategy parameter presets with JSONB storage."""

    __tablename__ = "strategy_presets"
    __table_args__ = (
        UniqueConstraint("strategy_id", "name", name="uq_strategy_preset"),
        Index("idx_strategy_presets_strategy_id", "strategy_id"),
        Index("idx_strategy_presets_name", "name"),
        Index("idx_strategy_presets_is_default", "is_default"),
    )

    id: int | None = Field(default=None, primary_key=True)
    strategy_id: int = Field(foreign_key="strategies.id", description="Strategy foreign key")
    name: str = Field(description="Preset name (e.g., 'default', 'conservative', 'aggressive')")
    description: str | None = Field(default=None, description="Preset description")
    is_default: bool = Field(default=False, description="Whether this is the default preset")

    # Use JSONB for better parameter querying and performance
    parameters_jsonb: dict[str, Any] | None = Field(
        default=None,
        sa_column=Column(JSONB, nullable=True),
        description="JSONB parameters for better querying",
    )

    # Backward compatibility
    parameters_json: str | None = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
        description="JSON string of parameters (legacy)",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(TIMESTAMP(timezone=True), server_default=func.current_timestamp()),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(
            TIMESTAMP(timezone=True),
            server_default=func.current_timestamp(),
            onupdate=func.current_timestamp(),
        ),
    )

    # Relationships
    strategy: Strategy = Relationship(back_populates="presets")

    @property
    def parameters(self) -> dict[str, Any]:
        """Get parameters as dictionary, preferring JSONB over JSON."""
        if self.parameters_jsonb is not None:
            return self.parameters_jsonb
        elif self.parameters_json is not None:
            return cast(dict[str, Any], json.loads(self.parameters_json))
        else:
            return {}

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Set parameters from dictionary, storing in both formats."""
        self.parameters_jsonb = params
        self.parameters_json = json.dumps(params, default=str)


class AutoTSModel(SQLModel, table=True):  # type: ignore[call-arg]
    """AutoTS model definitions and metadata with PostgreSQL optimizations.

    This model stores valid AutoTS models and their properties.
    Validation is performed before saving to ensure only valid models are stored.
    """

    __tablename__ = "autots_models"
    __table_args__ = (
        UniqueConstraint("name", name="uq_autots_model_name"),
        Index("idx_autots_models_name", "name"),
        Index("idx_autots_models_is_slow", "is_slow"),
        Index("idx_autots_models_is_gpu_enabled", "is_gpu_enabled"),
    )

    # Class-level registry of valid models (loaded from models.json)
    _valid_models: ClassVar[set[str] | None] = None
    _models_data: ClassVar[dict[str, dict] | None] = None

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(description="Model name (e.g., ARIMA, ETS)")
    description: str | None = Field(default=None, description="Model description")
    is_slow: bool = Field(default=False, description="Whether model is computationally expensive")
    is_gpu_enabled: bool = Field(default=False, description="Whether model can utilize GPU")
    requires_regressor: bool = Field(default=False, description="Whether model supports external regressors")
    min_data_points: int = Field(default=10, description="Minimum data points required")

    # Use JSONB for categories
    categories_jsonb: list[str] | None = Field(
        default=None,
        sa_column=Column(JSONB, nullable=True),
        description="JSONB array of categories",
    )

    # Backward compatibility
    categories: str = Field(default="[]", description="JSON array of categories (legacy)")

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(TIMESTAMP(timezone=True), server_default=func.current_timestamp()),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(
            TIMESTAMP(timezone=True),
            server_default=func.current_timestamp(),
            onupdate=func.current_timestamp(),
        ),
    )

    @classmethod
    def load_valid_models(cls, force_reload: bool = False) -> None:
        """Load valid models from AutoTS library.

        Args:
            force_reload: Force reload even if already loaded
        """
        if cls._valid_models is not None and not force_reload:
            return

        try:
            # Get the authoritative model list from AutoTS itself
            from autots.models.model_list import model_lists

            # The model_lists is a dictionary of presets
            # The 'all' preset contains all available models
            all_models = set(model_lists.get("all", []))

            cls._valid_models = all_models
            cls._models_data = model_lists  # Store the full model_lists for reference

        except ImportError:
            # Fallback if AutoTS is not available (shouldn't happen in normal operation)
            # Use a minimal set of known models
            cls._valid_models = {
                "ARIMA",
                "ETS",
                "FBProphet",
                "GluonTS",
                "VAR",
                "VECM",
                "Theta",
                "UnivariateMotif",
                "MultivariateMotif",
                "LastValueNaive",
                "ConstantNaive",
                "AverageValueNaive",
                "SeasonalNaive",
                "GLM",
                "GLS",
                "RollingRegression",
                "UnobservedComponents",
                "DynamicFactor",
                "WindowRegression",
                "DatepartRegression",
                "UnivariateRegression",
                "NVAR",
                "MultivariateRegression",
                "ARDL",
                "NeuralProphet",
            }
            cls._models_data = {}

    @classmethod
    def is_valid_model(cls, name: str) -> bool:
        """Check if a model name is valid.

        Args:
            name: Model name to validate

        Returns:
            True if model is valid, False otherwise
        """
        cls.load_valid_models()
        return name in (cls._valid_models or set())

    @classmethod
    def get_valid_models(cls) -> set[str]:
        """Get set of all valid model names.

        Returns:
            Set of valid model names
        """
        cls.load_valid_models()
        return cls._valid_models.copy() if cls._valid_models else set()

    @classmethod
    def validate_model_list(cls, models: list[str]) -> tuple[bool, list[str]]:
        """Validate a list of model names.

        Args:
            models: List of model names to validate

        Returns:
            Tuple of (all_valid, invalid_models)
        """
        cls.load_valid_models()
        invalid = [m for m in models if not cls.is_valid_model(m)]
        return len(invalid) == 0, invalid

    def validate_model(self) -> None:
        """Validate the model before saving.

        Raises:
            ValueError: If the model name is not a valid AutoTS model
        """
        if not self.is_valid_model(self.name):
            valid_models = self.get_valid_models()
            raise ValueError(
                f"'{self.name}' is not a valid AutoTS model. "
                f"Valid models include: {', '.join(sorted(list(valid_models)[:10]))}..."
            )

    @property
    def category_list(self) -> list[str]:
        """Get categories as list, preferring JSONB over JSON."""
        if self.categories_jsonb is not None:
            return self.categories_jsonb
        else:
            return cast(list[str], json.loads(self.categories))

    def set_categories(self, categories: list[str]) -> None:
        """Set categories from list, storing in both formats."""
        self.categories_jsonb = categories
        self.categories = json.dumps(categories)


class AutoTSPreset(SQLModel, table=True):  # type: ignore[call-arg]
    """AutoTS model preset configurations with PostgreSQL optimizations.

    This model stores preset configurations that group models together.
    Validation ensures all models in a preset are valid AutoTS models.
    """

    __tablename__ = "autots_presets"
    __table_args__ = (
        UniqueConstraint("name", name="uq_autots_preset_name"),
        Index("idx_autots_presets_name", "name"),
        Index("idx_autots_presets_use_case", "use_case"),
    )

    # Class-level registry of valid presets
    _valid_presets: ClassVar[set[str] | None] = None

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(description="Preset name (e.g., fast, superfast)")
    description: str | None = Field(default=None, description="Preset description")
    use_case: str | None = Field(default=None, description="Recommended use case")

    # Use JSONB for models
    models_jsonb: list[str] | dict[str, float] | None = Field(
        default=None,
        sa_column=Column(JSONB, nullable=True),
        description="JSONB array of model names or dict with weights",
    )

    # Backward compatibility
    models: str | None = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
        description="JSON array of model names or dict with weights (legacy)",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(TIMESTAMP(timezone=True), server_default=func.current_timestamp()),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(
            TIMESTAMP(timezone=True),
            server_default=func.current_timestamp(),
            onupdate=func.current_timestamp(),
        ),
    )

    @classmethod
    def load_valid_presets(cls, force_reload: bool = False) -> None:
        """Load valid presets from AutoTS library.

        Args:
            force_reload: Force reload even if already loaded
        """
        if cls._valid_presets is not None and not force_reload:
            return

        try:
            # Get presets directly from AutoTS
            from autots.models.model_list import model_lists

            # All keys in model_lists are valid presets
            cls._valid_presets = set(model_lists.keys())
        except ImportError:
            # Fallback to known AutoTS presets
            cls._valid_presets = {
                "fast",
                "superfast",
                "default",
                "parallel",
                "fast_parallel",
                "scalable",
                "probabilistic",
                "multivariate",
                "univariate",
                "best",
                "slow",
                "gpu",
                "regressor",
                "motifs",
                "regressions",
                "all",
                "no_shared",
                "experimental",
            }

    @classmethod
    def is_valid_preset(cls, name: str) -> bool:
        """Check if a preset name is valid.

        Args:
            name: Preset name to validate

        Returns:
            True if preset is valid, False otherwise
        """
        cls.load_valid_presets()
        return name in (cls._valid_presets or set())

    def validate_model(self) -> None:
        """Validate the preset before saving.

        Raises:
            ValueError: If preset contains invalid models
        """
        # Parse the models
        model_list = self.model_list

        # Extract model names depending on format
        if isinstance(model_list, list):
            model_names = model_list
        elif isinstance(model_list, dict):
            model_names = list(model_list.keys())
        else:
            raise ValueError(f"Invalid models format: {type(model_list)}")

        # Validate all models in the preset
        is_valid, invalid_models = AutoTSModel.validate_model_list(model_names)
        if not is_valid:
            raise ValueError(f"Preset '{self.name}' contains invalid models: {', '.join(invalid_models)}")

    @property
    def model_list(self) -> list[str] | dict[str, float]:
        """Get models as list or dict, preferring JSONB over JSON."""
        if self.models_jsonb is not None:
            return self.models_jsonb
        elif self.models is not None:
            data = json.loads(self.models)
            return cast("list[str] | dict[str, float]", data)
        else:
            return []

    def set_models(self, models: list[str] | dict[str, float]) -> None:
        """Set models from list or dict, storing in both formats."""
        self.models_jsonb = models
        self.models = json.dumps(models)


# Continuous aggregates for common time-series queries
class PriceAggregatesDaily(SQLModel, table=True):  # type: ignore[call-arg]
    """Continuous aggregate for daily price statistics."""

    __tablename__ = "price_aggregates_daily"
    __table_args__ = (
        Index("idx_price_aggregates_daily_symbol_day", "symbol", "day"),
        Index("idx_price_aggregates_daily_day", "day"),
    )

    symbol: str = Field(primary_key=True, description="Stock ticker symbol")
    day: DateType = Field(primary_key=True, description="Date")

    # OHLCV aggregates
    open_price: float | None = Field(description="Opening price")
    high_price: float | None = Field(description="Highest price")
    low_price: float | None = Field(description="Lowest price")
    close_price: float | None = Field(description="Closing price")
    avg_price: float | None = Field(description="Average price")
    volume: int | None = Field(description="Total volume")

    # Additional statistics
    price_volatility: float | None = Field(description="Price volatility (stddev)")
    trade_count: int | None = Field(description="Number of price updates")

    @classmethod
    def create_continuous_aggregate_sql(cls) -> str:
        """Generate SQL to create continuous aggregate."""
        return f"""
        CREATE MATERIALIZED VIEW {cls.__tablename__}
        WITH (timescaledb.continuous) AS
        SELECT
            symbol,
            time_bucket('1 day', timestamp) AS day,
            first(open_price, timestamp) AS open_price,
            max(high_price) AS high_price,
            min(low_price) AS low_price,
            last(close_price, timestamp) AS close_price,
            avg(close_price) AS avg_price,
            sum(volume) AS volume,
            stddev(close_price) AS price_volatility,
            count(*) AS trade_count
        FROM price_history
        WHERE timestamp >= NOW() - INTERVAL '1 year'
        GROUP BY symbol, time_bucket('1 day', timestamp);
        """


class PriceAggregatesHourly(SQLModel, table=True):  # type: ignore[call-arg]
    """Continuous aggregate for hourly price statistics."""

    __tablename__ = "price_aggregates_hourly"
    __table_args__ = (
        Index("idx_price_aggregates_hourly_symbol_hour", "symbol", "hour"),
        Index("idx_price_aggregates_hourly_hour", "hour"),
    )

    symbol: str = Field(primary_key=True, description="Stock ticker symbol")
    hour: datetime = Field(primary_key=True, description="Hour bucket")

    # OHLCV aggregates
    open_price: float | None = Field(description="Opening price")
    high_price: float | None = Field(description="Highest price")
    low_price: float | None = Field(description="Lowest price")
    close_price: float | None = Field(description="Closing price")
    avg_price: float | None = Field(description="Average price")
    volume: int | None = Field(description="Total volume")

    # Additional statistics
    price_volatility: float | None = Field(description="Price volatility (stddev)")
    trade_count: int | None = Field(description="Number of price updates")

    @classmethod
    def create_continuous_aggregate_sql(cls) -> str:
        """Generate SQL to create continuous aggregate."""
        return f"""
        CREATE MATERIALIZED VIEW {cls.__tablename__}
        WITH (timescaledb.continuous) AS
        SELECT
            symbol,
            time_bucket('1 hour', timestamp) AS hour,
            first(open_price, timestamp) AS open_price,
            max(high_price) AS high_price,
            min(low_price) AS low_price,
            last(close_price, timestamp) AS close_price,
            avg(close_price) AS avg_price,
            sum(volume) AS volume,
            stddev(close_price) AS price_volatility,
            count(*) AS trade_count
        FROM price_history
        WHERE timestamp >= NOW() - INTERVAL '30 days'
        GROUP BY symbol, time_bucket('1 hour', timestamp);
        """


# Helper functions for TimescaleDB setup
def get_hypertable_models() -> list[type[SQLModel]]:
    """Get all models that should be configured as hypertables."""
    return [
        PriceHistory,
        Dividend,
        Split,
        OptionsCall,
        OptionsPut,
    ]


def get_continuous_aggregate_models() -> list[type[SQLModel]]:
    """Get all continuous aggregate models."""
    return [
        PriceAggregatesDaily,
        PriceAggregatesHourly,
    ]


def get_timescale_setup_sql() -> list[str]:
    """Generate complete SQL setup for TimescaleDB features.

    Returns:
        List of SQL commands to set up TimescaleDB features
    """
    sql_commands = []

    # Create hypertables
    for model in get_hypertable_models():
        if hasattr(model, "create_hypertable_sql"):
            sql_commands.append(model.create_hypertable_sql())

            # Add compression and retention policies
            if hasattr(model, "create_compression_policy_sql"):
                sql_commands.append(model.create_compression_policy_sql())

                # Add retention policy for price history
                if model == PriceHistory and hasattr(model, "create_retention_policy_sql"):
                    sql_commands.append(model.create_retention_policy_sql())

    # Create continuous aggregates
    for model in get_continuous_aggregate_models():
        if hasattr(model, "create_continuous_aggregate_sql"):
            sql_commands.append(model.create_continuous_aggregate_sql())

    return sql_commands


def get_all_models() -> list[type[SQLModel]]:
    """Get all model classes for metadata creation.

    Returns:
        List of all SQLModel classes
    """
    return [
        Stock,
        PriceHistory,
        Dividend,
        Split,
        OptionsCall,
        OptionsPut,
        StockInfo,
        Strategy,
        StrategyPreset,
        AutoTSModel,
        AutoTSPreset,
        PriceAggregatesDaily,
        PriceAggregatesHourly,
    ]


# Backward compatibility aliases for timescale_models.py imports
# These aliases allow existing code to import TimescaleXXX classes seamlessly

# Core model aliases
TimescaleStock = Stock
TimescalePriceHistory = PriceHistory
TimescaleDividend = Dividend
TimescaleSplit = Split
TimescaleOptionsCall = OptionsCall
TimescaleOptionsPut = OptionsPut
TimescaleStockInfo = StockInfo

# Continuous aggregate aliases
TimescalePriceAggregatesDaily = PriceAggregatesDaily
TimescalePriceAggregatesHourly = PriceAggregatesHourly


# Helper function aliases for backward compatibility
def get_timescale_hypertable_models() -> list[type[SQLModel]]:
    """Backward compatibility function for timescale_models.py.

    Returns the same models as get_hypertable_models().
    """
    return get_hypertable_models()


def get_timescale_continuous_aggregates() -> list[type[SQLModel]]:
    """Backward compatibility function for timescale_models.py.

    Returns the same models as get_continuous_aggregate_models().
    """
    return get_continuous_aggregate_models()


# Note: get_timescale_setup_sql() is already defined above and is the main function
