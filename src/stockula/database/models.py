"""SQLModel models for the Stockula database.

This module defines database models using SQLModel, which combines
SQLAlchemy ORM with Pydantic validation.
"""

import json
from datetime import UTC, datetime
from datetime import date as DateType
from typing import ClassVar, Optional

from sqlalchemy import DateTime as SQLADateTime
from sqlalchemy import func
from sqlmodel import Column, Field, Index, Relationship, SQLModel, Text, UniqueConstraint


class Stock(SQLModel, table=True):  # type: ignore[call-arg]
    """Basic stock metadata."""

    __tablename__ = "stocks"

    symbol: str = Field(primary_key=True, description="Stock ticker symbol")
    name: str | None = Field(default=None, description="Company name")
    sector: str | None = Field(default=None, description="Business sector")
    industry: str | None = Field(default=None, description="Industry classification")
    market_cap: float | None = Field(default=None, description="Market capitalization")
    exchange: str | None = Field(default=None, description="Stock exchange")
    currency: str | None = Field(default=None, description="Trading currency")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(SQLADateTime, server_default=func.current_timestamp()),
        description="Timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(
            SQLADateTime,
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
    """Historical OHLCV price data."""

    __tablename__ = "price_history"
    __table_args__ = (
        UniqueConstraint("symbol", "date", "interval", name="uq_price_history"),
        Index("idx_price_history_symbol_date", "symbol", "date"),
    )

    id: int | None = Field(default=None, primary_key=True, description="Primary key")
    symbol: str = Field(foreign_key="stocks.symbol", description="Stock ticker symbol")
    date: DateType = Field(description="Trading date")
    open_price: float | None = Field(default=None, description="Opening price")
    high_price: float | None = Field(default=None, description="Highest price")
    low_price: float | None = Field(default=None, description="Lowest price")
    close_price: float | None = Field(default=None, description="Closing price")
    volume: int | None = Field(default=None, description="Trading volume")
    interval: str = Field(default="1d", description="Data interval (1d, 1h, etc.)")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(SQLADateTime, server_default=func.current_timestamp()),
        description="Timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(
            SQLADateTime,
            server_default=func.current_timestamp(),
            onupdate=func.current_timestamp(),
        ),
        description="Timestamp when the record was last updated",
    )

    # Relationships
    stock: Stock = Relationship(back_populates="price_history")


class Dividend(SQLModel, table=True):  # type: ignore[call-arg]
    """Dividend payment history."""

    __tablename__ = "dividends"
    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_dividends"),
        Index("idx_dividends_symbol_date", "symbol", "date"),
    )

    id: int | None = Field(default=None, primary_key=True, description="Primary key")
    symbol: str = Field(foreign_key="stocks.symbol", description="Stock ticker symbol")
    date: DateType = Field(description="Dividend payment date")
    amount: float = Field(description="Dividend amount per share")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(SQLADateTime, server_default=func.current_timestamp()),
        description="Timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(
            SQLADateTime,
            server_default=func.current_timestamp(),
            onupdate=func.current_timestamp(),
        ),
        description="Timestamp when the record was last updated",
    )

    # Relationships
    stock: Stock = Relationship(back_populates="dividends")


class Split(SQLModel, table=True):  # type: ignore[call-arg]
    """Stock split history."""

    __tablename__ = "splits"
    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_splits"),
        Index("idx_splits_symbol_date", "symbol", "date"),
    )

    id: int | None = Field(default=None, primary_key=True, description="Primary key")
    symbol: str = Field(foreign_key="stocks.symbol", description="Stock ticker symbol")
    date: DateType = Field(description="Split date")
    ratio: float = Field(description="Split ratio (e.g., 2.0 for 2:1 split)")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(SQLADateTime, server_default=func.current_timestamp()),
        description="Timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(
            SQLADateTime,
            server_default=func.current_timestamp(),
            onupdate=func.current_timestamp(),
        ),
        description="Timestamp when the record was last updated",
    )

    # Relationships
    stock: Stock = Relationship(back_populates="splits")


class OptionsCall(SQLModel, table=True):  # type: ignore[call-arg]
    """Call options data."""

    __tablename__ = "options_calls"
    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "expiration_date",
            "strike",
            "contract_symbol",
            name="uq_options_calls",
        ),
        Index("idx_options_calls_symbol_exp", "symbol", "expiration_date"),
    )

    id: int | None = Field(default=None, primary_key=True, description="Primary key")
    symbol: str = Field(foreign_key="stocks.symbol", description="Stock ticker symbol")
    expiration_date: DateType = Field(description="Option expiration date")
    strike: float = Field(description="Strike price")
    last_price: float | None = Field(default=None, description="Last traded price")
    bid: float | None = Field(default=None, description="Bid price")
    ask: float | None = Field(default=None, description="Ask price")
    volume: int | None = Field(default=None, description="Trading volume")
    open_interest: int | None = Field(default=None, description="Open interest")
    implied_volatility: float | None = Field(default=None, description="Implied volatility")
    in_the_money: bool | None = Field(default=None, description="Whether option is in the money")
    contract_symbol: str | None = Field(default=None, description="Option contract symbol")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(SQLADateTime, server_default=func.current_timestamp()),
        description="Timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(
            SQLADateTime,
            server_default=func.current_timestamp(),
            onupdate=func.current_timestamp(),
        ),
        description="Timestamp when the record was last updated",
    )

    # Relationships
    stock: Stock = Relationship(back_populates="options_calls")


class OptionsPut(SQLModel, table=True):  # type: ignore[call-arg]
    """Put options data."""

    __tablename__ = "options_puts"
    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "expiration_date",
            "strike",
            "contract_symbol",
            name="uq_options_puts",
        ),
        Index("idx_options_puts_symbol_exp", "symbol", "expiration_date"),
    )

    id: int | None = Field(default=None, primary_key=True, description="Primary key")
    symbol: str = Field(foreign_key="stocks.symbol", description="Stock ticker symbol")
    expiration_date: DateType = Field(description="Option expiration date")
    strike: float = Field(description="Strike price")
    last_price: float | None = Field(default=None, description="Last traded price")
    bid: float | None = Field(default=None, description="Bid price")
    ask: float | None = Field(default=None, description="Ask price")
    volume: int | None = Field(default=None, description="Trading volume")
    open_interest: int | None = Field(default=None, description="Open interest")
    implied_volatility: float | None = Field(default=None, description="Implied volatility")
    in_the_money: bool | None = Field(default=None, description="Whether option is in the money")
    contract_symbol: str | None = Field(default=None, description="Option contract symbol")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(SQLADateTime, server_default=func.current_timestamp()),
        description="Timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(
            SQLADateTime,
            server_default=func.current_timestamp(),
            onupdate=func.current_timestamp(),
        ),
        description="Timestamp when the record was last updated",
    )

    # Relationships
    stock: Stock = Relationship(back_populates="options_puts")


class StockInfo(SQLModel, table=True):  # type: ignore[call-arg]
    """Raw yfinance info data stored as JSON."""

    __tablename__ = "stock_info"

    symbol: str = Field(foreign_key="stocks.symbol", primary_key=True, description="Stock ticker symbol")
    info_json: str = Field(
        sa_column=Column(Text, nullable=False),
        description="JSON-encoded stock information from yfinance",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(SQLADateTime, server_default=func.current_timestamp()),
        description="Timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(
            SQLADateTime,
            server_default=func.current_timestamp(),
            onupdate=func.current_timestamp(),
        ),
        description="Timestamp when the record was last updated",
    )

    # Relationships
    stock: Stock = Relationship(back_populates="stock_info")

    @property
    def info_dict(self) -> dict:
        """Parse JSON info to dictionary."""
        return json.loads(self.info_json)

    def set_info(self, info: dict) -> None:
        """Set info from dictionary."""
        self.info_json = json.dumps(info, default=str)


class Strategy(SQLModel, table=True):
    """Model for trading strategies."""

    __tablename__ = "strategies"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    class_name: str = Field()  # Full class name (e.g., 'SMACrossStrategy')
    module_path: str = Field()  # Module path for importing
    description: str | None = Field(default=None)
    category: str | None = Field(default=None)  # e.g., 'momentum', 'trend'
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Relationships
    presets: list["StrategyPreset"] = Relationship(back_populates="strategy")


class StrategyPreset(SQLModel, table=True):
    """Model for strategy parameter presets."""

    __tablename__ = "strategy_presets"

    id: int | None = Field(default=None, primary_key=True)
    strategy_id: int = Field(foreign_key="strategies.id", index=True)
    name: str = Field(index=True)  # e.g., 'default', 'conservative', 'aggressive'
    parameters_json: str = Field()  # JSON string of parameters
    description: str | None = Field(default=None)
    is_default: bool = Field(default=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Relationships
    strategy: Strategy = Relationship(back_populates="presets")

    # Unique constraint on strategy_id + name
    __table_args__ = (UniqueConstraint("strategy_id", "name", name="unique_strategy_preset"),)

    @property
    def parameters(self) -> dict:
        """Parse JSON parameters to dictionary."""
        return json.loads(self.parameters_json)

    def set_parameters(self, params: dict) -> None:
        """Set parameters from dictionary."""
        self.parameters_json = json.dumps(params, default=str)


class AutoTSModel(SQLModel, table=True):
    """AutoTS model definitions and metadata.

    This model stores valid AutoTS models and their properties.
    Validation is performed before saving to ensure only valid models are stored.
    """

    __tablename__ = "autots_models"

    # Class-level registry of valid models (loaded from models.json)
    _valid_models: ClassVar[set[str] | None] = None
    _models_data: ClassVar[dict[str, dict] | None] = None

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True, description="Model name (e.g., ARIMA, ETS)")
    categories: str = Field(default="[]", description="JSON array of categories (univariate, multivariate, etc.)")
    is_slow: bool = Field(default=False, description="Whether model is computationally expensive")
    is_gpu_enabled: bool = Field(default=False, description="Whether model can utilize GPU")
    requires_regressor: bool = Field(default=False, description="Whether model supports external regressors")
    min_data_points: int = Field(default=10, description="Minimum data points required")
    description: str | None = Field(default=None, description="Model description")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def load_valid_models(cls, force_reload: bool = False) -> None:
        """Load valid models from models.json file.

        Args:
            force_reload: Force reload even if already loaded
        """
        if cls._valid_models is not None and not force_reload:
            return

        from pathlib import Path

        # Try to load from models.json
        models_json_path = Path(__file__).parent.parent.parent / "data" / "models.json"
        if models_json_path.exists():
            with open(models_json_path) as f:
                data = json.load(f)

            # Extract all unique models
            all_models = set(data.get("all", []))
            for preset_name, preset_data in data.items():
                if preset_name == "all":
                    continue
                if isinstance(preset_data, list):
                    all_models.update(preset_data)
                elif isinstance(preset_data, dict):
                    all_models.update(preset_data.keys())

            cls._valid_models = all_models
            cls._models_data = data
        else:
            # Fallback to a comprehensive set of known valid models
            # This list should match the models in data/models.json["all"]
            cls._valid_models = {
                "ARIMA",
                "ETS",
                "FBProphet",
                "GluonTS",
                "VAR",
                "VECM",
                "VARMAX",  # Added missing model
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
                "ARCH",
                "RRVAR",
                "MAR",
                "TMF",
                "LATC",
                "KalmanStateSpace",
                "MetricMotif",
                "Cassandra",
                "SeasonalityMotif",
                "MLEnsemble",
                "PreprocessingRegression",
                "FFT",
                "DMD",
                "BasicLinearModel",
                "TVVAR",
                "MotifSimulation",
                "SectionalMotif",
                "DynamicFactorMQ",
                "PytorchForecasting",
                "BallTreeMultivariateMotif",
                "TiDE",
                "NeuralForecast",
                "BallTreeRegressionMotif",
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

    def validate(self) -> None:
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
        """Parse JSON categories to list."""
        return json.loads(self.categories)

    def set_categories(self, categories: list[str]) -> None:
        """Set categories from list."""
        self.categories = json.dumps(categories)


class AutoTSPreset(SQLModel, table=True):
    """AutoTS model preset configurations.

    This model stores preset configurations that group models together.
    Validation ensures all models in a preset are valid AutoTS models.
    """

    __tablename__ = "autots_presets"

    # Class-level registry of valid presets
    _valid_presets: ClassVar[set[str] | None] = None

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True, description="Preset name (e.g., fast, superfast)")
    models: str = Field(description="JSON array of model names or dict with weights")
    description: str | None = Field(default=None, description="Preset description")
    use_case: str | None = Field(default=None, description="Recommended use case")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def load_valid_presets(cls, force_reload: bool = False) -> None:
        """Load valid presets from models.json file.

        Args:
            force_reload: Force reload even if already loaded
        """
        if cls._valid_presets is not None and not force_reload:
            return

        from pathlib import Path

        # Try to load from models.json
        models_json_path = Path(__file__).parent.parent.parent / "data" / "models.json"
        if models_json_path.exists():
            with open(models_json_path) as f:
                data = json.load(f)

            # Extract preset names (excluding special ones)
            presets = set()
            for key in data.keys():
                if key != "all" and not key.startswith("all_") and not key.startswith("no_"):
                    presets.add(key)

            cls._valid_presets = presets
        else:
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
        # Check both our presets and AutoTS built-in presets
        autots_builtins = {
            "fast",
            "superfast",
            "default",
            "parallel",
            "fast_parallel",
            "scalable",
            "probabilistic",
            "multivariate",
            "univariate",
            "all",
            "no_shared",
            "motifs",
            "regressor",
            "best",
            "slow",
            "gpu",
        }
        return name in (cls._valid_presets or set()) or name in autots_builtins

    def validate(self) -> None:
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
        """Parse JSON models to list or dict."""
        data = json.loads(self.models)
        return data

    def set_models(self, models: list[str] | dict[str, float]) -> None:
        """Set models from list or dict."""
        self.models = json.dumps(models)
