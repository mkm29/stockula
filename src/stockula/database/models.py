"""SQLModel models for the Stockula database.

This module defines database models using SQLModel, which combines
SQLAlchemy ORM with Pydantic validation.
"""

from datetime import date as DateType, datetime, timezone
from typing import Optional, List
import json

from sqlmodel import (
    Field,
    SQLModel,
    Relationship,
    Column,
    Text,
    Index,
    UniqueConstraint,
)
from sqlalchemy import func, DateTime as SQLADateTime


class Stock(SQLModel, table=True):
    """Basic stock metadata."""

    __tablename__ = "stocks"

    symbol: str = Field(primary_key=True, description="Stock ticker symbol")
    name: Optional[str] = Field(default=None, description="Company name")
    sector: Optional[str] = Field(default=None, description="Business sector")
    industry: Optional[str] = Field(default=None, description="Industry classification")
    market_cap: Optional[float] = Field(
        default=None, description="Market capitalization"
    )
    exchange: Optional[str] = Field(default=None, description="Stock exchange")
    currency: Optional[str] = Field(default=None, description="Trading currency")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(SQLADateTime, server_default=func.current_timestamp()),
        description="Timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(
            SQLADateTime,
            server_default=func.current_timestamp(),
            onupdate=func.current_timestamp(),
        ),
        description="Timestamp when the record was last updated",
    )

    # Relationships
    price_history: List["PriceHistory"] = Relationship(
        back_populates="stock", cascade_delete=True
    )
    dividends: List["Dividend"] = Relationship(
        back_populates="stock", cascade_delete=True
    )
    splits: List["Split"] = Relationship(back_populates="stock", cascade_delete=True)
    options_calls: List["OptionsCall"] = Relationship(
        back_populates="stock", cascade_delete=True
    )
    options_puts: List["OptionsPut"] = Relationship(
        back_populates="stock", cascade_delete=True
    )
    stock_info: Optional["StockInfo"] = Relationship(
        back_populates="stock", cascade_delete=True
    )


class PriceHistory(SQLModel, table=True):
    """Historical OHLCV price data."""

    __tablename__ = "price_history"
    __table_args__ = (
        UniqueConstraint("symbol", "date", "interval", name="uq_price_history"),
        Index("idx_price_history_symbol_date", "symbol", "date"),
    )

    id: Optional[int] = Field(default=None, primary_key=True, description="Primary key")
    symbol: str = Field(foreign_key="stocks.symbol", description="Stock ticker symbol")
    date: DateType = Field(description="Trading date")
    open_price: Optional[float] = Field(default=None, description="Opening price")
    high_price: Optional[float] = Field(default=None, description="Highest price")
    low_price: Optional[float] = Field(default=None, description="Lowest price")
    close_price: Optional[float] = Field(default=None, description="Closing price")
    volume: Optional[int] = Field(default=None, description="Trading volume")
    interval: str = Field(default="1d", description="Data interval (1d, 1h, etc.)")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(SQLADateTime, server_default=func.current_timestamp()),
        description="Timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(
            SQLADateTime,
            server_default=func.current_timestamp(),
            onupdate=func.current_timestamp(),
        ),
        description="Timestamp when the record was last updated",
    )

    # Relationships
    stock: Stock = Relationship(back_populates="price_history")


class Dividend(SQLModel, table=True):
    """Dividend payment history."""

    __tablename__ = "dividends"
    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_dividends"),
        Index("idx_dividends_symbol_date", "symbol", "date"),
    )

    id: Optional[int] = Field(default=None, primary_key=True, description="Primary key")
    symbol: str = Field(foreign_key="stocks.symbol", description="Stock ticker symbol")
    date: DateType = Field(description="Dividend payment date")
    amount: float = Field(description="Dividend amount per share")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(SQLADateTime, server_default=func.current_timestamp()),
        description="Timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(
            SQLADateTime,
            server_default=func.current_timestamp(),
            onupdate=func.current_timestamp(),
        ),
        description="Timestamp when the record was last updated",
    )

    # Relationships
    stock: Stock = Relationship(back_populates="dividends")


class Split(SQLModel, table=True):
    """Stock split history."""

    __tablename__ = "splits"
    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_splits"),
        Index("idx_splits_symbol_date", "symbol", "date"),
    )

    id: Optional[int] = Field(default=None, primary_key=True, description="Primary key")
    symbol: str = Field(foreign_key="stocks.symbol", description="Stock ticker symbol")
    date: DateType = Field(description="Split date")
    ratio: float = Field(description="Split ratio (e.g., 2.0 for 2:1 split)")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(SQLADateTime, server_default=func.current_timestamp()),
        description="Timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(
            SQLADateTime,
            server_default=func.current_timestamp(),
            onupdate=func.current_timestamp(),
        ),
        description="Timestamp when the record was last updated",
    )

    # Relationships
    stock: Stock = Relationship(back_populates="splits")


class OptionsCall(SQLModel, table=True):
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

    id: Optional[int] = Field(default=None, primary_key=True, description="Primary key")
    symbol: str = Field(foreign_key="stocks.symbol", description="Stock ticker symbol")
    expiration_date: DateType = Field(description="Option expiration date")
    strike: float = Field(description="Strike price")
    last_price: Optional[float] = Field(default=None, description="Last traded price")
    bid: Optional[float] = Field(default=None, description="Bid price")
    ask: Optional[float] = Field(default=None, description="Ask price")
    volume: Optional[int] = Field(default=None, description="Trading volume")
    open_interest: Optional[int] = Field(default=None, description="Open interest")
    implied_volatility: Optional[float] = Field(
        default=None, description="Implied volatility"
    )
    in_the_money: Optional[bool] = Field(
        default=None, description="Whether option is in the money"
    )
    contract_symbol: Optional[str] = Field(
        default=None, description="Option contract symbol"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(SQLADateTime, server_default=func.current_timestamp()),
        description="Timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(
            SQLADateTime,
            server_default=func.current_timestamp(),
            onupdate=func.current_timestamp(),
        ),
        description="Timestamp when the record was last updated",
    )

    # Relationships
    stock: Stock = Relationship(back_populates="options_calls")


class OptionsPut(SQLModel, table=True):
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

    id: Optional[int] = Field(default=None, primary_key=True, description="Primary key")
    symbol: str = Field(foreign_key="stocks.symbol", description="Stock ticker symbol")
    expiration_date: DateType = Field(description="Option expiration date")
    strike: float = Field(description="Strike price")
    last_price: Optional[float] = Field(default=None, description="Last traded price")
    bid: Optional[float] = Field(default=None, description="Bid price")
    ask: Optional[float] = Field(default=None, description="Ask price")
    volume: Optional[int] = Field(default=None, description="Trading volume")
    open_interest: Optional[int] = Field(default=None, description="Open interest")
    implied_volatility: Optional[float] = Field(
        default=None, description="Implied volatility"
    )
    in_the_money: Optional[bool] = Field(
        default=None, description="Whether option is in the money"
    )
    contract_symbol: Optional[str] = Field(
        default=None, description="Option contract symbol"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(SQLADateTime, server_default=func.current_timestamp()),
        description="Timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(
            SQLADateTime,
            server_default=func.current_timestamp(),
            onupdate=func.current_timestamp(),
        ),
        description="Timestamp when the record was last updated",
    )

    # Relationships
    stock: Stock = Relationship(back_populates="options_puts")


class StockInfo(SQLModel, table=True):
    """Raw yfinance info data stored as JSON."""

    __tablename__ = "stock_info"

    symbol: str = Field(
        foreign_key="stocks.symbol", primary_key=True, description="Stock ticker symbol"
    )
    info_json: str = Field(
        sa_column=Column(Text, nullable=False),
        description="JSON-encoded stock information from yfinance",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(SQLADateTime, server_default=func.current_timestamp()),
        description="Timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
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
