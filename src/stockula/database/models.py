"""SQLAlchemy models for the Stockula database."""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    Date,
    DateTime,
    Text,
    ForeignKey,
    Index,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Stock(Base):
    """Basic stock metadata."""

    __tablename__ = "stocks"

    symbol = Column(String, primary_key=True)
    name = Column(Text)
    sector = Column(Text)
    industry = Column(Text)
    market_cap = Column(Float)
    exchange = Column(Text)
    currency = Column(Text)
    created_at = Column(DateTime, server_default=func.current_timestamp())
    updated_at = Column(
        DateTime,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )

    # Relationships
    price_history = relationship(
        "PriceHistory", back_populates="stock", cascade="all, delete-orphan"
    )
    dividends = relationship(
        "Dividend", back_populates="stock", cascade="all, delete-orphan"
    )
    splits = relationship("Split", back_populates="stock", cascade="all, delete-orphan")
    options_calls = relationship(
        "OptionsCall", back_populates="stock", cascade="all, delete-orphan"
    )
    options_puts = relationship(
        "OptionsPut", back_populates="stock", cascade="all, delete-orphan"
    )
    stock_info = relationship(
        "StockInfo", back_populates="stock", uselist=False, cascade="all, delete-orphan"
    )


class PriceHistory(Base):
    """Historical OHLCV price data."""

    __tablename__ = "price_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, ForeignKey("stocks.symbol"), nullable=False)
    date = Column(Date, nullable=False)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Integer)
    interval = Column(String, default="1d")
    created_at = Column(DateTime, server_default=func.current_timestamp())

    # Relationships
    stock = relationship("Stock", back_populates="price_history")

    # Constraints
    __table_args__ = (
        UniqueConstraint("symbol", "date", "interval", name="uq_price_history"),
        Index("idx_price_history_symbol_date", "symbol", "date"),
    )


class Dividend(Base):
    """Dividend payment history."""

    __tablename__ = "dividends"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, ForeignKey("stocks.symbol"), nullable=False)
    date = Column(Date, nullable=False)
    amount = Column(Float, nullable=False)
    created_at = Column(DateTime, server_default=func.current_timestamp())

    # Relationships
    stock = relationship("Stock", back_populates="dividends")

    # Constraints
    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_dividends"),
        Index("idx_dividends_symbol_date", "symbol", "date"),
    )


class Split(Base):
    """Stock split history."""

    __tablename__ = "splits"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, ForeignKey("stocks.symbol"), nullable=False)
    date = Column(Date, nullable=False)
    ratio = Column(Float, nullable=False)
    created_at = Column(DateTime, server_default=func.current_timestamp())

    # Relationships
    stock = relationship("Stock", back_populates="splits")

    # Constraints
    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_splits"),
        Index("idx_splits_symbol_date", "symbol", "date"),
    )


class OptionsCall(Base):
    """Call options data."""

    __tablename__ = "options_calls"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, ForeignKey("stocks.symbol"), nullable=False)
    expiration_date = Column(Date, nullable=False)
    strike = Column(Float, nullable=False)
    last_price = Column(Float)
    bid = Column(Float)
    ask = Column(Float)
    volume = Column(Integer)
    open_interest = Column(Integer)
    implied_volatility = Column(Float)
    in_the_money = Column(Boolean)
    contract_symbol = Column(Text)
    created_at = Column(DateTime, server_default=func.current_timestamp())

    # Relationships
    stock = relationship("Stock", back_populates="options_calls")

    # Constraints
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


class OptionsPut(Base):
    """Put options data."""

    __tablename__ = "options_puts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, ForeignKey("stocks.symbol"), nullable=False)
    expiration_date = Column(Date, nullable=False)
    strike = Column(Float, nullable=False)
    last_price = Column(Float)
    bid = Column(Float)
    ask = Column(Float)
    volume = Column(Integer)
    open_interest = Column(Integer)
    implied_volatility = Column(Float)
    in_the_money = Column(Boolean)
    contract_symbol = Column(Text)
    created_at = Column(DateTime, server_default=func.current_timestamp())

    # Relationships
    stock = relationship("Stock", back_populates="options_puts")

    # Constraints
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


class StockInfo(Base):
    """Raw yfinance info data stored as JSON."""

    __tablename__ = "stock_info"

    symbol = Column(String, ForeignKey("stocks.symbol"), primary_key=True)
    info_json = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.current_timestamp())
    updated_at = Column(
        DateTime,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )

    # Relationships
    stock = relationship("Stock", back_populates="stock_info")
