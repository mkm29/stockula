"""Ticker domain model with singleton registry."""

from dataclasses import InitVar, dataclass, field
from typing import Any


@dataclass
class Ticker:
    """Represents a tradable ticker/symbol with metadata."""

    symbol_init: InitVar[str]
    sector_init: InitVar[str | None] = None
    market_cap_init: InitVar[float | None] = None  # in billions
    category_init: InitVar[str | None] = None  # momentum, growth, value, speculative, etc.
    price_range_init: InitVar[dict[str, float] | None] = None  # open, high, low, close
    metadata_init: InitVar[dict[str, Any] | None] = None
    _symbol: str = field(init=False, repr=False)
    _sector: str | None = field(init=False, repr=False)
    _market_cap: float | None = field(init=False, repr=False)
    _category: str | None = field(init=False, repr=False)
    _price_range: dict[str, float] | None = field(init=False, repr=False)
    _metadata: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(
        self,
        symbol_init: str,
        sector_init: str | None,
        market_cap_init: float | None,
        category_init: str | None,
        price_range_init: dict[str, float] | None,
        metadata_init: dict[str, Any] | None,
    ):
        """Initialize and validate symbol."""
        self._symbol = symbol_init.upper()  # Always store symbols in uppercase
        self._sector = sector_init
        self._market_cap = market_cap_init
        self._category = category_init
        self._price_range = price_range_init
        self._metadata = metadata_init if metadata_init is not None else {}

    @property
    def symbol(self) -> str:  # noqa: F811
        """Get ticker symbol (read-only)."""
        return self._symbol

    @property
    def sector(self) -> str | None:  # noqa: F811
        """Get ticker sector (read-only)."""
        return self._sector

    @property
    def market_cap(self) -> float | None:  # noqa: F811
        """Get market capitalization (read-only)."""
        return self._market_cap

    @property
    def category(self) -> str | None:  # noqa: F811
        """Get ticker category (read-only)."""
        return self._category

    @property
    def price_range(self) -> dict[str, float] | None:  # noqa: F811
        """Get price range data (read-only)."""
        return self._price_range

    @property
    def metadata(self) -> dict[str, Any]:  # noqa: F811
        """Get ticker metadata (read-only)."""
        return self._metadata

    def __hash__(self):
        """Make Ticker hashable based on symbol."""
        return hash(self._symbol)

    def __eq__(self, other):
        """Tickers are equal if symbols match."""
        if isinstance(other, Ticker):
            return self._symbol == other._symbol
        return False

    def __str__(self):
        """String representation."""
        return f"Ticker({self._symbol})"

    def __repr__(self):
        """Detailed representation."""
        sector_repr = f"'{self._sector}'" if self._sector is not None else "None"
        category_repr = f"'{self._category}'" if self._category is not None else "None"
        return (
            f"Ticker(symbol='{self._symbol}', sector={sector_repr}, "
            f"market_cap={self._market_cap}, category={category_repr})"
        )

    @classmethod
    def create(
        cls,
        symbol: str,
        sector: str | None = None,
        market_cap: float | None = None,
        category: str | None = None,
        price_range: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Ticker":
        """Factory method to create Ticker with intuitive parameter names."""
        return cls(
            symbol_init=symbol,
            sector_init=sector,
            market_cap_init=market_cap,
            category_init=category,
            price_range_init=price_range,
            metadata_init=metadata,
        )


class TickerRegistry:
    """Singleton registry for managing ticker instances."""

    _instances: dict[type, "TickerRegistry"] = {}
    _tickers: dict[str, Ticker]

    def __new__(cls):
        """Ensure only one instance exists per class."""
        if cls not in cls._instances:
            instance = super().__new__(cls)
            instance._tickers = {}
            cls._instances[cls] = instance
        return cls._instances[cls]

    def get_or_create(self, symbol: str, **kwargs) -> Ticker:
        """Get existing ticker or create new one.

        Args:
            symbol: Ticker symbol
            **kwargs: Additional ticker attributes

        Returns:
            Ticker instance (existing or newly created)
        """
        symbol = symbol.upper()

        if symbol not in self._tickers:
            # Map kwargs to the _init parameter names
            init_kwargs = {
                "symbol_init": symbol,
                "sector_init": kwargs.get("sector"),
                "market_cap_init": kwargs.get("market_cap"),
                "category_init": kwargs.get("category"),
                "price_range_init": kwargs.get("price_range"),
                "metadata_init": kwargs.get("metadata", {}),
            }
            self._tickers[symbol] = Ticker(**init_kwargs)
        else:
            # If ticker exists and new values are provided, create a new instance
            # with merged attributes (since Ticker is now immutable)
            existing = self._tickers[symbol]
            if any(v is not None for v in kwargs.values()):
                # Merge existing values with new ones
                merged_kwargs = {
                    "symbol_init": symbol,
                    "sector_init": kwargs.get("sector", existing.sector),
                    "market_cap_init": kwargs.get("market_cap", existing.market_cap),
                    "category_init": kwargs.get("category", existing.category),
                    "price_range_init": kwargs.get("price_range", existing.price_range),
                    "metadata_init": kwargs.get("metadata", existing.metadata),
                }
                self._tickers[symbol] = Ticker(**merged_kwargs)

        return self._tickers[symbol]

    def get(self, symbol: str) -> Ticker | None:
        """Get ticker by symbol if it exists."""
        return self._tickers.get(symbol.upper())

    def all(self) -> dict[str, Ticker]:
        """Get all registered tickers."""
        return self._tickers.copy()

    def _clear(self):
        """Clear all registered tickers (internal method, useful for testing)."""
        self._tickers.clear()

    def __len__(self):
        """Number of registered tickers."""
        return len(self._tickers)

    def __contains__(self, symbol: str):
        """Check if ticker is registered."""
        return symbol.upper() in self._tickers
