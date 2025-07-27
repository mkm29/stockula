"""Ticker domain model with singleton registry."""

from typing import Dict, Optional, Any
from dataclasses import dataclass, field, InitVar


@dataclass
class Ticker:
    """Represents a tradable ticker/symbol with metadata."""

    symbol: InitVar[str]
    sector: InitVar[Optional[str]] = None
    market_cap: InitVar[Optional[float]] = None  # in billions
    category: InitVar[Optional[str]] = (
        None  # momentum, growth, value, speculative, etc.
    )
    price_range: InitVar[Optional[Dict[str, float]]] = None  # open, high, low, close
    metadata: InitVar[Dict[str, Any]] = field(default_factory=dict)
    _symbol: str = field(init=False, repr=False)
    _sector: Optional[str] = field(init=False, repr=False)
    _market_cap: Optional[float] = field(init=False, repr=False)
    _category: Optional[str] = field(init=False, repr=False)
    _price_range: Optional[Dict[str, float]] = field(init=False, repr=False)
    _metadata: Dict[str, Any] = field(init=False, repr=False)

    def __post_init__(
        self,
        symbol: str,
        sector: Optional[str],
        market_cap: Optional[float],
        category: Optional[str],
        price_range: Optional[Dict[str, float]],
        metadata: Dict[str, Any],
    ):
        """Initialize and validate symbol."""
        self._symbol = symbol.upper()  # Always store symbols in uppercase
        self._sector = sector
        self._market_cap = market_cap
        self._category = category
        self._price_range = price_range
        self._metadata = metadata

    @property
    def symbol(self) -> str:  # noqa: F811
        """Get ticker symbol (read-only)."""
        return self._symbol

    @property
    def sector(self) -> Optional[str]:  # noqa: F811
        """Get ticker sector (read-only)."""
        return self._sector

    @property
    def market_cap(self) -> Optional[float]:  # noqa: F811
        """Get market capitalization (read-only)."""
        return self._market_cap

    @property
    def category(self) -> Optional[str]:  # noqa: F811
        """Get ticker category (read-only)."""
        return self._category

    @property
    def price_range(self) -> Optional[Dict[str, float]]:  # noqa: F811
        """Get price range data (read-only)."""
        return self._price_range

    @property
    def metadata(self) -> Dict[str, Any]:  # noqa: F811
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
        return (
            f"Ticker(symbol='{self._symbol}', sector='{self._sector}', "
            f"market_cap={self._market_cap}, category='{self._category}')"
        )


class TickerRegistry:
    """Singleton registry for managing ticker instances."""

    _instance: Optional["TickerRegistry"] = None
    _tickers: Dict[str, Ticker] = {}

    def __new__(cls):
        """Ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tickers = {}
        return cls._instance

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
            self._tickers[symbol] = Ticker(symbol=symbol, **kwargs)
        else:
            # If ticker exists and new values are provided, create a new instance
            # with merged attributes (since Ticker is now immutable)
            existing = self._tickers[symbol]
            if any(v is not None for v in kwargs.values()):
                # Merge existing values with new ones
                merged_kwargs = {
                    "sector": kwargs.get("sector", existing.sector),
                    "market_cap": kwargs.get("market_cap", existing.market_cap),
                    "category": kwargs.get("category", existing.category),
                    "price_range": kwargs.get("price_range", existing.price_range),
                    "metadata": kwargs.get("metadata", existing.metadata),
                }
                self._tickers[symbol] = Ticker(symbol=symbol, **merged_kwargs)

        return self._tickers[symbol]

    def get(self, symbol: str) -> Optional[Ticker]:
        """Get ticker by symbol if it exists."""
        return self._tickers.get(symbol.upper())

    def all(self) -> Dict[str, Ticker]:
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
