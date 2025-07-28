"""Wrapper for Ticker to provide backward compatibility."""

from typing import Any

from .ticker import Ticker as _Ticker


def Ticker(
    symbol: str,
    sector: str | None = None,
    market_cap: float | None = None,
    category: str | None = None,
    price_range: dict[str, float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> _Ticker:
    """Create a Ticker with backward-compatible parameter names.

    This wrapper function allows using the intuitive parameter names
    while internally mapping to the dataclass InitVar names.
    """
    return _Ticker(
        symbol_init=symbol,
        sector_init=sector,
        market_cap_init=market_cap,
        category_init=category,
        price_range_init=price_range,
        metadata_init=metadata,
    )


# Make Ticker look like a class for type hints
Ticker.__name__ = "Ticker"
Ticker.__module__ = _Ticker.__module__
