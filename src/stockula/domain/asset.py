"""Asset domain model representing a position in a portfolio."""

from dataclasses import InitVar, dataclass, field

from .category import Category
from .ticker import Ticker


@dataclass
class Asset:
    """Represents an asset position within a portfolio.

    An asset combines a ticker reference with quantity (shares) information
    and an optional category for classification.
    """

    ticker_init: InitVar[Ticker]
    quantity_init: InitVar[float]  # Number of shares
    category_init: InitVar[Category | None] = None  # Asset category
    _ticker: Ticker = field(init=False, repr=False)
    _quantity: float = field(init=False, repr=False)
    _category: Category | None = field(init=False, repr=False)

    def __post_init__(self, ticker_init: Ticker, quantity_init: float, category_init: Category | None):
        """Validate constraints and set private attributes."""
        self._ticker = ticker_init
        if quantity_init <= 0:
            raise ValueError(f"Asset {ticker_init.symbol} quantity must be positive")
        self._quantity = quantity_init
        self._category = category_init

    @property
    def ticker(self) -> Ticker:
        """Get ticker reference (read-only)."""
        return self._ticker

    @property
    def quantity(self) -> float:
        """Get number of shares (read-only)."""
        return self._quantity

    @property
    def category(self) -> Category | None:
        """Get asset category (read-only)."""
        return self._category

    @property
    def symbol(self) -> str:
        """Convenience property for ticker symbol."""
        return self._ticker.symbol

    def get_value(self, current_price: float) -> float:
        """Calculate current value of this position.

        Args:
            current_price: Current price per share

        Returns:
            Total value of the position
        """
        return self._quantity * current_price

    def calculate_percentage(self, asset_value: float, total_value: float) -> float:
        """Calculate this asset's percentage of total value.

        Args:
            asset_value: Current value of this asset
            total_value: Total portfolio value

        Returns:
            Percentage (0-100) of total value
        """
        if total_value <= 0:
            return 0.0
        return (asset_value / total_value) * 100.0

    def __str__(self):
        """String representation."""
        category_str = f", {self._category}" if self._category else ""
        return f"Asset({self._ticker.symbol}, {self._quantity:.2f} shares{category_str})"

    def __repr__(self):
        """Detailed representation."""
        return f"Asset(ticker={self._ticker}, quantity={self._quantity}, category={self._category})"
