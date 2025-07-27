"""Category enumeration for asset classification."""

from enum import Enum, auto


class Category(Enum):
    """Asset categories for portfolio organization."""

    # Sector-based categories
    TECHNOLOGY = auto()
    HEALTHCARE = auto()
    FINANCIAL = auto()
    CONSUMER = auto()
    INDUSTRIAL = auto()
    ENERGY = auto()
    UTILITIES = auto()
    REAL_ESTATE = auto()
    MATERIALS = auto()
    COMMUNICATION = auto()

    # Strategy-based categories
    GROWTH = auto()
    VALUE = auto()
    DIVIDEND = auto()
    MOMENTUM = auto()

    # Risk-based categories
    LARGE_CAP = auto()
    MID_CAP = auto()
    SMALL_CAP = auto()
    INTERNATIONAL = auto()
    EMERGING_MARKETS = auto()

    # Asset class categories
    EQUITY = auto()
    BOND = auto()
    COMMODITY = auto()
    CRYPTO = auto()
    REIT = auto()

    # Special categories
    INDEX = auto()
    ETF = auto()
    SPECULATIVE = auto()
    DEFENSIVE = auto()

    def __str__(self) -> str:
        """Return human-readable name."""
        return self.name.replace("_", " ").title()
