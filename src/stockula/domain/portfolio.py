"""Portfolio domain model for managing asset allocations."""

from dataclasses import InitVar, dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from dependency_injector.wiring import Provide, inject

from ..interfaces import IDataFetcher, ILoggingManager
from .asset import Asset
from .category import Category
from .ticker import Ticker

if TYPE_CHECKING:
    pass


# Cached function for allocation calculations
@lru_cache(maxsize=128)
def _calculate_allocations_cached(
    assets_tuple: tuple[tuple[str, float, float], ...],  # (symbol, quantity, price)
    total_value: float,
) -> dict[str, dict[str, float]]:
    """Cached calculation of asset allocations."""
    allocations = {}
    for symbol, quantity, price in assets_tuple:
        value = quantity * price
        percentage = (value / total_value) * 100.0 if total_value > 0 else 0.0
        allocations[symbol] = {
            "value": value,
            "percentage": percentage,
            "quantity": quantity,
        }
    return allocations


@dataclass
class Portfolio:
    """Represents a complete investment portfolio with assets."""

    # InitVar fields without defaults must come before those with defaults
    initial_capital_init: InitVar[float]
    name_init: InitVar[str] = "Main Portfolio"
    allocation_method_init: InitVar[str] = "equal_weight"  # equal_weight, market_cap, custom
    logging_manager_init: InitVar[ILoggingManager] = None
    # Regular fields
    _name: str = field(init=False, repr=False)
    _initial_capital: float = field(init=False, repr=False)
    _allocation_method: str = field(init=False, repr=False)
    _logger: ILoggingManager = field(init=False, repr=False)
    assets: list[Asset] = field(default_factory=list)
    rebalance_frequency: str | None = "monthly"
    max_position_size: float | None = None  # Max % per position
    stop_loss_pct: float | None = None  # Global stop loss %

    @inject
    def __post_init__(
        self,
        initial_capital_init: float,
        name_init: str,
        allocation_method_init: str,
        logging_manager_init: ILoggingManager | None = None,
        logging_manager: ILoggingManager = Provide["logging_manager"],
    ):
        """Validate portfolio constraints and set private attributes."""
        self._name = name_init
        if initial_capital_init <= 0:
            raise ValueError("Initial capital must be positive")
        self._initial_capital = initial_capital_init
        self._allocation_method = allocation_method_init
        self._logger = logging_manager_init or logging_manager

        if self.max_position_size is not None:
            if not 0 < self.max_position_size <= 100:
                raise ValueError("Max position size must be between 0 and 100")

        if self.stop_loss_pct is not None:
            if not 0 < self.stop_loss_pct <= 100:
                raise ValueError("Stop loss percentage must be between 0 and 100")

    @property
    def name(self) -> str:
        """Get portfolio name (read-only)."""
        return self._name

    @property
    def initial_capital(self) -> float:
        """Get initial portfolio capital (read-only)."""
        return self._initial_capital

    @property
    def allocation_method(self) -> str:
        """Get allocation method (read-only)."""
        return self._allocation_method

    def add_asset(self, asset: Asset) -> None:
        """Add an asset to the portfolio."""
        # Check for duplicate
        if self.has_asset(asset.symbol):
            raise ValueError(f"Asset with symbol {asset.symbol} already exists in portfolio")
        self.assets.append(asset)

    def get_asset(self, symbol: str) -> Asset | None:
        """Get asset by symbol."""
        for asset in self.assets:
            if asset.symbol == symbol:
                return asset
        return None

    def get_all_assets(self) -> list[Asset]:
        """Get all assets in the portfolio."""
        return self.assets.copy()

    def has_asset(self, symbol: str) -> bool:
        """Check if portfolio has an asset with given symbol."""
        return self.get_asset(symbol) is not None

    def get_asset_count(self) -> int:
        """Get number of assets in the portfolio."""
        return len(self.assets)

    def get_all_tickers(self) -> list[Ticker]:
        """Get unique tickers across all assets."""
        tickers = set()
        for asset in self.assets:
            tickers.add(asset.ticker)
        return list(tickers)

    def get_asset_by_symbol(self, symbol: str) -> Asset | None:
        """Find asset by ticker symbol.

        Delegates to inherited get_asset method.
        """
        return self.get_asset(symbol)

    def get_assets_by_category(self, category: Category) -> list[Asset]:
        """Get all assets belonging to a specific category.

        Args:
            category: The category to filter by

        Returns:
            List of assets with the specified category
        """
        return [asset for asset in self.assets if asset.category == category]

    def get_portfolio_value(self, prices: dict[str, float]) -> float:
        """Calculate total portfolio value based on current prices."""
        total = 0.0
        for asset in self.assets:
            if asset.symbol in prices:
                total += asset.get_value(prices[asset.symbol])
        return total

    def get_asset_allocations(self, prices: dict[str, float]) -> dict[str, dict[str, float]]:
        """Calculate current allocation percentages for all assets.

        Args:
            prices: Dictionary mapping symbols to prices

        Returns:
            Dictionary with asset symbols as keys and dict of value/percentage as values
        """
        total_value = self.get_portfolio_value(prices)

        # Build tuple for caching
        assets_data = []
        for asset in self.assets:
            if asset.symbol in prices:
                assets_data.append((asset.symbol, asset.quantity, prices[asset.symbol]))

        assets_tuple = tuple(sorted(assets_data))
        return _calculate_allocations_cached(assets_tuple, total_value)

    def get_asset_percentage(self, symbol: str, prices: dict[str, float]) -> float | None:
        """Get the current percentage allocation of an asset by symbol.

        Args:
            symbol: Ticker symbol to look up
            prices: Current prices for all assets

        Returns:
            Percentage (0-100) of portfolio allocated to this asset, or None if not found
        """
        allocations = self.get_asset_allocations(prices)

        if symbol in allocations:
            return allocations[symbol]["percentage"]
        return None

    def get_all_asset_percentages(self, prices: dict[str, float]) -> dict[str, float]:
        """Get current percentage allocations for all assets.

        Args:
            prices: Current prices for all assets

        Returns:
            Dictionary mapping ticker symbols to their percentage allocations
        """
        allocations = self.get_asset_allocations(prices)
        return {symbol: data["percentage"] for symbol, data in allocations.items()}

    def get_portfolio_summary(self, prices: dict[str, float]) -> dict[str, Any]:
        """Get comprehensive portfolio summary.

        Args:
            prices: Current prices for all assets

        Returns:
            Dictionary with portfolio summary
        """
        total_value = self.get_portfolio_value(prices)
        allocations = self.get_asset_allocations(prices)

        return {
            "total_value": total_value,
            "initial_capital": self._initial_capital,
            "total_return": total_value - self._initial_capital,
            "return_percentage": ((total_value - self._initial_capital) / self._initial_capital) * 100,
            "allocations": allocations,
            "asset_count": len(self.assets),
            "by_category": self.get_allocation_by_category(prices),
        }

    def get_allocation_by_category(self, prices: dict[str, float]) -> dict[str, dict[str, Any]]:
        """Calculate allocations grouped by category.

        Args:
            prices: Current prices for all assets

        Returns:
            Dictionary with categories as keys and allocation info as values
        """
        category_allocations: dict[str, dict[str, Any]] = {}
        total_value = self.get_portfolio_value(prices)

        # Group assets by category
        for asset in self.assets:
            if asset.symbol in prices:
                value = asset.get_value(prices[asset.symbol])
                category_name = str(asset.category) if asset.category else "Uncategorized"

                if category_name not in category_allocations:
                    category_allocations[category_name] = {
                        "value": 0.0,
                        "percentage": 0.0,
                        "assets": [],
                        "total_quantity": 0.0,
                    }

                category_allocations[category_name]["value"] += value
                category_allocations[category_name]["assets"].append(asset.symbol)
                category_allocations[category_name]["total_quantity"] += asset.quantity

        # Calculate percentages
        for category_data in category_allocations.values():
            category_data["percentage"] = (category_data["value"] / total_value * 100) if total_value > 0 else 0.0

        return category_allocations

    @property
    def symbols(self) -> list[str]:
        """Get list of all ticker symbols in the portfolio."""
        return [asset.symbol for asset in self.assets]

    def get_current_value(self, fetcher: "IDataFetcher | None" = None) -> float:
        """Get current portfolio value by fetching latest prices.

        This is a convenience method that fetches current prices.
        For better performance when prices are already available,
        use get_portfolio_value(prices) instead.

        Args:
            fetcher: Optional data fetcher instance. If not provided, creates a new one.

        Returns:
            Current total portfolio value
        """
        if fetcher is None:
            from ..data.fetcher import DataFetcher

            data_fetcher = DataFetcher()
            prices = data_fetcher.get_current_prices(self.symbols)
        else:
            prices = fetcher.get_current_prices(self.symbols)
        return self.get_portfolio_value(prices)

    def validate_capital_sufficiency(
        self,
        validation_prices: dict[str, float] | None = None,
        fetcher: "IDataFetcher | None" = None,
    ) -> None:
        """Validate that initial capital is sufficient to cover the specified asset quantities.

        Args:
            validation_prices: Optional price dictionary to use for validation. If not provided,
                             fetches current prices. This allows validation with historical prices.
            fetcher: Optional data fetcher instance.

        Raises:
            ValueError: If initial capital is less than the required portfolio value
            Warning: If initial capital significantly exceeds portfolio value (>50% over)
        """
        if not self.assets:
            return  # No assets to validate

        if validation_prices is None:
            if fetcher is None:
                from ..data.fetcher import DataFetcher

                data_fetcher = DataFetcher()
                validation_prices = data_fetcher.get_current_prices(self.symbols)
            else:
                validation_prices = fetcher.get_current_prices(self.symbols)

        # Calculate required capital based on asset quantities
        required_capital = self.get_portfolio_value(validation_prices)

        if required_capital == 0:
            self._logger.warning("Could not fetch prices for portfolio validation")
            return

        if self._initial_capital < required_capital:
            raise ValueError(
                f"Initial capital (${self._initial_capital:,.2f}) is insufficient to cover "
                f"specified asset quantities (${required_capital:,.2f}). "
                f"Either increase initial_capital or reduce asset quantities."
            )

        # Warn if capital is significantly higher than needed (optional)
        excess_ratio = (self._initial_capital - required_capital) / required_capital
        if excess_ratio > 0.5:  # More than 50% excess
            self._logger.warning(
                f"Initial capital (${self._initial_capital:,.2f}) significantly "
                f"exceeds required capital (${required_capital:,.2f}). "
                f"Consider adjusting asset quantities or initial capital."
            )

    def validate_allocation_constraints(
        self,
        prices: dict[str, float] | None = None,
        fetcher: "IDataFetcher | None" = None,
    ) -> None:
        """Validate portfolio allocation constraints against risk management rules.

        Args:
            prices: Optional price dictionary, will fetch current prices if not provided
            fetcher: Optional data fetcher instance.

        Raises:
            ValueError: If any allocation constraint is violated
        """
        if not self.assets:
            return  # No assets to validate

        if prices is None:
            if fetcher is None:
                from ..data.fetcher import DataFetcher

                data_fetcher = DataFetcher()
                prices = data_fetcher.get_current_prices(self.symbols)
            else:
                prices = fetcher.get_current_prices(self.symbols)

        if not prices:
            self._logger.warning("Could not fetch prices for allocation validation")
            return

        # Check max position size constraint
        if self.max_position_size is not None:
            allocations = self.get_asset_allocations(prices)
            for symbol, allocation_data in allocations.items():
                if allocation_data["percentage"] > self.max_position_size:
                    raise ValueError(
                        f"Asset {symbol} allocation ({allocation_data['percentage']:.1f}%) "
                        f"exceeds maximum position size ({self.max_position_size}%). "
                        f"Reduce quantity or increase max_position_size."
                    )

        # Validate total allocation doesn't exceed 100% (with small tolerance for rounding)
        total_allocation = sum(data["percentage"] for data in self.get_asset_allocations(prices).values())
        if total_allocation > 100.1:  # 0.1% tolerance for rounding
            self._logger.warning(
                f"Total allocation ({total_allocation:.1f}%) exceeds 100%. This may indicate overleveraging."
            )

        # Check if we have sufficient capital utilization (warn if too low)
        portfolio_value = self.get_portfolio_value(prices)
        utilization_ratio = portfolio_value / self._initial_capital
        if utilization_ratio < 0.5:  # Less than 50% utilization
            self._logger.warning(
                f"Low capital utilization ({utilization_ratio:.1%}). "
                f"Portfolio value (${portfolio_value:,.2f}) is much lower than "
                f"initial capital (${self._initial_capital:,.2f})."
            )

    def __str__(self):
        """String representation."""
        return f"Portfolio(name='{self._name}', capital=${self._initial_capital:,.2f}, {len(self.assets)} assets)"
