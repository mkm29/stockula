"""Factory for creating domain objects from configuration."""

from datetime import date
from typing import TYPE_CHECKING

from dependency_injector.wiring import Provide, inject

from ..allocation import Allocator
from ..config import StockulaConfig, TickerConfig
from ..interfaces import ILoggingManager
from .asset import Asset
from .category import Category
from .portfolio import Portfolio
from .ticker import Ticker, TickerRegistry

if TYPE_CHECKING:
    from ..interfaces import IDataFetcher


def date_to_string(date_value: str | date | None) -> str | None:
    """Convert date or string to string format."""
    if date_value is None:
        return None
    if isinstance(date_value, str):
        return date_value
    return date_value.strftime("%Y-%m-%d")


class DomainFactory:
    """Factory for creating domain objects from configuration."""

    @inject
    def __init__(
        self,
        config: StockulaConfig | None = None,
        fetcher: "IDataFetcher | None" = None,
        ticker_registry: TickerRegistry | None = None,
        allocator: Allocator | None = None,
        logging_manager: ILoggingManager = Provide["logging_manager"],
    ):
        """Initialize factory with dependencies.

        Args:
            config: Configuration object
            fetcher: Data fetcher instance
            ticker_registry: Ticker registry instance
            allocator: Allocator instance (if not provided, will be created from fetcher)
            logging_manager: Injected logging manager
        """
        self.config = config
        self.fetcher = fetcher
        self.ticker_registry = ticker_registry or TickerRegistry()
        # If allocator is provided, use it; otherwise create one if fetcher is available
        self.allocator = allocator if allocator is not None else (Allocator(fetcher) if fetcher else None)
        self.logger = logging_manager

    def _create_ticker(self, ticker_config: TickerConfig) -> Ticker:
        """Create or get ticker from configuration (internal method).

        Args:
            ticker_config: Ticker configuration

        Returns:
            Ticker instance (singleton per symbol)
        """
        return self.ticker_registry.get_or_create(
            symbol=ticker_config.symbol,
            sector=ticker_config.sector,
            market_cap=ticker_config.market_cap,
            category=ticker_config.category,
            price_range=ticker_config.price_range,
        )

    def _create_asset(self, ticker_config: TickerConfig, calculated_quantity: float | None = None) -> Asset:
        """Create asset from ticker configuration (internal method).

        Args:
            ticker_config: Ticker configuration with quantity or allocation info
            calculated_quantity: Dynamically calculated quantity (overrides ticker_config.quantity)

        Returns:
            Asset instance
        """
        ticker = self._create_ticker(ticker_config)

        # Convert category string to Category enum if provided
        category = None
        if ticker_config.category:
            try:
                # Try to find matching Category enum by name
                category = Category[ticker_config.category.upper()]
            except KeyError:
                # If not found, leave as None
                pass

        # Use calculated quantity if provided, otherwise use configured quantity
        quantity = calculated_quantity if calculated_quantity is not None else ticker_config.quantity

        if quantity is None:
            raise ValueError(f"No quantity specified for ticker {ticker_config.symbol}")

        return Asset(ticker_init=ticker, quantity_init=quantity, category_init=category)

    def create_portfolio(self, config: StockulaConfig) -> Portfolio:
        """Create complete portfolio from configuration.

        Args:
            config: Complete Stockula configuration

        Returns:
            Portfolio instance
        """
        portfolio = Portfolio(
            name_init=config.portfolio.name,
            initial_capital_init=config.portfolio.initial_capital,
            allocation_method_init=config.portfolio.allocation_method,
            logging_manager_init=self.logger,
            rebalance_frequency=config.portfolio.rebalance_frequency,
            max_position_size=config.portfolio.max_position_size,
            stop_loss_pct=config.portfolio.stop_loss_pct,
        )

        # Add tickers from portfolio config
        tickers_to_add = config.portfolio.tickers
        calculated_quantity: float | None = None

        # Handle different allocation modes
        if config.portfolio.auto_allocate:
            self.logger.info(
                "Using auto-allocation - optimizing quantities based on category ratios "
                "and capital utilization target..."
            )
            if not self.allocator:
                raise ValueError("Allocator not configured - data fetcher required for dynamic allocation")
            calculated_quantities = self.allocator.calculate_auto_allocation_quantities(config, tickers_to_add)

            for ticker_config in tickers_to_add:
                calculated_quantity = calculated_quantities.get(ticker_config.symbol, 0.0)
                # Skip tickers with 0 allocation (e.g., from categories with 0% ratio)
                if calculated_quantity > 0:
                    asset = self._create_asset(ticker_config, calculated_quantity)
                    portfolio.add_asset(asset)
                else:
                    self.logger.debug(f"Skipping {ticker_config.symbol} - 0 shares allocated")
        elif config.portfolio.dynamic_allocation or config.portfolio.allocation_method == "backtest_optimized":
            # For backtest_optimized, treat it like dynamic allocation
            if config.portfolio.allocation_method == "backtest_optimized":
                self.logger.info(
                    "Using backtest-optimized allocation - calculating quantities based on historical performance..."
                )
            else:
                self.logger.info(
                    "Using dynamic allocation - calculating quantities based on allocation percentages/amounts..."
                )
            if not self.allocator:
                raise ValueError("Allocator not configured - data fetcher required for dynamic allocation")
            calculated_quantities = self.allocator.calculate_dynamic_quantities(config, tickers_to_add)

            for ticker_config in tickers_to_add:
                calculated_quantity = calculated_quantities.get(ticker_config.symbol)
                if calculated_quantity is not None and calculated_quantity > 0:
                    asset = self._create_asset(ticker_config, calculated_quantity)
                    portfolio.add_asset(asset)
                    self.logger.debug(f"  {ticker_config.symbol}: {calculated_quantity:.4f} shares")
                else:
                    self.logger.debug(f"Skipping {ticker_config.symbol} - 0 shares allocated")
        else:
            # Use static quantities from configuration
            for ticker_config in tickers_to_add:
                asset = self._create_asset(ticker_config)
                portfolio.add_asset(asset)

        # Skip validation for dynamic allocations since they're calculated to fit within capital
        if not (config.portfolio.auto_allocate or config.portfolio.dynamic_allocation):
            # Validate that initial capital is sufficient for the specified asset quantities
            portfolio.validate_capital_sufficiency()

            # Validate allocation constraints against risk management rules
            portfolio.validate_allocation_constraints()

        return portfolio

    def get_all_tickers(self) -> list[Ticker]:
        """Get all registered tickers.

        Returns:
            List of all ticker instances
        """
        return list(self.ticker_registry.all().values())
