"""Factory for creating domain objects from configuration."""

import logging
from datetime import date
from typing import TYPE_CHECKING

from ..config import StockulaConfig, TickerConfig
from .asset import Asset
from .category import Category
from .portfolio import Portfolio
from .ticker import Ticker, TickerRegistry

if TYPE_CHECKING:
    from ..interfaces import IDataFetcher

# Create logger
logger = logging.getLogger(__name__)


def date_to_string(date_value: str | date | None) -> str | None:
    """Convert date or string to string format."""
    if date_value is None:
        return None
    if isinstance(date_value, str):
        return date_value
    return date_value.strftime("%Y-%m-%d")


class DomainFactory:
    """Factory for creating domain objects from configuration."""

    def __init__(
        self,
        config: StockulaConfig | None = None,
        fetcher: "IDataFetcher | None" = None,
        ticker_registry: TickerRegistry | None = None,
    ):
        """Initialize factory with dependencies.

        Args:
            config: Configuration object
            fetcher: Data fetcher instance
            ticker_registry: Ticker registry instance
        """
        self.config = config
        self.fetcher = fetcher
        self.ticker_registry = ticker_registry or TickerRegistry()

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

    def _calculate_dynamic_quantities(
        self, config: StockulaConfig, tickers_to_add: list[TickerConfig]
    ) -> dict[str, float]:
        """Calculate quantities dynamically based on allocation percentages/amounts.

        Args:
            config: Stockula configuration
            tickers_to_add: List of ticker configurations

        Returns:
            Dictionary mapping ticker symbols to calculated quantities
        """
        # Use injected fetcher
        fetcher = self.fetcher
        if fetcher is None:
            raise ValueError("Data fetcher not configured")
        symbols = [ticker.symbol for ticker in tickers_to_add]

        if config.data.start_date:
            # Use start date prices for backtesting to ensure portfolio value matches at start
            start_date_str = date_to_string(config.data.start_date)
            logger.debug(
                f"Calculating quantities using start date prices ({start_date_str}) for accurate portfolio value..."
            )

            calculation_prices = {}
            for symbol in symbols:
                try:
                    data = fetcher.get_stock_data(symbol, start=start_date_str, end=start_date_str)
                    if not data.empty:
                        calculation_prices[symbol] = data["Close"].iloc[0]
                    else:
                        # Fallback to a few days later if no data on exact date
                        from datetime import timedelta

                        import pandas as pd

                        if isinstance(config.data.start_date, str):
                            start_dt = pd.to_datetime(config.data.start_date)
                            end_date = (start_dt + timedelta(days=7)).strftime("%Y-%m-%d")
                        else:
                            end_date = (config.data.start_date + timedelta(days=7)).strftime("%Y-%m-%d")
                        data = fetcher.get_stock_data(symbol, start=start_date_str, end=end_date)
                        if not data.empty:
                            calculation_prices[symbol] = data["Close"].iloc[0]
                        else:
                            # Last resort: use current prices
                            current_prices = fetcher.get_current_prices([symbol])
                            if symbol in current_prices:
                                calculation_prices[symbol] = current_prices[symbol]
                                logger.warning(f"Using current price for {symbol} (no historical data available)")
                except Exception as e:
                    logger.error(f"Error fetching start date price for {symbol}: {e}")
                    # Fallback to current prices
                    current_prices = fetcher.get_current_prices([symbol])
                    if symbol in current_prices:
                        calculation_prices[symbol] = current_prices[symbol]
        else:
            # No start date specified, use current prices
            calculation_prices = fetcher.get_current_prices(symbols)

        calculated_quantities = {}

        for ticker_config in tickers_to_add:
            if ticker_config.symbol not in calculation_prices:
                raise ValueError(f"Could not fetch price for {ticker_config.symbol}")

            price = calculation_prices[ticker_config.symbol]

            # Calculate allocation amount
            if ticker_config.allocation_pct is not None:
                allocation_amount = (ticker_config.allocation_pct / 100.0) * config.portfolio.initial_capital
            elif ticker_config.allocation_amount is not None:
                allocation_amount = ticker_config.allocation_amount
            else:
                # Should not happen due to validation, but handle gracefully
                raise ValueError(f"No allocation specified for {ticker_config.symbol}")

            # Calculate quantity
            raw_quantity = allocation_amount / price

            if config.portfolio.allow_fractional_shares:
                calculated_quantities[ticker_config.symbol] = raw_quantity
            else:
                # Round down to nearest integer (conservative approach)
                calculated_quantities[ticker_config.symbol] = max(1, int(raw_quantity))

        return calculated_quantities

    def _calculate_auto_allocation_quantities(
        self, config: StockulaConfig, tickers_to_add: list[TickerConfig]
    ) -> dict[str, float]:
        """Calculate quantities using auto-allocation based on category ratios and capital utilization target.

        This method optimizes for maximum capital utilization while respecting category allocation ratios.

        Args:
            config: Stockula configuration
            tickers_to_add: List of ticker configurations (should only have category specified)

        Returns:
            Dictionary mapping ticker symbols to calculated quantities
        """
        if not config.portfolio.category_ratios:
            raise ValueError("Auto-allocation requires category_ratios to be specified")

        # Use injected fetcher
        fetcher = self.fetcher
        if fetcher is None:
            raise ValueError("Data fetcher not configured")
        symbols = [ticker.symbol for ticker in tickers_to_add]

        if config.data.start_date:
            # Use start date prices for backtesting to ensure portfolio value matches at start
            start_date_str = date_to_string(config.data.start_date)
            logger.debug(f"Using start date prices ({start_date_str}) for auto-allocation calculations...")

            calculation_prices = {}
            for symbol in symbols:
                try:
                    data = fetcher.get_stock_data(symbol, start=start_date_str, end=start_date_str)
                    if not data.empty:
                        calculation_prices[symbol] = data["Close"].iloc[0]
                    else:
                        # Fallback to a few days later if no data on exact date
                        from datetime import timedelta

                        import pandas as pd

                        if isinstance(config.data.start_date, str):
                            start_dt = pd.to_datetime(config.data.start_date)
                            end_date = (start_dt + timedelta(days=7)).strftime("%Y-%m-%d")
                        else:
                            end_date = (config.data.start_date + timedelta(days=7)).strftime("%Y-%m-%d")
                        data = fetcher.get_stock_data(symbol, start=start_date_str, end=end_date)
                        if not data.empty:
                            calculation_prices[symbol] = data["Close"].iloc[0]
                        else:
                            # Last resort: use current prices
                            current_prices = fetcher.get_current_prices([symbol])
                            if symbol in current_prices:
                                calculation_prices[symbol] = current_prices[symbol]
                                logger.warning(f"Using current price for {symbol} (no historical data available)")
                except Exception as e:
                    logger.error(f"Error fetching start date price for {symbol}: {e}")
                    # Fallback to current prices
                    current_prices = fetcher.get_current_prices([symbol])
                    if symbol in current_prices:
                        calculation_prices[symbol] = current_prices[symbol]
        else:
            # No start date specified, use current prices
            calculation_prices = fetcher.get_current_prices(symbols)

        # Group tickers by category
        tickers_by_category: dict[str, list[TickerConfig]] = {}
        for ticker_config in tickers_to_add:
            if ticker_config.symbol not in calculation_prices:
                raise ValueError(f"Could not fetch price for {ticker_config.symbol}")

            if not ticker_config.category:
                raise ValueError(f"Ticker {ticker_config.symbol} must have category specified for auto-allocation")

            category = ticker_config.category.upper()
            if category not in tickers_by_category:
                tickers_by_category[category] = []
            tickers_by_category[category].append(ticker_config)

        # Calculate target capital per category
        target_capital = config.portfolio.initial_capital * config.portfolio.capital_utilization_target
        calculated_quantities: dict[str, float] = {}

        # Initialize all tickers with 0 quantity
        for ticker_config in tickers_to_add:
            calculated_quantities[ticker_config.symbol] = 0.0

        logger.debug(
            f"Auto-allocation target capital: ${target_capital:,.2f} "
            f"({config.portfolio.capital_utilization_target:.1%} of ${config.portfolio.initial_capital:,.2f})"
        )

        # First pass: Calculate basic allocations per category
        category_allocations = {}
        for category, ratio in config.portfolio.category_ratios.items():
            category_upper = category.upper()
            if category_upper not in tickers_by_category:
                logger.warning(f"No tickers found for category {category}")
                continue

            # Skip categories with 0% allocation
            if ratio == 0:
                logger.debug(f"Skipping {category} - 0% allocation")
                continue

            category_capital = target_capital * ratio
            category_tickers = tickers_by_category[category_upper]
            category_allocations[category] = {
                "capital": category_capital,
                "tickers": category_tickers,
                "quantities": {},
            }

            logger.debug(
                f"\n{category} allocation: ${category_capital:,.2f} ({ratio:.1%}) "
                f"across {len(category_tickers)} tickers"
            )

        # Aggressive allocation algorithm to maximize capital utilization
        total_allocated = 0
        category_unused: dict[str, float] = {}

        # First pass: Allocate within each category
        for category, allocation_info in category_allocations.items():
            cat_capital: float = allocation_info["capital"]  # type: ignore[assignment]
            cat_tickers: list[TickerConfig] = allocation_info["tickers"]  # type: ignore[assignment]

            if config.portfolio.allow_fractional_shares:
                # Simple equal allocation for fractional shares
                capital_per_ticker = cat_capital / len(cat_tickers)
                for ticker_config in cat_tickers:
                    price = calculation_prices[ticker_config.symbol]
                    quantity = capital_per_ticker / price
                    calculated_quantities[ticker_config.symbol] = quantity
                    actual_cost = quantity * price
                    total_allocated += actual_cost
                    logger.debug(f"  {ticker_config.symbol}: {quantity:.4f} shares × ${price:.2f} = ${actual_cost:.2f}")
                category_unused[category] = 0  # No unused capital with fractional shares
            else:
                # Integer shares: optimize allocation for balanced portfolio
                remaining_capital = cat_capital
                ticker_quantities = {}

                # Calculate target value per ticker for balanced allocation
                target_value_per_ticker = cat_capital / len(cat_tickers)

                # Sort tickers by price to allocate more expensive ones first
                sorted_tickers = sorted(
                    cat_tickers,
                    key=lambda t: calculation_prices[t.symbol],
                    reverse=True,
                )

                # First pass: Try to get each ticker close to its target value
                for ticker_config in sorted_tickers:
                    price = calculation_prices[ticker_config.symbol]
                    symbol = ticker_config.symbol

                    # Calculate ideal number of shares for this ticker
                    ideal_shares = target_value_per_ticker / price

                    # Round to nearest integer, but at least 1 if we can afford it
                    if remaining_capital >= price:
                        # For expensive stocks, round down to conserve capital
                        # For cheap stocks, round to nearest to better match target
                        if price > target_value_per_ticker * 0.5:
                            target_shares = max(1, int(ideal_shares))
                        else:
                            target_shares = max(1, round(ideal_shares))

                        # Don't exceed available capital
                        max_affordable = int(remaining_capital / price)
                        actual_shares = min(target_shares, max_affordable)

                        ticker_quantities[symbol] = actual_shares
                        remaining_capital -= actual_shares * price
                    else:
                        ticker_quantities[symbol] = 0

                # Second pass: Distribute remaining capital more evenly
                # Sort by how far each ticker is from its target value
                ticker_distances = []
                for ticker_config in cat_tickers:
                    symbol = ticker_config.symbol
                    price = calculation_prices[symbol]
                    current_value = ticker_quantities[symbol] * price
                    distance_from_target = target_value_per_ticker - current_value
                    if distance_from_target > price:  # Can add at least one more share
                        ticker_distances.append((symbol, distance_from_target, price))

                # Sort by distance from target (largest distance first)
                ticker_distances.sort(key=lambda x: x[1], reverse=True)

                # Allocate remaining capital to stocks furthest from their target
                for symbol, distance, price in ticker_distances:
                    while remaining_capital >= price and distance > price:
                        ticker_quantities[symbol] += 1
                        remaining_capital -= price
                        distance -= price

                # Store results and calculate actual costs
                for ticker_config in cat_tickers:
                    symbol = ticker_config.symbol
                    quantity = ticker_quantities[symbol]
                    price = calculation_prices[symbol]
                    actual_cost = quantity * price

                    calculated_quantities[symbol] = quantity
                    total_allocated += actual_cost
                    logger.debug(f"  {symbol}: {quantity:.0f} shares × ${price:.2f} = ${actual_cost:.2f}")

                category_unused[category] = remaining_capital
                logger.debug(f"  Category unused capital: ${remaining_capital:.2f}")

        # Second pass: Redistribute remaining capital for better balance
        if not config.portfolio.allow_fractional_shares:
            # Calculate remaining capital from initial investment (not just category leftovers)
            remaining_capital = float(config.portfolio.initial_capital - total_allocated)
            logger.debug(f"\nRedistributing remaining capital for balance: ${remaining_capital:.2f}")

            # Calculate current portfolio values and identify underweight positions
            ticker_values = {}
            total_value = 0
            for symbol, quantity in calculated_quantities.items():
                if quantity > 0:
                    value = quantity * calculation_prices[symbol]
                    ticker_values[symbol] = value
                    total_value += value

            # Calculate average position value (excluding zero positions)
            positions_with_shares = sum(1 for q in calculated_quantities.values() if q > 0)
            if positions_with_shares > 0:
                avg_position_value = total_value / positions_with_shares
            else:
                avg_position_value = 0

            # Redistribute to positions below average
            max_iterations = 100  # Prevent infinite loops
            iteration = 0
            while remaining_capital > 0 and iteration < max_iterations:
                iteration += 1
                any_allocation = False

                # Find positions below average that we can add to
                underweight_positions = []
                for symbol, quantity in calculated_quantities.items():
                    if quantity > 0:  # Only consider positions we already have
                        current_value = ticker_values.get(symbol, 0)
                        price = calculation_prices[symbol]

                        # Check if below average and we can afford more
                        if current_value < avg_position_value * 0.9 and price <= remaining_capital:
                            distance_from_avg = avg_position_value - current_value
                            underweight_positions.append((symbol, distance_from_avg, price))

                # Sort by distance from average (most underweight first)
                underweight_positions.sort(key=lambda x: x[1], reverse=True)

                # Add shares to most underweight positions
                for symbol, _distance, price in underweight_positions:
                    if price <= remaining_capital:
                        calculated_quantities[symbol] += 1
                        remaining_capital -= price
                        total_allocated += price
                        ticker_values[symbol] += price
                        any_allocation = True
                        logger.debug(f"  Balanced redistribution: +1 {symbol} share (${price:.2f})")
                        break  # Recalculate after each addition

                if not any_allocation:
                    # If no underweight positions, add to smallest positions
                    smallest_positions = sorted(
                        [
                            (s, ticker_values.get(s, 0), calculation_prices[s])
                            for s in calculated_quantities.keys()
                            if calculated_quantities[s] > 0 and calculation_prices[s] <= remaining_capital
                        ],
                        key=lambda x: x[1],
                    )

                    if smallest_positions:
                        symbol, current_value, price = smallest_positions[0]
                        calculated_quantities[symbol] += 1
                        remaining_capital -= price
                        total_allocated += price
                        ticker_values[symbol] = current_value + price
                        logger.debug(f"  Final redistribution: +1 {symbol} share (${price:.2f})")
                    else:
                        break  # Can't afford any more shares

            logger.debug(f"Final unused capital: ${remaining_capital:.2f}")

        # Calculate final utilization statistics
        actual_utilization = total_allocated / config.portfolio.initial_capital

        logger.info(f"\nTotal portfolio cost: ${total_allocated:,.2f}")
        logger.info(f"Capital utilization: {actual_utilization:.1%}")
        logger.info(f"Remaining cash: ${config.portfolio.initial_capital - total_allocated:,.2f}")

        return calculated_quantities

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
            rebalance_frequency=config.portfolio.rebalance_frequency,
            max_position_size=config.portfolio.max_position_size,
            stop_loss_pct=config.portfolio.stop_loss_pct,
        )

        # Add tickers from portfolio config
        tickers_to_add = config.portfolio.tickers
        calculated_quantity: float | None = None

        # Handle different allocation modes
        if config.portfolio.auto_allocate:
            logger.info(
                "Using auto-allocation - optimizing quantities based on category ratios "
                "and capital utilization target..."
            )
            calculated_quantities = self._calculate_auto_allocation_quantities(config, tickers_to_add)

            for ticker_config in tickers_to_add:
                calculated_quantity = calculated_quantities.get(ticker_config.symbol, 0.0)
                # Skip tickers with 0 allocation (e.g., from categories with 0% ratio)
                if calculated_quantity > 0:
                    asset = self._create_asset(ticker_config, calculated_quantity)
                    portfolio.add_asset(asset)
                else:
                    logger.debug(f"Skipping {ticker_config.symbol} - 0 shares allocated")
        elif config.portfolio.dynamic_allocation:
            logger.info("Using dynamic allocation - calculating quantities based on allocation percentages/amounts...")
            calculated_quantities = self._calculate_dynamic_quantities(config, tickers_to_add)

            for ticker_config in tickers_to_add:
                calculated_quantity = calculated_quantities.get(ticker_config.symbol)
                asset = self._create_asset(ticker_config, calculated_quantity)
                portfolio.add_asset(asset)
                if calculated_quantity is not None:
                    logger.debug(f"  {ticker_config.symbol}: {calculated_quantity:.4f} shares")
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
