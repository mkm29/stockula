"""Asset allocation strategies for portfolio construction."""

import logging
from datetime import date, timedelta
from typing import TYPE_CHECKING

import pandas as pd

from ..config import StockulaConfig, TickerConfig

if TYPE_CHECKING:
    from ..data.fetcher import DataFetcher

logger = logging.getLogger(__name__)


def date_to_string(date_value: str | date | None) -> str | None:
    """Convert date or string to string format."""
    if date_value is None:
        return None
    if isinstance(date_value, str):
        return date_value
    return date_value.strftime("%Y-%m-%d")


class Allocator:
    """Handles asset allocation strategies for portfolio construction."""

    def __init__(self, fetcher: "DataFetcher"):
        """Initialize allocator with data fetcher.

        Args:
            fetcher: Data fetcher instance for price lookups
        """
        self.fetcher = fetcher

    def calculate_dynamic_quantities(
        self, config: StockulaConfig, tickers_to_add: list[TickerConfig]
    ) -> dict[str, float]:
        """Calculate quantities dynamically based on allocation percentages/amounts.

        Args:
            config: Stockula configuration
            tickers_to_add: List of ticker configurations

        Returns:
            Dictionary mapping ticker symbols to calculated quantities
        """
        if self.fetcher is None:
            raise ValueError("Data fetcher not configured")

        symbols = [ticker.symbol for ticker in tickers_to_add]
        calculation_prices = self._get_calculation_prices(config, symbols)

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

    def calculate_auto_allocation_quantities(
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
        if self.fetcher is None:
            raise ValueError("Data fetcher not configured")

        symbols = [ticker.symbol for ticker in tickers_to_add]
        calculation_prices = self._get_calculation_prices(config, symbols)

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

                    # Skip if we can't afford even one share
                    if price > remaining_capital:
                        ticker_quantities[ticker_config.symbol] = 0
                        continue

                    # Calculate ideal quantity based on target value
                    ideal_quantity = target_value_per_ticker / price
                    quantity = max(1, int(ideal_quantity))

                    # Ensure we don't exceed remaining capital
                    while quantity * price > remaining_capital and quantity > 0:
                        quantity -= 1

                    if quantity > 0:
                        ticker_quantities[ticker_config.symbol] = quantity
                        cost = quantity * price
                        remaining_capital -= cost
                        total_allocated += cost
                        logger.debug(
                            f"  {ticker_config.symbol}: {quantity} shares × ${price:.2f} = ${cost:.2f} "
                            f"(target: ${target_value_per_ticker:.2f})"
                        )
                    else:
                        ticker_quantities[ticker_config.symbol] = 0

                # Update calculated quantities
                for symbol, quantity in ticker_quantities.items():
                    calculated_quantities[symbol] = quantity

                category_unused[category] = remaining_capital
                logger.debug(f"  Unused capital in {category}: ${remaining_capital:.2f}")

        # Second pass: Aggressive redistribution of all unused capital
        remaining_capital = sum(category_unused.values())
        if remaining_capital > 0 and not config.portfolio.allow_fractional_shares:
            logger.debug(f"\nRedistributing unused capital: ${remaining_capital:.2f}")

            # Calculate position values for balancing
            ticker_values = {}
            for symbol, quantity in calculated_quantities.items():
                if quantity > 0:
                    ticker_values[symbol] = quantity * calculation_prices[symbol]

            # Calculate average position value for balance targeting
            if ticker_values:
                avg_position_value = sum(ticker_values.values()) / len(ticker_values)
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

    def _get_calculation_prices(self, config: StockulaConfig, symbols: list[str]) -> dict[str, float]:
        """Get prices for calculation (historical or current).

        Args:
            config: Stockula configuration
            symbols: List of ticker symbols

        Returns:
            Dictionary mapping symbols to prices
        """
        if config.data.start_date:
            # Use start date prices for backtesting to ensure portfolio value matches at start
            start_date_str = date_to_string(config.data.start_date)
            logger.debug(
                f"Calculating quantities using start date prices ({start_date_str}) for accurate portfolio value..."
            )
            calculation_prices = {}

            for symbol in symbols:
                try:
                    data = self.fetcher.get_stock_data(symbol, start=start_date_str, end=start_date_str)
                    if not data.empty:
                        calculation_prices[symbol] = data["Close"].iloc[0]
                    else:
                        # Fallback to a few days later if no data on exact date
                        if isinstance(config.data.start_date, str):
                            start_dt = pd.to_datetime(config.data.start_date)
                            end_date = (start_dt + timedelta(days=7)).strftime("%Y-%m-%d")
                        else:
                            end_date = (config.data.start_date + timedelta(days=7)).strftime("%Y-%m-%d")

                        data = self.fetcher.get_stock_data(symbol, start=start_date_str, end=end_date)
                        if not data.empty:
                            calculation_prices[symbol] = data["Close"].iloc[0]
                        else:
                            # Last resort: use current prices
                            current_prices = self.fetcher.get_current_prices([symbol])
                            if symbol in current_prices:
                                calculation_prices[symbol] = current_prices[symbol]
                                logger.warning(f"Using current price for {symbol} (no historical data available)")
                except Exception as e:
                    logger.error(f"Error fetching start date price for {symbol}: {e}")
                    # Fallback to current prices
                    current_prices = self.fetcher.get_current_prices([symbol])
                    if symbol in current_prices:
                        calculation_prices[symbol] = current_prices[symbol]
        else:
            # No start date specified, use current prices
            calculation_prices = self.fetcher.get_current_prices(symbols)

        return calculation_prices
