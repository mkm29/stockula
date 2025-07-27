"""Factory for creating domain objects from configuration."""

from typing import List, Optional, Dict
from ..config import StockulaConfig, TickerConfig
from .ticker import Ticker, TickerRegistry
from .asset import Asset
from .portfolio import Portfolio
from .category import Category


class DomainFactory:
    """Factory for creating domain objects from configuration."""

    def __init__(self):
        """Initialize factory with ticker registry."""
        self.ticker_registry = TickerRegistry()

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

    def _create_asset(
        self, ticker_config: TickerConfig, calculated_quantity: Optional[float] = None
    ) -> Asset:
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
        quantity = (
            calculated_quantity
            if calculated_quantity is not None
            else ticker_config.quantity
        )

        if quantity is None:
            raise ValueError(f"No quantity specified for ticker {ticker_config.symbol}")

        return Asset(ticker=ticker, quantity=quantity, category=category)

    def _calculate_dynamic_quantities(
        self, config: StockulaConfig, tickers_to_add: List[TickerConfig]
    ) -> Dict[str, float]:
        """Calculate quantities dynamically based on allocation percentages/amounts.

        Args:
            config: Stockula configuration
            tickers_to_add: List of ticker configurations

        Returns:
            Dictionary mapping ticker symbols to calculated quantities
        """
        from ..data.fetcher import DataFetcher

        # Fetch prices for calculation - use start date if available for backtesting
        fetcher = DataFetcher()
        symbols = [ticker.symbol for ticker in tickers_to_add]

        if config.data.start_date:
            # Use start date prices for backtesting to ensure portfolio value matches at start
            start_date_str = config.data.start_date.strftime("%Y-%m-%d")
            print(
                f"Calculating quantities using start date prices ({start_date_str}) for accurate portfolio value..."
            )

            calculation_prices = {}
            for symbol in symbols:
                try:
                    data = fetcher.get_stock_data(
                        symbol, start=start_date_str, end=start_date_str
                    )
                    if not data.empty:
                        calculation_prices[symbol] = data["Close"].iloc[0]
                    else:
                        # Fallback to a few days later if no data on exact date
                        from datetime import timedelta

                        end_date = (
                            config.data.start_date + timedelta(days=7)
                        ).strftime("%Y-%m-%d")
                        data = fetcher.get_stock_data(
                            symbol, start=start_date_str, end=end_date
                        )
                        if not data.empty:
                            calculation_prices[symbol] = data["Close"].iloc[0]
                        else:
                            # Last resort: use current prices
                            current_prices = fetcher.get_current_prices([symbol])
                            if symbol in current_prices:
                                calculation_prices[symbol] = current_prices[symbol]
                                print(
                                    f"Warning: Using current price for {symbol} (no historical data available)"
                                )
                except Exception as e:
                    print(f"Error fetching start date price for {symbol}: {e}")
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
                allocation_amount = (
                    ticker_config.allocation_pct / 100.0
                ) * config.portfolio.initial_capital
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
        self, config: StockulaConfig, tickers_to_add: List[TickerConfig]
    ) -> Dict[str, float]:
        """Calculate quantities using auto-allocation based on category ratios and capital utilization target.

        This method optimizes for maximum capital utilization while respecting category allocation ratios.

        Args:
            config: Stockula configuration
            tickers_to_add: List of ticker configurations (should only have category specified)

        Returns:
            Dictionary mapping ticker symbols to calculated quantities
        """
        from ..data.fetcher import DataFetcher

        if not config.portfolio.category_ratios:
            raise ValueError("Auto-allocation requires category_ratios to be specified")

        # Fetch prices for calculation - use start date if available for backtesting
        fetcher = DataFetcher()
        symbols = [ticker.symbol for ticker in tickers_to_add]

        if config.data.start_date:
            # Use start date prices for backtesting to ensure portfolio value matches at start
            start_date_str = config.data.start_date.strftime("%Y-%m-%d")
            print(
                f"Using start date prices ({start_date_str}) for auto-allocation calculations..."
            )

            calculation_prices = {}
            for symbol in symbols:
                try:
                    data = fetcher.get_stock_data(
                        symbol, start=start_date_str, end=start_date_str
                    )
                    if not data.empty:
                        calculation_prices[symbol] = data["Close"].iloc[0]
                    else:
                        # Fallback to a few days later if no data on exact date
                        from datetime import timedelta

                        end_date = (
                            config.data.start_date + timedelta(days=7)
                        ).strftime("%Y-%m-%d")
                        data = fetcher.get_stock_data(
                            symbol, start=start_date_str, end=end_date
                        )
                        if not data.empty:
                            calculation_prices[symbol] = data["Close"].iloc[0]
                        else:
                            # Last resort: use current prices
                            current_prices = fetcher.get_current_prices([symbol])
                            if symbol in current_prices:
                                calculation_prices[symbol] = current_prices[symbol]
                                print(
                                    f"Warning: Using current price for {symbol} (no historical data available)"
                                )
                except Exception as e:
                    print(f"Error fetching start date price for {symbol}: {e}")
                    # Fallback to current prices
                    current_prices = fetcher.get_current_prices([symbol])
                    if symbol in current_prices:
                        calculation_prices[symbol] = current_prices[symbol]
        else:
            # No start date specified, use current prices
            calculation_prices = fetcher.get_current_prices(symbols)

        # Group tickers by category
        tickers_by_category = {}
        for ticker_config in tickers_to_add:
            if ticker_config.symbol not in calculation_prices:
                raise ValueError(f"Could not fetch price for {ticker_config.symbol}")

            if not ticker_config.category:
                raise ValueError(
                    f"Ticker {ticker_config.symbol} must have category specified for auto-allocation"
                )

            category = ticker_config.category.upper()
            if category not in tickers_by_category:
                tickers_by_category[category] = []
            tickers_by_category[category].append(ticker_config)

        # Calculate target capital per category
        target_capital = (
            config.portfolio.initial_capital
            * config.portfolio.capital_utilization_target
        )
        calculated_quantities = {}

        print(
            f"Auto-allocation target capital: ${target_capital:,.2f} ({config.portfolio.capital_utilization_target:.1%} of ${config.portfolio.initial_capital:,.2f})"
        )

        for category, ratio in config.portfolio.category_ratios.items():
            category_upper = category.upper()
            if category_upper not in tickers_by_category:
                print(f"Warning: No tickers found for category {category}")
                continue

            category_capital = target_capital * ratio
            category_tickers = tickers_by_category[category_upper]

            print(
                f"\n{category} allocation: ${category_capital:,.2f} ({ratio:.1%}) across {len(category_tickers)} tickers"
            )

            # Equal allocation within category
            capital_per_ticker = category_capital / len(category_tickers)

            for ticker_config in category_tickers:
                price = calculation_prices[ticker_config.symbol]
                raw_quantity = capital_per_ticker / price

                if config.portfolio.allow_fractional_shares:
                    quantity = raw_quantity
                else:
                    quantity = max(1, int(raw_quantity))

                calculated_quantities[ticker_config.symbol] = quantity
                actual_cost = quantity * price
                print(
                    f"  {ticker_config.symbol}: {quantity:.4f} shares × ${price:.2f} = ${actual_cost:.2f}"
                )

        # Calculate actual total cost and utilization
        total_cost = sum(
            calculated_quantities[symbol] * calculation_prices[symbol]
            for symbol in calculated_quantities
        )
        actual_utilization = total_cost / config.portfolio.initial_capital

        print(f"\nTotal portfolio cost: ${total_cost:,.2f}")
        print(f"Capital utilization: {actual_utilization:.1%}")
        print(f"Remaining cash: ${config.portfolio.initial_capital - total_cost:,.2f}")

        return calculated_quantities

    def create_portfolio(self, config: StockulaConfig) -> Portfolio:
        """Create complete portfolio from configuration.

        Args:
            config: Complete Stockula configuration

        Returns:
            Portfolio instance
        """
        portfolio = Portfolio(
            name=config.portfolio.name,
            initial_capital=config.portfolio.initial_capital,
            allocation_method=config.portfolio.allocation_method,
            rebalance_frequency=config.portfolio.rebalance_frequency,
            max_position_size=config.portfolio.max_position_size,
            stop_loss_pct=config.portfolio.stop_loss_pct,
        )

        # Add tickers from portfolio config
        tickers_to_add = config.portfolio.tickers

        # Handle different allocation modes
        if config.portfolio.auto_allocate:
            print(
                "Using auto-allocation - optimizing quantities based on category ratios and capital utilization target..."
            )
            calculated_quantities = self._calculate_auto_allocation_quantities(
                config, tickers_to_add
            )

            for ticker_config in tickers_to_add:
                calculated_quantity = calculated_quantities.get(ticker_config.symbol)
                asset = self._create_asset(ticker_config, calculated_quantity)
                portfolio.add_asset(asset)
        elif config.portfolio.dynamic_allocation:
            print(
                "Using dynamic allocation - calculating quantities based on allocation percentages/amounts..."
            )
            calculated_quantities = self._calculate_dynamic_quantities(
                config, tickers_to_add
            )

            for ticker_config in tickers_to_add:
                calculated_quantity = calculated_quantities.get(ticker_config.symbol)
                asset = self._create_asset(ticker_config, calculated_quantity)
                portfolio.add_asset(asset)
                print(f"  {ticker_config.symbol}: {calculated_quantity:.4f} shares")
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

    def get_all_tickers(self) -> List[Ticker]:
        """Get all registered tickers.

        Returns:
            List of all ticker instances
        """
        return list(self.ticker_registry.all().values())
