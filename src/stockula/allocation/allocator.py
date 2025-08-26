"""Asset allocation strategies for portfolio construction."""

from typing import TYPE_CHECKING

from dependency_injector.wiring import Provide, inject

from ..config import StockulaConfig, TickerConfig
from ..interfaces import ILoggingManager
from .base_allocator import BaseAllocator

if TYPE_CHECKING:
    from ..data.fetcher import DataFetcher


class Allocator(BaseAllocator):
    """Standard allocator that handles various asset allocation strategies.

    This allocator supports:
    - Equal weight allocation
    - Market cap weighted allocation
    - Custom allocation (fixed amounts/percentages)
    - Dynamic allocation (based on prices and targets)
    - Auto allocation (category-based)
    """

    @inject
    def __init__(
        self,
        fetcher: "DataFetcher",
        logging_manager: ILoggingManager = Provide["logging_manager"],
    ):
        """Initialize allocator with data fetcher and logging manager.

        Args:
            fetcher: Data fetcher instance for price lookups
            logging_manager: Injected logging manager
        """
        super().__init__(fetcher, logging_manager)

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
        self._validate_fetcher()

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

            # Calculate quantity using base class method
            quantity = self._calculate_quantity_for_allocation(
                allocation_amount, price, config.portfolio.allow_fractional_shares
            )
            calculated_quantities[ticker_config.symbol] = quantity

        return calculated_quantities

    def calculate_auto_allocation_quantities(
        self, config: StockulaConfig, tickers_to_add: list[TickerConfig]
    ) -> dict[str, float]:
        """Calculate quantities using auto-allocation based on category ratios and capital utilization target.

        This method optimizes for maximum capital utilization while respecting category allocation ratios.
        The implementation delegates steps to helpers to keep cognitive complexity low.
        """
        self._validate_fetcher()

        symbols = [ticker.symbol for ticker in tickers_to_add]
        calculation_prices = self._get_calculation_prices(config, symbols)

        # Basic validation and grouping
        tickers_by_category = self._group_tickers_by_category(tickers_to_add, calculation_prices)

        if config.portfolio.category_ratios is None:
            raise ValueError("Category ratios must be specified for auto-allocation")

        target_capital = config.portfolio.initial_capital * config.portfolio.capital_utilization_target
        calculated_quantities: dict[str, float] = {t.symbol: 0.0 for t in tickers_to_add}

        self.logger.debug(
            f"Auto-allocation target capital: ${target_capital:,.2f} "
            f"({config.portfolio.capital_utilization_target:.1%} of ${config.portfolio.initial_capital:,.2f})"
        )

        # Prepare category allocations
        category_allocations = self._init_category_allocations(config, tickers_by_category, target_capital)

        total_allocated = 0.0
        category_unused: dict[str, float] = {}

        # Allocate per category using the appropriate method
        for category, allocation_info in category_allocations.items():
            if config.portfolio.allow_fractional_shares:
                allocated, unused = self._allocate_category_fractional(
                    allocation_info, calculation_prices, calculated_quantities
                )
            else:
                allocated, unused = self._allocate_category_integer(
                    allocation_info, calculation_prices, calculated_quantities
                )

            total_allocated += allocated
            category_unused[category] = unused
            self.logger.debug(f"  Unused capital in {category}: ${unused:.2f}")

        # Redistribute unused capital if needed (only for integer shares)
        remaining_capital = sum(category_unused.values())
        if remaining_capital > 0 and not config.portfolio.allow_fractional_shares:
            self.logger.debug(f"\nRedistributing unused capital: ${remaining_capital:.2f}")
            redistributed, remaining_capital = self._redistribute_unused_capital(
                calculated_quantities, calculation_prices, remaining_capital, config
            )
            total_allocated += redistributed
            self.logger.debug(f"Final unused capital: ${remaining_capital:.2f}")

        # Final stats
        actual_utilization = total_allocated / config.portfolio.initial_capital
        self.logger.info(f"\nTotal portfolio cost: ${total_allocated:,.2f}")
        self.logger.info(f"Capital utilization: {actual_utilization:.1%}")
        self.logger.info(f"Remaining cash: ${config.portfolio.initial_capital - total_allocated:,.2f}")

        return calculated_quantities

    def _group_tickers_by_category(
        self, tickers_to_add: list[TickerConfig], calculation_prices: dict[str, float]
    ) -> dict[str, list[TickerConfig]]:
        tickers_by_category: dict[str, list[TickerConfig]] = {}
        for ticker_config in tickers_to_add:
            if ticker_config.symbol not in calculation_prices:
                raise ValueError(f"Could not fetch price for {ticker_config.symbol}")

            if not ticker_config.category:
                raise ValueError(f"Ticker {ticker_config.symbol} must have category specified for auto-allocation")

            category = ticker_config.category.upper()
            tickers_by_category.setdefault(category, []).append(ticker_config)
        return tickers_by_category

    def _init_category_allocations(
        self, config: StockulaConfig, tickers_by_category: dict[str, list[TickerConfig]], target_capital: float
    ) -> dict[str, dict]:
        category_allocations: dict[str, dict] = {}
        for category, ratio in config.portfolio.category_ratios.items():
            category_upper = category.upper()
            if category_upper not in tickers_by_category:
                self.logger.warning(f"No tickers found for category {category}")
                continue

            if ratio == 0:
                self.logger.debug(f"Skipping {category} - 0% allocation")
                continue

            category_capital = target_capital * ratio
            category_tickers = tickers_by_category[category_upper]
            category_allocations[category] = {
                "capital": category_capital,
                "tickers": category_tickers,
            }

            self.logger.debug(
                f"\n{category} allocation: ${category_capital:,.2f} ({ratio:.1%}) "
                f"across {len(category_tickers)} tickers"
            )
        return category_allocations

    def _allocate_category_fractional(
        self, allocation_info: dict, calculation_prices: dict[str, float], calculated_quantities: dict[str, float]
    ) -> tuple[float, float]:
        cat_capital: float = allocation_info["capital"]
        cat_tickers: list[TickerConfig] = allocation_info["tickers"]
        capital_per_ticker = cat_capital / len(cat_tickers)
        allocated = 0.0
        for ticker_config in cat_tickers:
            price = calculation_prices[ticker_config.symbol]
            quantity = capital_per_ticker / price
            calculated_quantities[ticker_config.symbol] = quantity
            cost = quantity * price
            allocated += cost
            self.logger.debug(f"  {ticker_config.symbol}: {quantity:.4f} shares × ${price:.2f} = ${cost:.2f}")
        unused = 0.0
        return allocated, unused

    def _allocate_category_integer(
        self, allocation_info: dict, calculation_prices: dict[str, float], calculated_quantities: dict[str, float]
    ) -> tuple[float, float]:
        cat_capital: float = allocation_info["capital"]
        cat_tickers: list[TickerConfig] = allocation_info["tickers"]
        remaining_capital = cat_capital
        allocated = 0.0

        target_value_per_ticker = cat_capital / len(cat_tickers)
        sorted_tickers = sorted(
            cat_tickers,
            key=lambda t: calculation_prices[t.symbol],
            reverse=True,
        )

        ticker_quantities: dict[str, int] = {}
        for ticker_config in sorted_tickers:
            price = calculation_prices[ticker_config.symbol]

            if price > remaining_capital:
                ticker_quantities[ticker_config.symbol] = 0
                continue

            ideal_quantity = target_value_per_ticker / price
            quantity = max(1, int(ideal_quantity))

            while quantity * price > remaining_capital and quantity > 0:
                quantity -= 1

            if quantity > 0:
                ticker_quantities[ticker_config.symbol] = quantity
                cost = quantity * price
                remaining_capital -= cost
                allocated += cost
                self.logger.debug(
                    f"  {ticker_config.symbol}: {quantity} shares × ${price:.2f} = ${cost:.2f} "
                    f"(target: ${target_value_per_ticker:.2f})"
                )
            else:
                ticker_quantities[ticker_config.symbol] = 0

        for symbol, qty in ticker_quantities.items():
            calculated_quantities[symbol] = qty

        return allocated, remaining_capital

    def _redistribute_unused_capital(
        self,
        calculated_quantities: dict[str, float],
        calculation_prices: dict[str, float],
        remaining_capital: float,
    ) -> tuple[float, float]:
        # Prepare current position values and accumulation
        ticker_values = self._prepare_ticker_values_for_redistribution(calculated_quantities, calculation_prices)
        redistributed = 0.0

        avg_position_value = self._average_ticker_value(ticker_values)
        max_iterations = 100
        iteration = 0

        while remaining_capital > 0 and iteration < max_iterations:
            iteration += 1

            # Try allocating to underweight positions first; if that fails, try the smallest affordable position.
            allocated, cost = self._try_allocate_underweights_for_redistribution(
                calculated_quantities, ticker_values, calculation_prices, avg_position_value, remaining_capital
            )
            if not allocated:
                allocated, cost = self._try_allocate_smallest_for_redistribution(
                    calculated_quantities, ticker_values, calculation_prices, remaining_capital
                )

            if not allocated:
                break

            remaining_capital -= cost
            redistributed += cost

            # Recalculate average after changes
            avg_position_value = self._average_ticker_value(ticker_values)

        return redistributed, remaining_capital

    def _try_allocate_underweights_for_redistribution(
        self,
        calculated_quantities: dict[str, float],
        ticker_values: dict[str, float],
        calculation_prices: dict[str, float],
        avg_position_value: float,
        avail_cash: float,
    ) -> tuple[bool, float]:
        """Attempt to allocate one share to the largest underweight position that is affordable.

        Returns (allocated_flag, cost) where cost is 0.0 if nothing was allocated.
        """
        underweights = self._find_underweight_positions_for_redistribution(
            calculated_quantities, ticker_values, calculation_prices, avg_position_value, avail_cash
        )
        if not underweights:
            return False, 0.0

        for symbol, _dist, price in underweights:
            if price <= avail_cash:
                cost = self._allocate_one_share_for_redistribution(calculated_quantities, ticker_values, symbol, price)
                self.logger.debug(f"  Balanced redistribution: +1 {symbol} share (${price:.2f})")
                return True, cost

        return False, 0.0

    def _try_allocate_smallest_for_redistribution(
        self,
        calculated_quantities: dict[str, float],
        ticker_values: dict[str, float],
        calculation_prices: dict[str, float],
        avail_cash: float,
    ) -> tuple[bool, float]:
        """Attempt to allocate one share to the smallest affordable position.

        Returns (allocated_flag, cost) where cost is 0.0 if nothing was allocated.
        """
        smallest = self._find_affordable_smallest_for_redistribution(
            calculated_quantities, ticker_values, calculation_prices, avail_cash
        )
        if not smallest:
            return False, 0.0

        symbol, _current_value, price = smallest[0]
        cost = self._allocate_one_share_for_redistribution(calculated_quantities, ticker_values, symbol, price)
        self.logger.debug(f"  Final redistribution: +1 {symbol} share (${price:.2f})")
        return True, cost

    def _prepare_ticker_values_for_redistribution(
        self, calculated_quantities: dict[str, float], calculation_prices: dict[str, float]
    ) -> dict[str, float]:
        return {s: q * calculation_prices[s] for s, q in calculated_quantities.items() if q > 0}

    def _average_ticker_value(self, ticker_values: dict[str, float]) -> float:
        return (sum(ticker_values.values()) / len(ticker_values)) if ticker_values else 0.0

    def _find_underweight_positions_for_redistribution(
        self,
        calculated_quantities: dict[str, float],
        ticker_values: dict[str, float],
        calculation_prices: dict[str, float],
        avg_val: float,
        avail_cash: float,
    ) -> list[tuple[str, float, float]]:
        positions: list[tuple[str, float, float]] = []
        for symbol, qty in calculated_quantities.items():
            if qty <= 0:
                continue
            current_value = ticker_values.get(symbol, 0.0)
            price = calculation_prices[symbol]
            if current_value < avg_val * 0.9 and price <= avail_cash:
                positions.append((symbol, avg_val - current_value, price))
        return sorted(positions, key=lambda x: x[1], reverse=True)

    def _find_affordable_smallest_for_redistribution(
        self,
        calculated_quantities: dict[str, float],
        ticker_values: dict[str, float],
        calculation_prices: dict[str, float],
        avail_cash: float,
    ) -> list[tuple[str, float, float]]:
        items = [
            (s, ticker_values.get(s, 0.0), calculation_prices[s])
            for s in calculated_quantities.keys()
            if calculated_quantities[s] > 0 and calculation_prices[s] <= avail_cash
        ]
        return sorted(items, key=lambda x: x[1])

    def _allocate_one_share_for_redistribution(
        self,
        calculated_quantities: dict[str, float],
        ticker_values: dict[str, float],
        symbol: str,
        price: float,
    ) -> float:
        # allocate one share and return the cost (so caller updates remaining_capital and redistributed)
        calculated_quantities[symbol] += 1
        ticker_values[symbol] = ticker_values.get(symbol, 0.0) + price
        return price

    def calculate_equal_weight_quantities(
        self, config: StockulaConfig, tickers: list[TickerConfig]
    ) -> dict[str, float]:
        """Calculate equal weight quantities for each ticker.

        Args:
            config: Stockula configuration
            tickers: List of ticker configurations

        Returns:
            Dictionary mapping ticker symbols to calculated quantities
        """
        self._validate_fetcher()

        symbols = [ticker.symbol for ticker in tickers]
        calculation_prices = self._get_calculation_prices(config, symbols)

        # Calculate equal allocation per ticker
        num_tickers = len(tickers)
        if num_tickers == 0:
            return {}

        allocation_per_ticker = config.portfolio.initial_capital / num_tickers
        calculated_quantities = {}

        for ticker_config in tickers:
            if ticker_config.symbol not in calculation_prices:
                raise ValueError(f"Could not fetch price for {ticker_config.symbol}")

            price = calculation_prices[ticker_config.symbol]
            quantity = self._calculate_quantity_for_allocation(
                allocation_per_ticker, price, config.portfolio.allow_fractional_shares
            )
            calculated_quantities[ticker_config.symbol] = quantity

        return calculated_quantities

    def calculate_market_cap_quantities(self, config: StockulaConfig, tickers: list[TickerConfig]) -> dict[str, float]:
        """Calculate market cap weighted quantities for each ticker.

        Args:
            config: Stockula configuration
            tickers: List of ticker configurations

        Returns:
            Dictionary mapping ticker symbols to calculated quantities
        """
        self._validate_fetcher()

        symbols = [ticker.symbol for ticker in tickers]

        market_caps, total_market_cap = self._fetch_market_caps(symbols)
        if total_market_cap == 0:
            self.logger.warning("No market cap data available, falling back to equal weight allocation")
            return self.calculate_equal_weight_quantities(config, tickers)

        weights = self._compute_market_cap_weights(symbols, market_caps, total_market_cap)

        calculation_prices = self._get_calculation_prices(config, symbols)
        calculated_quantities: dict[str, float] = {}

        for ticker_config in tickers:
            if ticker_config.symbol not in calculation_prices:
                raise ValueError(f"Could not fetch price for {ticker_config.symbol}")

            price = calculation_prices[ticker_config.symbol]
            weight = weights.get(ticker_config.symbol, 0.0)
            allocation_amount = config.portfolio.initial_capital * weight

            quantity = self._calculate_quantity_for_allocation(
                allocation_amount, price, config.portfolio.allow_fractional_shares
            )
            calculated_quantities[ticker_config.symbol] = quantity

        return calculated_quantities

    def _fetch_market_caps(self, symbols: list[str]) -> tuple[dict[str, float | None], float]:
        """Fetch market caps for symbols; return dict and total market cap."""
        market_caps: dict[str, float | None] = {}
        total_market_cap = 0.0

        for symbol in symbols:
            try:
                info = self.fetcher.get_info(symbol)
                if info and "marketCap" in info and info["marketCap"]:
                    market_cap = info["marketCap"]
                    market_caps[symbol] = market_cap
                    total_market_cap += market_cap
                else:
                    self.logger.warning(f"Could not fetch market cap for {symbol}, using placeholder")
                    market_caps[symbol] = None
            except Exception as e:
                self.logger.error(f"Error fetching market cap for {symbol}: {e}")
                market_caps[symbol] = None

        return market_caps, total_market_cap

    def _compute_market_cap_weights(
        self, symbols: list[str], market_caps: dict[str, float | None], total_market_cap: float
    ) -> dict[str, float]:
        """Compute normalized weights from market caps, using average weight for missing data."""
        if total_market_cap <= 0:
            # Defensive: return equal weights if nothing available
            return {s: 1.0 / len(symbols) for s in symbols}

        weights: dict[str, float] = {}
        for symbol in symbols:
            mc = market_caps.get(symbol)
            if mc is not None:
                weights[symbol] = mc / total_market_cap
            else:
                weights[symbol] = 1.0 / len(symbols)

        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {symbol: weight / total_weight for symbol, weight in weights.items()}

        return weights

    def calculate_quantities(
        self,
        config: StockulaConfig,
        tickers: list[TickerConfig],
        **kwargs,
    ) -> dict[str, float]:
        """Calculate quantities based on the configured allocation method.

        Args:
            config: Stockula configuration
            tickers: List of ticker configurations
            **kwargs: Additional parameters (unused in standard allocator)

        Returns:
            Dictionary mapping ticker symbols to calculated quantities
        """
        allocation_method = config.portfolio.allocation_method

        if allocation_method == "equal_weight":
            return self.calculate_equal_weight_quantities(config, tickers)
        elif allocation_method == "market_cap":
            return self.calculate_market_cap_quantities(config, tickers)
        elif allocation_method == "custom":
            # Custom allocation is handled by ticker configs
            return {ticker.symbol: ticker.quantity for ticker in tickers if ticker.quantity}
        elif allocation_method == "dynamic":
            return self.calculate_dynamic_quantities(config, tickers)
        elif allocation_method == "auto":
            return self.calculate_auto_allocation_quantities(config, tickers)
        elif allocation_method == "backtest_optimized":
            # For backtest_optimized, we need the BacktestOptimizedAllocator
            # This is a placeholder - in practice, the container should inject the right allocator
            raise ValueError(
                "backtest_optimized allocation method requires BacktestOptimizedAllocator. "
                "Please use --mode optimize-allocation to calculate optimal quantities."
            )
        else:
            raise ValueError(f"Unknown allocation method: {allocation_method}")
