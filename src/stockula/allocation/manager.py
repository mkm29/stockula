"""Allocator Manager - Manages and coordinates different allocation strategies."""

from typing import TYPE_CHECKING, Any, cast

from dependency_injector.wiring import Provide, inject

from ..config import StockulaConfig, TickerConfig
from ..interfaces import ILoggingManager
from .allocator import Allocator
from .backtest_allocator import BacktestOptimizedAllocator
from .base_allocator import BaseAllocator

if TYPE_CHECKING:
    from ..backtesting.runner import BacktestRunner
    from ..data.fetcher import DataFetcher


class AllocatorManager:
    """Manages different allocation strategies and provides unified interface.

    This manager coordinates between different allocator implementations:
    - Standard allocator for equal weight, market cap, custom, dynamic, and auto allocation
    - Backtest optimized allocator for allocation based on backtest results
    """

    @inject
    def __init__(
        self,
        data_fetcher: "DataFetcher" = Provide["data_fetcher"],
        backtest_runner: "BacktestRunner" = Provide["backtest_runner"],
        logging_manager: ILoggingManager = Provide["logging_manager"],
        forecast_manager=None,  # Optional ForecastingManager
    ):
        """Initialize the allocator manager.

        Args:
            data_fetcher: Data fetcher for price lookups
            backtest_runner: Backtest runner for optimization
            logging_manager: Logging manager for structured logging
            forecast_manager: Optional ForecastingManager for forecast-aware allocation
        """
        self.data_fetcher = data_fetcher
        self.backtest_runner = backtest_runner
        self.logger = logging_manager
        self.forecast_manager = forecast_manager

        # Initialize allocators
        self.standard_allocator = Allocator(
            fetcher=data_fetcher,
            logging_manager=logging_manager,
        )

        self.backtest_allocator = BacktestOptimizedAllocator(
            fetcher=data_fetcher,
            logging_manager=logging_manager,
            backtest_runner=backtest_runner,
            forecast_manager=forecast_manager,
        )

        # Map allocation methods to allocators
        self.allocator_map = {
            "equal_weight": self.standard_allocator,
            "market_cap": self.standard_allocator,
            "custom": self.standard_allocator,
            "dynamic": self.standard_allocator,
            "auto": self.standard_allocator,
            "backtest_optimized": self.backtest_allocator,
        }

    def get_allocator(self, allocation_method: str) -> BaseAllocator:
        """Get the appropriate allocator for the given allocation method.

        Args:
            allocation_method: The allocation method to use

        Returns:
            The appropriate allocator instance

        Raises:
            ValueError: If the allocation method is unknown
        """
        allocator = self.allocator_map.get(allocation_method)
        if allocator is None:
            raise ValueError(f"Unknown allocation method: {allocation_method}")
        return allocator  # type: ignore[no-any-return]

    def calculate_quantities(
        self,
        config: StockulaConfig,
        tickers: list[TickerConfig],
        **kwargs: Any,
    ) -> dict[str, float]:
        """Calculate quantities based on the configured allocation method.

        This method delegates to the appropriate allocator based on the
        allocation method specified in the configuration.

        Args:
            config: Stockula configuration
            tickers: List of ticker configurations
            **kwargs: Additional parameters passed to the allocator

        Returns:
            Dictionary mapping ticker symbols to calculated quantities
        """
        allocation_method = config.portfolio.allocation_method
        self.logger.info(f"Calculating quantities using {allocation_method} allocation method")

        allocator = self.get_allocator(allocation_method)
        return allocator.calculate_quantities(config, tickers, **kwargs)

    def calculate_equal_weight_quantities(
        self,
        config: StockulaConfig,
        tickers: list[TickerConfig],
    ) -> dict[str, float]:
        """Calculate equal weight quantities.

        Args:
            config: Stockula configuration
            tickers: List of ticker configurations

        Returns:
            Dictionary mapping ticker symbols to calculated quantities
        """
        return cast(dict[str, float], self.standard_allocator.calculate_equal_weight_quantities(config, tickers))

    def calculate_market_cap_quantities(
        self,
        config: StockulaConfig,
        tickers: list[TickerConfig],
    ) -> dict[str, float]:
        """Calculate market cap weighted quantities.

        Args:
            config: Stockula configuration
            tickers: List of ticker configurations

        Returns:
            Dictionary mapping ticker symbols to calculated quantities
        """
        return cast(dict[str, float], self.standard_allocator.calculate_market_cap_quantities(config, tickers))

    def calculate_dynamic_quantities(
        self,
        config: StockulaConfig,
        tickers: list[TickerConfig],
    ) -> dict[str, float]:
        """Calculate dynamic quantities based on allocation percentages/amounts.

        Args:
            config: Stockula configuration
            tickers: List of ticker configurations

        Returns:
            Dictionary mapping ticker symbols to calculated quantities
        """
        return cast(dict[str, float], self.standard_allocator.calculate_dynamic_quantities(config, tickers))

    def calculate_auto_allocation_quantities(
        self,
        config: StockulaConfig,
        tickers: list[TickerConfig],
    ) -> dict[str, float]:
        """Calculate auto-allocation quantities based on category ratios.

        Args:
            config: Stockula configuration
            tickers: List of ticker configurations

        Returns:
            Dictionary mapping ticker symbols to calculated quantities
        """
        return cast(dict[str, float], self.standard_allocator.calculate_auto_allocation_quantities(config, tickers))

    def calculate_backtest_optimized_quantities(
        self,
        config: StockulaConfig,
        tickers: list[TickerConfig],
    ) -> dict[str, float]:
        """Calculate backtest optimized quantities.

        Args:
            config: Stockula configuration
            tickers: List of ticker configurations

        Returns:
            Dictionary mapping ticker symbols to calculated quantities
        """
        return self.backtest_allocator.calculate_quantities(config, tickers)

    def get_calculation_prices(
        self,
        config: StockulaConfig,
        symbols: list[str],
        use_start_date: bool = True,
    ) -> dict[str, float]:
        """Get prices for calculation based on configuration.

        This is a convenience method that delegates to the base allocator's
        price calculation logic.

        Args:
            config: Stockula configuration
            symbols: List of ticker symbols
            use_start_date: Whether to use start_date (True) or end_date (False) for pricing

        Returns:
            Dictionary mapping symbols to prices
        """
        # Use any allocator to get prices (they all have the same logic)
        return cast(
            dict[str, float],
            self.standard_allocator._get_calculation_prices(config, symbols, use_start_date),
        )

    def validate_allocation_config(self, config: StockulaConfig) -> None:
        """Validate allocation configuration.

        Args:
            config: Stockula configuration to validate

        Raises:
            ValueError: If configuration is invalid
        """
        allocation_method = config.portfolio.allocation_method

        if allocation_method == "backtest_optimized":
            if not config.backtest_optimization:
                raise ValueError(
                    "backtest_optimization configuration is required for backtest_optimized allocation method"
                )
        elif allocation_method == "auto":
            if not config.portfolio.category_ratios:
                raise ValueError("category_ratios must be specified for auto allocation method")
            if config.portfolio.capital_utilization_target <= 0:
                raise ValueError("capital_utilization_target must be positive for auto allocation method")
        elif allocation_method == "dynamic":
            # Check that all tickers have either allocation_pct or allocation_amount
            for ticker in config.portfolio.tickers:
                if ticker.allocation_pct is None and ticker.allocation_amount is None:
                    raise ValueError(
                        f"Ticker {ticker.symbol} must have either allocation_pct or "
                        "allocation_amount for dynamic allocation"
                    )
