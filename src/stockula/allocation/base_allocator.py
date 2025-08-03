"""Base allocator class for asset allocation strategies."""

from abc import ABC, abstractmethod
from datetime import date
from typing import TYPE_CHECKING

from ..config import StockulaConfig, TickerConfig
from ..interfaces import ILoggingManager

if TYPE_CHECKING:
    from ..data.fetcher import DataFetcher


def date_to_string(date_value: str | date | None) -> str | None:
    """Convert date or string to string format."""
    if date_value is None:
        return None
    if isinstance(date_value, str):
        return date_value
    return date_value.strftime("%Y-%m-%d")


class BaseAllocator(ABC):
    """Abstract base class for all allocator implementations.

    This class provides common functionality for asset allocation strategies,
    including data fetching, logging, and utility methods.
    """

    def __init__(
        self,
        fetcher: "DataFetcher",
        logging_manager: ILoggingManager,
    ):
        """Initialize base allocator with common dependencies.

        Args:
            fetcher: Data fetcher instance for price lookups
            logging_manager: Logging manager for structured logging
        """
        self.fetcher = fetcher
        self.logger = logging_manager

    def _validate_fetcher(self) -> None:
        """Validate that data fetcher is configured.

        Raises:
            ValueError: If data fetcher is not configured
        """
        if self.fetcher is None:
            raise ValueError("Data fetcher not configured")

    def _get_calculation_prices(
        self, config: StockulaConfig, symbols: list[str], use_start_date: bool = True
    ) -> dict[str, float]:
        """Get prices for calculation based on configuration.

        Args:
            config: Stockula configuration
            symbols: List of ticker symbols
            use_start_date: Whether to use start_date (True) or end_date (False) for pricing

        Returns:
            Dictionary mapping symbols to prices
        """
        self._validate_fetcher()

        # Determine which date to use for price calculation
        if config.data:
            target_date = None
            if use_start_date and config.data.start_date:
                target_date = date_to_string(config.data.start_date)
                self.logger.debug(
                    f"Calculating quantities using start date prices ({target_date}) for accurate portfolio value..."
                )
            elif not use_start_date and config.data.end_date:
                target_date = date_to_string(config.data.end_date)
                self.logger.debug(f"Calculating quantities using end date prices ({target_date})...")

            if target_date:
                # Get historical prices for that date
                prices = {}
                for symbol in symbols:
                    try:
                        data = self.fetcher.get_stock_data(
                            symbol,
                            start=target_date,
                            end=target_date,
                            interval="1d",
                        )
                        if not data.empty and "Close" in data.columns:
                            prices[symbol] = data["Close"].iloc[-1]
                        else:
                            # Try extending the date range
                            from datetime import timedelta

                            import pandas as pd

                            target_dt = pd.to_datetime(target_date)
                            extended_end = (target_dt + timedelta(days=7)).strftime("%Y-%m-%d")

                            data = self.fetcher.get_stock_data(
                                symbol,
                                start=target_date,
                                end=extended_end,
                                interval="1d",
                            )
                            if not data.empty and "Close" in data.columns:
                                prices[symbol] = data["Close"].iloc[0]  # Use first available price
                            else:
                                # Fallback to current price if historical data unavailable
                                current_prices = self.fetcher.get_current_prices([symbol])
                                if symbol in current_prices:
                                    prices[symbol] = current_prices[symbol]
                                    self.logger.warning(
                                        f"Using current price for {symbol} (no historical data available)"
                                    )
                    except Exception as e:
                        self.logger.error(f"Error fetching price for {symbol}: {e}")
                        # Fallback to current prices
                        current_prices = self.fetcher.get_current_prices([symbol])
                        if symbol in current_prices:
                            prices[symbol] = current_prices[symbol]

                return prices

        # Default to current prices
        return self.fetcher.get_current_prices(symbols)

    def _calculate_quantity_for_allocation(
        self,
        allocation_amount: float,
        price: float,
        allow_fractional: bool,
    ) -> float:
        """Calculate quantity based on allocation amount and price.

        Args:
            allocation_amount: Dollar amount to allocate
            price: Price per share
            allow_fractional: Whether to allow fractional shares

        Returns:
            Calculated quantity
        """
        raw_quantity = allocation_amount / price

        if allow_fractional:
            return raw_quantity
        else:
            # Round down to nearest integer (conservative approach)
            return max(1, int(raw_quantity))

    @abstractmethod
    def calculate_quantities(
        self,
        config: StockulaConfig,
        tickers: list[TickerConfig],
        **kwargs,
    ) -> dict[str, float]:
        """Calculate quantities for each ticker based on the allocation strategy.

        This is the main method that subclasses must implement to define
        their specific allocation logic.

        Args:
            config: Stockula configuration
            tickers: List of ticker configurations
            **kwargs: Additional strategy-specific parameters

        Returns:
            Dictionary mapping ticker symbols to calculated quantities
        """
        pass
