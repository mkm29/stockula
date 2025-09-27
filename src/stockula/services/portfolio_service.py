"""
Portfolio Service following SRP.
Single Responsibility: Portfolio management and operations.
"""

import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, cast

import pandas as pd
from rich.console import Console

from ..config import StockulaConfig
from ..container import Container
from ..domain import Category, Portfolio
from ..utils import get_console

logger = logging.getLogger(__name__)


class PortfolioService:
    """Manages portfolio operations - Single Responsibility: Portfolio Management."""

    def __init__(
        self,
        config: StockulaConfig,
        container: Container,
        console: Optional[Console] = None,
    ):
        """Initialize portfolio service.

        Args:
            config: Configuration object
            container: Dependency injection container
            console: Rich console for output (optional)
        """
        self.config = config
        self.container = container
        self.console = get_console(console)
        self.log_manager = container.logging_manager()

    def create_portfolio(self) -> Portfolio:
        """Create portfolio from configuration.

        Returns:
            Portfolio instance
        """
        factory = self.container.domain_factory()
        portfolio = factory.create_portfolio(self.config)
        return cast(Portfolio, portfolio)

    def get_portfolio_value_at_date(
        self, portfolio: Portfolio, start_date_str: Optional[str]
    ) -> Tuple[float, Dict[str, float]]:
        """Get portfolio value at a specific date.

        Args:
            portfolio: Portfolio instance
            start_date_str: Date string or None

        Returns:
            Tuple of (portfolio_value, prices_dict)
        """
        fetcher = self.container.data_fetcher()
        symbols = [asset.symbol for asset in portfolio.get_all_assets()]

        if start_date_str:
            self.log_manager.debug(f"\nFetching prices at start date ({start_date_str})...")
            return self._get_historical_portfolio_value(portfolio, symbols, start_date_str, fetcher)
        else:
            self.log_manager.debug("\nFetching current prices...")
            current_prices = fetcher.get_current_prices(symbols, show_progress=True)
            return portfolio.get_portfolio_value(current_prices), current_prices

    def categorize_assets(self, portfolio: Portfolio) -> Tuple[List[Any], List[Any], Set[Category]]:
        """Categorize assets into tradeable and hold-only.

        Args:
            portfolio: Portfolio instance

        Returns:
            Tuple of (tradeable_assets, hold_only_assets, hold_only_categories)
        """
        # Get hold-only categories from config
        hold_only_category_names = set(self.config.backtest.hold_only_categories)
        hold_only_categories = set()

        for category_name in hold_only_category_names:
            try:
                hold_only_categories.add(Category[category_name])
            except KeyError:
                self.log_manager.warning(f"Unknown category '{category_name}' in hold_only_categories")

        tradeable_assets = []
        hold_only_assets = []

        for asset in portfolio.get_all_assets():
            if asset.category in hold_only_categories:
                hold_only_assets.append(asset)
            else:
                tradeable_assets.append(asset)

        if hold_only_assets:
            self.log_manager.info("\nHold-only assets (excluded from backtesting):")
            for asset in hold_only_assets:
                self.log_manager.info(f"  {asset.symbol} ({asset.category})")

        return tradeable_assets, hold_only_assets, hold_only_categories

    def calculate_portfolio_returns(
        self, portfolio: Portfolio, start_date_str: Optional[str]
    ) -> Tuple[float, float, float]:
        """Calculate portfolio returns since inception.

        Args:
            portfolio: Portfolio instance
            start_date_str: Start date string for analysis

        Returns:
            Tuple of (initial_portfolio_value, initial_return, initial_return_pct)
        """
        # Get portfolio value at start of backtest period
        initial_portfolio_value, _ = self.get_portfolio_value_at_date(portfolio, start_date_str)

        # Calculate returns
        initial_return = initial_portfolio_value - portfolio.initial_capital
        initial_return_pct = (initial_return / portfolio.initial_capital) * 100

        self.log_manager.info(f"Initial Capital: ${portfolio.initial_capital:,.2f}")
        self.log_manager.info(f"Return Since Inception: ${initial_return:,.2f} ({initial_return_pct:+.2f}%)")

        return initial_portfolio_value, initial_return, initial_return_pct

    def _get_historical_portfolio_value(
        self, portfolio: Portfolio, symbols: List[str], start_date_str: str, fetcher
    ) -> Tuple[float, Dict[str, float]]:
        """Get portfolio value at a historical date.

        Args:
            portfolio: Portfolio instance
            symbols: List of symbols
            start_date_str: Start date string
            fetcher: Data fetcher instance

        Returns:
            Tuple of (portfolio_value, prices_dict)
        """
        # Fetch one day of data at the start date to get opening prices
        start_prices = {}
        for symbol in symbols:
            try:
                data = fetcher.get_stock_data(symbol, start=start_date_str, end=start_date_str)
                if not data.empty:
                    start_prices[symbol] = data["Close"].iloc[0]
                else:
                    # If no data on exact date, get the next available date
                    start_dt = pd.to_datetime(start_date_str)
                    end_date = (start_dt + timedelta(days=7)).strftime("%Y-%m-%d")
                    data = fetcher.get_stock_data(symbol, start=start_date_str, end=end_date)
                    if not data.empty:
                        start_prices[symbol] = data["Close"].iloc[0]
            except Exception as e:
                self.log_manager.warning(f"Could not get start price for {symbol}: {e}")

        return portfolio.get_portfolio_value(start_prices), start_prices
