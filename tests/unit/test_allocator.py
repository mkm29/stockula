"""Tests for the Allocator class."""

from unittest.mock import Mock

import pandas as pd
import pytest

from stockula.allocation import Allocator
from stockula.config import DataConfig, PortfolioConfig, StockulaConfig, TickerConfig


@pytest.fixture
def mock_logging_manager():
    """Create a mock logging manager."""
    logger = Mock()
    logger.debug = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    return logger


class TestAllocator:
    """Test the Allocator class."""

    def test_allocator_creation(self, mock_data_fetcher, mock_logging_manager):
        """Test creating an allocator instance."""
        allocator = Allocator(mock_data_fetcher, logging_manager=mock_logging_manager)
        assert allocator.fetcher == mock_data_fetcher
        assert allocator.logger == mock_logging_manager

    def test_calculate_dynamic_quantities_no_fetcher(self, mock_logging_manager):
        """Test dynamic quantities without data fetcher raises error."""
        allocator = Allocator(None, logging_manager=mock_logging_manager)
        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test", initial_capital=100000, tickers=[TickerConfig(symbol="AAPL", allocation_pct=50)]
            )
        )

        with pytest.raises(ValueError, match="Data fetcher not configured"):
            allocator.calculate_dynamic_quantities(config, config.portfolio.tickers)

    def test_calculate_dynamic_quantities_with_percentage(self, mock_data_fetcher, mock_logging_manager):
        """Test dynamic quantities with percentage allocation."""
        allocator = Allocator(mock_data_fetcher, logging_manager=mock_logging_manager)
        mock_data_fetcher.get_current_prices = Mock(return_value={"AAPL": 150.0, "GOOGL": 100.0})

        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test",
                initial_capital=100000,
                allow_fractional_shares=True,
                tickers=[
                    TickerConfig(symbol="AAPL", allocation_pct=60),  # $60,000
                    TickerConfig(symbol="GOOGL", allocation_pct=40),  # $40,000
                ],
            )
        )

        quantities = allocator.calculate_dynamic_quantities(config, config.portfolio.tickers)
        assert quantities["AAPL"] == pytest.approx(400.0, rel=0.01)  # 60000 / 150
        assert quantities["GOOGL"] == pytest.approx(400.0, rel=0.01)  # 40000 / 100

    def test_calculate_dynamic_quantities_with_amount(self, mock_data_fetcher, mock_logging_manager):
        """Test dynamic quantities with fixed amount allocation."""
        allocator = Allocator(mock_data_fetcher, logging_manager=mock_logging_manager)
        mock_data_fetcher.get_current_prices = Mock(return_value={"AAPL": 150.0})

        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test",
                initial_capital=100000,
                allow_fractional_shares=True,
                tickers=[TickerConfig(symbol="AAPL", allocation_amount=30000)],
            )
        )

        quantities = allocator.calculate_dynamic_quantities(config, config.portfolio.tickers)
        assert quantities["AAPL"] == pytest.approx(200.0, rel=0.01)  # 30000 / 150

    def test_calculate_dynamic_quantities_no_fractional(self, mock_data_fetcher, mock_logging_manager):
        """Test dynamic quantities without fractional shares."""
        allocator = Allocator(mock_data_fetcher, logging_manager=mock_logging_manager)
        mock_data_fetcher.get_current_prices = Mock(return_value={"AAPL": 150.0})

        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test",
                initial_capital=100000,
                allow_fractional_shares=False,
                tickers=[TickerConfig(symbol="AAPL", allocation_amount=200)],  # Small amount
            )
        )

        quantities = allocator.calculate_dynamic_quantities(config, config.portfolio.tickers)
        assert quantities["AAPL"] == 1  # Should round down but minimum 1

    def test_calculate_dynamic_quantities_missing_price(self, mock_data_fetcher, mock_logging_manager):
        """Test dynamic quantities with missing price data."""
        allocator = Allocator(mock_data_fetcher, logging_manager=mock_logging_manager)
        mock_data_fetcher.get_current_prices = Mock(return_value={})  # No prices

        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test", initial_capital=100000, tickers=[TickerConfig(symbol="AAPL", allocation_pct=50)]
            )
        )

        with pytest.raises(ValueError, match="Could not fetch price for AAPL"):
            allocator.calculate_dynamic_quantities(config, config.portfolio.tickers)

    def test_calculate_dynamic_quantities_no_allocation(self, mock_data_fetcher, mock_logging_manager):
        """Test dynamic quantities without allocation specified."""
        allocator = Allocator(mock_data_fetcher, logging_manager=mock_logging_manager)
        mock_data_fetcher.get_current_prices = Mock(return_value={"AAPL": 150.0})

        # Create a ticker config with quantity, then override the allocation fields
        ticker_config = TickerConfig(symbol="AAPL", quantity=10)
        ticker_config.allocation_pct = None
        ticker_config.allocation_amount = None

        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test",
                initial_capital=100000,
                tickers=[ticker_config],
            )
        )

        with pytest.raises(ValueError, match="No allocation specified for AAPL"):
            allocator.calculate_dynamic_quantities(config, config.portfolio.tickers)

    def test_calculate_auto_allocation_quantities_basic(self, mock_data_fetcher, mock_logging_manager):
        """Test auto allocation with basic category ratios."""
        allocator = Allocator(mock_data_fetcher, logging_manager=mock_logging_manager)
        mock_data_fetcher.get_current_prices = Mock(return_value={"AAPL": 150.0, "GOOGL": 100.0, "SPY": 400.0})

        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test",
                initial_capital=100000,
                auto_allocate=True,
                category_ratios={"MOMENTUM": 0.5, "INDEX": 0.5},
                capital_utilization_target=0.95,
                allow_fractional_shares=True,
                tickers=[
                    TickerConfig(symbol="AAPL", category="MOMENTUM"),
                    TickerConfig(symbol="GOOGL", category="MOMENTUM"),
                    TickerConfig(symbol="SPY", category="INDEX"),
                ],
            )
        )

        quantities = allocator.calculate_auto_allocation_quantities(config, config.portfolio.tickers)

        # Should allocate approximately 50% to MOMENTUM (split between AAPL and GOOGL)
        # and 50% to INDEX (SPY)
        assert quantities["AAPL"] > 0
        assert quantities["GOOGL"] > 0
        assert quantities["SPY"] > 0

        # Total allocation should be close to 95% of capital
        total_cost = quantities["AAPL"] * 150.0 + quantities["GOOGL"] * 100.0 + quantities["SPY"] * 400.0
        assert total_cost == pytest.approx(95000, rel=0.01)

    def test_calculate_auto_allocation_no_category(self, mock_data_fetcher, mock_logging_manager):
        """Test auto allocation without category specified."""
        allocator = Allocator(mock_data_fetcher, logging_manager=mock_logging_manager)
        mock_data_fetcher.get_current_prices = Mock(return_value={"AAPL": 150.0})

        # Create a ticker config with quantity, then remove category
        ticker_config = TickerConfig(symbol="AAPL", quantity=10)
        ticker_config.category = None

        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test",
                initial_capital=100000,
                auto_allocate=True,
                category_ratios={"MOMENTUM": 1.0},
                tickers=[ticker_config],
            )
        )

        with pytest.raises(ValueError, match="must have category specified for auto-allocation"):
            allocator.calculate_auto_allocation_quantities(config, config.portfolio.tickers)

    def test_calculate_auto_allocation_integer_shares(self, mock_data_fetcher, mock_logging_manager):
        """Test auto allocation with integer shares only."""
        allocator = Allocator(mock_data_fetcher, logging_manager=mock_logging_manager)
        mock_data_fetcher.get_current_prices = Mock(
            return_value={
                "AAPL": 150.0,
                "GOOGL": 100.0,
            }
        )

        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test",
                initial_capital=10000,  # Smaller capital
                auto_allocate=True,
                category_ratios={"MOMENTUM": 1.0},
                capital_utilization_target=0.95,
                allow_fractional_shares=False,
                tickers=[
                    TickerConfig(symbol="AAPL", category="MOMENTUM"),
                    TickerConfig(symbol="GOOGL", category="MOMENTUM"),
                ],
            )
        )

        quantities = allocator.calculate_auto_allocation_quantities(config, config.portfolio.tickers)

        # Both should have integer quantities
        assert quantities["AAPL"] == int(quantities["AAPL"])
        assert quantities["GOOGL"] == int(quantities["GOOGL"])
        assert quantities["AAPL"] > 0
        assert quantities["GOOGL"] > 0

    def test_calculate_auto_allocation_zero_ratio_category(self, mock_data_fetcher, mock_logging_manager):
        """Test auto allocation with zero ratio category."""
        allocator = Allocator(mock_data_fetcher, logging_manager=mock_logging_manager)
        mock_data_fetcher.get_current_prices = Mock(return_value={"AAPL": 150.0, "GOOGL": 100.0})

        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test",
                initial_capital=100000,
                auto_allocate=True,
                category_ratios={"MOMENTUM": 1.0, "INDEX": 0.0},  # INDEX has 0 allocation
                tickers=[
                    TickerConfig(symbol="AAPL", category="MOMENTUM"),
                    TickerConfig(symbol="GOOGL", category="INDEX"),  # Should get 0 shares
                ],
            )
        )

        quantities = allocator.calculate_auto_allocation_quantities(config, config.portfolio.tickers)
        assert quantities["AAPL"] > 0
        assert quantities["GOOGL"] == 0  # Should have 0 allocation

    def test_get_calculation_prices_with_start_date(self, mock_data_fetcher, mock_logging_manager):
        """Test getting calculation prices with start date."""
        allocator = Allocator(mock_data_fetcher, logging_manager=mock_logging_manager)

        # Mock successful historical data retrieval
        mock_data_fetcher.get_stock_data = Mock(return_value=pd.DataFrame({"Close": [145.0]}))

        config = StockulaConfig(
            portfolio=PortfolioConfig(name="Test", initial_capital=100000),
            data=DataConfig(start_date="2023-01-01"),
        )

        prices = allocator._get_calculation_prices(config, ["AAPL"])
        assert prices["AAPL"] == 145.0

    def test_get_calculation_prices_fallback_to_current(self, mock_data_fetcher, mock_logging_manager):
        """Test falling back to current prices when historical data unavailable."""
        allocator = Allocator(mock_data_fetcher, logging_manager=mock_logging_manager)

        # Mock empty historical data
        mock_data_fetcher.get_stock_data = Mock(return_value=pd.DataFrame())
        mock_data_fetcher.get_current_prices = Mock(return_value={"AAPL": 150.0})

        config = StockulaConfig(
            portfolio=PortfolioConfig(name="Test", initial_capital=100000),
            data=DataConfig(start_date="2023-01-01"),
        )

        prices = allocator._get_calculation_prices(config, ["AAPL"])
        assert prices["AAPL"] == 150.0

    def test_get_calculation_prices_no_start_date(self, mock_data_fetcher, mock_logging_manager):
        """Test getting current prices when no start date."""
        allocator = Allocator(mock_data_fetcher, logging_manager=mock_logging_manager)
        mock_data_fetcher.get_current_prices = Mock(return_value={"AAPL": 150.0})

        config = StockulaConfig(
            portfolio=PortfolioConfig(name="Test", initial_capital=100000),
        )

        prices = allocator._get_calculation_prices(config, ["AAPL"])
        assert prices["AAPL"] == 150.0
