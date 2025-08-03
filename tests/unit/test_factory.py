"""Comprehensive tests for DomainFactory to improve coverage."""

from datetime import date
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from stockula.allocation import Allocator
from stockula.config import DataConfig, PortfolioConfig, StockulaConfig, TickerConfig
from stockula.domain import DomainFactory, Portfolio, TickerRegistry


@pytest.fixture
def mock_logging_manager():
    """Create a mock logging manager."""
    logger = Mock()
    logger.debug = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    return logger


@pytest.fixture
def mock_allocator(mock_data_fetcher, mock_logging_manager):
    """Create a mock allocator with mocked logging manager."""
    allocator = Allocator(mock_data_fetcher, logging_manager=mock_logging_manager)
    return allocator


class TestDomainFactoryDateConversion:
    """Test date_to_string utility function."""

    def test_date_to_string_with_date_object(self):
        """Test converting date object to string."""
        from stockula.domain.factory import date_to_string

        test_date = date(2023, 1, 1)
        result = date_to_string(test_date)
        assert result == "2023-01-01"

    def test_date_to_string_with_string(self):
        """Test converting string to string (passthrough)."""
        from stockula.domain.factory import date_to_string

        test_date = "2023-01-01"
        result = date_to_string(test_date)
        assert result == "2023-01-01"

    def test_date_to_string_with_none(self):
        """Test converting None returns None."""
        from stockula.domain.factory import date_to_string

        result = date_to_string(None)
        assert result is None


class TestDomainFactoryAssetCreation:
    """Test asset creation edge cases."""

    def test_create_asset_with_no_quantity(self):
        """Test creating asset with no quantity raises error."""
        factory = DomainFactory()
        # Create a ticker config that passes validation but has None quantity
        ticker_config = TickerConfig(symbol="AAPL", sector="Technology", category="MOMENTUM")
        ticker_config.quantity = None  # Override after creation

        with pytest.raises(ValueError, match="No quantity specified for"):
            factory._create_asset(ticker_config)

    def test_create_asset_with_calculated_quantity(self):
        """Test creating asset with calculated quantity overrides config quantity."""
        factory = DomainFactory()
        ticker_config = TickerConfig(symbol="AAPL", sector="Technology", quantity=10.0)

        # Create asset with calculated quantity that overrides config
        asset = factory._create_asset(ticker_config, calculated_quantity=20.0)
        assert asset.quantity == 20.0

    def test_create_asset_with_invalid_category(self):
        """Test creating asset with invalid category string."""
        factory = DomainFactory()
        ticker_config = TickerConfig(
            symbol="AAPL",
            sector="Technology",
            quantity=10.0,
            category="INVALID_CATEGORY",  # This should not match any enum
        )

        # Should create asset with None category when invalid
        asset = factory._create_asset(ticker_config)
        assert asset.category is None


class TestDomainFactoryDynamicQuantities:
    """Test dynamic quantity calculation."""

    def test_calculate_dynamic_quantities_no_allocator(self, mock_logging_manager):
        """Test dynamic quantities without allocator raises error."""
        factory = DomainFactory(fetcher=None, logging_manager=mock_logging_manager)
        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test",
                initial_capital=100000,
                dynamic_allocation=True,
                tickers=[TickerConfig(symbol="AAPL", allocation_pct=50)],
            )
        )

        with pytest.raises(ValueError, match="Allocator not configured"):
            factory.create_portfolio(config)

    def test_dynamic_allocation_with_start_date_no_data(self, mock_data_fetcher, mock_allocator, mock_logging_manager):
        """Test dynamic allocation when historical data is empty."""
        # Mock empty historical data
        mock_data_fetcher.get_stock_data = Mock(return_value=pd.DataFrame())
        mock_data_fetcher.get_current_prices = Mock(return_value={"AAPL": 150.0})
        mock_allocator.fetcher = mock_data_fetcher

        factory = DomainFactory(
            fetcher=mock_data_fetcher, allocator=mock_allocator, logging_manager=mock_logging_manager
        )

        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test",
                initial_capital=100000,
                dynamic_allocation=True,
                tickers=[TickerConfig(symbol="AAPL", allocation_pct=50)],
            ),
            data=DataConfig(start_date="2023-01-01"),
        )

        # Should fall back to current prices
        portfolio = factory.create_portfolio(config)
        assert len(portfolio.assets) == 1
        assert portfolio.assets[0].quantity == pytest.approx(333.33, rel=0.01)  # 50000 / 150

    def test_calculate_dynamic_quantities_with_date_string(self, mock_data_fetcher, mock_allocator):
        """Test dynamic quantities with start date as string."""
        # First call returns empty, second call returns data
        call_count = 0

        def mock_get_stock_data(symbol, start=None, end=None, interval="1d"):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return pd.DataFrame()  # Empty on first call
            else:
                # Return data on second call (with extended date range)
                dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
                return pd.DataFrame({"Close": [155.0, 156.0, 157.0, 158.0, 159.0]}, index=dates)

        mock_data_fetcher.get_stock_data = Mock(side_effect=mock_get_stock_data)
        mock_allocator.fetcher = mock_data_fetcher

        factory = DomainFactory(fetcher=mock_data_fetcher, allocator=mock_allocator)

        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test",
                initial_capital=100000,
                dynamic_allocation=True,
                allow_fractional_shares=True,  # Enable fractional shares for this test
                tickers=[TickerConfig(symbol="AAPL", allocation_pct=50)],
            ),
            data=DataConfig(start_date="2023-01-01"),  # String date
        )

        quantities = factory.allocator.calculate_dynamic_quantities(config, config.portfolio.tickers)
        assert "AAPL" in quantities
        assert quantities["AAPL"] == pytest.approx(322.58, rel=0.01)  # 50000 / 155

    def test_calculate_dynamic_quantities_no_allocation(self, mock_data_fetcher):
        """Test dynamic quantities with no allocation specified."""
        factory = DomainFactory(fetcher=mock_data_fetcher)
        mock_data_fetcher.get_current_prices = Mock(return_value={"AAPL": 150.0})

        # Create a ticker config with quantity, then override the allocation fields
        ticker_config = TickerConfig(symbol="AAPL", quantity=10)
        ticker_config.allocation_pct = None
        ticker_config.allocation_amount = None

        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test",
                initial_capital=100000,
                dynamic_allocation=True,
                tickers=[ticker_config],
            )
        )

        with pytest.raises(ValueError, match="No allocation specified for AAPL"):
            factory.allocator.calculate_dynamic_quantities(config, config.portfolio.tickers)

    def test_calculate_dynamic_quantities_fractional_shares_disabled(self, mock_data_fetcher):
        """Test dynamic quantities with fractional shares disabled."""
        factory = DomainFactory(fetcher=mock_data_fetcher)
        mock_data_fetcher.get_current_prices = Mock(return_value={"AAPL": 150.0})

        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test",
                initial_capital=100000,
                allow_fractional_shares=False,
                dynamic_allocation=True,
                tickers=[TickerConfig(symbol="AAPL", allocation_amount=200)],  # Small amount
            )
        )

        quantities = factory.allocator.calculate_dynamic_quantities(config, config.portfolio.tickers)
        assert quantities["AAPL"] == 1  # Should round down but minimum 1

    def test_calculate_dynamic_quantities_exception_handling(self, mock_data_fetcher, mock_allocator):
        """Test dynamic quantities with exception during data fetch."""
        # Mock exception on first call, then return current prices
        mock_data_fetcher.get_stock_data = Mock(side_effect=Exception("API Error"))
        mock_data_fetcher.get_current_prices = Mock(return_value={"AAPL": 150.0})
        mock_allocator.fetcher = mock_data_fetcher

        factory = DomainFactory(fetcher=mock_data_fetcher, allocator=mock_allocator)

        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test",
                initial_capital=100000,
                dynamic_allocation=True,
                tickers=[TickerConfig(symbol="AAPL", allocation_pct=50)],
            ),
            data=DataConfig(start_date="2023-01-01"),
        )

        # Should handle exception and fall back to current prices
        quantities = factory.allocator.calculate_dynamic_quantities(config, config.portfolio.tickers)
        assert "AAPL" in quantities

    def test_calculate_dynamic_quantities_missing_price(self, mock_data_fetcher):
        """Test dynamic quantities when price is not available."""
        factory = DomainFactory(fetcher=mock_data_fetcher)
        mock_data_fetcher.get_current_prices = Mock(return_value={})  # No prices

        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test",
                initial_capital=100000,
                dynamic_allocation=True,
                tickers=[TickerConfig(symbol="AAPL", allocation_pct=50)],
            )
        )

        with pytest.raises(ValueError, match="Could not fetch price for AAPL"):
            factory.allocator.calculate_dynamic_quantities(config, config.portfolio.tickers)


class TestDomainFactoryAutoAllocation:
    """Test auto allocation functionality."""

    def test_auto_allocation_no_fetcher(self, mock_logging_manager):
        """Test auto allocation without data fetcher."""
        factory = DomainFactory(fetcher=None, logging_manager=mock_logging_manager)
        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test",
                initial_capital=100000,
                auto_allocate=True,
                category_ratios={"MOMENTUM": 1.0},
                tickers=[TickerConfig(symbol="AAPL", category="MOMENTUM")],
            )
        )

        with pytest.raises(ValueError, match="Allocator not configured"):
            factory.create_portfolio(config)

    def test_auto_allocation_no_category_ratios(self, mock_data_fetcher):
        """Test auto allocation without category ratios."""
        DomainFactory(fetcher=mock_data_fetcher)

        # Test that the validation error happens during config creation
        with pytest.raises(ValueError, match="auto_allocate=True requires category_ratios"):
            StockulaConfig(
                portfolio=PortfolioConfig(
                    name="Test",
                    initial_capital=100000,
                    auto_allocate=True,
                    tickers=[TickerConfig(symbol="AAPL", category="MOMENTUM")],
                )
            )

    def test_auto_allocation_missing_category(self, mock_data_fetcher):
        """Test auto allocation with ticker missing category."""
        factory = DomainFactory(fetcher=mock_data_fetcher)
        mock_data_fetcher.get_current_prices = Mock(return_value={"AAPL": 150.0})

        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test",
                initial_capital=100000,
                auto_allocate=True,
                category_ratios={"MOMENTUM": 1.0},
                tickers=[TickerConfig(symbol="AAPL", quantity=10)],  # No category
            )
        )

        with pytest.raises(ValueError, match="must have category specified for auto-allocation"):
            factory.allocator.calculate_auto_allocation_quantities(config, config.portfolio.tickers)

    def test_auto_allocation_missing_price(self, mock_data_fetcher):
        """Test auto allocation when price is not available."""
        factory = DomainFactory(fetcher=mock_data_fetcher)
        mock_data_fetcher.get_current_prices = Mock(return_value={})  # No prices

        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test",
                initial_capital=100000,
                auto_allocate=True,
                category_ratios={"MOMENTUM": 1.0},
                tickers=[TickerConfig(symbol="AAPL", category="MOMENTUM")],
            )
        )

        with pytest.raises(ValueError, match="Could not fetch price for AAPL"):
            factory.allocator.calculate_auto_allocation_quantities(config, config.portfolio.tickers)

    def test_auto_allocation_no_tickers_in_category(self, mock_data_fetcher, mock_allocator):
        """Test auto allocation with no tickers in specified category."""
        mock_data_fetcher.get_current_prices = Mock(return_value={"AAPL": 150.0})
        mock_allocator.fetcher = mock_data_fetcher

        factory = DomainFactory(fetcher=mock_data_fetcher, allocator=mock_allocator)

        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test",
                initial_capital=100000,
                auto_allocate=True,
                category_ratios={"GROWTH": 0.5, "MOMENTUM": 0.5},  # Two categories
                capital_utilization_target=0.95,
                tickers=[TickerConfig(symbol="AAPL", category="MOMENTUM")],  # Only MOMENTUM
            )
        )

        # Should handle missing category gracefully
        quantities = factory.allocator.calculate_auto_allocation_quantities(config, config.portfolio.tickers)
        assert "AAPL" in quantities

    def test_auto_allocation_zero_ratio_category(self, mock_data_fetcher, mock_allocator):
        """Test auto allocation with zero ratio category."""
        mock_data_fetcher.get_current_prices = Mock(return_value={"AAPL": 150.0, "GOOGL": 120.0})
        mock_allocator.fetcher = mock_data_fetcher

        factory = DomainFactory(fetcher=mock_data_fetcher, allocator=mock_allocator)

        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test",
                initial_capital=100000,
                auto_allocate=True,
                category_ratios={"GROWTH": 0.0, "MOMENTUM": 1.0},  # GROWTH has 0%
                capital_utilization_target=0.95,
                tickers=[
                    TickerConfig(symbol="AAPL", category="MOMENTUM"),
                    TickerConfig(symbol="GOOGL", category="GROWTH"),
                ],
            )
        )

        quantities = factory.allocator.calculate_auto_allocation_quantities(config, config.portfolio.tickers)
        assert quantities["AAPL"] > 0
        assert quantities["GOOGL"] == 0  # Should have 0 allocation

    def test_auto_allocation_integer_shares_expensive_stock(self, mock_data_fetcher, mock_allocator):
        """Test auto allocation with expensive stocks and integer shares."""
        mock_data_fetcher.get_current_prices = Mock(
            return_value={
                "BRK.A": 500000.0,  # Very expensive stock
                "AAPL": 150.0,
            }
        )
        mock_allocator.fetcher = mock_data_fetcher

        factory = DomainFactory(fetcher=mock_data_fetcher, allocator=mock_allocator)

        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test",
                initial_capital=1000000,  # 1M capital
                auto_allocate=True,
                allow_fractional_shares=False,
                category_ratios={"VALUE": 1.0},
                capital_utilization_target=0.95,
                tickers=[TickerConfig(symbol="BRK.A", category="VALUE"), TickerConfig(symbol="AAPL", category="VALUE")],
            )
        )

        quantities = factory.allocator.calculate_auto_allocation_quantities(config, config.portfolio.tickers)
        # Should allocate at least 1 share of expensive stock if affordable
        assert quantities["BRK.A"] >= 1
        assert quantities["AAPL"] > 0

    def test_auto_allocation_fractional_shares(self, mock_data_fetcher, mock_allocator):
        """Test auto allocation with fractional shares enabled."""
        mock_data_fetcher.get_current_prices = Mock(return_value={"AAPL": 150.0, "GOOGL": 120.0})
        mock_allocator.fetcher = mock_data_fetcher

        factory = DomainFactory(fetcher=mock_data_fetcher, allocator=mock_allocator)

        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test",
                initial_capital=100000,
                auto_allocate=True,
                allow_fractional_shares=True,
                category_ratios={"TECH": 1.0},
                capital_utilization_target=0.95,
                tickers=[TickerConfig(symbol="AAPL", category="TECH"), TickerConfig(symbol="GOOGL", category="TECH")],
            )
        )

        quantities = factory.allocator.calculate_auto_allocation_quantities(config, config.portfolio.tickers)
        # With fractional shares, should have precise allocation
        assert quantities["AAPL"] > 0
        assert quantities["GOOGL"] > 0
        # Verify fractional shares
        assert quantities["AAPL"] % 1 != 0 or quantities["GOOGL"] % 1 != 0

    def test_auto_allocation_redistribution_small_positions(self, mock_data_fetcher, mock_allocator):
        """Test redistribution to smallest positions."""
        # Create a scenario where we have leftover capital
        mock_data_fetcher.get_current_prices = Mock(return_value={"AAPL": 150.0, "GOOGL": 120.0, "MSFT": 300.0})
        mock_allocator.fetcher = mock_data_fetcher

        factory = DomainFactory(fetcher=mock_data_fetcher, allocator=mock_allocator)

        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test",
                initial_capital=10000,  # Small capital to ensure leftovers
                auto_allocate=True,
                allow_fractional_shares=False,
                category_ratios={"TECH": 1.0},
                capital_utilization_target=0.95,
                tickers=[
                    TickerConfig(symbol="AAPL", category="TECH"),
                    TickerConfig(symbol="GOOGL", category="TECH"),
                    TickerConfig(symbol="MSFT", category="TECH"),
                ],
            )
        )

        quantities = factory.allocator.calculate_auto_allocation_quantities(config, config.portfolio.tickers)
        # Should have allocated to all positions
        assert all(q > 0 for q in quantities.values())

    def test_auto_allocation_with_historical_prices(self, mock_data_fetcher, mock_allocator):
        """Test auto allocation using historical start date prices."""
        # Mock historical data fetch
        historical_data = pd.DataFrame(
            {
                "Close": [145.0]  # Historical price different from current
            },
            index=pd.date_range(start="2023-01-01", periods=1),
        )

        mock_data_fetcher.get_stock_data = Mock(return_value=historical_data)
        mock_data_fetcher.get_current_prices = Mock(return_value={"AAPL": 150.0})  # Current price
        mock_allocator.fetcher = mock_data_fetcher

        factory = DomainFactory(fetcher=mock_data_fetcher, allocator=mock_allocator)

        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test",
                initial_capital=100000,
                auto_allocate=True,
                category_ratios={"TECH": 1.0},
                capital_utilization_target=0.95,
                tickers=[TickerConfig(symbol="AAPL", category="TECH")],
            ),
            data=DataConfig(start_date="2023-01-01"),
        )

        quantities = factory.allocator.calculate_auto_allocation_quantities(config, config.portfolio.tickers)
        # Should use historical price (145) not current price (150)
        # Auto-allocation includes redistribution phase which can add extra shares
        # Just verify it used historical price by checking allocation is greater than
        # what we'd get with current price
        shares_with_current_price = int((100000 * 0.95) / 150.0)  # 633
        shares_with_historical_price = int((100000 * 0.95) / 145.0)  # 655

        # Should have at least the base allocation from historical price
        assert quantities["AAPL"] >= shares_with_historical_price
        # And should be more than what we'd get with current price
        assert quantities["AAPL"] > shares_with_current_price


class TestDomainFactoryPortfolioCreation:
    """Test portfolio creation scenarios."""

    def test_create_portfolio_with_dynamic_allocation_zero_quantity(self, mock_data_fetcher, mock_logging_manager):
        """Test portfolio creation raises error for zero quantity assets."""
        factory = DomainFactory(fetcher=mock_data_fetcher, logging_manager=mock_logging_manager)

        # Mock zero allocation for one ticker
        def mock_calculate_dynamic(config, tickers):
            return {"AAPL": 10.0, "GOOGL": 0.0}  # GOOGL has 0 shares

        factory.allocator.calculate_dynamic_quantities = Mock(side_effect=mock_calculate_dynamic)

        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test",
                initial_capital=100000,
                dynamic_allocation=True,
                tickers=[
                    TickerConfig(symbol="AAPL", allocation_pct=50),
                    TickerConfig(symbol="GOOGL", allocation_pct=50),
                ],
            )
        )

        # With the refactoring, zero quantity assets are now skipped during portfolio creation
        # So this should not raise an error
        portfolio = factory.create_portfolio(config)
        assert len(portfolio.assets) == 1  # Only AAPL with non-zero quantity
        assert portfolio.assets[0].symbol == "AAPL"

    def test_create_portfolio_static_allocation_with_validation(self, mock_data_fetcher):
        """Test portfolio creation with static allocation triggers validation."""
        factory = DomainFactory(fetcher=mock_data_fetcher)

        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test", initial_capital=100000, tickers=[TickerConfig(symbol="AAPL", quantity=10)]
            )
        )

        # Mock portfolio methods
        with (
            patch.object(Portfolio, "validate_capital_sufficiency") as mock_capital_validate,
            patch.object(Portfolio, "validate_allocation_constraints") as mock_allocation_validate,
        ):
            factory.create_portfolio(config)

            # Verify validations were called for static allocation
            mock_capital_validate.assert_called_once()
            mock_allocation_validate.assert_called_once()


class TestDomainFactoryIntegration:
    """Integration tests for factory."""

    def test_factory_with_custom_ticker_registry(self):
        """Test factory with custom ticker registry."""
        custom_registry = TickerRegistry()
        factory = DomainFactory(ticker_registry=custom_registry)

        assert factory.ticker_registry is custom_registry

    def test_get_all_tickers_empty(self):
        """Test get_all_tickers with empty registry."""
        factory = DomainFactory()
        factory.ticker_registry._clear()  # Clear any existing tickers

        tickers = factory.get_all_tickers()
        assert tickers == []
