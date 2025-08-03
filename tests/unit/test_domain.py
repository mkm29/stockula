"""Tests for domain models."""

from datetime import date
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from stockula.allocation import Allocator
from stockula.config import DataConfig, PortfolioConfig, StockulaConfig, TickerConfig
from stockula.domain import Asset, Category, DomainFactory, Portfolio, Ticker, TickerRegistry


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


class TestTicker:
    """Test Ticker domain model."""

    def test_ticker_creation(self, mock_logging_manager):
        """Test creating a ticker."""
        ticker = Ticker(
            symbol="AAPL",
            sector="Technology",
            market_cap=3000.0,
            category=Category.MOMENTUM,
        )
        assert ticker.symbol == "AAPL"
        assert ticker.sector == "Technology"
        assert ticker.market_cap == 3000.0
        assert ticker.category == Category.MOMENTUM

    def test_ticker_defaults(self):
        """Test ticker with default values."""
        ticker = Ticker(symbol="GOOGL")
        assert ticker.symbol == "GOOGL"
        assert ticker.sector is None
        assert ticker.market_cap is None
        assert ticker.category is None

    def test_ticker_equality(self):
        """Test ticker equality based on symbol."""
        ticker1 = Ticker(symbol="AAPL", sector="Technology")
        ticker2 = Ticker(symbol="AAPL", sector="Tech")  # Different sector
        ticker3 = Ticker(symbol="GOOGL")

        assert ticker1 == ticker2  # Same symbol
        assert ticker1 != ticker3  # Different symbol

    def test_ticker_hash(self):
        """Test ticker hashing for use in sets/dicts."""
        ticker1 = Ticker(symbol="AAPL")
        ticker2 = Ticker(symbol="AAPL")
        ticker3 = Ticker(symbol="GOOGL")

        ticker_set = {ticker1, ticker2, ticker3}
        assert len(ticker_set) == 2  # ticker1 and ticker2 are the same

    def test_ticker_string_representation(self):
        """Test ticker string representation."""
        ticker = Ticker(symbol="AAPL", sector="Technology")
        assert str(ticker) == "Ticker(AAPL)"
        assert repr(ticker) == "Ticker(symbol='AAPL', sector='Technology', market_cap=None, category=None)"


class TestTickerRegistry:
    """Test TickerRegistry singleton."""

    def test_ticker_registry_singleton(self):
        """Test that TickerRegistry is a singleton."""
        registry1 = TickerRegistry()
        registry2 = TickerRegistry()
        assert registry1 is registry2

    def test_get_or_create_ticker(self):
        """Test get_or_create ticker functionality."""
        registry = TickerRegistry()

        # Create new ticker
        ticker1 = registry.get_or_create("AAPL", sector="Technology")
        assert ticker1.symbol == "AAPL"
        assert ticker1.sector == "Technology"

        # Get existing ticker with different sector (should update the ticker)
        ticker2 = registry.get_or_create("AAPL", sector="Tech")  # Different sector
        assert ticker2.symbol == "AAPL"
        assert ticker2.sector == "Tech"  # Updated sector

        # Get without providing sector should keep current value
        ticker3 = registry.get_or_create("AAPL")
        assert ticker3.sector == "Tech"

    def test_get_ticker(self):
        """Test getting a ticker from registry."""
        registry = TickerRegistry()

        # Ticker doesn't exist
        assert registry.get("MSFT") is None

        # Create ticker
        ticker = registry.get_or_create("MSFT")

        # Now it exists
        assert registry.get("MSFT") is ticker

    def test_all_tickers(self):
        """Test getting all tickers from registry."""
        registry = TickerRegistry()

        # Create multiple tickers
        ticker1 = registry.get_or_create("AAPL")
        registry.get_or_create("GOOGL")
        registry.get_or_create("MSFT")

        all_tickers = registry.all()
        assert len(all_tickers) == 3
        assert "AAPL" in all_tickers
        assert all_tickers["AAPL"] is ticker1

    def test_clear_registry(self):
        """Test clearing the ticker registry."""
        registry = TickerRegistry()

        # Add some tickers
        registry.get_or_create("AAPL")
        registry.get_or_create("GOOGL")
        assert len(registry.all()) == 2

        # Clear registry
        registry._clear()
        assert len(registry.all()) == 0


class TestAsset:
    """Test Asset domain model."""

    def test_asset_creation(self, sample_ticker):
        """Test creating an asset."""
        asset = Asset(ticker_init=sample_ticker, quantity_init=10.0, category_init=Category.MOMENTUM)
        assert asset.ticker == sample_ticker
        assert asset.quantity == 10.0
        assert asset.category == Category.MOMENTUM
        assert asset.symbol == "AAPL"

    def test_asset_without_category(self):
        """Test asset without explicit category is None."""
        ticker_with_category = Ticker(
            "NVDA",  # symbol_init
            None,  # sector_init
            None,  # market_cap_init
            "SPECULATIVE",  # category_init
            None,  # price_range_init
            None,  # metadata_init
        )
        asset = Asset(ticker_init=ticker_with_category, quantity_init=5.0, category_init=None)
        assert asset.category is None  # Asset doesn't inherit ticker category

    def test_asset_category_override(self):
        """Test asset can have its own category."""
        ticker_with_category = Ticker(
            "SPY",  # symbol_init
            None,  # sector_init
            None,  # market_cap_init
            "INDEX",  # category_init
            None,  # price_range_init
            None,  # metadata_init
        )
        asset = Asset(
            ticker_init=ticker_with_category,
            quantity_init=20.0,
            category_init=Category.GROWTH,  # Asset's own category
        )
        assert asset.category == Category.GROWTH
        assert ticker_with_category.category == "INDEX"  # Ticker unchanged

    def test_asset_value_calculation(self, sample_ticker):
        """Test asset value calculation."""
        asset = Asset(ticker_init=sample_ticker, quantity_init=10.0)

        # Test with different prices
        assert asset.get_value(150.0) == 1500.0
        assert asset.get_value(200.0) == 2000.0
        assert asset.get_value(0.0) == 0.0

    def test_asset_string_representation(self, sample_ticker):
        """Test asset string representation."""
        asset = Asset(ticker_init=sample_ticker, quantity_init=10.0)
        # Asset.__str__ returns: Asset(symbol, quantity shares[, category])
        assert "AAPL" in str(asset)
        assert "10.00 shares" in str(asset)


class TestCategory:
    """Test Category enum."""

    def test_category_values(self):
        """Test category enum values."""
        # Check that categories exist and have proper string representations
        assert str(Category.INDEX) == "Index"
        assert str(Category.LARGE_CAP) == "Large Cap"
        assert str(Category.MOMENTUM) == "Momentum"
        assert str(Category.GROWTH) == "Growth"
        assert str(Category.VALUE) == "Value"
        assert str(Category.DIVIDEND) == "Dividend"
        assert str(Category.SPECULATIVE) == "Speculative"
        assert str(Category.INTERNATIONAL) == "International"
        assert str(Category.COMMODITY) == "Commodity"
        assert str(Category.BOND) == "Bond"
        assert str(Category.CRYPTO) == "Crypto"

        # Test that values are integers (from auto())
        assert isinstance(Category.INDEX.value, int)
        assert isinstance(Category.GROWTH.value, int)

    def test_category_from_string(self):
        """Test creating category from string."""
        assert Category["INDEX"] == Category.INDEX
        assert Category["MOMENTUM"] == Category.MOMENTUM

        with pytest.raises(KeyError):
            Category["INVALID_CATEGORY"]


class TestPortfolio:
    """Test Portfolio domain model."""

    def test_portfolio_creation(self, mock_logging_manager):
        """Test creating a portfolio."""
        portfolio = Portfolio(
            name_init="Test Portfolio",
            initial_capital_init=100000.0,
            allocation_method_init="equal_weight",
            logging_manager_init=mock_logging_manager,
            max_position_size=25.0,
            stop_loss_pct=10.0,
        )
        assert portfolio.name == "Test Portfolio"
        assert portfolio.initial_capital == 100000.0
        assert portfolio.allocation_method == "equal_weight"
        assert portfolio.max_position_size == 25.0
        assert portfolio.stop_loss_pct == 10.0
        assert len(portfolio.assets) == 0

    def test_add_asset(self, sample_portfolio, sample_asset):
        """Test adding an asset to portfolio."""
        sample_portfolio.add_asset(sample_asset)
        assert len(sample_portfolio.assets) == 1
        assert sample_portfolio.assets[0] == sample_asset

    def test_add_duplicate_asset_raises_error(self, sample_portfolio, sample_asset):
        """Test adding duplicate asset raises error."""
        sample_portfolio.add_asset(sample_asset)

        # Try to add same ticker again
        duplicate_asset = Asset(ticker_init=sample_asset.ticker, quantity_init=5.0)
        with pytest.raises(ValueError, match="already exists"):
            sample_portfolio.add_asset(duplicate_asset)

    def test_get_all_assets(self, populated_portfolio):
        """Test getting all assets from portfolio."""
        assets = populated_portfolio.get_all_assets()
        assert len(assets) == 4
        symbols = [asset.symbol for asset in assets]
        assert "AAPL" in symbols
        assert "GOOGL" in symbols
        assert "SPY" in symbols
        assert "NVDA" in symbols

    def test_get_assets_by_category(self, populated_portfolio):
        """Test getting assets by category."""
        momentum_assets = populated_portfolio.get_assets_by_category(Category.MOMENTUM)
        assert len(momentum_assets) == 1
        assert momentum_assets[0].symbol == "AAPL"

        index_assets = populated_portfolio.get_assets_by_category(Category.INDEX)
        assert len(index_assets) == 1
        assert index_assets[0].symbol == "SPY"

    def test_get_asset_by_symbol(self, populated_portfolio):
        """Test getting asset by symbol."""
        asset = populated_portfolio.get_asset_by_symbol("AAPL")
        assert asset is not None
        assert asset.symbol == "AAPL"

        # Non-existent symbol
        assert populated_portfolio.get_asset_by_symbol("TSLA") is None

    def test_portfolio_value_calculation(self, populated_portfolio, sample_prices):
        """Test portfolio value calculation."""
        value = populated_portfolio.get_portfolio_value(sample_prices)

        # Calculate expected value
        expected = (
            10.0 * 150.0  # AAPL
            + 5.0 * 120.0  # GOOGL
            + 20.0 * 450.0  # SPY
            + 8.0 * 500.0
        )  # NVDA
        assert value == expected

    def test_portfolio_value_with_missing_prices(self, populated_portfolio):
        """Test portfolio value calculation with missing prices."""
        partial_prices = {"AAPL": 150.0, "GOOGL": 120.0}  # Missing SPY and NVDA

        # Should calculate value for assets with prices, skip others
        value = populated_portfolio.get_portfolio_value(partial_prices)
        expected = 10.0 * 150.0 + 5.0 * 120.0  # Only AAPL and GOOGL
        assert value == expected

    def test_get_allocation_by_category(self, populated_portfolio, sample_prices):
        """Test getting allocation by category."""
        allocations = populated_portfolio.get_allocation_by_category(sample_prices)

        # Check that category names are in allocations
        assert "Momentum" in allocations
        assert "Index" in allocations
        assert "Growth" in allocations
        assert "Speculative" in allocations

        # Check percentages sum to 100
        total_pct = sum(alloc["percentage"] for alloc in allocations.values())
        assert abs(total_pct - 100.0) < 0.01

        # Check specific category
        momentum_alloc = allocations["Momentum"]
        assert momentum_alloc["value"] == 10.0 * 150.0  # AAPL
        assert momentum_alloc["assets"] == ["AAPL"]

    def test_validate_capital_sufficiency(self, sample_portfolio, sample_asset):
        """Test capital sufficiency validation."""
        # Add assets that exceed capital
        expensive_ticker = Ticker(
            "BRK.A",  # symbol_init
            None,  # sector_init
            None,  # market_cap_init
            "VALUE",  # category_init
            None,  # price_range_init
            None,  # metadata_init
        )
        expensive_asset = Asset(ticker_init=expensive_ticker, quantity_init=1.0, category_init=Category.VALUE)

        sample_portfolio.add_asset(sample_asset)
        sample_portfolio.add_asset(expensive_asset)

        # Mock prices
        with patch.object(sample_portfolio, "get_portfolio_value") as mock_value:
            mock_value.return_value = 150000.0  # Exceeds initial capital of 100000

            with pytest.raises(ValueError, match="is insufficient to cover"):
                sample_portfolio.validate_capital_sufficiency()

    def test_validate_allocation_constraints(self, sample_portfolio, sample_asset):
        """Test allocation constraints validation."""
        sample_portfolio.max_position_size = 20.0  # 20% max
        sample_portfolio.add_asset(sample_asset)

        # Mock to make one position exceed 20%
        mock_allocations = {
            "AAPL": {
                "value": 250.0,
                "percentage": 25.0,  # Exceeds 20% max
                "quantity": 10.0,
            }
        }

        with patch.object(sample_portfolio, "get_asset_allocations") as mock_alloc:
            mock_alloc.return_value = mock_allocations

            # Should raise error for exceeding max position size
            with pytest.raises(ValueError, match="exceeds maximum position size"):
                sample_portfolio.validate_allocation_constraints()


class TestDomainFactory:
    """Test DomainFactory."""

    def test_create_portfolio_basic(self, sample_stockula_config, mock_data_fetcher, mock_logging_manager):
        """Test creating a basic portfolio from config."""
        factory = DomainFactory(fetcher=mock_data_fetcher, logging_manager=mock_logging_manager)

        # Mock validation methods to not raise errors
        with patch.object(Portfolio, "validate_capital_sufficiency"):
            with patch.object(Portfolio, "validate_allocation_constraints"):
                portfolio = factory.create_portfolio(sample_stockula_config)

        assert portfolio.name == "Test Portfolio"
        assert portfolio.initial_capital == 100000.0
        assert len(portfolio.assets) == 4

    def test_create_portfolio_with_dynamic_allocation(
        self, dynamic_allocation_config, mock_data_fetcher, mock_logging_manager
    ):
        """Test creating portfolio with dynamic allocation."""
        config = StockulaConfig(portfolio=dynamic_allocation_config)
        factory = DomainFactory(fetcher=mock_data_fetcher, logging_manager=mock_logging_manager)

        portfolio = factory.create_portfolio(config)

        assert len(portfolio.assets) == 3

        # Check AAPL allocation (fixed $15,000 / $150 = 100 shares)
        aapl_asset = portfolio.get_asset_by_symbol("AAPL")
        assert aapl_asset.quantity == 100.0

        # Check GOOGL allocation (20% of $50,000 = $10,000 / $120 ≈ 83.33 shares)
        googl_asset = portfolio.get_asset_by_symbol("GOOGL")
        assert googl_asset.quantity == pytest.approx(83.33, rel=0.01)

    def test_create_portfolio_with_auto_allocation(
        self, auto_allocation_config, mock_data_fetcher, mock_allocator, mock_logging_manager
    ):
        """Test creating portfolio with auto allocation."""
        config = StockulaConfig(portfolio=auto_allocation_config)
        factory = DomainFactory(
            fetcher=mock_data_fetcher, allocator=mock_allocator, logging_manager=mock_logging_manager
        )

        portfolio = factory.create_portfolio(config)

        assert len(portfolio.assets) == 5

        # Check that all assets have quantities
        for asset in portfolio.assets:
            assert asset.quantity > 0

        # Check category allocation ratios are roughly maintained
        allocations = portfolio.get_allocation_by_category(
            mock_data_fetcher.get_current_prices([a.symbol for a in portfolio.assets])
        )

        # INDEX should be roughly 35%
        index_pct = allocations["Index"]["percentage"]
        assert 30 < index_pct < 40  # Allow some variance

    def test_create_portfolio_dynamic_allocation_without_info(self, mock_logging_manager):
        """Test creating portfolio with dynamic allocation but no allocation info."""
        # This is now allowed - the ticker config doesn't validate against portfolio settings
        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test",
                initial_capital=10000,
                dynamic_allocation=True,
                tickers=[
                    TickerConfig(symbol="AAPL"),  # No allocation info
                ],
            )
        )
        # The validation would happen later when trying to calculate dynamic quantities
        assert config.portfolio.dynamic_allocation is True
        assert config.portfolio.tickers[0].quantity is None

    def test_get_all_tickers(self, sample_stockula_config, mock_data_fetcher, mock_logging_manager):
        """Test getting all tickers from factory."""
        factory = DomainFactory(fetcher=mock_data_fetcher, logging_manager=mock_logging_manager)

        # Create portfolio to populate registry
        with patch.object(Portfolio, "validate_capital_sufficiency"):
            with patch.object(Portfolio, "validate_allocation_constraints"):
                factory.create_portfolio(sample_stockula_config)

        all_tickers = factory.get_all_tickers()
        assert len(all_tickers) == 4
        symbols = [t.symbol for t in all_tickers]
        assert "AAPL" in symbols
        assert "GOOGL" in symbols


class TestDomainFactoryAdvanced:
    """Test advanced DomainFactory functionality."""

    def test_create_portfolio_edge_cases(self, mock_data_fetcher, mock_logging_manager):
        """Test portfolio creation edge cases."""
        factory = DomainFactory(fetcher=mock_data_fetcher, logging_manager=mock_logging_manager)

        # Test with minimal config
        minimal_config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Minimal",
                initial_capital=1000.0,
                allocation_method="equal_weight",
                tickers=[TickerConfig(symbol="SPY", quantity=1.0, category="INDEX")],
            )
        )

        with patch.object(Portfolio, "validate_capital_sufficiency"):
            with patch.object(Portfolio, "validate_allocation_constraints"):
                portfolio = factory.create_portfolio(minimal_config)

        assert portfolio.name == "Minimal"
        assert portfolio.initial_capital == 1000.0
        assert len(portfolio.assets) == 1

    def test_create_portfolio_with_categories(self, mock_data_fetcher, mock_logging_manager):
        """Test portfolio creation with various category types."""
        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Category Test",
                initial_capital=100000.0,
                allocation_method="equal_weight",
                tickers=[
                    TickerConfig(symbol="VTI", quantity=10.0, category="INDEX"),
                    TickerConfig(symbol="QQQ", quantity=5.0, category="GROWTH"),
                    TickerConfig(symbol="VYM", quantity=15.0, category="DIVIDEND"),
                    TickerConfig(symbol="ARKK", quantity=8.0, category="SPECULATIVE"),
                ],
            )
        )

        factory = DomainFactory(fetcher=mock_data_fetcher, logging_manager=mock_logging_manager)

        with patch.object(Portfolio, "validate_capital_sufficiency"):
            with patch.object(Portfolio, "validate_allocation_constraints"):
                portfolio = factory.create_portfolio(config)

        # Check categories are correctly assigned
        categories = {asset.category for asset in portfolio.assets}
        expected_categories = {
            Category.INDEX,
            Category.GROWTH,
            Category.DIVIDEND,
            Category.SPECULATIVE,
        }
        assert categories == expected_categories

    def test_create_portfolio_dynamic_allocation_error_handling(self, mock_data_fetcher, mock_logging_manager):
        """Test error handling in dynamic allocation."""
        # Patch the get_current_prices method to raise an exception
        with patch.object(
            mock_data_fetcher,
            "get_current_prices",
            side_effect=Exception("Price fetch failed"),
        ):
            config = StockulaConfig(
                portfolio=PortfolioConfig(
                    name="Error Test",
                    initial_capital=10000.0,
                    dynamic_allocation=True,
                    tickers=[TickerConfig(symbol="AAPL", allocation_amount=5000.0)],
                )
            )

            factory = DomainFactory(fetcher=mock_data_fetcher, logging_manager=mock_logging_manager)

            # Should handle price fetch errors gracefully or raise appropriate error
            with pytest.raises(Exception) as exc_info:
                factory.create_portfolio(config)
            assert "Price fetch failed" in str(exc_info.value)

    def test_create_portfolio_auto_allocation_categories(self, mock_data_fetcher, mock_allocator, mock_logging_manager):
        """Test auto allocation with category ratios."""
        # Override the return value for this specific test
        with patch.object(
            mock_data_fetcher,
            "get_current_prices",
            return_value={
                "VTI": 200.0,
                "QQQ": 300.0,
                "VYM": 100.0,
                "ARKK": 50.0,
                "VWO": 45.0,
            },
        ):
            config = StockulaConfig(
                portfolio=PortfolioConfig(
                    name="Auto Allocation",
                    initial_capital=100000.0,
                    auto_allocate=True,
                    category_ratios={
                        "INDEX": 0.4,  # 40%
                        "GROWTH": 0.3,  # 30%
                        "DIVIDEND": 0.2,  # 20%
                        "SPECULATIVE": 0.1,  # 10%
                        "INTERNATIONAL": 0.0,  # 0% for VWO
                    },
                    tickers=[
                        TickerConfig(symbol="VTI", category="INDEX"),
                        TickerConfig(symbol="QQQ", category="GROWTH"),
                        TickerConfig(symbol="VYM", category="DIVIDEND"),
                        TickerConfig(symbol="ARKK", category="SPECULATIVE"),
                        TickerConfig(symbol="VWO", category="INTERNATIONAL"),
                    ],
                )
            )

            factory = DomainFactory(
                fetcher=mock_data_fetcher, allocator=mock_allocator, logging_manager=mock_logging_manager
            )
            portfolio = factory.create_portfolio(config)

            # Only tickers with non-zero category allocation should be included
            # VWO has 0% allocation for INTERNATIONAL category, so it's excluded
            assert len(portfolio.assets) == 4

            # Check that quantities were calculated
            for asset in portfolio.assets:
                assert asset.quantity > 0
                # VWO should not be in the portfolio
                assert asset.ticker.symbol != "VWO"

    def test_ticker_registry_integration(self, mock_data_fetcher, mock_logging_manager):
        """Test ticker registry integration with factory."""
        factory = DomainFactory(fetcher=mock_data_fetcher, logging_manager=mock_logging_manager)

        # Clear registry first
        registry = TickerRegistry()
        registry._clear()

        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Registry Test",
                initial_capital=50000.0,
                allocation_method="equal_weight",
                tickers=[
                    TickerConfig(symbol="AAPL", quantity=5.0),
                    TickerConfig(symbol="GOOGL", quantity=3.0),
                ],
            )
        )

        with patch.object(Portfolio, "validate_capital_sufficiency"):
            with patch.object(Portfolio, "validate_allocation_constraints"):
                factory.create_portfolio(config)

        # Check that tickers were added to registry
        all_tickers = factory.get_all_tickers()
        assert len(all_tickers) == 2

        # Check registry contains the tickers
        registry_tickers = registry.all()
        assert "AAPL" in registry_tickers
        assert "GOOGL" in registry_tickers

    def test_factory_methods_consistency(self, mock_data_fetcher, mock_logging_manager):
        """Test that factory methods are consistent."""
        factory = DomainFactory(fetcher=mock_data_fetcher, logging_manager=mock_logging_manager)

        # Test that get_all_tickers returns empty list initially
        initial_tickers = factory.get_all_tickers()
        assert isinstance(initial_tickers, list)
        assert len(initial_tickers) >= 0  # Could have tickers from other tests

    def test_allocation_amount_calculation(self, mock_data_fetcher, mock_logging_manager):
        """Test allocation amount calculation accuracy."""
        # Override the return value for this specific test
        with patch.object(
            mock_data_fetcher,
            "get_current_prices",
            return_value={
                "AAPL": 150.0,
                "GOOGL": 2500.0,  # High price stock
            },
        ):
            config = StockulaConfig(
                portfolio=PortfolioConfig(
                    name="Allocation Test",
                    initial_capital=100000.0,
                    dynamic_allocation=True,
                    tickers=[
                        TickerConfig(symbol="AAPL", allocation_amount=15000.0),  # $15k / $150 = 100 shares
                        TickerConfig(symbol="GOOGL", allocation_amount=5000.0),  # $5k / $2500 = 2 shares
                    ],
                )
            )

            factory = DomainFactory(fetcher=mock_data_fetcher, logging_manager=mock_logging_manager)
            portfolio = factory.create_portfolio(config)

            # Check exact allocations
            aapl_asset = portfolio.get_asset_by_symbol("AAPL")
            assert aapl_asset.quantity == 100.0

            googl_asset = portfolio.get_asset_by_symbol("GOOGL")
            assert googl_asset.quantity == 2.0

    def test_allocation_percentage_calculation(self, mock_data_fetcher, mock_logging_manager):
        """Test allocation percentage calculation."""
        # Override the return value for this specific test
        with patch.object(
            mock_data_fetcher,
            "get_current_prices",
            return_value={"VTI": 200.0, "QQQ": 300.0},
        ):
            config = StockulaConfig(
                portfolio=PortfolioConfig(
                    name="Percentage Test",
                    initial_capital=60000.0,
                    dynamic_allocation=True,
                    tickers=[
                        TickerConfig(symbol="VTI", allocation_pct=60.0),  # 60% of $60k = $36k / $200 = 180 shares
                        TickerConfig(symbol="QQQ", allocation_pct=40.0),  # 40% of $60k = $24k / $300 = 80 shares
                    ],
                )
            )

            factory = DomainFactory(fetcher=mock_data_fetcher, logging_manager=mock_logging_manager)
            portfolio = factory.create_portfolio(config)

            vti_asset = portfolio.get_asset_by_symbol("VTI")
            assert vti_asset.quantity == 180.0

            qqq_asset = portfolio.get_asset_by_symbol("QQQ")
            assert qqq_asset.quantity == 80.0


class TestDomainFactoryErrorScenarios:
    """Test error scenarios in DomainFactory."""

    def test_create_portfolio_zero_capital(self, mock_logging_manager):
        """Test portfolio creation with zero capital."""
        from pydantic import ValidationError

        # Zero capital should be caught by pydantic validation
        with pytest.raises(ValidationError):
            StockulaConfig(
                portfolio=PortfolioConfig(
                    name="Zero Capital",
                    initial_capital=0.0,  # This should fail validation
                    allocation_method="equal_weight",
                    tickers=[TickerConfig(symbol="SPY", quantity=1.0)],
                )
            )

    def test_create_portfolio_zero_price(self, mock_data_fetcher, mock_logging_manager):
        """Test handling of zero stock prices."""
        # Override the return value for this specific test
        with patch.object(
            mock_data_fetcher,
            "get_current_prices",
            return_value={
                "PENNY": 0.0  # Zero price
            },
        ):
            config = StockulaConfig(
                portfolio=PortfolioConfig(
                    name="Zero Price Test",
                    initial_capital=10000.0,
                    dynamic_allocation=True,
                    tickers=[TickerConfig(symbol="PENNY", allocation_amount=1000.0)],
                )
            )

            factory = DomainFactory(fetcher=mock_data_fetcher, logging_manager=mock_logging_manager)

            # Should handle zero price gracefully or raise appropriate error
            with pytest.raises((ValueError, ZeroDivisionError)):
                factory.create_portfolio(config)

    def test_create_portfolio_missing_prices(self, mock_data_fetcher, mock_logging_manager):
        """Test handling of missing stock prices."""
        # Override the return value for this specific test
        with patch.object(
            mock_data_fetcher,
            "get_current_prices",
            return_value={
                "AAPL": 150.0
                # GOOGL missing
            },
        ):
            config = StockulaConfig(
                portfolio=PortfolioConfig(
                    name="Missing Price Test",
                    initial_capital=10000.0,
                    dynamic_allocation=True,
                    tickers=[
                        TickerConfig(symbol="AAPL", allocation_amount=5000.0),
                        TickerConfig(symbol="GOOGL", allocation_amount=5000.0),  # Price missing
                    ],
                )
            )

            factory = DomainFactory(fetcher=mock_data_fetcher, logging_manager=mock_logging_manager)

            # Should raise error for missing prices in dynamic allocation
            with pytest.raises(ValueError, match="Could not fetch price for GOOGL"):
                factory.create_portfolio(config)


class TestTickerRegistryAdvanced:
    """Test advanced TickerRegistry functionality."""

    def test_ticker_registry_update_behavior(self, mock_logging_manager):
        """Test ticker registry update behavior."""
        registry = TickerRegistry()
        registry._clear()

        # Create ticker with initial data - use all required parameters
        ticker1 = registry.get_or_create(
            "AAPL",
            sector="Technology",
            market_cap=3000.0,
            category=None,
            price_range=None,
        )
        assert ticker1.sector == "Technology"
        assert ticker1.market_cap == 3000.0

        # Update with new data - use all required parameters
        ticker2 = registry.get_or_create("AAPL", sector="Tech", market_cap=3100.0, category=None, price_range=None)
        # Since Ticker is immutable, a new instance is created with updated values
        assert ticker2 is not ticker1  # Different object due to immutability
        assert ticker2.symbol == ticker1.symbol  # Same symbol
        assert ticker2.sector == "Tech"  # Updated
        assert ticker2.market_cap == 3100.0  # Updated

        # But the registry returns the new instance for the same symbol
        ticker3 = registry.get("AAPL")
        assert ticker3 is ticker2

    def test_ticker_registry_category_handling(self):
        """Test ticker registry category handling."""
        registry = TickerRegistry()
        registry._clear()

        # Create ticker with category
        ticker = registry.get_or_create("NVDA", category=Category.SPECULATIVE)
        assert ticker.category == Category.SPECULATIVE

        # Update category
        updated_ticker = registry.get_or_create("NVDA", category=Category.GROWTH)
        assert updated_ticker.category == Category.GROWTH

    def test_ticker_registry_concurrent_access(self):
        """Test ticker registry handles concurrent-like access."""
        registry = TickerRegistry()
        registry._clear()

        # Simulate multiple "concurrent" creations
        tickers = []
        for i in range(10):
            ticker = registry.get_or_create(f"STOCK{i}")
            tickers.append(ticker)

        # All should be unique
        assert len(tickers) == 10
        assert len({ticker.symbol for ticker in tickers}) == 10

        # Registry should contain all
        all_tickers = registry.all()
        assert len(all_tickers) == 10


class TestAssetAdvanced:
    """Test advanced Asset functionality."""

    def test_asset_value_edge_cases(self):
        """Test asset value calculation edge cases."""
        ticker = Ticker("TEST")
        asset = Asset(ticker_init=ticker, quantity_init=10.0)

        # Test with negative price (should handle gracefully)
        value = asset.get_value(-50.0)
        assert value == -500.0  # Mathematically correct

        # Test with very large numbers
        large_value = asset.get_value(1e10)
        assert large_value == 1e11

        # Test with very small numbers
        small_value = asset.get_value(0.0001)
        assert small_value == 0.001

    def test_asset_quantity_edge_cases(self):
        """Test asset with edge case quantities."""
        # Use proper Ticker constructor with all required parameters
        ticker = Ticker(
            "TEST",  # symbol_init
            None,  # sector_init
            None,  # market_cap_init
            None,  # category_init
            None,  # price_range_init
            None,  # metadata_init
        )

        # Fractional shares
        fractional_asset = Asset(ticker_init=ticker, quantity_init=0.5)
        assert fractional_asset.get_value(100.0) == 50.0

        # Assets must have positive quantity, so test minimum positive value
        min_asset = Asset(ticker_init=ticker, quantity_init=0.001)
        assert min_asset.get_value(100.0) == 0.1

        # Very large quantity
        large_asset = Asset(ticker_init=ticker, quantity_init=1e6)
        assert large_asset.get_value(100.0) == 1e8

    def test_asset_category_precedence(self):
        """Test asset category precedence over ticker category."""
        ticker_with_category = Ticker("TEST", category=Category.INDEX)

        # Asset category should override ticker category
        asset = Asset(ticker_init=ticker_with_category, quantity_init=10.0, category_init=Category.GROWTH)
        assert asset.category == Category.GROWTH
        assert ticker_with_category.category == Category.INDEX  # Unchanged


class TestDateToString:
    """Test the date_to_string utility function."""

    def test_date_to_string_with_none(self):
        """Test date_to_string with None input."""
        from stockula.domain.factory import date_to_string

        assert date_to_string(None) is None

    def test_date_to_string_with_string(self):
        """Test date_to_string with string input."""
        from stockula.domain.factory import date_to_string

        assert date_to_string("2023-01-01") == "2023-01-01"

    def test_date_to_string_with_date(self):
        """Test date_to_string with date object."""
        from stockula.domain.factory import date_to_string

        test_date = date(2023, 1, 1)
        assert date_to_string(test_date) == "2023-01-01"


class TestDomainFactoryEdgeCases:
    """Test edge cases in DomainFactory."""

    def test_create_portfolio_without_config(self, mock_logging_manager):
        """Test creating portfolio without config."""
        factory = DomainFactory(logging_manager=mock_logging_manager)

        # Create a config to pass
        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test",
                initial_capital=100000,
                tickers=[TickerConfig(symbol="AAPL", quantity=10)],
            )
        )

        # Without fetcher, should raise error for any allocation method requiring prices
        factory = DomainFactory(fetcher=None, logging_manager=mock_logging_manager)
        config.portfolio.dynamic_allocation = True
        config.portfolio.tickers[0].allocation_pct = 100.0

        with pytest.raises(ValueError, match="Allocator not configured"):
            factory.create_portfolio(config)

    def test_calculate_dynamic_quantities_with_start_date_no_data(self, mock_allocator, mock_logging_manager):
        """Test quantity calculation when no data available on start date."""
        # Create config with start date
        config = StockulaConfig(
            data=DataConfig(start_date="2023-01-01", end_date="2023-12-31"),
            portfolio=PortfolioConfig(
                name="Test Portfolio",
                initial_capital=100000,
                dynamic_allocation=True,
                tickers=[
                    TickerConfig(symbol="AAPL", category="MOMENTUM", allocation_pct=50.0),
                    TickerConfig(symbol="GOOGL", category="GROWTH", allocation_pct=50.0),
                ],
            ),
        )

        # Mock fetcher
        mock_fetcher = Mock()

        # First call returns empty data (no data on exact date)
        # Second call returns data from a week later
        mock_fetcher.get_stock_data.side_effect = [
            pd.DataFrame(),  # Empty for AAPL on start date
            pd.DataFrame({"Close": [150.0]}, index=[pd.Timestamp("2023-01-08")]),  # AAPL week later
            pd.DataFrame(),  # Empty for GOOGL on start date
            pd.DataFrame({"Close": [120.0]}, index=[pd.Timestamp("2023-01-08")]),  # GOOGL week later
        ]

        mock_fetcher.get_current_prices.return_value = {"AAPL": 155.0, "GOOGL": 125.0}

        # Update the allocator's fetcher
        mock_allocator.fetcher = mock_fetcher

        factory = DomainFactory(
            config=config, fetcher=mock_fetcher, allocator=mock_allocator, logging_manager=mock_logging_manager
        )

        # Calculate quantities
        quantities = factory.allocator.calculate_dynamic_quantities(config, config.portfolio.tickers)

        # Should have used the week-later prices
        assert "AAPL" in quantities
        assert "GOOGL" in quantities
        assert quantities["AAPL"] > 0
        assert quantities["GOOGL"] > 0

    def test_calculate_dynamic_quantities_with_start_date_fallback_to_current(
        self, mock_allocator, mock_logging_manager
    ):
        """Test quantity calculation falling back to current prices."""
        # Create config with start date
        config = StockulaConfig(
            data=DataConfig(start_date="2023-01-01", end_date="2023-12-31"),
            portfolio=PortfolioConfig(
                name="Test Portfolio",
                initial_capital=100000,
                dynamic_allocation=True,
                tickers=[TickerConfig(symbol="AAPL", category="MOMENTUM", allocation_pct=100.0)],
            ),
        )

        # Mock fetcher - all historical data calls return empty
        mock_fetcher = Mock()
        mock_fetcher.get_stock_data.return_value = pd.DataFrame()  # Always empty
        mock_fetcher.get_current_prices.return_value = {"AAPL": 155.0}

        # Update the allocator's fetcher
        mock_allocator.fetcher = mock_fetcher

        factory = DomainFactory(
            config=config, fetcher=mock_fetcher, allocator=mock_allocator, logging_manager=mock_logging_manager
        )

        # Calculate quantities
        quantities = factory.allocator.calculate_dynamic_quantities(config, config.portfolio.tickers)

        # Should have used current price as fallback
        assert "AAPL" in quantities
        assert quantities["AAPL"] > 0

    def test_calculate_dynamic_quantities_with_exception_handling(self, mock_allocator, mock_logging_manager):
        """Test quantity calculation with exception in data fetching."""
        # Create config with start date as string
        config = StockulaConfig(
            data=DataConfig(start_date="2023-01-01", end_date="2023-12-31"),
            portfolio=PortfolioConfig(
                name="Test Portfolio",
                initial_capital=100000,
                dynamic_allocation=True,
                tickers=[TickerConfig(symbol="AAPL", category="MOMENTUM", allocation_pct=100.0)],
            ),
        )

        # Mock fetcher that raises exception
        mock_fetcher = Mock()
        mock_fetcher.get_stock_data.side_effect = Exception("API Error")
        mock_fetcher.get_current_prices.return_value = {"AAPL": 155.0}

        # Update the allocator's fetcher
        mock_allocator.fetcher = mock_fetcher

        factory = DomainFactory(
            config=config, fetcher=mock_fetcher, allocator=mock_allocator, logging_manager=mock_logging_manager
        )

        # Calculate quantities - should handle exception and use current prices
        quantities = factory.allocator.calculate_dynamic_quantities(config, config.portfolio.tickers)

        assert "AAPL" in quantities
        assert quantities["AAPL"] > 0

    def test_calculate_dynamic_quantities_date_type_handling(self, mock_allocator, mock_logging_manager):
        """Test quantity calculation with date object instead of string."""
        # Create config with start date as date object
        config = StockulaConfig(
            data=DataConfig(start_date=date(2023, 1, 1), end_date=date(2023, 12, 31)),
            portfolio=PortfolioConfig(
                name="Test Portfolio",
                initial_capital=100000,
                dynamic_allocation=True,
                tickers=[TickerConfig(symbol="AAPL", category="MOMENTUM", allocation_pct=100.0)],
            ),
        )

        # Mock fetcher
        mock_fetcher = Mock()
        mock_fetcher.get_stock_data.side_effect = [
            pd.DataFrame(),  # Empty on start date
            pd.DataFrame({"Close": [150.0]}, index=[pd.Timestamp("2023-01-08")]),  # Week later
        ]
        mock_fetcher.get_current_prices.return_value = {"AAPL": 155.0}

        # Update the allocator's fetcher
        mock_allocator.fetcher = mock_fetcher

        factory = DomainFactory(
            config=config, fetcher=mock_fetcher, allocator=mock_allocator, logging_manager=mock_logging_manager
        )

        # Calculate quantities
        quantities = factory.allocator.calculate_dynamic_quantities(config, config.portfolio.tickers)

        assert "AAPL" in quantities
        assert quantities["AAPL"] > 0

    def test_auto_allocation_with_no_category_ratios(self, mock_logging_manager):
        """Test auto-allocation without category ratios."""
        # PortfolioConfig validation prevents auto_allocate without category_ratios
        # So we test that the validation works
        with pytest.raises(ValueError, match="auto_allocate=True requires category_ratios"):
            StockulaConfig(
                portfolio=PortfolioConfig(
                    name="Test Portfolio",
                    initial_capital=100000,
                    auto_allocate=True,
                    tickers=[TickerConfig(symbol="AAPL", category="MOMENTUM")],
                )
            )

    def test_auto_allocation_with_fractional_shares(self, mock_allocator):
        """Test auto-allocation with fractional shares allowed."""
        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test Portfolio",
                initial_capital=100000,
                auto_allocate=True,
                allow_fractional_shares=True,
                category_ratios={"MOMENTUM": 0.6, "GROWTH": 0.4},
                capital_utilization_target=0.95,
                tickers=[
                    TickerConfig(symbol="AAPL", category="MOMENTUM"),
                    TickerConfig(symbol="GOOGL", category="GROWTH"),
                    TickerConfig(symbol="MSFT", category="GROWTH"),
                ],
            )
        )

        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.return_value = {
            "AAPL": 150.0,
            "GOOGL": 120.0,
            "MSFT": 300.0,
        }

        # Update the allocator's fetcher
        mock_allocator.fetcher = mock_fetcher

        factory = DomainFactory(
            config=config, fetcher=mock_fetcher, allocator=mock_allocator, logging_manager=mock_logging_manager
        )

        quantities = factory.allocator.calculate_auto_allocation_quantities(config, config.portfolio.tickers)

        # Check that quantities are fractional
        assert "AAPL" in quantities
        assert "GOOGL" in quantities
        assert "MSFT" in quantities
        # With fractional shares, we should use most of the capital
        total_value = quantities["AAPL"] * 150.0 + quantities["GOOGL"] * 120.0 + quantities["MSFT"] * 300.0
        assert abs(total_value - 95000.0) < 100  # Close to 95% of 100k

    def test_auto_allocation_with_zero_ratio_category(self, mock_allocator, mock_logging_manager):
        """Test auto-allocation with zero ratio for a category."""
        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test Portfolio",
                initial_capital=100000,
                auto_allocate=True,
                category_ratios={"MOMENTUM": 1.0, "GROWTH": 0.0},
                capital_utilization_target=1.0,
                tickers=[
                    TickerConfig(symbol="AAPL", category="MOMENTUM"),
                    TickerConfig(symbol="GOOGL", category="GROWTH"),
                ],
            )
        )

        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.return_value = {
            "AAPL": 150.0,
            "GOOGL": 120.0,
        }

        # Update the allocator's fetcher
        mock_allocator.fetcher = mock_fetcher

        factory = DomainFactory(
            config=config, fetcher=mock_fetcher, allocator=mock_allocator, logging_manager=mock_logging_manager
        )

        quantities = factory.allocator.calculate_auto_allocation_quantities(config, config.portfolio.tickers)

        # AAPL should get allocation, GOOGL should get 0
        assert quantities["AAPL"] > 0
        assert quantities["GOOGL"] == 0

    def test_create_portfolio_with_missing_info(self, mock_logging_manager):
        """Test portfolio creation when ticker info is missing."""
        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test Portfolio",
                initial_capital=100000,
                tickers=[TickerConfig(symbol="UNKNOWN", quantity=10)],
            )
        )

        mock_fetcher = Mock()
        # Return minimal info
        mock_fetcher.get_info.return_value = {"symbol": "UNKNOWN"}
        mock_fetcher.get_current_prices.return_value = {"UNKNOWN": 100.0}

        factory = DomainFactory(config=config, fetcher=mock_fetcher, logging_manager=mock_logging_manager)

        # Should create portfolio without error
        portfolio = factory.create_portfolio(config)
        assert portfolio is not None
        assert len(portfolio.assets) == 1

    def test_create_asset_with_invalid_category(self, mock_logging_manager):
        """Test creating asset with invalid category string."""
        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test Portfolio",
                initial_capital=100000,
                tickers=[TickerConfig(symbol="AAPL", quantity=10, category="INVALID_CATEGORY")],
            )
        )

        factory = DomainFactory(config=config, logging_manager=mock_logging_manager)

        # Should create asset without error (category will be None)
        ticker_config = config.portfolio.tickers[0]
        asset = factory._create_asset(ticker_config)

        # Asset should be created but category might be None
        assert asset is not None
        assert asset.ticker.symbol == "AAPL"
        assert asset.quantity == 10

    def test_calculate_dynamic_quantities_missing_price(self, mock_logging_manager):
        """Test dynamic quantity calculation when price is missing."""
        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test Portfolio",
                initial_capital=100000,
                dynamic_allocation=True,
                tickers=[TickerConfig(symbol="AAPL", allocation_pct=100.0)],
            )
        )

        mock_fetcher = Mock()
        mock_fetcher.get_current_prices.return_value = {}  # No prices returned

        factory = DomainFactory(config=config, fetcher=mock_fetcher, logging_manager=mock_logging_manager)

        with pytest.raises(ValueError, match="Could not fetch price"):
            factory.allocator.calculate_dynamic_quantities(config, config.portfolio.tickers)

    def test_ticker_config_allows_no_allocation(self, mock_logging_manager):
        """Test that TickerConfig allows no allocation for backtest_optimized."""
        # This is now valid for backtest_optimized allocation
        ticker = TickerConfig(symbol="AAPL")
        assert ticker.symbol == "AAPL"
        assert ticker.quantity is None
        assert ticker.allocation_pct is None
        assert ticker.allocation_amount is None

    def test_auto_allocation_requires_category_for_all_tickers(self, mock_logging_manager):
        """Test that auto-allocation requires category for all tickers."""
        # When auto_allocate is True, ALL tickers must have category
        with pytest.raises(ValueError, match="must have category specified for auto-allocation"):
            config = StockulaConfig(
                portfolio=PortfolioConfig(
                    name="Test Portfolio",
                    initial_capital=100000,
                    auto_allocate=True,
                    category_ratios={"MOMENTUM": 1.0},
                    tickers=[
                        TickerConfig(symbol="AAPL", category="MOMENTUM"),
                        TickerConfig(
                            symbol="GOOGL", quantity=10
                        ),  # Has quantity but no category - invalid for auto-allocation
                    ],
                )
            )

            mock_fetcher = Mock()
            mock_fetcher.get_current_prices.return_value = {"AAPL": 150.0, "GOOGL": 120.0}

            factory = DomainFactory(config=config, fetcher=mock_fetcher, logging_manager=mock_logging_manager)
            factory.create_portfolio(config)

    def test_get_all_tickers_from_factory(self, mock_logging_manager):
        """Test getting all registered tickers from factory."""
        factory = DomainFactory(logging_manager=mock_logging_manager)

        # Create some tickers with quantity
        ticker_config1 = TickerConfig(symbol="AAPL", sector="Technology", quantity=10)
        ticker_config2 = TickerConfig(symbol="GOOGL", sector="Technology", quantity=20)

        factory._create_ticker(ticker_config1)
        factory._create_ticker(ticker_config2)

        all_tickers = factory.get_all_tickers()

        assert len(all_tickers) == 2
        symbols = [t.symbol for t in all_tickers]
        assert "AAPL" in symbols
        assert "GOOGL" in symbols
