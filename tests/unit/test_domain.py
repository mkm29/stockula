"""Tests for domain models."""

import pytest
from unittest.mock import Mock, patch

from stockula.domain import (
    Ticker,
    TickerRegistry,
    Asset,
    Portfolio,
    Category,
    DomainFactory,
)
from stockula.config import StockulaConfig, PortfolioConfig, TickerConfig


class TestTicker:
    """Test Ticker domain model."""

    def test_ticker_creation(self):
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
        assert (
            repr(ticker)
            == "Ticker(symbol='AAPL', sector='Technology', market_cap=None, category=None)"
        )


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
        ticker2 = registry.get_or_create("GOOGL")
        ticker3 = registry.get_or_create("MSFT")

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
        asset = Asset(ticker=sample_ticker, quantity=10.0, category=Category.MOMENTUM)
        assert asset.ticker == sample_ticker
        assert asset.quantity == 10.0
        assert asset.category == Category.MOMENTUM
        assert asset.symbol == "AAPL"

    def test_asset_without_category(self, sample_ticker):
        """Test asset without explicit category is None."""
        ticker_with_category = Ticker(
            "NVDA",  # symbol_init
            None,  # sector_init
            None,  # market_cap_init
            "SPECULATIVE",  # category_init
            None,  # price_range_init
            None,  # metadata_init
        )
        asset = Asset(ticker_with_category, quantity=5.0, category=None)
        assert asset.category is None  # Asset doesn't inherit ticker category

    def test_asset_category_override(self, sample_ticker):
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
            ticker=ticker_with_category,
            quantity=20.0,
            category=Category.GROWTH,  # Asset's own category
        )
        assert asset.category == Category.GROWTH
        assert ticker_with_category.category == "INDEX"  # Ticker unchanged

    def test_asset_value_calculation(self, sample_ticker):
        """Test asset value calculation."""
        asset = Asset(ticker=sample_ticker, quantity=10.0)

        # Test with different prices
        assert asset.get_value(150.0) == 1500.0
        assert asset.get_value(200.0) == 2000.0
        assert asset.get_value(0.0) == 0.0

    def test_asset_string_representation(self, sample_ticker):
        """Test asset string representation."""
        asset = Asset(ticker=sample_ticker, quantity=10.0)
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

    def test_portfolio_creation(self):
        """Test creating a portfolio."""
        portfolio = Portfolio(
            name="Test Portfolio",
            initial_capital=100000.0,
            allocation_method="equal_weight",
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
        duplicate_asset = Asset(ticker=sample_asset.ticker, quantity=5.0)
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
        expensive_asset = Asset(
            ticker=expensive_ticker, quantity=1.0, category=Category.VALUE
        )

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

    def test_create_portfolio_basic(self, sample_stockula_config):
        """Test creating a basic portfolio from config."""
        factory = DomainFactory()

        # Mock validation methods to not raise errors
        with patch.object(Portfolio, "validate_capital_sufficiency"):
            with patch.object(Portfolio, "validate_allocation_constraints"):
                portfolio = factory.create_portfolio(sample_stockula_config)

        assert portfolio.name == "Test Portfolio"
        assert portfolio.initial_capital == 100000.0
        assert len(portfolio.assets) == 4

    def test_create_portfolio_with_dynamic_allocation(
        self, dynamic_allocation_config, mock_data_fetcher
    ):
        """Test creating portfolio with dynamic allocation."""
        config = StockulaConfig(portfolio=dynamic_allocation_config)
        factory = DomainFactory()

        with patch("stockula.data.fetcher.DataFetcher", return_value=mock_data_fetcher):
            portfolio = factory.create_portfolio(config)

        assert len(portfolio.assets) == 3

        # Check AAPL allocation (fixed $15,000 / $150 = 100 shares)
        aapl_asset = portfolio.get_asset_by_symbol("AAPL")
        assert aapl_asset.quantity == 100.0

        # Check GOOGL allocation (20% of $50,000 = $10,000 / $120 ≈ 83.33 shares)
        googl_asset = portfolio.get_asset_by_symbol("GOOGL")
        assert googl_asset.quantity == pytest.approx(83.33, rel=0.01)

    def test_create_portfolio_with_auto_allocation(
        self, auto_allocation_config, mock_data_fetcher
    ):
        """Test creating portfolio with auto allocation."""
        config = StockulaConfig(portfolio=auto_allocation_config)
        factory = DomainFactory()

        with patch("stockula.data.fetcher.DataFetcher", return_value=mock_data_fetcher):
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

    def test_create_portfolio_insufficient_allocation(self):
        """Test creating portfolio with insufficient allocation info raises error."""
        from pydantic import ValidationError

        # This should raise validation error at config creation time
        with pytest.raises(ValidationError, match="must specify exactly one"):
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

    def test_get_all_tickers(self, sample_stockula_config):
        """Test getting all tickers from factory."""
        factory = DomainFactory()

        # Create portfolio to populate registry
        with patch.object(Portfolio, "validate_capital_sufficiency"):
            with patch.object(Portfolio, "validate_allocation_constraints"):
                factory.create_portfolio(sample_stockula_config)

        all_tickers = factory.get_all_tickers()
        assert len(all_tickers) == 4
        symbols = [t.symbol for t in all_tickers]
        assert "AAPL" in symbols
        assert "GOOGL" in symbols
