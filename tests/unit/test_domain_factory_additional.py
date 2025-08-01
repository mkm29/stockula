"""Additional tests for domain factory to improve coverage."""

from datetime import date
from unittest.mock import Mock

import pandas as pd
import pytest

from stockula.config import DataConfig, PortfolioConfig, StockulaConfig, TickerConfig
from stockula.domain import DomainFactory


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

    def test_create_portfolio_without_config(self):
        """Test creating portfolio without config."""
        factory = DomainFactory()

        # Create a config to pass
        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test",
                initial_capital=100000,
                tickers=[TickerConfig(symbol="AAPL", quantity=10)],
            )
        )

        # Without fetcher, should raise error for any allocation method requiring prices
        factory = DomainFactory(fetcher=None)
        config.portfolio.dynamic_allocation = True
        config.portfolio.tickers[0].allocation_pct = 100.0

        with pytest.raises(ValueError, match="Data fetcher not configured"):
            factory.create_portfolio(config)

    def test_calculate_dynamic_quantities_with_start_date_no_data(self):
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

        factory = DomainFactory(config=config, fetcher=mock_fetcher)

        # Calculate quantities
        quantities = factory._calculate_dynamic_quantities(config, config.portfolio.tickers)

        # Should have used the week-later prices
        assert "AAPL" in quantities
        assert "GOOGL" in quantities
        assert quantities["AAPL"] > 0
        assert quantities["GOOGL"] > 0

    def test_calculate_dynamic_quantities_with_start_date_fallback_to_current(self):
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

        factory = DomainFactory(config=config, fetcher=mock_fetcher)

        # Calculate quantities
        quantities = factory._calculate_dynamic_quantities(config, config.portfolio.tickers)

        # Should have used current price as fallback
        assert "AAPL" in quantities
        assert quantities["AAPL"] > 0

    def test_calculate_dynamic_quantities_with_exception_handling(self):
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

        factory = DomainFactory(config=config, fetcher=mock_fetcher)

        # Calculate quantities - should handle exception and use current prices
        quantities = factory._calculate_dynamic_quantities(config, config.portfolio.tickers)

        assert "AAPL" in quantities
        assert quantities["AAPL"] > 0

    def test_calculate_dynamic_quantities_date_type_handling(self):
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

        factory = DomainFactory(config=config, fetcher=mock_fetcher)

        # Calculate quantities
        quantities = factory._calculate_dynamic_quantities(config, config.portfolio.tickers)

        assert "AAPL" in quantities
        assert quantities["AAPL"] > 0

    def test_auto_allocation_with_no_category_ratios(self):
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

    def test_auto_allocation_with_fractional_shares(self):
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

        factory = DomainFactory(config=config, fetcher=mock_fetcher)

        quantities = factory._calculate_auto_allocation_quantities(config, config.portfolio.tickers)

        # Check that quantities are fractional
        assert "AAPL" in quantities
        assert "GOOGL" in quantities
        assert "MSFT" in quantities
        # With fractional shares, we should use most of the capital
        total_value = quantities["AAPL"] * 150.0 + quantities["GOOGL"] * 120.0 + quantities["MSFT"] * 300.0
        assert abs(total_value - 95000.0) < 100  # Close to 95% of 100k

    def test_auto_allocation_with_zero_ratio_category(self):
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

        factory = DomainFactory(config=config, fetcher=mock_fetcher)

        quantities = factory._calculate_auto_allocation_quantities(config, config.portfolio.tickers)

        # AAPL should get allocation, GOOGL should get 0
        assert quantities["AAPL"] > 0
        assert quantities["GOOGL"] == 0

    def test_create_portfolio_with_missing_info(self):
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

        factory = DomainFactory(config=config, fetcher=mock_fetcher)

        # Should create portfolio without error
        portfolio = factory.create_portfolio(config)
        assert portfolio is not None
        assert len(portfolio.assets) == 1

    def test_create_asset_with_invalid_category(self):
        """Test creating asset with invalid category string."""
        config = StockulaConfig(
            portfolio=PortfolioConfig(
                name="Test Portfolio",
                initial_capital=100000,
                tickers=[TickerConfig(symbol="AAPL", quantity=10, category="INVALID_CATEGORY")],
            )
        )

        factory = DomainFactory(config=config)

        # Should create asset without error (category will be None)
        ticker_config = config.portfolio.tickers[0]
        asset = factory._create_asset(ticker_config)

        # Asset should be created but category might be None
        assert asset is not None
        assert asset.ticker.symbol == "AAPL"
        assert asset.quantity == 10

    def test_calculate_dynamic_quantities_missing_price(self):
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

        factory = DomainFactory(config=config, fetcher=mock_fetcher)

        with pytest.raises(ValueError, match="Could not fetch price"):
            factory._calculate_dynamic_quantities(config, config.portfolio.tickers)

    def test_ticker_config_requires_allocation(self):
        """Test that TickerConfig requires some form of allocation."""
        # TickerConfig validation requires one of: quantity, allocation_pct, allocation_amount, or category
        with pytest.raises(ValueError, match="Ticker AAPL must specify exactly one of"):
            TickerConfig(symbol="AAPL")

    def test_auto_allocation_requires_category_for_all_tickers(self):
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

            factory = DomainFactory(config=config, fetcher=mock_fetcher)
            factory.create_portfolio(config)

    def test_get_all_tickers(self):
        """Test getting all registered tickers."""
        factory = DomainFactory()

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
