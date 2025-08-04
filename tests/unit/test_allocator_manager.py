"""Unit tests for AllocatorManager."""

from unittest.mock import MagicMock

import pytest

from stockula.allocation import AllocatorManager
from stockula.config import TickerConfig


@pytest.fixture
def mock_data_fetcher():
    """Create a mock data fetcher."""
    fetcher = MagicMock()
    fetcher.get_current_prices.return_value = {"AAPL": 150.0, "GOOGL": 2800.0, "MSFT": 300.0}
    fetcher.get_stock_data.return_value = MagicMock()
    fetcher.get_stock_info.return_value = {"marketCap": 1000000000}
    return fetcher


@pytest.fixture
def mock_backtest_runner():
    """Create a mock backtest runner."""
    return MagicMock()


@pytest.fixture
def mock_logging_manager():
    """Create a mock logging manager."""
    return MagicMock()


@pytest.fixture
def sample_config():
    """Create a sample configuration."""
    config = MagicMock()
    # Mock portfolio attributes
    config.portfolio = MagicMock()
    config.portfolio.initial_capital = 100000
    config.portfolio.allocation_method = "equal_weight"
    config.portfolio.allow_fractional_shares = True
    config.portfolio.capital_utilization_target = 0.95
    config.portfolio.category_ratios = {"TECH": 0.6, "FINANCE": 0.4}
    config.portfolio.tickers = []
    # Mock data attributes
    config.data = MagicMock()
    config.data.start_date = "2024-01-01"
    config.data.end_date = "2024-12-31"
    # Mock backtest_optimization
    config.backtest_optimization = MagicMock()
    return config


@pytest.fixture
def sample_tickers():
    """Create sample ticker configurations."""
    return [
        TickerConfig(symbol="AAPL", category="TECH", allocation_pct=50),
        TickerConfig(symbol="GOOGL", category="TECH", allocation_pct=30),
        TickerConfig(symbol="MSFT", category="TECH", allocation_pct=20),
    ]


class TestAllocatorManager:
    """Test cases for AllocatorManager."""

    def test_init(self, mock_data_fetcher, mock_backtest_runner, mock_logging_manager):
        """Test AllocatorManager initialization."""
        manager = AllocatorManager(
            data_fetcher=mock_data_fetcher,
            backtest_runner=mock_backtest_runner,
            logging_manager=mock_logging_manager,
        )

        assert manager.data_fetcher == mock_data_fetcher
        assert manager.backtest_runner == mock_backtest_runner
        assert manager.logger == mock_logging_manager
        assert manager.standard_allocator is not None
        assert manager.backtest_allocator is not None
        assert len(manager.allocator_map) == 6

    def test_get_allocator(self, mock_data_fetcher, mock_backtest_runner, mock_logging_manager):
        """Test getting allocator by method."""
        manager = AllocatorManager(
            data_fetcher=mock_data_fetcher,
            backtest_runner=mock_backtest_runner,
            logging_manager=mock_logging_manager,
        )

        # Test standard allocator methods
        assert manager.get_allocator("equal_weight") == manager.standard_allocator
        assert manager.get_allocator("market_cap") == manager.standard_allocator
        assert manager.get_allocator("custom") == manager.standard_allocator
        assert manager.get_allocator("dynamic") == manager.standard_allocator
        assert manager.get_allocator("auto") == manager.standard_allocator

        # Test backtest allocator
        assert manager.get_allocator("backtest_optimized") == manager.backtest_allocator

        # Test unknown method
        with pytest.raises(ValueError, match="Unknown allocation method"):
            manager.get_allocator("unknown_method")

    def test_calculate_quantities(
        self, mock_data_fetcher, mock_backtest_runner, mock_logging_manager, sample_config, sample_tickers
    ):
        """Test calculate_quantities delegates to appropriate allocator."""
        manager = AllocatorManager(
            data_fetcher=mock_data_fetcher,
            backtest_runner=mock_backtest_runner,
            logging_manager=mock_logging_manager,
        )

        # Mock the standard allocator's calculate_quantities method
        expected_result = {"AAPL": 222.22, "GOOGL": 11.90, "MSFT": 111.11}
        manager.standard_allocator.calculate_quantities = MagicMock(return_value=expected_result)

        result = manager.calculate_quantities(sample_config, sample_tickers)

        assert result == expected_result
        manager.standard_allocator.calculate_quantities.assert_called_once_with(sample_config, sample_tickers)

    def test_calculate_equal_weight_quantities(
        self, mock_data_fetcher, mock_backtest_runner, mock_logging_manager, sample_config, sample_tickers
    ):
        """Test equal weight quantity calculation."""
        manager = AllocatorManager(
            data_fetcher=mock_data_fetcher,
            backtest_runner=mock_backtest_runner,
            logging_manager=mock_logging_manager,
        )

        result = manager.calculate_equal_weight_quantities(sample_config, sample_tickers)

        # With $100,000 split equally among 3 stocks
        # AAPL at $150: $33,333.33 / $150 = 222.22 shares
        # GOOGL at $2800: $33,333.33 / $2800 = 11.90 shares
        # MSFT at $300: $33,333.33 / $300 = 111.11 shares
        assert result["AAPL"] == pytest.approx(222.22, rel=0.01)
        assert result["GOOGL"] == pytest.approx(11.90, rel=0.01)
        assert result["MSFT"] == pytest.approx(111.11, rel=0.01)

    def test_calculate_market_cap_quantities(
        self, mock_data_fetcher, mock_backtest_runner, mock_logging_manager, sample_config, sample_tickers
    ):
        """Test market cap weighted quantity calculation."""
        # Set up market caps
        mock_data_fetcher.get_stock_info.side_effect = lambda symbol: {
            "AAPL": {"marketCap": 3000000000000},  # $3T
            "GOOGL": {"marketCap": 2000000000000},  # $2T
            "MSFT": {"marketCap": 2500000000000},  # $2.5T
        }.get(symbol, {})

        manager = AllocatorManager(
            data_fetcher=mock_data_fetcher,
            backtest_runner=mock_backtest_runner,
            logging_manager=mock_logging_manager,
        )

        result = manager.calculate_market_cap_quantities(sample_config, sample_tickers)

        # Total market cap: $7.5T
        # AAPL weight: 3/7.5 = 0.4, allocation: $40,000
        # GOOGL weight: 2/7.5 = 0.267, allocation: $26,667
        # MSFT weight: 2.5/7.5 = 0.333, allocation: $33,333
        assert "AAPL" in result
        assert "GOOGL" in result
        assert "MSFT" in result

    def test_calculate_dynamic_quantities(
        self, mock_data_fetcher, mock_backtest_runner, mock_logging_manager, sample_config, sample_tickers
    ):
        """Test dynamic quantity calculation."""
        manager = AllocatorManager(
            data_fetcher=mock_data_fetcher,
            backtest_runner=mock_backtest_runner,
            logging_manager=mock_logging_manager,
        )

        result = manager.calculate_dynamic_quantities(sample_config, sample_tickers)

        # AAPL: 50% of $100,000 = $50,000 / $150 = 333.33 shares
        # GOOGL: 30% of $100,000 = $30,000 / $2800 = 10.71 shares
        # MSFT: 20% of $100,000 = $20,000 / $300 = 66.67 shares
        assert result["AAPL"] == pytest.approx(333.33, rel=0.01)
        assert result["GOOGL"] == pytest.approx(10.71, rel=0.01)
        assert result["MSFT"] == pytest.approx(66.67, rel=0.01)

    def test_calculate_auto_allocation_quantities(
        self, mock_data_fetcher, mock_backtest_runner, mock_logging_manager, sample_config, sample_tickers
    ):
        """Test auto allocation quantity calculation."""
        manager = AllocatorManager(
            data_fetcher=mock_data_fetcher,
            backtest_runner=mock_backtest_runner,
            logging_manager=mock_logging_manager,
        )

        result = manager.calculate_auto_allocation_quantities(sample_config, sample_tickers)

        # All tickers are TECH, and TECH has 60% allocation
        # Target capital: $100,000 * 0.95 = $95,000
        # TECH allocation: $95,000 * 0.6 = $57,000
        # Split equally among 3 TECH stocks: $19,000 each
        assert "AAPL" in result
        assert "GOOGL" in result
        assert "MSFT" in result

    def test_calculate_backtest_optimized_quantities(
        self, mock_data_fetcher, mock_backtest_runner, mock_logging_manager, sample_config, sample_tickers
    ):
        """Test backtest optimized quantity calculation."""
        manager = AllocatorManager(
            data_fetcher=mock_data_fetcher,
            backtest_runner=mock_backtest_runner,
            logging_manager=mock_logging_manager,
        )

        # Mock the backtest allocator's calculate_quantities method
        expected_result = {"AAPL": 300, "GOOGL": 10, "MSFT": 100}
        manager.backtest_allocator.calculate_quantities = MagicMock(return_value=expected_result)

        result = manager.calculate_backtest_optimized_quantities(sample_config, sample_tickers)

        assert result == expected_result
        manager.backtest_allocator.calculate_quantities.assert_called_once_with(sample_config, sample_tickers)

    def test_get_calculation_prices(self, mock_data_fetcher, mock_backtest_runner, mock_logging_manager, sample_config):
        """Test getting calculation prices."""
        manager = AllocatorManager(
            data_fetcher=mock_data_fetcher,
            backtest_runner=mock_backtest_runner,
            logging_manager=mock_logging_manager,
        )

        symbols = ["AAPL", "GOOGL", "MSFT"]
        result = manager.get_calculation_prices(sample_config, symbols, use_start_date=True)

        # Should use the mocked prices
        assert result == {"AAPL": 150.0, "GOOGL": 2800.0, "MSFT": 300.0}

    def test_validate_allocation_config_backtest_optimized(
        self, mock_data_fetcher, mock_backtest_runner, mock_logging_manager
    ):
        """Test validation for backtest optimized allocation."""
        manager = AllocatorManager(
            data_fetcher=mock_data_fetcher,
            backtest_runner=mock_backtest_runner,
            logging_manager=mock_logging_manager,
        )

        config = MagicMock()
        config.portfolio = MagicMock()
        config.portfolio.allocation_method = "backtest_optimized"
        config.backtest_optimization = None

        with pytest.raises(ValueError, match="backtest_optimization configuration is required"):
            manager.validate_allocation_config(config)

    def test_validate_allocation_config_auto(self, mock_data_fetcher, mock_backtest_runner, mock_logging_manager):
        """Test validation for auto allocation."""
        manager = AllocatorManager(
            data_fetcher=mock_data_fetcher,
            backtest_runner=mock_backtest_runner,
            logging_manager=mock_logging_manager,
        )

        config = MagicMock()
        config.portfolio = MagicMock()
        config.portfolio.allocation_method = "auto"
        config.portfolio.category_ratios = None

        with pytest.raises(ValueError, match="category_ratios must be specified"):
            manager.validate_allocation_config(config)

    def test_validate_allocation_config_dynamic(self, mock_data_fetcher, mock_backtest_runner, mock_logging_manager):
        """Test validation for dynamic allocation."""
        manager = AllocatorManager(
            data_fetcher=mock_data_fetcher,
            backtest_runner=mock_backtest_runner,
            logging_manager=mock_logging_manager,
        )

        config = MagicMock()
        config.portfolio = MagicMock()
        config.portfolio.allocation_method = "dynamic"
        config.portfolio.tickers = [
            TickerConfig(symbol="AAPL"),  # No allocation_pct or allocation_amount
            TickerConfig(symbol="GOOGL", allocation_pct=50),
        ]

        with pytest.raises(ValueError, match="Ticker AAPL must have either allocation_pct"):
            manager.validate_allocation_config(config)
