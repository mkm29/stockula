"""Shared pytest fixtures for the test suite."""

import pytest
from datetime import datetime
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from stockula.config import (
    StockulaConfig,
    PortfolioConfig,
    TickerConfig,
    DataConfig,
    BacktestConfig,
    ForecastConfig,
)
from stockula.domain import Portfolio, Asset, Category, DomainFactory
from stockula.data.fetcher import DataFetcher
from stockula.database.manager import DatabaseManager
from stockula.container import Container
from stockula.utils import LoggingManager
from stockula.backtesting import BacktestRunner
from stockula.forecasting import StockForecaster


# ===== Configuration Fixtures =====


@pytest.fixture
def sample_ticker_config():
    """Create a sample ticker configuration."""
    return TickerConfig(
        symbol="AAPL",
        quantity=10.0,
        sector="Technology",
        market_cap=3000.0,
        category="MOMENTUM",
    )


@pytest.fixture
def sample_ticker_configs():
    """Create multiple ticker configurations for testing."""
    return [
        TickerConfig(symbol="AAPL", quantity=10.0, category="MOMENTUM"),
        TickerConfig(symbol="GOOGL", quantity=5.0, category="GROWTH"),
        TickerConfig(symbol="SPY", quantity=20.0, category="INDEX"),
        TickerConfig(symbol="NVDA", quantity=8.0, category="SPECULATIVE"),
    ]


@pytest.fixture
def sample_portfolio_config(sample_ticker_configs):
    """Create a sample portfolio configuration."""
    return PortfolioConfig(
        name="Test Portfolio",
        initial_capital=100000.0,
        allocation_method="equal_weight",
        tickers=sample_ticker_configs,
        max_position_size=25.0,
        stop_loss_pct=10.0,
    )


@pytest.fixture
def dynamic_allocation_config():
    """Create a portfolio config with dynamic allocation."""
    return PortfolioConfig(
        name="Dynamic Portfolio",
        initial_capital=50000.0,
        allocation_method="dynamic",
        dynamic_allocation=True,
        tickers=[
            TickerConfig(symbol="AAPL", allocation_amount=15000, category="MOMENTUM"),
            TickerConfig(symbol="GOOGL", allocation_pct=20.0, category="GROWTH"),
            TickerConfig(symbol="SPY", allocation_amount=20000, category="INDEX"),
        ],
    )


@pytest.fixture
def auto_allocation_config():
    """Create a portfolio config with auto allocation."""
    return PortfolioConfig(
        name="Auto Portfolio",
        initial_capital=100000.0,
        allocation_method="auto",
        auto_allocate=True,
        allow_fractional_shares=True,
        category_ratios={"INDEX": 0.35, "MOMENTUM": 0.40, "SPECULATIVE": 0.25},
        capital_utilization_target=0.95,
        tickers=[
            TickerConfig(symbol="SPY", category="INDEX"),
            TickerConfig(symbol="QQQ", category="INDEX"),
            TickerConfig(symbol="AAPL", category="MOMENTUM"),
            TickerConfig(symbol="NVDA", category="MOMENTUM"),
            TickerConfig(symbol="TSLA", category="SPECULATIVE"),
        ],
    )


@pytest.fixture
def sample_data_config():
    """Create a sample data configuration."""
    return DataConfig(start_date="2023-01-01", end_date="2023-12-31", interval="1d")


@pytest.fixture
def sample_stockula_config(sample_portfolio_config, sample_data_config):
    """Create a complete Stockula configuration."""
    return StockulaConfig(
        portfolio=sample_portfolio_config,
        data=sample_data_config,
        backtest=BacktestConfig(
            initial_cash=10000.0, commission=0.002, hold_only_categories=["INDEX"]
        ),
        forecast=ForecastConfig(forecast_length=30, model_list="fast"),
    )


# ===== Domain Model Fixtures =====


@pytest.fixture
def sample_ticker():
    """Create a sample ticker."""
    # Import the wrapper function from domain
    from stockula.domain import Ticker

    return Ticker(
        symbol="AAPL",
        sector="Technology",
        market_cap=3000.0,
        category=Category.MOMENTUM,
    )


@pytest.fixture
def sample_asset(sample_ticker):
    """Create a sample asset."""
    return Asset(ticker=sample_ticker, quantity=10.0, category=Category.MOMENTUM)


@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio."""
    return Portfolio(
        name="Test Portfolio",
        initial_capital=100000.0,
        allocation_method="equal_weight",
    )


@pytest.fixture
def populated_portfolio(sample_portfolio, sample_ticker_configs, mock_data_fetcher):
    """Create a portfolio with multiple assets."""
    factory = DomainFactory(fetcher=mock_data_fetcher)
    config = StockulaConfig(
        portfolio=PortfolioConfig(
            name="Test Portfolio",
            initial_capital=100000.0,
            allocation_method="equal_weight",
            tickers=sample_ticker_configs,
        )
    )
    return factory.create_portfolio(config)


# ===== Data Fixtures =====


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-01-31", freq="D")
    data = pd.DataFrame(
        {
            "Open": [150.0 + i * 0.5 for i in range(len(dates))],
            "High": [152.0 + i * 0.5 for i in range(len(dates))],
            "Low": [149.0 + i * 0.5 for i in range(len(dates))],
            "Close": [151.0 + i * 0.5 for i in range(len(dates))],
            "Volume": [1000000 + i * 10000 for i in range(len(dates))],
        },
        index=dates,
    )
    return data


@pytest.fixture
def sample_prices():
    """Create a sample price dictionary."""
    return {
        "AAPL": 150.0,
        "GOOGL": 120.0,
        "SPY": 450.0,
        "NVDA": 500.0,
        "TSLA": 200.0,
        "QQQ": 380.0,
    }


@pytest.fixture
def mock_yfinance_ticker():
    """Create a mock yfinance Ticker object."""
    mock_ticker = Mock()

    # Mock info property
    mock_ticker.info = {
        "longName": "Apple Inc.",
        "sector": "Technology",
        "marketCap": 3000000000000,
        "currentPrice": 150.0,
        "previousClose": 149.0,
        "volume": 50000000,
    }

    # Mock history method
    def mock_history(**kwargs):
        period = kwargs.get("period", "1mo")
        interval = kwargs.get("interval", "1d")

        if period == "1mo":
            dates = pd.date_range(end=datetime.now(), periods=22, freq="D")
        else:
            dates = pd.date_range(end=datetime.now(), periods=5, freq="D")

        return pd.DataFrame(
            {
                "Open": [150.0 + i * 0.5 for i in range(len(dates))],
                "High": [152.0 + i * 0.5 for i in range(len(dates))],
                "Low": [149.0 + i * 0.5 for i in range(len(dates))],
                "Close": [151.0 + i * 0.5 for i in range(len(dates))],
                "Volume": [1000000 + i * 10000 for i in range(len(dates))],
            },
            index=dates,
        )

    mock_ticker.history = mock_history

    return mock_ticker


@pytest.fixture
def mock_data_fetcher(mock_yfinance_ticker, sample_prices):
    """Create a mock DataFetcher."""
    with patch("stockula.data.fetcher.yf.Ticker") as mock_yf_ticker:
        mock_yf_ticker.return_value = mock_yfinance_ticker

        fetcher = DataFetcher(use_cache=False)

        # Mock get_current_prices to return our sample prices
        original_get_current_prices = fetcher.get_current_prices

        def mock_get_current_prices(symbols):
            if isinstance(symbols, str):
                symbols = [symbols]
            return {s: sample_prices.get(s, 100.0) for s in symbols}

        fetcher.get_current_prices = mock_get_current_prices

        yield fetcher


# ===== Database Fixtures =====


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path."""
    return str(tmp_path / "test_stockula.db")


@pytest.fixture
def test_database(temp_db_path):
    """Create a test database instance."""
    db = DatabaseManager(temp_db_path)
    yield db
    # Cleanup is automatic with tmp_path


@pytest.fixture
def populated_database(test_database, sample_ohlcv_data):
    """Create a database with sample data."""
    # Add stock info
    test_database.store_stock_info(
        "AAPL",
        {
            "longName": "Apple Inc.",
            "sector": "Technology",
            "marketCap": 3000000000000,
        },
    )

    # Add price history
    test_database.store_price_history("AAPL", sample_ohlcv_data, "1d")

    return test_database


# ===== File System Fixtures =====


@pytest.fixture
def temp_config_file(tmp_path, sample_stockula_config):
    """Create a temporary config file."""
    import yaml

    config_path = tmp_path / "test_config.yaml"

    config_dict = sample_stockula_config.model_dump()
    # Convert dates to strings for YAML serialization
    if config_dict["data"]["start_date"]:
        config_dict["data"]["start_date"] = config_dict["data"]["start_date"].strftime(
            "%Y-%m-%d"
        )
    if config_dict["data"]["end_date"]:
        config_dict["data"]["end_date"] = config_dict["data"]["end_date"].strftime(
            "%Y-%m-%d"
        )

    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    return str(config_path)


@pytest.fixture
def mock_env_variables(monkeypatch):
    """Set up mock environment variables."""
    monkeypatch.setenv("STOCKULA_CONFIG_FILE", "test_config.yaml")
    monkeypatch.setenv("STOCKULA_DEBUG", "true")
    monkeypatch.setenv("STOCKULA_LOG_LEVEL", "DEBUG")


# ===== Strategy Testing Fixtures =====


@pytest.fixture
def backtest_data():
    """Create data suitable for backtesting."""
    # Create 100 days of data with some trend
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

    # Create trending data with some noise
    trend = pd.Series(range(len(dates)), index=dates) * 0.5
    noise = pd.Series(
        [(-1) ** i * (i % 5) * 0.2 for i in range(len(dates))], index=dates
    )
    base_price = 100.0

    close_prices = base_price + trend + noise

    data = pd.DataFrame(
        {
            "Open": close_prices - 0.5,
            "High": close_prices + 1.0,
            "Low": close_prices - 1.0,
            "Close": close_prices,
            "Volume": [1000000] * len(dates),
        },
        index=dates,
    )

    return data


@pytest.fixture
def forecast_data():
    """Create data suitable for forecasting."""
    # Create 365 days of historical data with seasonality
    dates = pd.date_range(end=datetime.now(), periods=365, freq="D")

    # Add trend, seasonality, and noise
    trend = pd.Series(range(len(dates)), index=dates) * 0.1
    seasonal = pd.Series(
        [10 * np.sin(2 * np.pi * i / 30) for i in range(len(dates))], index=dates
    )
    noise = pd.Series(np.random.normal(0, 2, len(dates)), index=dates)

    values = 100 + trend + seasonal + noise

    return pd.DataFrame({"Close": values}, index=dates)


# ===== Container Fixtures =====


@pytest.fixture
def mock_container(mock_data_fetcher):
    """Create a mock container with all dependencies mocked."""
    container = Container()
    
    # Mock all dependencies
    mock_logging_manager = Mock(spec=LoggingManager)
    mock_logging_manager.setup = Mock()
    mock_logging_manager.info = Mock()
    mock_logging_manager.debug = Mock()
    mock_logging_manager.warning = Mock()
    mock_logging_manager.error = Mock()
    
    mock_database_manager = Mock(spec=DatabaseManager)
    
    # Use the existing mock_data_fetcher
    
    mock_domain_factory = Mock(spec=DomainFactory)
    mock_domain_factory.fetcher = mock_data_fetcher
    
    mock_backtest_runner = Mock(spec=BacktestRunner)
    mock_backtest_runner.data_fetcher = mock_data_fetcher
    
    mock_stock_forecaster = Mock(spec=StockForecaster)
    mock_stock_forecaster.data_fetcher = mock_data_fetcher
    
    # Override container providers with mocks
    container.logging_manager.override(mock_logging_manager)
    container.database_manager.override(mock_database_manager)
    container.data_fetcher.override(mock_data_fetcher)
    container.domain_factory.override(mock_domain_factory)
    container.backtest_runner.override(Mock(return_value=mock_backtest_runner))
    container.stock_forecaster.override(Mock(return_value=mock_stock_forecaster))
    
    # Wire the container
    container.wire(modules=["stockula.main"])
    
    return container


# ===== Cleanup Fixtures =====


@pytest.fixture(autouse=True)
def cleanup_singleton():
    """Clean up singleton instances between tests."""
    from stockula.domain.ticker import TickerRegistry

    # Reset the ticker registry singleton
    TickerRegistry._instances = {}
    yield
    TickerRegistry._instances = {}
