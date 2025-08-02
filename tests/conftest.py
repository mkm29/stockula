"""Shared pytest fixtures for the test suite."""

import glob
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest


# pytest-xdist support
def pytest_configure(config):
    """Configure pytest with xdist-specific markers."""
    config.addinivalue_line("markers", "xdist_group(name): mark test to run in the same xdist worker")


def pytest_sessionfinish(session, exitstatus):
    """Clean up temporary coverage files after test session completes."""
    # Only run cleanup on the main process (not on workers)
    if hasattr(session.config, "workerinput"):
        return

    # Get the root directory
    root_dir = Path(__file__).parent.parent

    # Find all temporary coverage files
    coverage_pattern = str(root_dir / ".coverage.*")
    temp_coverage_files = glob.glob(coverage_pattern)

    # Combine coverage data if coverage is installed and was used
    if temp_coverage_files and session.config.getoption("--cov", default=None):
        try:
            import coverage

            cov = coverage.Coverage(data_file=str(root_dir / ".coverage"))
            cov.combine(data_paths=temp_coverage_files, strict=True)
            cov.save()

            # Clean up temporary files after combining
            for temp_file in temp_coverage_files:
                try:
                    os.remove(temp_file)
                except OSError:
                    pass  # Ignore errors if file is already deleted

        except Exception:
            # If coverage combine fails, still try to clean up
            for temp_file in temp_coverage_files:
                try:
                    os.remove(temp_file)
                except OSError:
                    pass


@pytest.fixture(scope="session")
def worker_id(request):
    """Get the xdist worker id for parallel test execution."""
    return request.config.workerinput.get("workerid", "master") if hasattr(request.config, "workerinput") else "master"


from stockula.backtesting import BacktestRunner
from stockula.config import BacktestConfig, DataConfig, ForecastConfig, PortfolioConfig, StockulaConfig, TickerConfig
from stockula.container import Container
from stockula.data.fetcher import DataFetcher
from stockula.database.manager import DatabaseManager
from stockula.domain import Asset, Category, DomainFactory, Portfolio
from stockula.forecasting import StockForecaster
from stockula.utils import LoggingManager

# ===== Configuration Fixtures =====


@pytest.fixture(scope="session")
def sample_ticker_config():
    """Create a sample ticker configuration."""
    return TickerConfig(
        symbol="AAPL",
        quantity=10.0,
        sector="Technology",
        market_cap=3000.0,
        category="MOMENTUM",
    )


@pytest.fixture(scope="session")
def sample_ticker_configs():
    """Create multiple ticker configurations for testing."""
    return [
        TickerConfig(symbol="AAPL", quantity=10.0, category="MOMENTUM"),
        TickerConfig(symbol="GOOGL", quantity=5.0, category="GROWTH"),
        TickerConfig(symbol="SPY", quantity=20.0, category="INDEX"),
        TickerConfig(symbol="NVDA", quantity=8.0, category="SPECULATIVE"),
    ]


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def sample_data_config():
    """Create a sample data configuration."""
    return DataConfig(start_date="2023-01-01", end_date="2023-12-31", interval="1d")


@pytest.fixture(scope="session")
def sample_stockula_config(sample_portfolio_config, sample_data_config):
    """Create a complete Stockula configuration."""
    return StockulaConfig(
        portfolio=sample_portfolio_config,
        data=sample_data_config,
        backtest=BacktestConfig(initial_cash=10000.0, commission=0.002, hold_only_categories=["INDEX"]),
        forecast=ForecastConfig(forecast_length=30, model_list="fast"),
    )


# ===== Domain Model Fixtures =====


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="function")
def sample_asset(sample_ticker):
    """Create a sample asset."""
    return Asset(ticker_init=sample_ticker, quantity_init=10.0, category_init=Category.MOMENTUM)


@pytest.fixture
def mock_logging_manager():
    """Create a mock logging manager."""
    logger = Mock()
    logger.debug = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    return logger


@pytest.fixture(scope="function")
def sample_portfolio(mock_logging_manager):
    """Create a sample portfolio."""
    return Portfolio(
        name_init="Test Portfolio",
        initial_capital_init=100000.0,
        allocation_method_init="equal_weight",
        logging_manager_init=mock_logging_manager,
    )


@pytest.fixture(scope="function")
def populated_portfolio(sample_ticker_configs, mock_data_fetcher, mock_logging_manager):
    """Create a portfolio with multiple assets."""
    factory = DomainFactory(fetcher=mock_data_fetcher, logging_manager=mock_logging_manager)
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


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="module")
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
        kwargs.get("interval", "1d")

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


@pytest.fixture(scope="function")
def mock_data_fetcher(mock_yfinance_ticker, sample_prices):
    """Create a mock DataFetcher."""
    with patch("stockula.data.fetcher.yf.Ticker") as mock_yf_ticker:
        mock_yf_ticker.return_value = mock_yfinance_ticker

        fetcher = DataFetcher(use_cache=False)

        # Mock get_current_prices to return our sample prices
        def mock_get_current_prices(symbols, _show_progress=True):
            if isinstance(symbols, str):
                symbols = [symbols]
            return {s: sample_prices.get(s, 100.0) for s in symbols}

        fetcher.get_current_prices = mock_get_current_prices

        # Mock get_treasury_rates to return sample rates
        def mock_get_treasury_rates(start_date=None, end_date=None, duration="3_month"):
            # Return a simple Series with sample treasury rates
            dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
            rates = pd.Series([0.05] * len(dates), index=dates)  # 5% rate
            return rates

        fetcher.get_treasury_rates = mock_get_treasury_rates

        yield fetcher

        # Close the fetcher to clean up database connections
        fetcher.close()


# ===== Database Fixtures =====


@pytest.fixture(scope="session")
def test_db_path(tmp_path_factory, worker_id):
    """Create a persistent test database path for the session.

    When using pytest-xdist, each worker gets its own database file.
    """
    if worker_id == "master":
        # Not running in parallel mode
        db_path = tmp_path_factory.mktemp("data") / "test_stockula.db"
    else:
        # Running with pytest-xdist, create unique db per worker
        db_path = tmp_path_factory.mktemp(f"data_{worker_id}") / "test_stockula.db"
    return str(db_path)


@pytest.fixture(scope="session")
def test_database_session(test_db_path):
    """Create a test database instance for the entire test session.

    This fixture:
    1. Creates a test database in tests/data/test_stockula.db
    2. Creates all tables using SQLModel
    3. Seeds the database with test data
    4. Provides the database for all tests
    5. Cleans up after all tests are complete
    """
    import os
    from pathlib import Path

    # Set test environment to skip migrations
    os.environ["PYTEST_CURRENT_TEST"] = "true"

    # Remove existing test database if it exists
    db_file = Path(test_db_path)
    if db_file.exists():
        db_file.unlink()

    # Create database manager
    db = DatabaseManager(test_db_path)

    # Create all tables
    from sqlmodel import SQLModel

    SQLModel.metadata.create_all(db.engine)

    # Seed database with test data
    _seed_test_database(db)

    yield db

    # Close database connections
    db.close()

    # Cleanup: Remove test database after all tests
    if db_file.exists():
        db_file.unlink()


def _seed_test_database(db: DatabaseManager):
    """Seed the test database with sample data."""
    from datetime import date, timedelta

    import pandas as pd

    # Test stocks to seed
    test_stocks = [
        {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "market_cap": 3000000000000,
            "exchange": "NASDAQ",
            "currency": "USD",
        },
        {
            "symbol": "GOOGL",
            "name": "Alphabet Inc.",
            "sector": "Technology",
            "industry": "Internet Services",
            "market_cap": 2000000000000,
            "exchange": "NASDAQ",
            "currency": "USD",
        },
        {
            "symbol": "MSFT",
            "name": "Microsoft Corporation",
            "sector": "Technology",
            "industry": "Software",
            "market_cap": 2500000000000,
            "exchange": "NASDAQ",
            "currency": "USD",
        },
        {
            "symbol": "TSLA",
            "name": "Tesla Inc.",
            "sector": "Consumer Cyclical",
            "industry": "Auto Manufacturers",
            "market_cap": 800000000000,
            "exchange": "NASDAQ",
            "currency": "USD",
        },
        {
            "symbol": "SPY",
            "name": "SPDR S&P 500 ETF",
            "sector": "ETF",
            "industry": "Large Cap Blend",
            "market_cap": 400000000000,
            "exchange": "NYSE",
            "currency": "USD",
        },
    ]

    # Add stocks
    for stock_data in test_stocks:
        db.store_stock_info(stock_data["symbol"], stock_data)

    # Generate price history for each stock
    base_date = date.today() - timedelta(days=365)
    dates = pd.date_range(start=base_date, periods=365, freq="D")

    base_prices = {
        "AAPL": 150.0,
        "GOOGL": 120.0,
        "MSFT": 300.0,
        "TSLA": 200.0,
        "SPY": 400.0,
    }

    for symbol, base_price in base_prices.items():
        # Create realistic price movement
        np.random.seed(hash(symbol) % 2**32)  # Consistent random data per symbol

        # Generate price data with trend and volatility
        returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
        price_series = base_price * np.exp(np.cumsum(returns))

        # Create OHLCV data
        price_data = pd.DataFrame(
            {
                "Open": price_series * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
                "High": price_series * (1 + np.random.uniform(0, 0.02, len(dates))),
                "Low": price_series * (1 + np.random.uniform(-0.02, 0, len(dates))),
                "Close": price_series,
                "Volume": np.random.randint(1000000, 50000000, len(dates)),
            },
            index=dates,
        )

        # Ensure High >= Close >= Low
        price_data["High"] = price_data[["Open", "High", "Close"]].max(axis=1)
        price_data["Low"] = price_data[["Open", "Low", "Close"]].min(axis=1)

        db.store_price_history(symbol, price_data, "1d")

        # Add some dividends for dividend-paying stocks
        if symbol in ["AAPL", "MSFT"]:
            dividend_dates = pd.date_range(start=base_date, periods=4, freq="QE")
            dividend_amounts = [0.22, 0.23, 0.24, 0.25]
            # Filter dates that are within our data range
            # Convert to date for comparison
            last_date = dates[-1].date() if hasattr(dates[-1], "date") else dates[-1]
            valid_dates = [d for d in dividend_dates if d.date() <= last_date]
            if valid_dates:
                dividends = pd.Series(dividend_amounts[: len(valid_dates)], index=valid_dates)
                db.store_dividends(symbol, dividends)

        # Add a stock split for AAPL
        if symbol == "AAPL":
            split_date_ts = pd.Timestamp(base_date + timedelta(days=180))
            last_date = dates[-1].date() if hasattr(dates[-1], "date") else dates[-1]
            if split_date_ts.date() <= last_date:
                splits = pd.Series([4.0], index=[split_date_ts])  # 4:1 split
                db.store_splits(symbol, splits)


@pytest.fixture(scope="function")
def temp_db_path(tmp_path):
    """Create a temporary database path."""
    return str(tmp_path / "test_stockula.db")


@pytest.fixture(scope="function")
def in_memory_database():
    """Create an in-memory SQLite database for tests.

    This is much faster than file-based databases and provides
    complete isolation between tests.
    """
    import os

    # Set test environment to skip migrations
    os.environ["PYTEST_CURRENT_TEST"] = "true"

    # Create in-memory database
    db = DatabaseManager(":memory:")

    # Create all tables
    from sqlmodel import SQLModel

    SQLModel.metadata.create_all(db.engine)

    # Seed with minimal test data if needed
    _seed_test_database(db)

    yield db

    # Close the database even for in-memory
    db.close()


@pytest.fixture(scope="function")
def test_database(test_database_session):
    """Create a test database instance for individual tests.

    This uses the session-scoped database but provides isolation
    between tests by using transactions.
    """
    # For now, just return the session database
    # In the future, we could add transaction rollback here
    yield test_database_session


@pytest.fixture(scope="function")
def populated_database(test_database):
    """Return the test database which is already populated with data.

    The test_database fixture already contains seeded data from the
    session-scoped test_database_session fixture.
    """
    return test_database


# ===== File System Fixtures =====


@pytest.fixture(scope="function")
def temp_config_file(tmp_path, sample_stockula_config):
    """Create a temporary config file."""
    import yaml

    config_path = tmp_path / "test_config.yaml"

    config_dict = sample_stockula_config.model_dump()
    # Convert dates to strings for YAML serialization
    if config_dict["data"]["start_date"]:
        config_dict["data"]["start_date"] = config_dict["data"]["start_date"].strftime("%Y-%m-%d")
    if config_dict["data"]["end_date"]:
        config_dict["data"]["end_date"] = config_dict["data"]["end_date"].strftime("%Y-%m-%d")

    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    return str(config_path)


@pytest.fixture(scope="function")
def mock_env_variables(monkeypatch):
    """Set up mock environment variables."""
    monkeypatch.setenv("STOCKULA_CONFIG_FILE", "test_config.yaml")
    monkeypatch.setenv("STOCKULA_DEBUG", "true")
    monkeypatch.setenv("STOCKULA_LOG_LEVEL", "DEBUG")


# ===== Strategy Testing Fixtures =====


@pytest.fixture(scope="module")
def backtest_data():
    """Create data suitable for backtesting."""
    # Create 100 days of data with some trend
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

    # Create trending data with some noise
    trend = pd.Series(range(len(dates)), index=dates) * 0.5
    noise = pd.Series([(-1) ** i * (i % 5) * 0.2 for i in range(len(dates))], index=dates)
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


@pytest.fixture(scope="function")
def forecast_data():
    """Create data suitable for forecasting."""
    # Create 365 days of historical data with seasonality
    dates = pd.date_range(end=datetime.now(), periods=365, freq="D")

    # Add trend, seasonality, and noise
    trend = pd.Series(range(len(dates)), index=dates) * 0.1
    seasonal = pd.Series([10 * np.sin(2 * np.pi * i / 30) for i in range(len(dates))], index=dates)
    noise = pd.Series(np.random.normal(0, 2, len(dates)), index=dates)

    values = 100 + trend + seasonal + noise

    return pd.DataFrame({"Close": values}, index=dates)


# ===== Container Fixtures =====


@pytest.fixture(scope="function")
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


@pytest.fixture(scope="function", autouse=True)
def cleanup_singleton():
    """Clean up singleton instances between tests."""
    from stockula.domain.ticker import TickerRegistry

    # Reset the ticker registry singleton
    TickerRegistry._instances = {}
    yield
    TickerRegistry._instances = {}
