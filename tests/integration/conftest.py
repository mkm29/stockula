"""Integration test fixtures and utilities."""

import time
from unittest.mock import patch

import pytest

from stockula.container import Container
from stockula.data.fetcher import DataFetcher
from stockula.database.manager import DatabaseManager
from stockula.domain import DomainFactory
from stockula.forecasting import StockForecaster
from stockula.utils import LoggingManager


@pytest.fixture(scope="function")
def integration_container():
    """Create a container with all dependencies configured for integration tests."""
    from stockula.config import StockulaConfig

    container = Container()

    # Create real logging manager
    logging_manager = LoggingManager()
    # Create a minimal config for logging setup
    config = StockulaConfig()
    logging_manager.setup(config)

    # Create real database manager (in-memory for speed)
    database_manager = DatabaseManager(":memory:")

    # Create all tables
    from sqlmodel import SQLModel

    SQLModel.metadata.create_all(database_manager.engine)

    # Override container providers
    container.logging_manager.override(logging_manager)
    container.database_manager.override(database_manager)

    # Wire the container to all modules that need it
    container.wire(
        modules=[
            "stockula.main",
            "stockula.data.fetcher",
            "stockula.forecasting.forecaster",
            "stockula.domain.factory",
            "stockula.backtesting.runner",
            "stockula.manager",
        ]
    )

    yield container

    # Cleanup
    database_manager.close()
    container.unwire()


@pytest.fixture(scope="function")
def integration_logging_manager(integration_container):
    """Get the logging manager from the integration container."""
    return integration_container.logging_manager()


@pytest.fixture(scope="function")
def integration_data_fetcher(integration_container, mock_yfinance_ticker):
    """Create a DataFetcher with proper dependency injection for integration tests."""
    with patch("yfinance.Ticker", return_value=mock_yfinance_ticker):
        # Create fetcher with injected dependencies
        fetcher = DataFetcher(use_cache=False, logging_manager=integration_container.logging_manager())

        # Add rate limiting wrapper for yfinance calls
        original_get_stock_data = fetcher.get_stock_data

        def rate_limited_get_stock_data(*args, **kwargs):
            """Wrapper to add rate limiting to prevent hitting yfinance limits."""
            time.sleep(0.1)  # 100ms delay between calls
            return original_get_stock_data(*args, **kwargs)

        fetcher.get_stock_data = rate_limited_get_stock_data

        yield fetcher

        # Cleanup
        fetcher.close()


@pytest.fixture(scope="function")
def integration_stock_forecaster(integration_container, integration_data_fetcher):
    """Create a StockForecaster with proper dependency injection for integration tests."""
    forecaster = StockForecaster(
        forecast_length=7,
        frequency="D",
        data_fetcher=integration_data_fetcher,
        logging_manager=integration_container.logging_manager(),
    )
    return forecaster


@pytest.fixture(scope="function")
def integration_domain_factory(integration_container, integration_data_fetcher):
    """Create a DomainFactory with proper dependency injection for integration tests."""
    factory = DomainFactory(fetcher=integration_data_fetcher, logging_manager=integration_container.logging_manager())
    return factory


@pytest.fixture(scope="function")
def rate_limiter():
    """Provide a rate limiter for yfinance API calls."""

    class RateLimiter:
        def __init__(self, min_interval=0.5):
            self.min_interval = min_interval
            self.last_call = 0

        def wait(self):
            """Wait if necessary to maintain minimum interval between calls."""
            current_time = time.time()
            elapsed = current_time - self.last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call = time.time()

    return RateLimiter()


@pytest.fixture(autouse=True)
def mock_yfinance_download():
    """Mock yfinance.download to prevent real API calls in integration tests."""
    import numpy as np
    import pandas as pd

    def mock_download(tickers, start=None, end=None, **kwargs):
        """Create realistic mock data for yfinance.download."""
        # Generate mock data
        if isinstance(tickers, str):
            tickers = [tickers]

        # Create date range
        if start and end:
            dates = pd.date_range(start=start, end=end, freq="D")
        else:
            dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq="D")

        # Generate mock OHLCV data
        data = {}
        for ticker in tickers:
            np.random.seed(hash(ticker) % 2**32)
            base_price = 100 + np.random.uniform(-50, 200)
            returns = np.random.normal(0.0005, 0.02, len(dates))
            close_prices = base_price * np.exp(np.cumsum(returns))

            ticker_data = pd.DataFrame(
                {
                    "Open": close_prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
                    "High": close_prices * (1 + np.random.uniform(0, 0.02, len(dates))),
                    "Low": close_prices * (1 + np.random.uniform(-0.02, 0, len(dates))),
                    "Close": close_prices,
                    "Volume": np.random.randint(1000000, 50000000, len(dates)),
                    "Adj Close": close_prices,
                },
                index=dates,
            )

            # Ensure High >= Close >= Low
            ticker_data["High"] = ticker_data[["Open", "High", "Close"]].max(axis=1)
            ticker_data["Low"] = ticker_data[["Open", "Low", "Close"]].min(axis=1)

            if len(tickers) == 1:
                return ticker_data
            else:
                for col in ticker_data.columns:
                    data[(col, ticker)] = ticker_data[col]

        # Create multi-index columns for multiple tickers
        if len(tickers) > 1:
            result = pd.DataFrame(data, index=dates)
            result.columns = pd.MultiIndex.from_tuples(result.columns)
            return result

        return ticker_data

    with patch("yfinance.download", side_effect=mock_download):
        yield
