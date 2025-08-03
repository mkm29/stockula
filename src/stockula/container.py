"""Dependency injection container for Stockula."""

import threading

from dependency_injector import containers, providers

from .allocation import Allocator, BacktestOptimizedAllocator
from .backtesting.runner import BacktestRunner
from .config import load_config
from .data.fetcher import DataFetcher
from .database.manager import DatabaseManager
from .domain.factory import DomainFactory
from .forecasting.forecaster import StockForecaster
from .technical_analysis.indicators import TechnicalIndicators
from .utils.logging_manager import LoggingManager


class Container(containers.DeclarativeContainer):
    """Main dependency injection container for Stockula.

    Thread-safe singleton providers are used for shared components
    to ensure proper synchronization in multi-threading environments.
    """

    # Thread synchronization lock for singleton providers
    _lock = threading.RLock()

    # Configuration
    config = providers.Configuration()

    # Config file path
    config_path = providers.Object(None)

    # Logger - thread-safe singleton
    logging_manager = providers.ThreadSafeSingleton(LoggingManager, name="stockula")

    # Stockula configuration - thread-safe singleton
    stockula_config = providers.ThreadSafeSingleton(
        lambda config_path: load_config(config_path),
        config_path=config_path,
    )

    # Database manager - thread-safe singleton
    database_manager = providers.ThreadSafeSingleton(
        DatabaseManager,
        db_path=providers.Callable(lambda config: config.data.db_path, stockula_config),
    )

    # Data fetcher with injected database manager and logging - thread-safe singleton
    data_fetcher = providers.ThreadSafeSingleton(
        DataFetcher,
        use_cache=providers.Callable(lambda config: config.data.use_cache, stockula_config),
        db_path=providers.Callable(lambda config: config.data.db_path, stockula_config),
        database_manager=database_manager,
        logging_manager=logging_manager,
    )

    # Allocator - thread-safe singleton
    allocator = providers.ThreadSafeSingleton(Allocator, fetcher=data_fetcher, logging_manager=logging_manager)

    # Domain factory - thread-safe singleton
    domain_factory = providers.ThreadSafeSingleton(
        DomainFactory, config=stockula_config, fetcher=data_fetcher, allocator=allocator
    )

    # Backtesting runner
    backtest_runner = providers.Factory(
        BacktestRunner,
        cash=providers.Callable(lambda config: config.backtest.initial_cash, stockula_config),
        commission=providers.Callable(lambda config: config.backtest.commission, stockula_config),
        broker_config=providers.Callable(lambda config: config.backtest.broker_config, stockula_config),
        data_fetcher=data_fetcher,
    )

    # Backtest-optimized allocator - thread-safe singleton
    backtest_allocator = providers.ThreadSafeSingleton(
        BacktestOptimizedAllocator,
        fetcher=data_fetcher,
        logging_manager=logging_manager,
        backtest_runner=backtest_runner,
    )

    # Stock forecaster
    stock_forecaster = providers.Factory(
        StockForecaster,
        forecast_length=providers.Callable(lambda config: config.forecast.forecast_length, stockula_config),
        frequency=providers.Callable(lambda config: config.forecast.frequency, stockula_config),
        model_list=providers.Callable(lambda config: config.forecast.model_list, stockula_config),
        prediction_interval=providers.Callable(lambda config: config.forecast.prediction_interval, stockula_config),
        data_fetcher=data_fetcher,
        logging_manager=logging_manager,
    )

    # Technical indicators factory
    technical_indicators = providers.Factory(TechnicalIndicators)


def create_container(config_path: str | None = None) -> Container:
    """Create and configure the DI container.

    Args:
        config_path: Path to configuration file

    Returns:
        Configured container instance
    """
    container = Container()

    if config_path:
        container.config_path.override(config_path)

    # Wire the container to modules that need it
    container.wire(
        modules=[
            "stockula.main",
            "stockula.allocation.allocator",
            # Note: backtest_allocator doesn't need wiring as it doesn't use @inject
            "stockula.data.fetcher",
            "stockula.domain.factory",
            "stockula.domain.portfolio",
            "stockula.forecasting.forecaster",
        ]
    )

    return container
