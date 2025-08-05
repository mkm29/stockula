"""Dependency injection container for Stockula."""

import threading

from dependency_injector import containers, providers

from .allocation import Allocator, AllocatorManager, BacktestOptimizedAllocator
from .backtesting import BacktestingManager
from .backtesting.runner import BacktestRunner
from .config import load_config
from .data.manager import DataManager
from .database.manager import DatabaseManager
from .domain.factory import DomainFactory
from .forecasting import ForecastingManager, StockForecaster
from .technical_analysis import TechnicalAnalysisManager, TechnicalIndicators
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

    # Data manager - thread-safe singleton
    data_manager = providers.ThreadSafeSingleton(
        DataManager,
        db_manager=database_manager,
        logging_manager=logging_manager,
        use_cache=providers.Callable(lambda config: config.data.use_cache, stockula_config),
        db_path=providers.Callable(lambda config: config.data.db_path, stockula_config),
    )

    # Data fetcher extracted from data manager - thread-safe singleton
    data_fetcher = providers.ThreadSafeSingleton(
        lambda data_mgr: data_mgr.fetcher,
        data_mgr=data_manager,
    )

    # Allocator - thread-safe singleton
    allocator = providers.ThreadSafeSingleton(Allocator, fetcher=data_fetcher, logging_manager=logging_manager)

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

    # Allocator manager - thread-safe singleton
    allocator_manager = providers.ThreadSafeSingleton(
        AllocatorManager,
        data_fetcher=data_fetcher,
        backtest_runner=backtest_runner,
        logging_manager=logging_manager,
    )

    # Forecasting manager - thread-safe singleton
    forecasting_manager = providers.ThreadSafeSingleton(
        ForecastingManager,
        data_fetcher=data_fetcher,
        logging_manager=logging_manager,
    )

    # Technical analysis manager - thread-safe singleton
    technical_analysis_manager = providers.ThreadSafeSingleton(
        TechnicalAnalysisManager,
        data_fetcher=data_fetcher,
        logging_manager=logging_manager,
    )

    # Backtesting manager - thread-safe singleton
    backtesting_manager = providers.ThreadSafeSingleton(
        BacktestingManager,
        data_fetcher=data_fetcher,
        logging_manager=logging_manager,
    )

    # Domain factory - thread-safe singleton
    domain_factory = providers.ThreadSafeSingleton(
        DomainFactory, config=stockula_config, fetcher=data_fetcher, allocator_manager=allocator_manager
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
            "stockula.allocation.manager",
            # Note: backtest_allocator doesn't need wiring as it doesn't use @inject
            "stockula.data.fetcher",
            "stockula.data.manager",
            "stockula.domain.factory",
            "stockula.domain.portfolio",
            "stockula.forecasting.forecaster",
            "stockula.forecasting.manager",
            "stockula.technical_analysis.manager",
            "stockula.backtesting.manager",
        ]
    )

    # Initialize data manager to set up registry
    container.data_manager()

    return container
