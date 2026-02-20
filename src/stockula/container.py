"""Dependency injection container for Stockula."""

import threading
from typing import Any

from dependency_injector import containers, providers

from .allocation import Allocator, AllocatorManager, BacktestOptimizedAllocator
from .backtesting import BacktestingManager
from .backtesting.runner import BacktestRunner
from .config import load_config
from .data.fetcher import DataFetcher
from .data.manager import DataManager
from .database.manager import DatabaseManager
from .domain.factory import DomainFactory
from .forecasting import ForecastingManager
from .technical_analysis import TechnicalAnalysisManager, TechnicalIndicators
from .utils.logging_manager import LoggingManager


def _get_db_path(config: Any) -> str:
    """Extract db_path from configuration with a typed signature."""
    return str(config.data.db_path)


def _get_use_cache(config: Any) -> bool:
    """Extract use_cache from configuration with a typed signature."""
    return bool(config.data.use_cache)


def _get_data_fetcher(data_mgr: DataManager) -> DataFetcher:
    """Extract data fetcher from data manager with a typed signature."""
    return data_mgr.fetcher


def _get_strategy_repository(data_mgr: DataManager) -> Any:
    """Extract strategy repository from data manager with a typed signature."""
    return data_mgr.strategies


def _get_initial_cash(config: Any) -> float:
    """Extract initial_cash from configuration with a typed signature."""
    return float(config.backtest.initial_cash)


def _get_commission(config: Any) -> float:
    """Extract commission from configuration with a typed signature."""
    return float(config.backtest.commission)


def _get_broker_config(config: Any) -> Any:
    """Extract broker_config from configuration with a typed signature."""
    return config.backtest.broker_config


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
    config_path: providers.Provider[str | None] = providers.Object[str | None](None)

    # Logger - thread-safe singleton
    logging_manager = providers.ThreadSafeSingleton(LoggingManager, name="stockula")

    # Stockula configuration - thread-safe singleton
    stockula_config = providers.ThreadSafeSingleton(
        load_config,
        config_path=config_path,
    )

    # Database manager - thread-safe singleton
    database_manager = providers.ThreadSafeSingleton(
        DatabaseManager,
        db_path=providers.Callable(_get_db_path, stockula_config),
    )

    # Data manager - thread-safe singleton
    data_manager = providers.ThreadSafeSingleton(
        DataManager,
        db_manager=database_manager,
        logging_manager=logging_manager,
        use_cache=providers.Callable(_get_use_cache, stockula_config),
        db_path=providers.Callable(_get_db_path, stockula_config),
    )

    # Data fetcher extracted from data manager - thread-safe singleton
    data_fetcher = providers.ThreadSafeSingleton(
        _get_data_fetcher,
        data_mgr=data_manager,
    )

    # Strategy repository - thread-safe singleton via DataManager
    strategy_repository = providers.ThreadSafeSingleton(
        _get_strategy_repository,
        data_mgr=data_manager,
    )

    # Allocator - thread-safe singleton
    allocator = providers.ThreadSafeSingleton(Allocator, fetcher=data_fetcher, logging_manager=logging_manager)

    # Backtesting runner
    backtest_runner = providers.Factory(
        BacktestRunner,
        cash=providers.Callable(_get_initial_cash, stockula_config),
        commission=providers.Callable(_get_commission, stockula_config),
        broker_config=providers.Callable(_get_broker_config, stockula_config),
        data_fetcher=data_fetcher,
    )

    # Backtest-optimized allocator - thread-safe singleton
    backtest_allocator = providers.ThreadSafeSingleton(
        BacktestOptimizedAllocator,
        fetcher=data_fetcher,
        logging_manager=logging_manager,
        backtest_runner=backtest_runner,
    )

    # Forecasting manager - thread-safe singleton (defined before allocator_manager)
    forecasting_manager = providers.ThreadSafeSingleton(
        ForecastingManager,
        data_fetcher=data_fetcher,
        logging_manager=logging_manager,
    )

    # Allocator manager - thread-safe singleton (now with forecasting_manager)
    allocator_manager = providers.ThreadSafeSingleton(
        AllocatorManager,
        data_fetcher=data_fetcher,
        backtest_runner=backtest_runner,
        logging_manager=logging_manager,
        forecast_manager=forecasting_manager,
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
        strategy_repository=strategy_repository,
    )

    # Domain factory - thread-safe singleton
    domain_factory = providers.ThreadSafeSingleton(
        DomainFactory,
        config=stockula_config,
        fetcher=data_fetcher,
        allocator_manager=allocator_manager,
        logging_manager=logging_manager,
    )

    # Stock forecaster (removed - using ForecastingManager instead)

    # Technical indicators factory
    technical_indicators = providers.Factory(TechnicalIndicators)


def create_container(config_path: str | None = None) -> Container:
    """Create and configure the DI container.

    Args:
        config_path: Path to configuration file

    Returns:
        Configured container instance
    """
    container = Container(config_path=providers.Object(config_path))

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
            "stockula.forecasting.manager",
            "stockula.forecasting.factory",
            "stockula.forecasting.backends.base",
            "stockula.technical_analysis.manager",
            "stockula.backtesting.manager",
        ]
    )

    # Initialize data manager to set up registry
    container.data_manager()

    return container
