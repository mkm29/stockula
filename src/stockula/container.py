"""Dependency injection container for Stockula."""

from dependency_injector import containers, providers

from .utils.logging_manager import LoggingManager
from .config import StockulaConfig, load_config
from .database.manager import DatabaseManager
from .data.fetcher import DataFetcher
from .domain.factory import DomainFactory
from .technical_analysis.indicators import TechnicalIndicators
from .backtesting.runner import BacktestRunner
from .forecasting.forecaster import StockForecaster


class Container(containers.DeclarativeContainer):
    """Main dependency injection container for Stockula."""

    # Configuration
    config = providers.Configuration()

    # Config file path
    config_path = providers.Object(None)

    # Logger
    logging_manager = providers.Singleton(LoggingManager, name="stockula")

    # Stockula configuration
    stockula_config = providers.Singleton(
        lambda config_path: load_config(config_path)
        if config_path
        else StockulaConfig(),
        config_path=config_path,
    )

    # Database manager
    database_manager = providers.Singleton(
        DatabaseManager,
        db_path=providers.Callable(lambda config: config.data.db_path, stockula_config),
    )

    # Data fetcher with injected database manager
    data_fetcher = providers.Singleton(
        DataFetcher,
        use_cache=providers.Callable(
            lambda config: config.data.use_cache, stockula_config
        ),
        db_path=providers.Callable(lambda config: config.data.db_path, stockula_config),
        database_manager=database_manager,
    )

    # Domain factory
    domain_factory = providers.Singleton(
        DomainFactory, config=stockula_config, fetcher=data_fetcher
    )

    # Backtesting runner
    backtest_runner = providers.Factory(
        BacktestRunner,
        cash=providers.Callable(
            lambda config: config.backtest.initial_cash, stockula_config
        ),
        commission=providers.Callable(
            lambda config: config.backtest.commission, stockula_config
        ),
        broker_config=providers.Callable(
            lambda config: config.backtest.broker_config, stockula_config
        ),
        data_fetcher=data_fetcher,
    )

    # Stock forecaster
    stock_forecaster = providers.Factory(
        StockForecaster,
        forecast_length=providers.Callable(
            lambda config: config.forecast.forecast_length, stockula_config
        ),
        model_list=providers.Callable(
            lambda config: config.forecast.model_list, stockula_config
        ),
        prediction_interval=providers.Callable(
            lambda config: config.forecast.prediction_interval, stockula_config
        ),
        data_fetcher=data_fetcher,
    )

    # Technical indicators factory
    technical_indicators = providers.Factory(TechnicalIndicators)


def create_container(config_path: str = None) -> Container:
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
        ]
    )

    return container
