"""Factory for creating forecasting backends."""

from typing import TYPE_CHECKING

from dependency_injector.wiring import Provide, inject

from ..interfaces import ILoggingManager
from .backends import AutoGluonBackend, AutoTSBackend, ForecastBackend

if TYPE_CHECKING:
    from ..config import ForecastConfig


@inject
def create_forecast_backend(
    config: "ForecastConfig",
    logging_manager: ILoggingManager = Provide["logging_manager"],
) -> ForecastBackend:
    """Create a forecasting backend based on configuration.

    Args:
        config: Forecast configuration
        logging_manager: Injected logging manager

    Returns:
        Configured forecasting backend

    Raises:
        ValueError: If backend is not supported
    """
    backend_name = config.backend.lower()

    if backend_name == "autots":
        return AutoTSBackend(
            forecast_length=config.forecast_length,
            frequency=config.frequency,
            prediction_interval=config.prediction_interval,
            ensemble=config.ensemble,
            num_validations=config.num_validations,
            validation_method=config.validation_method,
            model_list=config.model_list,
            max_generations=config.max_generations,
            no_negatives=config.no_negatives,
            logging_manager=logging_manager,
        )
    elif backend_name == "autogluon":
        return AutoGluonBackend(
            forecast_length=config.forecast_length,
            frequency=config.frequency,
            prediction_interval=config.prediction_interval,
            preset=config.preset,
            time_limit=config.time_limit,
            eval_metric=config.eval_metric,
            no_negatives=config.no_negatives,
            logging_manager=logging_manager,
        )
    else:
        raise ValueError(f"Unsupported forecasting backend: {backend_name}. Supported: 'autots', 'autogluon'")
