"""Factory for creating forecasting backends."""

from typing import TYPE_CHECKING

from dependency_injector.wiring import Provide, inject

from ..interfaces import ILoggingManager
from .backends import AUTOGLUON_AVAILABLE, AutoGluonBackend, ForecastBackend, SimpleForecastBackend

if TYPE_CHECKING:
    from ..config import ForecastConfig


@inject
def create_forecast_backend(
    config: "ForecastConfig",
    logging_manager: ILoggingManager = Provide["logging_manager"],
) -> ForecastBackend:
    """Create forecasting backend (AutoGluon if available, otherwise Simple).

    Args:
        config: Forecast configuration
        logging_manager: Injected logging manager

    Returns:
        Configured forecasting backend
    """
    # Use default forecast_length of 7 if not specified
    forecast_length = config.forecast_length if config.forecast_length is not None else 7

    if AUTOGLUON_AVAILABLE:
        return AutoGluonBackend(
            forecast_length=forecast_length,
            frequency=config.frequency,
            prediction_interval=config.prediction_interval,
            preset=config.preset,
            time_limit=config.time_limit,
            eval_metric=config.eval_metric,
            no_negatives=config.no_negatives,
        )
    else:
        # Fall back to simple backend if AutoGluon is not available
        logging_manager.warning(
            "AutoGluon not available (requires Python < 3.13). Using simple linear regression for forecasting."
        )
        return SimpleForecastBackend(
            forecast_length=forecast_length,
            frequency=config.frequency,
            prediction_interval=config.prediction_interval,
            no_negatives=config.no_negatives,
        )
