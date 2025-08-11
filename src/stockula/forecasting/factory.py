"""Factory for creating forecasting backends."""

from typing import TYPE_CHECKING, cast

from dependency_injector.wiring import Provide, inject

from ..interfaces import ILoggingManager
from .backends import (
    AUTOGLUON_AVAILABLE,
    CHRONOS_AVAILABLE,
    AutoGluonBackend,
    ChronosBackend,
    ForecastBackend,
    SimpleForecastBackend,
)

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

    # Prefer AutoGluon+Chronos integration when explicitly requested via models setting
    # Accept models="zero_shot" or models list containing "Chronos"
    requested_models = getattr(config, "models", None)
    is_chronos_requested = (isinstance(requested_models, str) and requested_models.lower() == "zero_shot") or (
        isinstance(requested_models, list) and any(str(m).lower() == "chronos" for m in requested_models)
    )

    if is_chronos_requested and AUTOGLUON_AVAILABLE:
        # Use AutoGluon with Chronos model to leverage covariates/ensembling
        return cast(
            ForecastBackend,
            AutoGluonBackend(
                forecast_length=forecast_length,
                frequency=config.frequency,
                prediction_interval=config.prediction_interval,
                preset=config.preset,
                models=requested_models if requested_models is not None else "zero_shot",
                time_limit=config.time_limit,
                eval_metric=config.eval_metric,
                no_negatives=config.no_negatives,
                use_calendar_covariates=getattr(config, "use_calendar_covariates", True),
                past_covariate_columns=getattr(config, "past_covariate_columns", None),
            ),
        )

    if is_chronos_requested and CHRONOS_AVAILABLE:
        # Ensure runtime prerequisites are present (chronos + torch) for direct Chronos
        try:
            import importlib

            importlib.import_module("chronos")
            importlib.import_module("torch")
            chronos_ready = True
        except Exception:
            chronos_ready = False

        if chronos_ready:
            return ChronosBackend(
                forecast_length=forecast_length,
                frequency=config.frequency,
                prediction_interval=config.prediction_interval,
                no_negatives=config.no_negatives,
                model_name=next(
                    (
                        m
                        for m in (requested_models if isinstance(requested_models, list) else [])
                        if isinstance(m, str) and m.startswith("amazon/chronos-")
                    ),
                    None,
                ),
            )

    if AUTOGLUON_AVAILABLE:
        return cast(
            ForecastBackend,
            AutoGluonBackend(
                forecast_length=forecast_length,
                frequency=config.frequency,
                prediction_interval=config.prediction_interval,
                preset=config.preset,
                models=requested_models,
                time_limit=config.time_limit,
                eval_metric=config.eval_metric,
                no_negatives=config.no_negatives,
                use_calendar_covariates=getattr(config, "use_calendar_covariates", True),
                past_covariate_columns=getattr(config, "past_covariate_columns", None),
            ),
        )
    else:
        # Fall back to simple backend if AutoGluon is not available
        # Check if logging_manager is properly injected
        if hasattr(logging_manager, "warning"):
            logging_manager.warning(
                "AutoGluon not available (requires Python < 3.13). Using simple linear regression for forecasting."
            )
        return cast(
            ForecastBackend,
            SimpleForecastBackend(
                forecast_length=forecast_length,
                frequency=config.frequency,
                prediction_interval=config.prediction_interval,
                no_negatives=config.no_negatives,
            ),
        )
