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
    """Create forecasting backend (AutoGluon if available, otherwise Simple)."""

    forecast_length = config.forecast_length if config.forecast_length is not None else 7
    requested_models = getattr(config, "models", None)
    is_chronos_requested = _is_chronos_requested(requested_models)

    if is_chronos_requested and AUTOGLUON_AVAILABLE:
        return _create_autogluon_chronos_backend(config, forecast_length, requested_models)

    if is_chronos_requested and CHRONOS_AVAILABLE and _is_chronos_runtime_ready():
        return _create_chronos_backend(config, forecast_length, requested_models)

    if AUTOGLUON_AVAILABLE:
        return _create_autogluon_backend(config, forecast_length, requested_models)

    _warn_if_no_autogluon(logging_manager)
    return _create_simple_backend(config, forecast_length)


def _is_chronos_requested(requested_models):
    if isinstance(requested_models, str):
        return requested_models.lower() == "zero_shot"
    if isinstance(requested_models, list):
        return any(str(m).lower() == "chronos" for m in requested_models)
    return False


def _is_chronos_runtime_ready():
    try:
        import importlib

        importlib.import_module("chronos")
        importlib.import_module("torch")
        return True
    except Exception:
        return False


def _create_autogluon_chronos_backend(config, forecast_length, requested_models):
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


def _create_chronos_backend(config, forecast_length, requested_models):
    model_name = next(
        (
            m
            for m in (requested_models if isinstance(requested_models, list) else [])
            if isinstance(m, str) and m.startswith("amazon/chronos-")
        ),
        None,
    )
    return ChronosBackend(
        forecast_length=forecast_length,
        frequency=config.frequency,
        prediction_interval=config.prediction_interval,
        no_negatives=config.no_negatives,
        model_name=model_name,
    )


def _create_autogluon_backend(config, forecast_length, requested_models):
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


def _warn_if_no_autogluon(logging_manager):
    if hasattr(logging_manager, "__class__") and logging_manager.__class__.__name__ == "Provide":
        return
    logging_manager.warning(
        "AutoGluon not available (requires Python < 3.13). Using simple linear regression for forecasting."
    )


def _create_simple_backend(config, forecast_length):
    return SimpleForecastBackend(
        forecast_length=forecast_length,
        frequency=config.frequency,
        prediction_interval=config.prediction_interval,
        no_negatives=config.no_negatives,
    )
