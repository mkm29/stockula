"""AutoTS backend for time series forecasting."""

import signal
import sys
import time
from typing import Any

import pandas as pd
from autots import AutoTS
from dependency_injector.wiring import Provide, inject
from rich.progress import Progress, SpinnerColumn, TextColumn

from ...cli_manager import cli_manager
from ...interfaces import ILoggingManager
from ..forecaster import suppress_autots_output
from .base import ForecastBackend, ForecastResult


class AutoTSBackend(ForecastBackend):
    """AutoTS backend implementation for time series forecasting."""

    # Financial-appropriate models optimized for stock data
    FINANCIAL_MODEL_LIST = [
        "LastValueNaive",
        "AverageValueNaive",
        "SeasonalNaive",
        "ARIMA",
        "ETS",
        "DynamicFactor",
        "VAR",
        "UnivariateRegression",
        "MultivariateRegression",
        "WindowRegression",
        "DatepartRegression",
        "NVAR",
        "Theta",
    ]

    ULTRA_FAST_MODEL_LIST = [
        "LastValueNaive",
        "AverageValueNaive",
        "SeasonalNaive",
    ]

    CLEAN_MODEL_LIST = [
        "LastValueNaive",
        "AverageValueNaive",
        "SeasonalNaive",
        "ETS",
        "ARIMA",
        "UnivariateRegression",
        "WindowRegression",
        "Theta",
    ]

    FAST_MODEL_LIST = ULTRA_FAST_MODEL_LIST + [
        "ZeroesNaive",
        "ETS",
        "WindowRegression",
        "UnivariateRegression",
        "Theta",
        "ARIMA",
    ]

    @inject
    def __init__(
        self,
        forecast_length: int | None = None,
        frequency: str = "infer",
        prediction_interval: float = 0.95,
        ensemble: str | None = "auto",
        num_validations: int = 2,
        validation_method: str = "backwards",
        model_list: str | list[str] = "fast",
        max_generations: int = 5,
        no_negatives: bool = True,
        logging_manager: ILoggingManager = Provide["logging_manager"],
        **kwargs,
    ):
        """Initialize the AutoTS backend.

        Args:
            forecast_length: Number of periods to forecast
            frequency: Data frequency ('infer' to detect automatically)
            prediction_interval: Confidence interval for predictions (0-1)
            ensemble: Ensemble method or None
            num_validations: Number of validation splits
            validation_method: Validation method ('backwards', 'even', 'similarity')
            model_list: List of models or preset
            max_generations: Maximum generations for model evolution
            no_negatives: Constraint predictions to be non-negative
            logging_manager: Injected logging manager
            **kwargs: Additional parameters
        """
        super().__init__(
            forecast_length=forecast_length,
            frequency=frequency,
            prediction_interval=prediction_interval,
            no_negatives=no_negatives,
            logging_manager=logging_manager,
        )
        self.ensemble = ensemble
        self.num_validations = num_validations
        self.validation_method = validation_method
        self.model_list = model_list
        self.max_generations = max_generations
        self.prediction = None

    def _get_model_list(self, model_list: str | list[str]) -> list[str] | str:
        """Get the appropriate model list based on input.

        Args:
            model_list: String preset or list of model names

        Returns:
            List of model names or string preset for AutoTS
        """
        # If it's already a list, return as-is
        if isinstance(model_list, list):
            return model_list

        # Check if it's an AutoTS built-in preset
        autots_builtins = [
            "fast",
            "superfast",
            "default",
            "parallel",
            "fast_parallel",
            "scalable",
            "probabilistic",
            "multivariate",
            "univariate",
            "all",
            "no_shared",
            "motifs",
            "regressor",
            "best",
            "slow",
            "gpu",
        ]
        if model_list in autots_builtins:
            self.logger.info(f"Using AutoTS built-in preset '{model_list}'")
            return model_list

        # Map our custom presets to model lists
        preset_map = {
            "ultra_fast": self.ULTRA_FAST_MODEL_LIST,
            "clean": self.CLEAN_MODEL_LIST,
            "financial": self.FINANCIAL_MODEL_LIST,
            "fast_financial": [m for m in self.FAST_MODEL_LIST if m in self.FINANCIAL_MODEL_LIST],
        }

        if model_list in preset_map:
            models = preset_map[model_list]
            self.logger.info(f"Using {model_list} model list ({len(models)} models)")
            return models

        # If not recognized, pass through and let AutoTS handle it
        return model_list

    def fit(
        self,
        data: pd.DataFrame,
        target_column: str = "Close",
        show_progress: bool = True,
        model_list: str | list[str] | None = None,
        ensemble: str | None = None,
        max_generations: int | None = None,
        **kwargs,
    ) -> "AutoTSBackend":
        """Fit the AutoTS model on historical data.

        Args:
            data: DataFrame with time series data (index should be DatetimeIndex)
            target_column: Column to forecast
            show_progress: Whether to show progress bar
            model_list: Override default model list
            ensemble: Override default ensemble method
            max_generations: Override default max generations
            **kwargs: Additional parameters

        Returns:
            Self for chaining
        """
        # Validate input
        self.validate_input(data, target_column)

        # Use provided parameters or defaults
        model_list_to_use = model_list if model_list is not None else self.model_list
        ensemble = ensemble if ensemble is not None else self.ensemble
        max_generations = max_generations or self.max_generations

        # Get actual model list
        actual_model_list = self._get_model_list(model_list_to_use)

        # Prepare data for AutoTS
        data_for_model = data[[target_column]].copy()
        data_for_model = data_for_model.reset_index()
        data_for_model.columns = ["date", target_column]

        self.logger.debug(f"Fitting AutoTS model on {len(data_for_model)} data points")
        self.logger.debug(f"Date range: {data_for_model['date'].min()} to {data_for_model['date'].max()}")

        try:
            # Set signal handler for graceful interruption
            def signal_handler(sig, frame):
                print("Forecasting interrupted by user.")
                sys.exit(0)

            signal.signal(signal.SIGINT, signal_handler)

            if show_progress:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=cli_manager.get_console(),
                    transient=True,
                ) as progress:
                    task = progress.add_task("[cyan]Training AutoTS models on historical data...", total=None)

                    # Use a default forecast length if None
                    forecast_length_to_use = self.forecast_length if self.forecast_length is not None else 14

                    # Try to infer frequency from data if set to 'infer'
                    freq_to_use = self.frequency
                    if self.frequency == "infer":
                        try:
                            inferred_freq = pd.infer_freq(data_for_model["date"])
                            freq_to_use = inferred_freq if inferred_freq else "D"
                        except Exception:
                            freq_to_use = "D"

                    self.model = AutoTS(
                        forecast_length=forecast_length_to_use,
                        frequency=freq_to_use,
                        prediction_interval=self.prediction_interval,
                        ensemble=ensemble,
                        model_list=actual_model_list,
                        max_generations=max_generations,
                        num_validations=self.num_validations,
                        validation_method=self.validation_method,
                        verbose=0,
                        no_negatives=self.no_negatives,
                        drop_most_recent=0,
                        n_jobs="auto",
                        model_interrupt=False,
                    )

                    # Fit the model with suppressed output
                    with suppress_autots_output():
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(
                                self.model.fit,
                                data_for_model,
                                date_col="date",
                                value_col=target_column,
                                id_col=None,
                            )

                            # Update progress while waiting
                            dots = ["", ".", "..", "..."]
                            dot_index = 0
                            while not future.done():
                                desc = f"[cyan]Training AutoTS models on historical data{dots[dot_index]}"
                                progress.update(task, description=desc)
                                dot_index = (dot_index + 1) % len(dots)
                                time.sleep(0.2)

                            result = future.result()
                            if result is not None:
                                self.model = result

                        progress.update(task, description="[green]✓ AutoTS model training completed")
            else:
                # No progress display
                forecast_length_to_use = self.forecast_length if self.forecast_length is not None else 14
                freq_to_use = self.frequency
                if self.frequency == "infer":
                    try:
                        inferred_freq = pd.infer_freq(data_for_model["date"])
                        freq_to_use = inferred_freq if inferred_freq else "D"
                    except Exception:
                        freq_to_use = "D"

                self.model = AutoTS(
                    forecast_length=forecast_length_to_use,
                    frequency=freq_to_use,
                    prediction_interval=self.prediction_interval,
                    ensemble=ensemble,
                    model_list=actual_model_list,
                    max_generations=max_generations,
                    num_validations=self.num_validations,
                    validation_method=self.validation_method,
                    verbose=1 if self.logger.isEnabledFor(10) else 0,
                    no_negatives=self.no_negatives,
                    drop_most_recent=0,
                    n_jobs="auto",
                    model_interrupt=False,
                )

                # Fit directly
                result = self.model.fit(
                    data_for_model,
                    date_col="date",
                    value_col=target_column,
                    id_col=None,
                )
                if result is not None:
                    self.model = result

            self.is_fitted = True
            self.logger.debug("AutoTS model fitting completed")

        except KeyboardInterrupt:
            self.logger.warning("Model fitting interrupted by user.")
            raise
        except Exception as e:
            self.logger.error(f"Error during model fitting: {str(e)}")
            raise

        return self

    def predict(self, **kwargs) -> ForecastResult:
        """Generate predictions using fitted AutoTS model.

        Returns:
            ForecastResult with predictions and metadata
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        self.logger.debug("Generating AutoTS predictions...")

        with suppress_autots_output():
            self.prediction = self.model.predict()

        forecast = self.prediction.forecast
        upper_forecast = self.prediction.upper_forecast
        lower_forecast = self.prediction.lower_forecast

        # Combine into single DataFrame
        result_df = pd.DataFrame(
            {
                "forecast": forecast.iloc[:, 0],
                "lower_bound": lower_forecast.iloc[:, 0],
                "upper_bound": upper_forecast.iloc[:, 0],
            }
        )

        self.logger.debug(f"Generated {len(result_df)} forecast points")

        # Get model info
        model_info = self.get_model_info()

        return ForecastResult(
            forecast=result_df,
            model_name=model_info["model_name"],
            model_params=model_info["model_params"],
            metadata={
                "transformation": model_info.get("model_transformation"),
                "accuracy": model_info.get("model_accuracy"),
            },
        )

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the fitted AutoTS model.

        Returns:
            Dictionary with model information
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return {
            "model_name": self.model.best_model_name,
            "model_params": self.model.best_model_params,
            "model_transformation": self.model.best_model_transformation_params,
            "model_accuracy": getattr(self.model, "best_model_accuracy", "N/A"),
        }

    def get_available_models(self) -> list[str]:
        """Get list of available AutoTS models.

        Returns:
            List of model names
        """
        # Return a comprehensive list of AutoTS models
        return [
            "LastValueNaive",
            "AverageValueNaive",
            "SeasonalNaive",
            "ZeroesNaive",
            "ARIMA",
            "ETS",
            "GLM",
            "ARDL",
            "DynamicFactor",
            "VAR",
            "VECM",
            "UnivariateRegression",
            "MultivariateRegression",
            "WindowRegression",
            "DatepartRegression",
            "UnivariateMotif",
            "MultivariateMotif",
            "NVAR",
            "Theta",
            "FBProphet",
            "GLS",
            # Deep learning models (require additional dependencies)
            "RollingRegression",
            "PytorchForecasting",
            "Cassandra",
        ]

    def get_available_model_lists(self) -> dict[str, list[str]]:
        """Get available model lists and their descriptions.

        Returns:
            Dictionary of model list names and their models
        """
        return {
            "ultra_fast": self.ULTRA_FAST_MODEL_LIST,
            "fast": self.FAST_MODEL_LIST,
            "clean": self.CLEAN_MODEL_LIST,
            "financial": self.FINANCIAL_MODEL_LIST,
            "fast_financial": [m for m in self.FAST_MODEL_LIST if m in self.FINANCIAL_MODEL_LIST],
        }

    def create_financial_forecaster(
        self,
        forecast_length: int | None = None,
        frequency: str = "infer",
        prediction_interval: float = 0.95,
    ) -> "AutoTSBackend":
        """Create a new AutoTSBackend instance optimized for financial forecasting.

        Args:
            forecast_length: Number of periods to forecast
            frequency: Data frequency
            prediction_interval: Confidence interval for predictions

        Returns:
            New AutoTSBackend instance configured for financial data
        """
        return AutoTSBackend(
            forecast_length=forecast_length or self.forecast_length,
            frequency=frequency,
            prediction_interval=prediction_interval,
            model_list="financial",
            ensemble="distance",  # Better for financial data
            num_validations=3,  # More validation for robustness
            max_generations=self.max_generations,
            no_negatives=True,  # Stock prices can't be negative
            logging_manager=self.logger,
        )
