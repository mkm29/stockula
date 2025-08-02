"""Stock price forecasting using AutoTS - cleaned version without parallel processing."""

import os
import signal
import sys
import time
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import pandas as pd
from autots import AutoTS
from dependency_injector.wiring import Provide, inject
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..interfaces import ILoggingManager

console = Console()

if TYPE_CHECKING:
    from ..data import DataFetcher

# Set up warning suppression for AutoTS
warnings.filterwarnings("ignore", category=UserWarning, module="autots")
warnings.filterwarnings("ignore", category=FutureWarning, module="autots")
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
warnings.filterwarnings("ignore", message=".*DataConversionWarning.*")
warnings.filterwarnings("ignore", message=".*SVD did not converge.*")


class SuppressAutoTSOutput:
    """Context manager to suppress AutoTS verbose output and warnings."""

    def __init__(self):
        """Initialize the output suppressor."""
        self.original_stdout = None
        self.original_stderr = None
        self.null_file = None

    def __enter__(self):
        """Redirect stdout and stderr to suppress output."""
        # Configure warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*is not available.*")
        warnings.filterwarnings("ignore", message=".*SVD did not converge.*")
        warnings.filterwarnings("ignore", message=".*Ill-conditioned matrix.*")
        warnings.filterwarnings("ignore", message=".*invalid value encountered.*")
        warnings.filterwarnings("ignore", message=".*divide by zero.*")
        warnings.filterwarnings("ignore", message=".*overflow encountered.*")
        warnings.filterwarnings("ignore", message="Template Eval Error:")
        warnings.filterwarnings("ignore", message="Model Number:")
        warnings.filterwarnings("ignore", message="Ensembling")
        warnings.filterwarnings("ignore", message="Data frequency is:")

        # Suppress matplotlib font cache warnings
        warnings.filterwarnings("ignore", message="Matplotlib is building the font cache")
        # Ignore prophet plotly warnings
        warnings.filterwarnings("ignore", message="Importing plotly failed")

        # Prophet warnings
        warnings.filterwarnings("ignore", message=".*Optimization terminated abnormally.*", module="prophet")

        # Suppress numerical warnings from sklearn, numpy, and statsmodels
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

        # Suppress sklearn specific warnings
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*X does not have valid feature names.*",
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*X has feature names.*",
        )

        # Sklearn convergence warnings
        warnings.filterwarnings(
            "ignore",
            message=".*ConvergenceWarning.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=".*did not converge.*",
        )

        # Ignore GLM warnings
        warnings.filterwarnings(
            "ignore",
            message=".*DomainWarning.*",
        )

        # Statsmodels warnings
        warnings.filterwarnings(
            "ignore",
            message=".*maximum likelihood optimization failed.*",
        )
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")

        # ARDL model warnings
        warnings.filterwarnings(
            "ignore",
            message=".*divide by zero encountered in matmul.*",
            module="statsmodels.tsa.ardl",
        )
        warnings.filterwarnings(
            "ignore",
            message=".*overflow encountered in matmul.*",
            module="statsmodels.tsa.ardl",
        )
        warnings.filterwarnings(
            "ignore",
            message=".*invalid value encountered in matmul.*",
            module="statsmodels.tsa.ardl",
        )

        # Joblib warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
        warnings.filterwarnings("ignore", message=".*resource_tracker.*")

        # Data conversion warnings
        warnings.filterwarnings("ignore", message=".*Data was converted to boolean.*")
        warnings.filterwarnings("ignore", message=".*DataConversionWarning.*")

        # Pandas warnings
        warnings.filterwarnings("ignore", message=".*DataFrame.interpolate.*")
        warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

        # AutoTS specific patterns
        warnings.filterwarnings("ignore", message=".*failed validation.*")
        warnings.filterwarnings("ignore", message=".*New Generation.*")

        # Open null device
        self.null_file = open(os.devnull, "w")

        # Store original streams
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        # Create custom stream that filters AutoTS output
        class FilteredStream:
            def __init__(self, original_stream):
                self.original_stream = original_stream

            def write(self, s):
                # Completely suppress empty strings
                if not s or s == "\n":
                    return len(s)

                # Only filter non-debug messages
                if True:  # Always filter for now since we can't access logger here
                    # List of patterns to suppress
                    suppress_patterns = [
                        "Model Number:",
                        "New Generation:",
                        "Validation Round:",
                        "Ensembling",
                        "Data frequency is:",
                        "Model Index:",
                        "Template Eval Error:",
                        "Ensemble Weights:",
                        "Model Weights:",
                        "with avg smape",
                        "📈",
                        "AutoTS",
                        "failed validation",
                        # Suppress warnings from AutoTS
                        "UserWarning:",
                        "RuntimeWarning:",
                        "FutureWarning:",
                        "DomainWarning:",
                        "DataConversionWarning:",
                        "ConvergenceWarning:",
                        # Suppress model info
                        "Univariate",
                        "Multivariate",
                        # Suppress sklearn warnings
                        "sklearn",
                        "Convergence",
                        "resource_tracker:",
                        # Suppress error stack traces we handle
                        "Traceback",
                        "ValueError:",
                        "KeyError:",
                        "during handling",
                        # Suppress ensembling details
                        "Ensembling Error:",
                        "interpolating",
                        "SVD did not converge",
                        # Date outputs from AutoTS
                        "2025-",
                        "2024-",
                        "2023-",
                        "2022-",
                        "2021-",
                        "2020-",
                        # Prophet warnings
                        "Optimization terminated abnormally",
                        "Falling back to Newton",
                        "WARNI",
                        "prophet.models",
                        # Statsmodels ARDL warnings
                        "divide by zero encountered in matmul",
                        "overflow encountered in matmul",
                        "invalid value encountered in matmul",
                        "No anomalies detected",
                    ]

                    if any(pattern in s for pattern in suppress_patterns):
                        return len(s)

                    # Let other non-empty messages through
                    self.original_stream.write(s)
                else:
                    # In debug mode - only filter template eval errors
                    if "Template Eval Error:" not in s:
                        self.original_stream.write(s)
                return len(s)

            def flush(self):
                self.original_stream.flush()

            def __getattr__(self, name):
                # Delegate all other attributes to the original stream
                return getattr(self.original_stream, name)

        # Replace stdout and stderr with filtered versions
        sys.stdout = FilteredStream(self.original_stdout)
        sys.stderr = FilteredStream(self.original_stderr)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original stdout and stderr."""
        # Restore original streams
        if self.original_stdout:
            sys.stdout = self.original_stdout
        if self.original_stderr:
            sys.stderr = self.original_stderr

        # Close null file
        if self.null_file:
            self.null_file.close()

        # Reset warnings to default
        warnings.resetwarnings()


@contextmanager
def suppress_autots_output():
    """Context manager to suppress AutoTS output."""
    with SuppressAutoTSOutput():
        yield


class StockForecaster:
    """Stock price forecaster using AutoTS."""

    # Define financial-appropriate models that avoid problematic GLM configurations
    # These models are specifically chosen to work well with stock price data and avoid:
    # - GLM models with Gamma distribution and InversePower link (causes DomainWarning)
    # - Models that require specific data distributions not typical in financial data
    # - Models prone to numerical instability with financial time series
    FINANCIAL_MODEL_LIST = [
        "LastValueNaive",  # Simple but effective for financial data
        "AverageValueNaive",  # Moving average approach
        "SeasonalNaive",  # Captures seasonal patterns
        "ARIMA",  # Classic time series model
        "ETS",  # Exponential smoothing
        "DynamicFactor",  # Good for capturing trends
        "VAR",  # Vector autoregression
        "UnivariateRegression",  # Basic regression models
        "MultivariateRegression",  # Multiple regression
        "WindowRegression",  # Rolling window regression
        "DatepartRegression",  # Date-based features
        # Note: Motif models removed to avoid "k too large" warnings with small datasets
        # "UnivariateMotif",  # Pattern recognition
        # "MultivariateMotif",  # Multi-pattern recognition
        "NVAR",  # Neural VAR
        "Theta",  # Theta method
        # Note: ARDL removed due to numerical instability with financial data
        # "ARDL",  # Autoregressive distributed lag
    ]

    # Ultra-fast models for quick results
    ULTRA_FAST_MODEL_LIST = [
        "LastValueNaive",
        "AverageValueNaive",
        "SeasonalNaive",
    ]

    # Fast models that balance speed and accuracy
    FAST_MODEL_LIST = ULTRA_FAST_MODEL_LIST + [
        "ZeroesNaive",
        "ETS",
        "WindowRegression",
        "UnivariateRegression",
        "Theta",
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
        data_fetcher: "DataFetcher | None" = None,
        logging_manager: ILoggingManager = Provide["logging_manager"],
    ):
        """Initialize the stock forecaster.

        Args:
            forecast_length: Number of periods to forecast (None for evaluation mode)
            frequency: Data frequency ('infer' to detect automatically)
            prediction_interval: Confidence interval for predictions (0-1)
            ensemble: Ensemble method or None
            num_validations: Number of validation splits
            validation_method: Validation method ('backwards', 'even', 'similarity')
            model_list: List of models or preset ('fast', 'superfast', 'parallel', 'statistical',
                'probabilistic', 'multivariate')
            max_generations: Maximum generations for model search
            no_negatives: Constraint predictions to be non-negative
            data_fetcher: Optional DataFetcher instance for retrieving stock data
            logging_manager: Injected logging manager
        """
        self.forecast_length = forecast_length
        self.frequency = frequency
        self.prediction_interval = prediction_interval
        self.ensemble = ensemble
        self.num_validations = num_validations
        self.validation_method = validation_method
        self.model_list = model_list
        self.max_generations = max_generations
        self.no_negatives = no_negatives
        self.data_fetcher = data_fetcher
        self.logger = logging_manager
        self.model = None
        self.prediction = None

    def _get_model_list(self, model_list: str | list[str], target_column: str = "Close") -> list[str] | str:
        """Get the appropriate model list based on input.

        Args:
            model_list: String preset or list of model names
            target_column: Name of the target column for logging

        Returns:
            List of model names or string preset for AutoTS
        """
        if isinstance(model_list, list):
            return model_list

        # Map our presets to model lists
        if model_list == "ultra_fast":
            self.logger.info(
                f"Using ultra-fast model list ({len(self.ULTRA_FAST_MODEL_LIST)} models) for {target_column}"
            )
            return self.ULTRA_FAST_MODEL_LIST
        elif model_list == "financial":
            return self.FINANCIAL_MODEL_LIST
        elif model_list == "fast_financial":
            # Intersection of fast and financial models
            return [m for m in self.FAST_MODEL_LIST if m in self.FINANCIAL_MODEL_LIST]
        else:
            # Use AutoTS built-in presets
            return model_list

    def fit(
        self,
        data: pd.DataFrame,
        target_column: str = "Close",
        model_list: str | list[str] | None = None,
        ensemble: str | None = None,
        max_generations: int | None = None,
        show_progress: bool = True,
    ) -> "StockForecaster":
        """Fit the forecasting model on historical data.

        Args:
            data: DataFrame with time series data (index should be DatetimeIndex)
            target_column: Column to forecast
            model_list: Override default model list
            ensemble: Override default ensemble method
            max_generations: Override default max generations
            show_progress: Whether to show progress bar

        Returns:
            Self for chaining
        """
        # Validate input
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a DatetimeIndex")

        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        # Use provided parameters or defaults
        model_list_to_use = model_list if model_list is not None else self.model_list
        ensemble = ensemble if ensemble is not None else self.ensemble
        max_generations = max_generations or self.max_generations

        # Get actual model list
        actual_model_list = self._get_model_list(model_list_to_use, target_column)
        if actual_model_list is None:
            actual_model_list = self.model_list

        # Prepare data for AutoTS
        data_for_model = data[[target_column]].copy()

        # Reset index to have date as a column
        data_for_model = data_for_model.reset_index()
        data_for_model.columns = ["date", target_column]

        self.logger.debug(f"Fitting model on {len(data_for_model)} data points")
        self.logger.debug(f"Date range: {data_for_model['date'].min()} to {data_for_model['date'].max()}")

        try:
            # Set signal handler for graceful interruption
            def signal_handler(sig, frame):
                print("Forecasting interrupted by user.")
                sys.exit(0)

            signal.signal(signal.SIGINT, signal_handler)

            # Create progress context
            if show_progress:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True,
                ) as progress:
                    task = progress.add_task("[cyan]Training models on historical data...", total=None)

                    # Use a default forecast length if None (will be overridden in evaluation mode)
                    forecast_length_to_use = self.forecast_length if self.forecast_length is not None else 14

                    # Try to infer frequency from data if set to 'infer'
                    freq_to_use = self.frequency
                    if self.frequency == "infer":
                        try:
                            # Try to infer frequency from data
                            inferred_freq = pd.infer_freq(data_for_model["date"])
                            if inferred_freq:
                                freq_to_use = inferred_freq
                            else:
                                # If inference fails, default to 'D' to avoid AutoTS warnings
                                freq_to_use = "D"
                        except Exception:
                            # If inference fails, default to 'D' to avoid warnings
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
                        verbose=0,  # Disable verbose in progress mode
                        no_negatives=True,
                        drop_most_recent=0,
                        n_jobs="auto",
                        constraint=None,
                        drop_data_older_than_periods=None,
                        model_interrupt=False,
                    )

                    model_list_info = (
                        actual_model_list if isinstance(actual_model_list, str) else f"{len(actual_model_list)} models"
                    )
                    self.logger.debug(
                        f"Fitting AutoTS with parameters: "
                        f"model_list={model_list_info}, "
                        f"max_generations={max_generations}"
                    )

                    # Fit the model
                    with suppress_autots_output():
                        # Run fit in a separate thread to allow progress updates
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                            if self.model is None:
                                raise ValueError("Model not initialized")
                            future = executor.submit(
                                self.model.fit,
                                data_for_model,
                                date_col="date",
                                value_col=target_column,
                                id_col=None,
                            )

                            # Update progress while waiting for completion
                            dots = ["", ".", "..", "..."]
                            dot_index = 0

                            while not future.done():
                                desc = f"[cyan]Training models on historical data{dots[dot_index]}"
                                progress.update(task, description=desc)
                                dot_index = (dot_index + 1) % len(dots)
                                time.sleep(0.2)  # Small sleep to control update rate

                            # Get the result (raises any exceptions from the thread)
                            result = future.result()
                            if result is not None:
                                self.model = result

                        complete_msg = "[green]✓ Model training completed"
                        progress.update(task, description=complete_msg)
            else:
                # No progress display (for worker threads)
                n_jobs_value = "auto"

                # Use a default forecast length if None (will be overridden in evaluation mode)
                forecast_length_to_use = self.forecast_length if self.forecast_length is not None else 14

                # Try to infer frequency from data if set to 'infer'
                freq_to_use = self.frequency
                if self.frequency == "infer":
                    try:
                        # Try to infer frequency from data
                        inferred_freq = pd.infer_freq(data_for_model["date"])
                        if inferred_freq:
                            freq_to_use = inferred_freq
                        else:
                            # If inference fails, default to 'D' to avoid AutoTS warnings
                            freq_to_use = "D"
                    except Exception:
                        # If inference fails, default to 'D' to avoid warnings
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
                    verbose=1 if self.logger.isEnabledFor(10) else 0,  # Verbose only in debug mode
                    no_negatives=True,  # Stock prices can't be negative
                    drop_most_recent=0,  # Don't drop recent data
                    n_jobs=n_jobs_value,  # Limit to 1 in worker threads
                    constraint=None,  # No constraint to avoid some errors
                    drop_data_older_than_periods=None,  # Keep all data
                    model_interrupt=False,  # Don't interrupt on errors
                )

                model_list_info = (
                    actual_model_list if isinstance(actual_model_list, str) else f"{len(actual_model_list)} models"
                )
                self.logger.debug(
                    f"Fitting AutoTS with parameters: "
                    f"model_list={model_list_info}, "
                    f"max_generations={max_generations}, n_jobs={n_jobs_value}"
                )

                # Fit the model directly without progress
                if self.model is None:
                    raise ValueError("Model not initialized")
                result = self.model.fit(
                    data_for_model,
                    date_col="date",
                    value_col=target_column,
                    id_col=None,
                )
                if result is not None:
                    self.model = result

            self.logger.debug("Model fitting completed")

        except KeyboardInterrupt:
            self.logger.warning("Model fitting interrupted by user.")
            raise
        except Exception as e:
            self.logger.error(f"Error during model fitting: {str(e)}")
            raise

        return self

    def predict(self) -> pd.DataFrame:
        """Generate predictions using fitted model.

        Returns:
            DataFrame with predictions and confidence intervals
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        self.logger.debug("Generating predictions...")

        with suppress_autots_output():
            self.prediction = self.model.predict()

        forecast = self.prediction.forecast
        self.logger.debug(f"Forecast shape: {forecast.shape}")

        # Get upper and lower bounds
        upper_forecast = self.prediction.upper_forecast
        lower_forecast = self.prediction.lower_forecast

        # Combine into single DataFrame
        result = pd.DataFrame(
            {
                "forecast": forecast.iloc[:, 0],
                "lower_bound": lower_forecast.iloc[:, 0],
                "upper_bound": upper_forecast.iloc[:, 0],
            }
        )

        self.logger.debug(f"Generated {len(result)} forecast points")

        return result

    def fit_predict(self, data: pd.DataFrame, target_column: str = "Close", **kwargs) -> pd.DataFrame:
        """Fit model and generate predictions in one step.

        Args:
            data: DataFrame with time series data
            target_column: Column to forecast
            **kwargs: Additional arguments for fit()

        Returns:
            DataFrame with predictions
        """
        self.fit(data, target_column, **kwargs)
        return self.predict()

    def forecast_from_symbol_with_evaluation(
        self,
        symbol: str,
        train_start_date: str | None = None,
        train_end_date: str | None = None,
        test_start_date: str | None = None,
        test_end_date: str | None = None,
        target_column: str = "Close",
        **kwargs,
    ) -> dict[str, Any]:
        """Forecast stock prices using train/test split with evaluation.

        Args:
            symbol: Stock symbol to forecast
            train_start_date: Start date for training data
            train_end_date: End date for training data
            test_start_date: Start date for testing data
            test_end_date: End date for testing data
            target_column: Column to forecast
            **kwargs: Additional arguments for fit()

        Returns:
            Dictionary with predictions and evaluation metrics
        """
        if self.data_fetcher is None:
            raise ValueError("Data fetcher not configured")

        # Fetch training data
        train_data = self.data_fetcher.get_stock_data(symbol, train_start_date, train_end_date)

        if train_data.empty:
            raise ValueError(f"No training data available for symbol {symbol}")

        # Fetch test data if dates provided
        test_data = None
        if test_start_date and test_end_date:
            test_data = self.data_fetcher.get_stock_data(symbol, test_start_date, test_end_date)

            # If forecast_length is None (evaluation mode), calculate it from test period
            if self.forecast_length is None:
                # Calculate the number of business days in the test period
                import pandas as pd

                test_start = pd.to_datetime(test_start_date)
                test_end = pd.to_datetime(test_end_date)

                # Count business days between test start and end
                business_days = pd.bdate_range(start=test_start, end=test_end)
                temp_forecast_length = len(business_days)

                self.logger.debug(f"Calculated forecast length from test period: {temp_forecast_length} business days")

                # Temporarily set forecast length for this evaluation
                original_forecast_length = self.forecast_length
                self.forecast_length = temp_forecast_length

                # Fit model on training data with calculated forecast length
                self.fit(train_data, target_column, **kwargs)

                # Restore original forecast length
                self.forecast_length = original_forecast_length
            else:
                # Use specified forecast length
                self.fit(train_data, target_column, **kwargs)
        else:
            # No test data, just fit normally
            self.fit(train_data, target_column, **kwargs)

        # Generate predictions
        predictions = self.predict()

        # If test data available, calculate evaluation metrics
        evaluation_metrics = None
        if test_data is not None and not test_data.empty:
            # Align predictions with test data
            common_dates = predictions.index.intersection(test_data.index)
            if len(common_dates) > 0:
                actual_values = test_data.loc[common_dates, target_column]
                forecast_values = predictions.loc[common_dates, "forecast"]

                # Calculate residuals and metrics
                residuals = actual_values - forecast_values

                import numpy as np
                from sklearn.metrics import mean_absolute_error, mean_squared_error

                mae = mean_absolute_error(actual_values, forecast_values)
                mse = mean_squared_error(actual_values, forecast_values)
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((actual_values - forecast_values) / actual_values)) * 100

                evaluation_metrics = {
                    "mae": mae,
                    "mse": mse,
                    "rmse": rmse,
                    "mape": mape,
                    "residuals": residuals.tolist(),
                    "actual_values": actual_values.tolist(),
                    "forecast_values": forecast_values.tolist(),
                    "evaluation_dates": common_dates.strftime("%Y-%m-%d").tolist(),
                }

        return {
            "predictions": predictions,
            "evaluation_metrics": evaluation_metrics,
            "train_period": {
                "start": train_data.index[0].strftime("%Y-%m-%d"),
                "end": train_data.index[-1].strftime("%Y-%m-%d"),
                "size": len(train_data),
            },
            "test_period": {
                "start": test_data.index[0].strftime("%Y-%m-%d") if test_data is not None else None,
                "end": test_data.index[-1].strftime("%Y-%m-%d") if test_data is not None else None,
                "size": len(test_data) if test_data is not None else 0,
            }
            if test_data is not None
            else None,
        }

    def forecast_from_symbol(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        target_column: str = "Close",
        **kwargs,
    ) -> pd.DataFrame:
        """Forecast stock prices by fetching data for a symbol.

        Args:
            symbol: Stock symbol to forecast
            start_date: Start date for historical data
            end_date: End date for historical data
            target_column: Column to forecast
            **kwargs: Additional arguments for fit()

        Returns:
            DataFrame with predictions
        """
        if self.data_fetcher is None:
            raise ValueError("Data fetcher not configured")

        # Fetch historical data
        data = self.data_fetcher.get_stock_data(symbol, start_date, end_date)

        if data.empty:
            raise ValueError(f"No data available for symbol {symbol}")

        # Fit and predict
        return self.fit_predict(data, target_column, **kwargs)

    def get_best_model(self) -> dict[str, Any]:
        """Get information about the best model found.

        Returns:
            Dictionary with model information
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        model_info = {
            "model_name": self.model.best_model_name,
            "model_params": self.model.best_model_params,
            "model_transformation": self.model.best_model_transformation_params,
            "model_accuracy": getattr(self.model, "best_model_accuracy", "N/A"),
        }

        self.logger.debug(f"Best model: {model_info['model_name']}")

        return model_info

    def plot_forecast(self, historical_data: pd.DataFrame | None = None, n_historical: int = 100) -> None:
        """Plot the forecast with historical data.

        Args:
            historical_data: Historical data to plot alongside forecast
            n_historical: Number of historical points to show
        """
        if self.prediction is None:
            raise ValueError("No predictions available. Call predict() first.")

        # Use AutoTS built-in plotting
        self.prediction.plot(
            self.model.df_wide_numeric,
            remove_zero_series=False,
            start_date=(-n_historical if n_historical else None),
        )

    def forecast(self, data: pd.DataFrame, target_column: str = "Close", **kwargs) -> pd.DataFrame:
        """Alias for fit_predict method.

        Args:
            data: Historical time series data
            target_column: Name of the column to forecast
            **kwargs: Additional arguments passed to fit_predict

        Returns:
            DataFrame with forecast predictions
        """
        return self.fit_predict(data, target_column, **kwargs)

    def evaluate_forecast(self, actual_data: pd.DataFrame, target_column: str = "Close") -> dict[str, float]:
        """Evaluate forecast accuracy against actual data.

        Args:
            actual_data: DataFrame with actual values
            target_column: Column to compare

        Returns:
            Dictionary with evaluation metrics
        """
        if self.prediction is None:
            raise ValueError("No predictions available. Call predict() first.")

        # Get forecast values
        forecast = self.prediction.forecast.iloc[:, 0]

        # Align actual data with forecast dates
        actual_values = actual_data.loc[forecast.index, target_column]

        # Calculate metrics
        import numpy as np
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        mae = mean_absolute_error(actual_values, forecast)
        mse = mean_squared_error(actual_values, forecast)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual_values - forecast) / actual_values)) * 100

        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mape": mape,
        }

    @classmethod
    def forecast_multiple_symbols(
        cls,
        symbols: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        forecast_length: int = 14,
        model_list: str = "fast",
        ensemble: str = "auto",
        max_generations: int = 5,
        data_fetcher=None,
        show_progress: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """Forecast multiple stock symbols sequentially.

        Args:
            symbols: List of stock symbols to forecast
            start_date: Start date for historical data
            end_date: End date for historical data
            forecast_length: Number of periods to forecast
            model_list: Model list preset or custom list
            ensemble: Ensemble method
            max_generations: Maximum generations for model search
            data_fetcher: Data fetcher instance
            show_progress: Whether to show progress

        Returns:
            Dictionary mapping symbols to their forecast results
        """
        import logging

        logger = logging.getLogger(__name__)

        results = {}

        logger.info(f"Starting sequential forecast for {len(symbols)} symbols")

        for idx, symbol in enumerate(symbols, 1):
            try:
                logger.info(f"Processing {symbol} ({idx}/{len(symbols)})")

                # Create a forecaster instance
                forecaster = cls(
                    forecast_length=forecast_length,
                    data_fetcher=data_fetcher,
                )

                # Get data and forecast
                data = data_fetcher.get_stock_data(symbol, start_date, end_date)

                if data.empty:
                    raise ValueError(f"No data available for symbol {symbol}")

                predictions = forecaster.fit_predict(
                    data,
                    target_column="Close",
                    model_list=model_list,
                    ensemble=ensemble,
                    max_generations=max_generations,
                    show_progress=show_progress,
                )

                # Get best model info
                model_info = forecaster.get_best_model()

                # Store results
                results[symbol] = {
                    "ticker": symbol,
                    "current_price": float(predictions["forecast"].iloc[0]),
                    "forecast_price": float(predictions["forecast"].iloc[-1]),
                    "lower_bound": float(predictions["lower_bound"].iloc[-1]),
                    "upper_bound": float(predictions["upper_bound"].iloc[-1]),
                    "forecast_length": forecast_length,
                    "best_model": model_info["model_name"],
                    "model_params": model_info.get("model_params", {}),
                }

                logger.info(f"Completed {symbol} using {model_info['model_name']}")

            except Exception as e:
                logger.error(f"Error forecasting {symbol}: {e}")
                results[symbol] = {
                    "ticker": symbol,
                    "error": str(e),
                }

        return results
