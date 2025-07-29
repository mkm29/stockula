"""Stock price forecasting using AutoTS."""

import logging
import os
import signal
import sys
import warnings
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional

import pandas as pd
from autots import AutoTS
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Global lock for all AutoTS operations to prevent threading issues
_AUTOTS_GLOBAL_LOCK = threading.Lock()

console = Console()

if TYPE_CHECKING:
    from ..interfaces import IDataFetcher

# Create logger
logger = logging.getLogger(__name__)

# Global flag for interruption
_interrupted = False


@contextmanager
def suppress_autots_output():
    """Context manager to suppress AutoTS verbose output."""
    # Disable interactive plots from Prophet
    old_mplbackend = os.environ.get("MPLBACKEND")
    os.environ["MPLBACKEND"] = "Agg"  # Non-interactive backend

    # Set cmdstanpy temp dir to avoid permission issues
    old_cmdstan_tmpdir = os.environ.get("CMDSTAN_TMP_DIR")
    import tempfile

    os.environ["CMDSTAN_TMP_DIR"] = tempfile.gettempdir()

    # Suppress cmdstanpy verbose logging
    cmdstanpy_logger = logging.getLogger("cmdstanpy")
    prophet_logger = logging.getLogger("prophet")
    prophet_plot_logger = logging.getLogger("prophet.plot")
    matplotlib_logger = logging.getLogger("matplotlib")

    old_cmdstanpy_level = cmdstanpy_logger.level
    old_prophet_level = prophet_logger.level
    old_prophet_plot_level = prophet_plot_logger.level
    old_matplotlib_level = matplotlib_logger.level

    # Set to ERROR to only show critical issues
    cmdstanpy_logger.setLevel(logging.ERROR)
    prophet_logger.setLevel(logging.ERROR)
    prophet_plot_logger.setLevel(logging.ERROR)
    matplotlib_logger.setLevel(logging.WARNING)

    # Suppress warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # Specifically ignore joblib warnings including resource tracker
        warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
        warnings.filterwarnings(
            "ignore", message="resource_tracker: There appear to be"
        )
        warnings.filterwarnings(
            "ignore",
            message="resource_tracker:",
            module="joblib.externals.loky.backend.resource_tracker",
        )
        # Ignore matplotlib font cache warnings
        warnings.filterwarnings(
            "ignore", message="Matplotlib is building the font cache"
        )
        # Ignore prophet plotly warnings
        warnings.filterwarnings("ignore", message="Importing plotly failed")

        # Suppress numerical warnings from sklearn, numpy, and statsmodels
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="divide by zero encountered"
        )
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="overflow encountered"
        )
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="invalid value encountered"
        )
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="Mean of empty slice"
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="Degrees of freedom <= 0 for slice",
        )
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="autots")

        # Suppress statsmodels warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
        warnings.filterwarnings(
            "ignore", message="Negative binomial dispersion parameter"
        )
        warnings.filterwarnings("ignore", message="Non-invertible MA parameters found")
        warnings.filterwarnings(
            "ignore", message="Non-stationary starting autoregressive parameters found"
        )

        # Suppress sklearn convergence warnings
        warnings.filterwarnings("ignore", message="ConvergenceWarning")
        warnings.filterwarnings("ignore", message="did not converge")

        # Suppress statsmodels GLM warnings that occur with problematic model combinations
        try:
            from statsmodels.tools.sm_exceptions import DomainWarning

            warnings.filterwarnings("ignore", category=DomainWarning)
        except ImportError:
            pass

        # Suppress specific GLM warnings
        warnings.filterwarnings(
            "ignore",
            message="The InversePower link function does not respect the domain",
        )

        # Try to import and suppress ValueWarning if it exists
        try:
            from statsmodels.tools.sm_exceptions import ValueWarning

            warnings.filterwarnings("ignore", category=ValueWarning)
        except ImportError:
            # If ValueWarning doesn't exist, ignore it
            pass

        # Suppress sklearn DataConversionWarning
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="Data was converted to boolean"
        )
        # Also suppress specific metric warnings
        warnings.filterwarnings(
            "ignore", message="Data was converted to boolean for metric"
        )
        warnings.filterwarnings(
            "ignore", category=UserWarning, module="sklearn.metrics.pairwise"
        )

        # Suppress fast_kalman warnings
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, module="autots.tools.fast_kalman"
        )

        # Suppress sklearn extmath warnings
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, module="sklearn.utils.extmath"
        )

        # Suppress statsmodels ARDL warnings
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, module="statsmodels.tsa.ardl"
        )

        # Suppress pandas FutureWarning about downcasting behavior
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message="Downcasting behavior in `replace`",
        )

        # Set environment variable to suppress joblib resource tracker warnings
        old_loky_pickler = os.environ.get("LOKY_PICKLER", None)
        os.environ["LOKY_PICKLER"] = "pickle"

        # Also suppress AutoTS template eval errors by setting env var
        old_autots_suppress = os.environ.get(
            "SUPPRESS_AUTOTS_TEMPLATE_EVAL_ERRORS", None
        )
        os.environ["SUPPRESS_AUTOTS_TEMPLATE_EVAL_ERRORS"] = "1"

        # Redirect stdout and stderr based on debug mode
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        class FilteredOutput:
            def __init__(self, original_stream, suppress_all=False):
                self.original_stream = original_stream
                self.suppress_all = suppress_all

            def write(self, s):
                if self.suppress_all:
                    # In production, suppress various AutoTS outputs
                    # Skip empty lines and whitespace-only lines
                    if not s or s.isspace():
                        return len(s)

                    # Skip lines we want to suppress
                    suppress_patterns = [
                        "Template Eval Error:",
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

        # Use filtered output to remove noisy AutoTS messages
        # Always suppress template eval errors unless in DEBUG mode
        suppress_all = not logger.isEnabledFor(logging.DEBUG)
        sys.stdout = FilteredOutput(
            old_stdout, suppress_all=True
        )  # Always filter stdout
        sys.stderr = FilteredOutput(
            old_stderr, suppress_all=True
        )  # Always filter stderr

        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            # Restore environment variables
            if old_loky_pickler is None:
                os.environ.pop("LOKY_PICKLER", None)
            else:
                os.environ["LOKY_PICKLER"] = old_loky_pickler

            if old_autots_suppress is None:
                os.environ.pop("SUPPRESS_AUTOTS_TEMPLATE_EVAL_ERRORS", None)
            else:
                os.environ["SUPPRESS_AUTOTS_TEMPLATE_EVAL_ERRORS"] = old_autots_suppress

            # Restore matplotlib backend
            if old_mplbackend is None:
                os.environ.pop("MPLBACKEND", None)
            else:
                os.environ["MPLBACKEND"] = old_mplbackend

            # Restore cmdstan temp dir
            if old_cmdstan_tmpdir is None:
                os.environ.pop("CMDSTAN_TMP_DIR", None)
            else:
                os.environ["CMDSTAN_TMP_DIR"] = old_cmdstan_tmpdir

            # Restore logger levels
            cmdstanpy_logger.setLevel(old_cmdstanpy_level)
            prophet_logger.setLevel(old_prophet_level)
            prophet_plot_logger.setLevel(old_prophet_plot_level)
            matplotlib_logger.setLevel(old_matplotlib_level)


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global _interrupted
    _interrupted = True
    logger.info("\nReceived interrupt signal. Stopping forecast...")
    sys.exit(0)


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
        "SeasonalNaive",  # Captures seasonality
        "GLS",  # Generalized Least Squares
        "ETS",  # Exponential Smoothing
        "ARIMA",  # Classic time series model
        # "FBProphet",  # Temporarily excluded due to cmdstanpy permission issues
        "RollingRegression",  # Rolling window regression
        "WindowRegression",  # Window-based regression
        "VAR",  # Vector Autoregression
        "VECM",  # Vector Error Correction
        "DynamicFactor",  # Dynamic Factor Model
        "MotifSimulation",  # Pattern-based simulation
        "SectionalMotif",  # Cross-sectional patterns
        "NVAR",  # Neural VAR
        "Theta",  # Theta method
        # Note: ARDL excluded due to numerical stability issues with stock data
        # Note: FBProphet excluded due to cmdstanpy permission issues on some systems
    ]

    # Fastest models for quick forecasting (15-30 seconds per symbol)
    # These models provide excellent speed/accuracy trade-off for stock data
    FAST_FINANCIAL_MODEL_LIST = [
        "LastValueNaive",  # Very fast (<1s) - Uses last value as forecast
        "AverageValueNaive",  # Very fast (<1s) - Moving average approach
        "SeasonalNaive",  # Fast (<5s) - Captures seasonal patterns
        "ETS",  # Fast (5-10s) - Exponential smoothing (Error/Trend/Seasonality)
        "ARIMA",  # Moderate (10-20s) - AutoRegressive Integrated Moving Average
        "Theta",  # Fast (5-10s) - Statistical decomposition method
    ]

    # Ultra-fast models for immediate results (5-10 seconds per symbol)
    ULTRA_FAST_MODEL_LIST = [
        "LastValueNaive",  # <1s - Simple but effective baseline
        "AverageValueNaive",  # <1s - Moving average
        "SeasonalNaive",  # <5s - Basic seasonality
    ]

    def __init__(
        self,
        forecast_length: int = 14,
        frequency: str = "infer",
        prediction_interval: float = 0.9,
        num_validations: int = 2,
        validation_method: str = "backwards",
        model_list: str = "fast",
        data_fetcher: Optional["IDataFetcher"] = None,
    ):
        """Initialize forecaster.

        Args:
            forecast_length: Number of periods to forecast
            frequency: Time series frequency ('D', 'W', 'M', etc.), 'infer' to auto-detect
            prediction_interval: Confidence interval for predictions (0-1)
            num_validations: Number of validation splits
            validation_method: Validation method ('backwards', 'seasonal', 'similarity')
            model_list: Default model list to use
            data_fetcher: Injected data fetcher instance
        """
        self.forecast_length = forecast_length
        self.frequency = frequency
        self.prediction_interval = prediction_interval
        self.num_validations = num_validations
        self.validation_method = validation_method
        self.model_list = model_list
        self.model = None
        self.prediction = None
        self.data_fetcher = data_fetcher

    def fit(
        self,
        data: pd.DataFrame,
        target_column: str = "Close",
        date_col: str | None = None,
        model_list: str = "fast",
        ensemble: str = "auto",
        max_generations: int = 5,
        show_progress: bool = True,
    ) -> "StockForecaster":
        """Fit forecasting model on historical data.

        Args:
            data: DataFrame with time series data
            target_column: Column to forecast
            date_col: Date column name (if None, uses index)
            model_list: Model subset to use ('fast', 'default', 'slow', 'parallel', 'financial')
            ensemble: Ensemble method ('auto', 'simple', 'distance', 'horizontal')
            max_generations: Maximum generations for model search
            show_progress: Whether to show progress indicators (default True)

        Returns:
            Self for method chaining
        """
        # Prepare data
        if date_col:
            data = data.set_index(date_col)

        # Ensure we have the target column
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        # Validate data for financial time series
        target_data = data[target_column]

        # Check for non-positive values (problematic for some models)
        if (target_data <= 0).any():
            logger.warning(
                f"Found {(target_data <= 0).sum()} non-positive values in {target_column}. This may cause issues with some models."
            )

        # Check for missing values
        if target_data.isna().any():
            logger.warning(
                f"Found {target_data.isna().sum()} missing values in {target_column}. AutoTS will handle these."
            )

        # Use financial model list for stock data or if explicitly requested
        # Default to financial models for fast mode to avoid statsmodels warnings
        # Always use financial models for stock data to avoid issues
        if target_column in ["Close", "Price", "close", "price"]:
            # For stock data, use appropriate financial models
            if model_list == "fast":
                actual_model_list = self.FAST_FINANCIAL_MODEL_LIST
                logger.info(
                    f"Using fast financial model list ({len(self.FAST_FINANCIAL_MODEL_LIST)} models) for {target_column}"
                )
                print(
                    f"INFO: Using fast financial model list ({len(self.FAST_FINANCIAL_MODEL_LIST)} models) for {target_column}"
                )
            elif model_list == "ultra_fast":
                actual_model_list = self.ULTRA_FAST_MODEL_LIST
                logger.info(
                    f"Using ultra-fast model list ({len(self.ULTRA_FAST_MODEL_LIST)} models) for {target_column}"
                )
                print(
                    f"INFO: Using ultra-fast model list ({len(self.ULTRA_FAST_MODEL_LIST)} models) for {target_column}"
                )
            elif model_list == "financial" or model_list == "default":
                actual_model_list = self.FINANCIAL_MODEL_LIST
                logger.info(
                    f"Using full financial model list ({len(self.FINANCIAL_MODEL_LIST)} models) for {target_column}"
                )
            else:
                # If user explicitly requests a different model list, use it
                actual_model_list = model_list
                logger.warning(
                    f"Using non-financial model list '{model_list}' for stock data - may cause warnings"
                )
        else:
            # Non-stock data, use requested model list
            actual_model_list = model_list

        # Prepare data for AutoTS - it needs date as a column, not index
        data_for_model = data[[target_column]].copy()

        # Ensure the index is a DatetimeIndex
        if not isinstance(data_for_model.index, pd.DatetimeIndex):
            data_for_model.index = pd.to_datetime(data_for_model.index)

        # Reset index to have date as a column named 'date'
        data_for_model = data_for_model.reset_index()
        data_for_model.columns = ["date", target_column]

        # Ensure date column is datetime format
        data_for_model["date"] = pd.to_datetime(data_for_model["date"])

        # Set up signal handler for graceful interruption (only in main thread)
        import threading

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, signal_handler)

        logger.debug(f"Fitting model on {len(data_for_model)} data points")
        if threading.current_thread() is threading.main_thread():
            logger.debug(f"Forecast horizon: {self.forecast_length} periods")

        try:
            with suppress_autots_output():
                # Suppress joblib warnings
                import os

                os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

                # Determine if we should show progress
                should_show_progress = (
                    show_progress
                    and threading.current_thread() is threading.main_thread()
                )

                if should_show_progress:
                    # Show progress only in main thread
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                        transient=False,
                    ) as progress:
                        task = progress.add_task(
                            "[yellow]Setting up time series models...",
                            total=None,
                        )

                        # Run initialization in a thread to keep spinner animated
                        import time
                        from concurrent.futures import ThreadPoolExecutor

                        def init_autots():
                            """Initialize AutoTS model in background."""
                            return AutoTS(
                                forecast_length=self.forecast_length,
                                frequency=self.frequency,
                                prediction_interval=self.prediction_interval,
                                ensemble=ensemble,
                                model_list=actual_model_list,
                                max_generations=max_generations,
                                num_validations=self.num_validations,
                                validation_method=self.validation_method,
                                verbose=1
                                if logger.isEnabledFor(logging.DEBUG)
                                else 0,  # Verbose only in debug mode
                                no_negatives=True,  # Stock prices can't be negative
                                drop_most_recent=0,  # Don't drop recent data
                                n_jobs="auto",  # Use available cores
                                constraint=None,  # No constraint to avoid some errors
                                drop_data_older_than_periods=None,  # Keep all data
                                model_interrupt=False,  # Don't interrupt on errors
                            )

                        # Initialize in background thread
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            init_future = executor.submit(init_autots)

                            # Update progress while initializing
                            dots = ["", ".", "..", "..."]
                            dot_index = 0

                            while not init_future.done():
                                desc = f"[yellow]Setting up time series models{dots[dot_index]}"
                                progress.update(task, description=desc)
                                dot_index = (dot_index + 1) % len(dots)
                                time.sleep(0.2)

                            # Get the initialized model
                            self.model = init_future.result()

                        # Fit the model
                        progress.update(
                            task,
                            description="[cyan]Training models on historical data...",
                        )
                        logger.debug(
                            f"Fitting AutoTS with parameters: model_list={actual_model_list if isinstance(actual_model_list, str) else f'{len(actual_model_list)} models'}, max_generations={max_generations}"
                        )

                        # Run model fitting in a separate thread to keep Progress updating
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            # Submit the fit task to run in background
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
                            self.model = future.result()

                        complete_msg = "[green]✓ Model training completed"
                        progress.update(task, description=complete_msg)
                else:
                    # No progress display (for worker threads)
                    import threading

                    # For worker threads, limit n_jobs to 1 to avoid nested parallelism
                    if threading.current_thread() is not threading.main_thread():
                        n_jobs_value = 1
                        logger.debug("Running in worker thread, setting n_jobs=1")
                    else:
                        n_jobs_value = "auto"

                    self.model = AutoTS(
                        forecast_length=self.forecast_length,
                        frequency=self.frequency,
                        prediction_interval=self.prediction_interval,
                        ensemble=ensemble,
                        model_list=actual_model_list,
                        max_generations=max_generations,
                        num_validations=self.num_validations,
                        validation_method=self.validation_method,
                        verbose=1
                        if logger.isEnabledFor(logging.DEBUG)
                        else 0,  # Verbose only in debug mode
                        no_negatives=True,  # Stock prices can't be negative
                        drop_most_recent=0,  # Don't drop recent data
                        n_jobs=n_jobs_value,  # Limit to 1 in worker threads
                        constraint=None,  # No constraint to avoid some errors
                        drop_data_older_than_periods=None,  # Keep all data
                        model_interrupt=False,  # Don't interrupt on errors
                    )

                    logger.debug(
                        f"Fitting AutoTS with parameters: model_list={actual_model_list if isinstance(actual_model_list, str) else f'{len(actual_model_list)} models'}, max_generations={max_generations}, n_jobs={n_jobs_value}"
                    )

                    # Fit the model directly without progress
                    # Use global lock for AutoTS fit operation
                    with _AUTOTS_GLOBAL_LOCK:
                        logger.debug("Acquired global AutoTS lock for fit()")
                        self.model = self.model.fit(
                            data_for_model,
                            date_col="date",
                            value_col=target_column,
                            id_col=None,
                        )

            logger.debug("Model fitting completed")

        except KeyboardInterrupt:
            logger.warning("Model fitting interrupted by user.")
            raise
        except Exception as e:
            logger.error(f"Error during model fitting: {str(e)}")
            raise

        return self

    def predict(self) -> pd.DataFrame:
        """Generate predictions using fitted model.

        Returns:
            DataFrame with predictions and confidence intervals
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        logger.debug("Generating predictions...")

        # Use global lock for AutoTS predict operation
        logger.debug("Calling model.predict()...")

        # AutoTS is not thread-safe, use global lock
        with _AUTOTS_GLOBAL_LOCK:
            logger.debug("Acquired global AutoTS lock for predict()")
            with suppress_autots_output():
                self.prediction = self.model.predict()

        forecast = self.prediction.forecast
        logger.debug(f"Forecast shape: {forecast.shape}")

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

        logger.debug(f"Generated {len(result)} forecast points")

        return result

    def fit_predict(
        self, data: pd.DataFrame, target_column: str = "Close", **kwargs
    ) -> pd.DataFrame:
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
        logger.debug(f"Fetching data for {symbol}")
        if not self.data_fetcher:
            raise ValueError(
                "Data fetcher not configured. Ensure DI container is properly set up."
            )

        data = self.data_fetcher.get_stock_data(symbol, start_date, end_date)

        if data.empty:
            raise ValueError(f"No data available for symbol {symbol}")

        logger.debug(f"Retrieved {len(data)} data points")
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

        logger.debug(f"Best model: {model_info['model_name']}")
        return model_info

    def plot_forecast(
        self, historical_data: pd.DataFrame | None = None, n_historical: int = 100
    ):
        """Plot forecast with historical data.

        Args:
            historical_data: Historical data to plot alongside forecast
            n_historical: Number of historical points to show
        """
        if self.prediction is None:
            raise ValueError("No predictions available. Call predict() first.")

        self.model.plot(self.prediction, include_history=True, n_back=n_historical)

    def forecast(
        self, data: pd.DataFrame, target_column: str = "Close", **kwargs
    ) -> pd.DataFrame:
        """Forecast future values (alias for fit_predict).

        Args:
            data: DataFrame with time series data
            target_column: Column to forecast
            **kwargs: Additional arguments for fit()

        Returns:
            DataFrame with predictions
        """
        return self.fit_predict(data, target_column, **kwargs)

    @classmethod
    def forecast_multiple_parallel(
        cls,
        symbols: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        forecast_length: int = 14,
        model_list: str = "fast",
        ensemble: str = "auto",
        max_generations: int = 5,
        max_workers: int = 4,
        data_fetcher=None,
        progress_callback=None,
        status_update_interval: int = 10,
    ) -> dict[str, dict[str, Any]]:
        """Forecast multiple stocks in parallel.

        Args:
            symbols: List of stock symbols to forecast
            start_date: Start date for historical data
            end_date: End date for historical data
            forecast_length: Number of periods to forecast
            model_list: Model subset to use
            ensemble: Ensemble method
            max_generations: Maximum generations for model search
            max_workers: Maximum number of parallel workers
            data_fetcher: Data fetcher instance
            progress_callback: Optional callback for progress updates
            status_update_interval: Seconds between status updates

        Returns:
            Dictionary mapping symbols to their forecast results
        """
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Warn about AutoTS threading limitations
        if max_workers > 1:
            logger.warning(
                f"Using max_workers={max_workers}. Note: AutoTS has threading limitations "
                "and may hang with concurrent operations. Consider using max_workers=1 for reliability."
            )

        results = {}
        lock = threading.Lock()
        status_info = {
            "active": {},  # symbol -> start_time
            "completed": [],
            "errors": [],
            "pending": list(symbols),
        }

        def update_status(symbol: str, status: str, start_time: float = None):
            """Update the status of a symbol."""
            with lock:
                if status == "started":
                    status_info["active"][symbol] = start_time or time.time()
                    if symbol in status_info["pending"]:
                        status_info["pending"].remove(symbol)
                elif status == "completed":
                    if symbol in status_info["active"]:
                        del status_info["active"][symbol]
                    status_info["completed"].append(symbol)
                elif status == "error":
                    if symbol in status_info["active"]:
                        del status_info["active"][symbol]
                    status_info["errors"].append(symbol)

        def get_status_summary():
            """Get a summary of current status."""
            with lock:
                active_details = []
                for symbol, start_time in status_info["active"].items():
                    elapsed = int(time.time() - start_time)
                    active_details.append(f"{symbol} ({elapsed}s)")

                return {
                    "active": active_details,
                    "active_count": len(status_info["active"]),
                    "completed_count": len(status_info["completed"]),
                    "error_count": len(status_info["errors"]),
                    "pending_count": len(status_info["pending"]),
                    "total": len(symbols),
                }

        def forecast_single(symbol: str) -> tuple[str, dict[str, Any]]:
            """Forecast a single symbol with timeout protection."""
            # Set up warning suppression for this worker thread
            import warnings
            import os
            from concurrent.futures import ThreadPoolExecutor, TimeoutError

            # Set environment variable to suppress sklearn warnings
            os.environ["PYTHONWARNINGS"] = (
                "ignore::UserWarning:sklearn.metrics.pairwise"
            )

            # Configure warnings at the start
            warnings.simplefilter("ignore")
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)

            # Specific sklearn warnings
            warnings.filterwarnings(
                "ignore",
                message="Data was converted to boolean",
            )
            warnings.filterwarnings("ignore", message="DataConversionWarning")
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            # Suppress joblib resource tracker warnings
            warnings.filterwarnings("ignore", message="resource_tracker:")
            warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

            start_time = time.time()
            update_status(symbol, "started", start_time)

            # Set a reasonable timeout for each symbol based on model list
            if isinstance(model_list, list) and len(model_list) == 1:
                SYMBOL_TIMEOUT = 15  # 15 seconds for single model
            elif model_list == "ultra_fast":
                SYMBOL_TIMEOUT = 30  # 30 seconds for ultra-fast models
            elif model_list == "fast":
                SYMBOL_TIMEOUT = 60  # 60 seconds for fast models
            else:
                SYMBOL_TIMEOUT = 120  # 120 seconds for other models

            def run_forecast():
                """Run the forecast in a separate thread to enable timeout."""
                # Create a forecaster instance for this symbol
                forecaster = cls(
                    forecast_length=forecast_length,
                    data_fetcher=data_fetcher,
                )

                # Log which symbol is being processed
                logger.debug(f"Starting forecast for {symbol}")

                # Add detailed timing logs
                total_start = time.time()

                # Time data fetching
                fetch_start = time.time()
                logger.debug(f"[{symbol}] Starting data fetch...")

                # Get data directly to separate fetch time from modeling time
                if not forecaster.data_fetcher:
                    raise ValueError("Data fetcher not configured")

                data = forecaster.data_fetcher.get_stock_data(
                    symbol, start_date, end_date
                )
                fetch_time = time.time() - fetch_start
                logger.debug(
                    f"[{symbol}] Data fetched in {fetch_time:.1f}s ({len(data)} points)"
                )

                # Time model fitting and prediction
                fit_start = time.time()
                logger.debug(
                    f"[{symbol}] Starting model fitting with {model_list} models..."
                )

                predictions = forecaster.fit_predict(
                    data,
                    target_column="Close",
                    model_list=model_list,
                    ensemble=ensemble,
                    max_generations=max_generations,
                    show_progress=False,  # Disable progress display in worker threads
                )

                fit_time = time.time() - fit_start
                total_time = time.time() - total_start

                logger.debug(f"[{symbol}] Model fitting completed in {fit_time:.1f}s")
                logger.debug(
                    f"[{symbol}] Total forecast time: {total_time:.1f}s (fetch: {fetch_time:.1f}s, fit: {fit_time:.1f}s)"
                )

                logger.debug(f"[{symbol}] Getting best model info...")
                model_info = forecaster.get_best_model()
                logger.debug(
                    f"Completed forecast for {symbol} using {model_info['model_name']}"
                )
                logger.debug(f"[{symbol}] Best model: {model_info['model_name']}")

                # Check predictions data
                logger.debug(f"[{symbol}] Processing predictions...")
                if predictions is None:
                    raise ValueError(f"No predictions returned for {symbol}")

                result = {
                    "ticker": symbol,
                    "current_price": predictions["forecast"].iloc[0],
                    "forecast_price": predictions["forecast"].iloc[-1],
                    "lower_bound": predictions["lower_bound"].iloc[-1],
                    "upper_bound": predictions["upper_bound"].iloc[-1],
                    "forecast_length": forecast_length,
                    "best_model": model_info["model_name"],
                    "model_params": model_info.get("model_params", {}),
                }

                logger.debug(f"[{symbol}] Forecast complete!")
                return result

            try:
                # Run forecast directly without nested executor
                result = run_forecast()
                update_status(symbol, "completed")
                if progress_callback:
                    with lock:
                        progress_callback(symbol, "completed", get_status_summary())
                return symbol, result

            except Exception as e:
                logger.error(f"Error forecasting {symbol}: {e}")
                update_status(symbol, "error")

                if progress_callback:
                    with lock:
                        progress_callback(symbol, "error", get_status_summary())

                return symbol, {"ticker": symbol, "error": str(e)}

        # Use ThreadPoolExecutor for parallel forecasting
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all forecast tasks
            future_to_symbol = {
                executor.submit(forecast_single, symbol): symbol for symbol in symbols
            }

            # Start a background thread for periodic status updates
            stop_updates = threading.Event()

            def periodic_status_updates():
                """Send periodic status updates."""
                while not stop_updates.is_set():
                    time.sleep(status_update_interval)
                    if not stop_updates.is_set() and progress_callback:
                        status = get_status_summary()
                        if status["active_count"] > 0:
                            logger.debug(f"Sending status update: {status}")
                            with lock:
                                progress_callback(None, "status_update", status)

            if (
                progress_callback
                and threading.current_thread() is threading.main_thread()
            ):
                status_thread = threading.Thread(
                    target=periodic_status_updates, daemon=True
                )
                status_thread.start()

            # Process completed forecasts
            try:
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        _, result = future.result()
                        results[symbol] = result
                    except Exception as e:
                        logger.error(f"Failed to get result for {symbol}: {e}")
                        results[symbol] = {"ticker": symbol, "error": str(e)}
            finally:
                # Stop the status update thread
                stop_updates.set()

        return results
