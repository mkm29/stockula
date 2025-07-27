"""Stock price forecasting using AutoTS."""

import pandas as pd
import warnings
import logging
import signal
import sys
from contextlib import contextmanager
from io import StringIO
from autots import AutoTS
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..interfaces import IDataFetcher

# Create logger
logger = logging.getLogger(__name__)

# Global flag for interruption
_interrupted = False


@contextmanager
def suppress_autots_output():
    """Context manager to suppress AutoTS verbose output."""
    # Suppress warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        # Redirect stdout and stderr if not in debug mode
        if not logger.isEnabledFor(logging.DEBUG):
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            try:
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        else:
            yield


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global _interrupted
    _interrupted = True
    logger.info("\nReceived interrupt signal. Stopping forecast...")
    sys.exit(0)


class StockForecaster:
    """Stock price forecaster using AutoTS."""

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
        date_col: Optional[str] = None,
        model_list: str = "fast",
        ensemble: str = "auto",
        max_generations: int = 5,
    ) -> "StockForecaster":
        """Fit forecasting model on historical data.

        Args:
            data: DataFrame with time series data
            target_column: Column to forecast
            date_col: Date column name (if None, uses index)
            model_list: Model subset to use ('fast', 'default', 'slow', 'parallel')
            ensemble: Ensemble method ('auto', 'simple', 'distance', 'horizontal')
            max_generations: Maximum generations for model search

        Returns:
            Self for method chaining
        """
        # Prepare data
        if date_col:
            data = data.set_index(date_col)

        # Ensure we have the target column
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

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

        # Set up signal handler for graceful interruption
        signal.signal(signal.SIGINT, signal_handler)

        logger.info(
            f"Starting AutoTS model fitting for {len(data_for_model)} data points..."
        )
        logger.info("This may take a few minutes. Press Ctrl+C to cancel.")

        try:
            with suppress_autots_output():
                # Initialize AutoTS with minimal verbosity
                self.model = AutoTS(
                    forecast_length=self.forecast_length,
                    frequency=self.frequency,
                    prediction_interval=self.prediction_interval,
                    ensemble=ensemble,
                    model_list=model_list,
                    max_generations=max_generations,
                    num_validations=self.num_validations,
                    validation_method=self.validation_method,
                    verbose=0,  # Minimal verbosity
                    no_negatives=True,  # Stock prices can't be negative
                    drop_most_recent=0,  # Don't drop recent data
                    n_jobs="auto",  # Use available cores
                )

                # Fit the model
                logger.debug(
                    f"Fitting AutoTS with parameters: model_list={model_list}, max_generations={max_generations}"
                )
                self.model = self.model.fit(
                    data_for_model,
                    date_col="date",  # Using 'date' column
                    value_col=target_column,
                    id_col=None,  # Single series
                )

            logger.info("Model fitting completed successfully.")

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

        with suppress_autots_output():
            self.prediction = self.model.predict()

        forecast = self.prediction.forecast

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
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
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
        logger.info(f"Fetching data for {symbol}...")
        if not self.data_fetcher:
            raise ValueError(
                "Data fetcher not configured. Ensure DI container is properly set up."
            )

        data = self.data_fetcher.get_stock_data(symbol, start_date, end_date)

        if data.empty:
            raise ValueError(f"No data available for symbol {symbol}")

        logger.info(f"Fetched {len(data)} data points for {symbol}")
        return self.fit_predict(data, target_column, **kwargs)

    def get_best_model(self) -> Dict[str, Any]:
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
        self, historical_data: Optional[pd.DataFrame] = None, n_historical: int = 100
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
