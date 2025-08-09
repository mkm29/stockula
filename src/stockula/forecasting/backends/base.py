"""Base abstract class for forecasting backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd
from dependency_injector.wiring import Provide, inject

from ...interfaces import ILoggingManager


@dataclass
class ForecastResult:
    """Result from a forecasting backend."""

    forecast: pd.DataFrame  # DataFrame with forecast, lower_bound, upper_bound columns
    model_name: str
    model_params: dict[str, Any]
    metrics: dict[str, float] | None = None
    metadata: dict[str, Any] | None = None


class ForecastBackend(ABC):
    """Abstract base class for forecasting backends."""

    @inject
    def __init__(
        self,
        forecast_length: int | None = None,
        frequency: str = "infer",
        prediction_interval: float = 0.95,
        no_negatives: bool = True,
        logging_manager: ILoggingManager = Provide["logging_manager"],
        **kwargs,
    ):
        """Initialize the forecasting backend.

        Args:
            forecast_length: Number of periods to forecast
            frequency: Data frequency ('infer' to detect automatically)
            prediction_interval: Confidence interval for predictions (0-1)
            no_negatives: Constraint predictions to be non-negative
            logging_manager: Injected logging manager
            **kwargs: Additional backend-specific parameters
        """
        self.forecast_length = forecast_length
        self.frequency = frequency
        self.prediction_interval = prediction_interval
        self.no_negatives = no_negatives
        self.logger = logging_manager
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        data: pd.DataFrame,
        target_column: str = "Close",
        show_progress: bool = True,
        **kwargs,
    ) -> "ForecastBackend":
        """Fit the forecasting model on historical data.

        Args:
            data: DataFrame with time series data (index should be DatetimeIndex)
            target_column: Column to forecast
            show_progress: Whether to show progress bar
            **kwargs: Additional backend-specific parameters

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, **kwargs) -> ForecastResult:
        """Generate predictions using fitted model.

        Args:
            **kwargs: Additional backend-specific parameters

        Returns:
            ForecastResult with predictions and metadata
        """
        pass

    def fit_predict(
        self,
        data: pd.DataFrame,
        target_column: str = "Close",
        show_progress: bool = True,
        **kwargs,
    ) -> ForecastResult:
        """Fit model and generate predictions in one step.

        Args:
            data: DataFrame with time series data
            target_column: Column to forecast
            show_progress: Whether to show progress bar
            **kwargs: Additional backend-specific parameters

        Returns:
            ForecastResult with predictions
        """
        self.fit(data, target_column, show_progress, **kwargs)
        return self.predict(**kwargs)

    def evaluate(
        self,
        actual_data: pd.DataFrame,
        target_column: str = "Close",
    ) -> dict[str, float]:
        """Evaluate forecast accuracy against actual data.

        Args:
            actual_data: DataFrame with actual values
            target_column: Column to compare

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get predictions
        result = self.predict()
        forecast = result.forecast["forecast"]

        # Align actual data with forecast dates
        common_dates = forecast.index.intersection(actual_data.index)
        if len(common_dates) == 0:
            raise ValueError("No overlapping dates between forecast and actual data")

        actual_values = actual_data.loc[common_dates, target_column]
        forecast_values = forecast.loc[common_dates]

        # Calculate metrics
        import numpy as np
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        mae = mean_absolute_error(actual_values, forecast_values)
        mse = mean_squared_error(actual_values, forecast_values)
        rmse = np.sqrt(mse)

        # Calculate MASE (Mean Absolute Scaled Error)
        # MASE = MAE / MAE_naive
        # where MAE_naive is the MAE of a naive forecast (using previous value)
        naive_forecast = actual_data[target_column].shift(1).dropna()
        actual_for_naive = actual_data[target_column].iloc[1:]
        mae_naive = mean_absolute_error(actual_for_naive, naive_forecast)

        # Avoid division by zero
        if mae_naive == 0 or np.isclose(mae_naive, 0, atol=1e-10):
            mase = np.inf if mae > 0 else 0.0
        else:
            mase = mae / mae_naive

        # Keep MAPE for backward compatibility but add deprecation note
        mape = np.mean(np.abs((actual_values - forecast_values) / actual_values)) * 100

        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mase": mase,  # Added MASE
            "mape": mape,  # Kept for backward compatibility
        }

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Get information about the fitted model.

        Returns:
            Dictionary with model information
        """
        pass

    @abstractmethod
    def get_available_models(self) -> list[str]:
        """Get list of available models for this backend.

        Returns:
            List of model names
        """
        pass

    def validate_input(self, data: pd.DataFrame, target_column: str) -> None:
        """Validate input data.

        Args:
            data: DataFrame to validate
            target_column: Target column name

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a DatetimeIndex")

        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        if data[target_column].isnull().any():
            raise ValueError(f"Target column '{target_column}' contains null values")

        if len(data) < 2:
            raise ValueError("Need at least 2 data points for forecasting")
