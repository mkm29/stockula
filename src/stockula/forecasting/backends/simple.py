"""Simple fallback forecasting backend for environments without AutoGluon/Chronos."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .base import ForecastBackend, ForecastResult


class SimpleForecastBackend(ForecastBackend):
    """Simple forecasting backend using linear regression as fallback."""

    model: LinearRegression
    last_values: np.ndarray | None
    trend: float | None
    std_dev: float | None

    def __init__(
        self,
        forecast_length: int | None = None,
        frequency: str = "infer",
        prediction_interval: float = 0.95,
        no_negatives: bool = True,
        **kwargs,
    ):
        """Initialize the simple forecasting backend.

        Args:
            forecast_length: Number of periods to forecast
            frequency: Data frequency (ignored in simple backend)
            prediction_interval: Confidence interval for predictions
            no_negatives: Constraint predictions to be non-negative
            **kwargs: Additional parameters (ignored)
        """
        super().__init__(
            forecast_length=forecast_length,
            frequency=frequency,
            prediction_interval=prediction_interval,
            no_negatives=no_negatives,
            **kwargs,
        )
        self.model = LinearRegression()
        self.last_values = None
        self.trend = None
        self.std_dev = None

    def fit(
        self,
        data: pd.DataFrame,
        target_column: str = "Close",
        show_progress: bool = True,
        **kwargs,
    ) -> "SimpleForecastBackend":
        """Fit a simple linear regression model on historical data.

        Args:
            data: DataFrame with time series data
            target_column: Column to forecast
            show_progress: Whether to show progress (ignored)
            **kwargs: Additional parameters (ignored)

        Returns:
            Self for method chaining
        """
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        # Prepare data
        y = data[target_column].values
        X = np.arange(len(y)).reshape(-1, 1)

        # Fit linear regression
        self.model.fit(X, y)

        # Store last values for forecasting
        self.last_values = y[-min(30, len(y)) :]  # Use last 30 values
        self.trend = self.model.coef_[0]

        # Calculate standard deviation for confidence intervals
        predictions = self.model.predict(X)
        residuals = y - predictions
        self.std_dev = np.std(residuals)

        self.is_fitted = True
        return self

    def predict(self, **kwargs) -> ForecastResult:
        """Generate predictions using the fitted model.

        Args:
            **kwargs: Additional parameters (ignored)

        Returns:
            ForecastResult with predictions and metadata
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Ensure required attributes are set (for type checking)
        assert self.last_values is not None
        assert self.std_dev is not None
        assert self.forecast_length is not None

        # Generate forecast points
        n_historical = len(self.last_values)
        future_X = np.arange(n_historical, n_historical + self.forecast_length).reshape(-1, 1)

        # Make predictions
        forecast_values = self.model.predict(future_X)

        # Apply non-negative constraint if requested
        if self.no_negatives:
            forecast_values = np.maximum(forecast_values, 0)

        # Calculate confidence intervals
        z_score = 1.96 if self.prediction_interval >= 0.95 else 1.645
        margin = z_score * self.std_dev * np.sqrt(1 + np.arange(1, self.forecast_length + 1) / 30)

        lower_bound = forecast_values - margin
        upper_bound = forecast_values + margin

        if self.no_negatives:
            lower_bound = np.maximum(lower_bound, 0)

        # Create forecast DataFrame
        forecast_df = pd.DataFrame(
            {
                "forecast": forecast_values,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }
        )

        # Ensure attributes are not None for type checking
        assert self.trend is not None
        assert self.std_dev is not None
        assert self.last_values is not None

        return ForecastResult(
            forecast=forecast_df,
            model_name="LinearRegression",
            model_params={
                "trend": float(self.trend),
                "intercept": float(self.model.intercept_),
            },
            metrics={
                "std_dev": float(self.std_dev),
                "r2_score": float(self.model.score(np.arange(len(self.last_values)).reshape(-1, 1), self.last_values)),
            },
        )

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the fitted model.

        Returns:
            Dictionary with model information
        """
        if not self.is_fitted:
            return {"model_name": "LinearRegression", "status": "not_fitted"}

        return {
            "model_name": "LinearRegression",
            "model_params": {
                "trend": float(self.trend) if self.trend is not None else None,
                "intercept": float(self.model.intercept_) if hasattr(self.model, "intercept_") else None,
            },
            "status": "fitted",
        }

    def get_available_models(self) -> list[str]:
        """Get list of available models for the simple backend.

        Returns:
            List of model names (just LinearRegression for simple backend)
        """
        return ["LinearRegression"]
