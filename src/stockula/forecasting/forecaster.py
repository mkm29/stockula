"""Stock price forecasting using AutoTS."""

import pandas as pd
from autots import AutoTS
from typing import Optional, Dict, Any
from ..data.fetcher import DataFetcher


class StockForecaster:
    """Stock price forecaster using AutoTS."""

    def __init__(
        self,
        forecast_length: int = 14,
        frequency: str = "infer",
        prediction_interval: float = 0.9,
        num_validations: int = 2,
        validation_method: str = "backwards",
    ):
        """Initialize forecaster.

        Args:
            forecast_length: Number of periods to forecast
            frequency: Time series frequency ('D', 'W', 'M', etc.), 'infer' to auto-detect
            prediction_interval: Confidence interval for predictions (0-1)
            num_validations: Number of validation splits
            validation_method: Validation method ('backwards', 'seasonal', 'similarity')
        """
        self.forecast_length = forecast_length
        self.frequency = frequency
        self.prediction_interval = prediction_interval
        self.num_validations = num_validations
        self.validation_method = validation_method
        self.model = None
        self.prediction = None

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

        # Reset index to have date as a column for AutoTS
        data_for_model = data[[target_column]].copy()
        # Ensure the index is a DatetimeIndex
        if not isinstance(data_for_model.index, pd.DatetimeIndex):
            data_for_model.index = pd.to_datetime(data_for_model.index)

        # Initialize AutoTS
        self.model = AutoTS(
            forecast_length=self.forecast_length,
            frequency=self.frequency,
            prediction_interval=self.prediction_interval,
            ensemble=ensemble,
            model_list=model_list,
            max_generations=max_generations,
            num_validations=self.num_validations,
            validation_method=self.validation_method,
            verbose=0,
        )

        # Fit the model
        self.model = self.model.fit(data_for_model)

        return self

    def predict(self) -> pd.DataFrame:
        """Generate predictions using fitted model.

        Returns:
            DataFrame with predictions and confidence intervals
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

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
        fetcher = DataFetcher()
        data = fetcher.get_stock_data(symbol, start_date, end_date)
        return self.fit_predict(data, target_column, **kwargs)

    def get_best_model(self) -> Dict[str, Any]:
        """Get information about the best model found.

        Returns:
            Dictionary with model information
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return {
            "model_name": self.model.best_model_name,
            "model_params": self.model.best_model_params,
            "model_transformation": self.model.best_model_transformation_params,
            "model_accuracy": getattr(self.model, "best_model_accuracy", "N/A"),
        }

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
