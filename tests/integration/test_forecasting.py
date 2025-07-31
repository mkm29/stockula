"""Tests for forecasting module."""

from datetime import timedelta
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from stockula.forecasting import StockForecaster


def create_mock_autots_prediction(start_date, periods=7, base_value=110.0):
    """Helper function to create mock AutoTS prediction with required attributes."""
    mock_prediction = Mock()
    mock_prediction.forecast = pd.DataFrame(
        {"CLOSE": [base_value] * periods},
        index=pd.date_range(start=start_date, periods=periods),
    )
    mock_prediction.upper_forecast = pd.DataFrame(
        {"CLOSE": [base_value + 5.0] * periods},
        index=pd.date_range(start=start_date, periods=periods),
    )
    mock_prediction.lower_forecast = pd.DataFrame(
        {"CLOSE": [base_value - 5.0] * periods},
        index=pd.date_range(start=start_date, periods=periods),
    )
    return mock_prediction


class TestStockForecaster:
    """Test StockForecaster class."""

    def test_initialization(self):
        """Test StockForecaster initialization."""
        forecaster = StockForecaster(forecast_length=30, frequency="D", prediction_interval=0.95)
        assert forecaster.forecast_length == 30
        assert forecaster.frequency == "D"
        assert forecaster.prediction_interval == 0.95

        # Test defaults
        forecaster_default = StockForecaster()
        assert forecaster_default.forecast_length is None  # Changed to None for mutual exclusivity
        assert forecaster_default.frequency == "infer"  # Default is 'infer' which gets converted to 'D' during fit
        assert forecaster_default.prediction_interval == 0.95

    def test_forecast_with_data(self, forecast_data):
        """Test forecasting with provided data."""
        forecaster = StockForecaster(forecast_length=7)

        # Mock AutoTS to avoid actual model training
        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            # Create mock model
            mock_model = Mock()

            # Create proper mock prediction
            start_date = forecast_data.index[-1] + timedelta(days=1)
            mock_prediction = create_mock_autots_prediction(start_date, periods=7)

            mock_model.predict.return_value = mock_prediction
            mock_model.best_model_name = "TestModel"
            mock_model.best_model_params = {}
            mock_model.best_model_transformation_params = {}

            mock_autots_instance = Mock()
            mock_autots_instance.fit.return_value = mock_model
            mock_autots.return_value = mock_autots_instance

            # Run forecast
            predictions = forecaster.fit_predict(forecast_data)

            # Check results
            assert isinstance(predictions, pd.DataFrame)
            assert len(predictions) == 7
            assert all(col in predictions.columns for col in ["forecast", "lower_bound", "upper_bound"])
            assert mock_autots_instance.fit.called
            assert mock_model.predict.called

    def test_forecast_from_symbol(self, mock_data_fetcher, forecast_data):
        """Test forecasting from symbol."""
        # Create a mock DataFetcher that returns our test data
        mock_fetcher = Mock()
        mock_fetcher.get_stock_data = Mock(return_value=forecast_data)

        # Pass the data fetcher to the forecaster
        forecaster = StockForecaster(forecast_length=14, data_fetcher=mock_fetcher)

        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            # Setup mock AutoTS
            mock_model = Mock()

            # Create proper mock prediction
            start_date = forecast_data.index[-1] + timedelta(days=1)
            mock_prediction = create_mock_autots_prediction(start_date, periods=14)

            mock_model.predict.return_value = mock_prediction
            mock_model.best_model_name = "TestModel"
            mock_model.best_model_params = {}
            mock_model.best_model_transformation_params = {}

            mock_autots_instance = Mock()
            mock_autots_instance.fit.return_value = mock_model
            mock_autots.return_value = mock_autots_instance

            # Run forecast
            predictions = forecaster.forecast_from_symbol("AAPL")

            # Check results
            assert isinstance(predictions, pd.DataFrame)
            assert len(predictions) == 14
            assert all(col in predictions.columns for col in ["forecast", "lower_bound", "upper_bound"])
            # Check values - the helper creates base_value ± 5.0
            assert predictions["forecast"].iloc[0] == 110.0
            assert predictions["lower_bound"].iloc[0] == 105.0
            assert predictions["upper_bound"].iloc[0] == 115.0

    def test_get_best_model(self):
        """Test getting best model information."""
        forecaster = StockForecaster()

        # Before fitting, should raise error
        with pytest.raises(ValueError, match="Model not fitted"):
            forecaster.get_best_model()

        # Mock a fitted model
        forecaster.model = Mock()
        forecaster.model.best_model_name = "ARIMA"
        forecaster.model.best_model_params = {"p": 1, "d": 1, "q": 1}
        forecaster.model.best_model_transformation_params = {}

        model_info = forecaster.get_best_model()
        assert model_info["model_name"] == "ARIMA"
        assert model_info["model_params"]["p"] == 1

    def test_model_list_parameter(self, forecast_data):
        """Test different model list parameters."""
        forecaster = StockForecaster(forecast_length=7)

        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            # Create a proper mock model that returns valid predictions
            mock_model = Mock()

            # Use the helper to create proper prediction structure
            start_date = forecast_data.index[-1] + timedelta(days=1)
            mock_prediction = create_mock_autots_prediction(start_date, periods=7)

            mock_model.predict.return_value = mock_prediction
            mock_model.best_model_name = "TestModel"
            mock_model.best_model_params = {}
            mock_model.best_model_transformation_params = {}

            mock_autots_instance = Mock()
            mock_autots_instance.fit.return_value = mock_model
            mock_autots.return_value = mock_autots_instance

            # Test with different model lists
            for model_list in ["fast", "default", "slow"]:
                forecaster.forecast(forecast_data, model_list=model_list)

                # Check that AutoTS was called with correct model_list
                call_args = mock_autots.call_args_list[-1]
                assert call_args[1]["model_list"] == model_list

    def test_ensemble_parameter(self, forecast_data):
        """Test ensemble parameter."""
        forecaster = StockForecaster(forecast_length=7)

        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            # Create a proper mock model that returns valid predictions
            mock_model = Mock()

            # Use the helper to create proper prediction structure
            start_date = forecast_data.index[-1] + timedelta(days=1)
            mock_prediction = create_mock_autots_prediction(start_date, periods=7)

            mock_model.predict.return_value = mock_prediction
            mock_model.best_model_name = "TestModel"
            mock_model.best_model_params = {}
            mock_model.best_model_transformation_params = {}

            mock_autots_instance = Mock()
            mock_autots_instance.fit.return_value = mock_model
            mock_autots.return_value = mock_autots_instance

            # Test with different ensemble methods
            for ensemble in ["simple", "distance", "horizontal"]:
                forecaster.forecast(forecast_data, ensemble=ensemble)

                # Check that AutoTS was called with correct ensemble
                call_args = mock_autots.call_args_list[-1]
                assert call_args[1]["ensemble"] == ensemble

    def test_frequency_inference(self):
        """Test automatic frequency inference."""
        # Daily data
        daily_data = pd.DataFrame(
            {"Close": range(100)},
            index=pd.date_range("2023-01-01", periods=100, freq="D"),
        )

        forecaster = StockForecaster(frequency="infer", forecast_length=7)

        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            # Create a proper mock model that returns valid predictions
            mock_model = Mock()

            # Use the helper to create proper prediction structure
            start_date = daily_data.index[-1] + timedelta(days=1)
            mock_prediction = create_mock_autots_prediction(start_date, periods=7)

            mock_model.predict.return_value = mock_prediction
            mock_model.best_model_name = "TestModel"
            mock_model.best_model_params = {}
            mock_model.best_model_transformation_params = {}

            mock_autots_instance = Mock()
            mock_autots_instance.fit.return_value = mock_model
            mock_autots.return_value = mock_autots_instance

            forecaster.fit_predict(daily_data)

            # AutoTS should be called with 'D' since we default infer to 'D'
            # The actual inference happens inside the fit method
            call_args = mock_autots.call_args_list[0]
            assert call_args[1]["frequency"] == "D"

    def test_validation_parameters(self, forecast_data):
        """Test validation parameters."""
        forecaster = StockForecaster(forecast_length=7, num_validations=3, validation_method="similarity")

        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            # Create a proper mock model that returns valid predictions
            mock_model = Mock()

            # Use the helper to create proper prediction structure
            start_date = forecast_data.index[-1] + timedelta(days=1)
            mock_prediction = create_mock_autots_prediction(start_date, periods=7)

            mock_model.predict.return_value = mock_prediction
            mock_model.best_model_name = "TestModel"
            mock_model.best_model_params = {}
            mock_model.best_model_transformation_params = {}

            mock_autots_instance = Mock()
            mock_autots_instance.fit.return_value = mock_model
            mock_autots.return_value = mock_autots_instance

            forecaster.forecast(forecast_data)

            # Check AutoTS parameters
            call_args = mock_autots.call_args_list[0]
            assert call_args[1]["num_validations"] == 3
            assert call_args[1]["validation_method"] == "similarity"

    def test_max_generations(self, forecast_data):
        """Test max generations parameter."""
        forecaster = StockForecaster(forecast_length=7, max_generations=10)

        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            # Create a proper mock model that returns valid predictions
            mock_model = Mock()

            # Use the helper to create proper prediction structure
            start_date = forecast_data.index[-1] + timedelta(days=1)
            mock_prediction = create_mock_autots_prediction(start_date, periods=7)

            mock_model.predict.return_value = mock_prediction
            mock_model.best_model_name = "TestModel"
            mock_model.best_model_params = {}
            mock_model.best_model_transformation_params = {}

            mock_autots_instance = Mock()
            mock_autots_instance.fit.return_value = mock_model
            mock_autots.return_value = mock_autots_instance

            forecaster.forecast(forecast_data, max_generations=10)

            # Should pass through to AutoTS
            call_args = mock_autots.call_args_list[0]
            assert call_args[1]["max_generations"] == 10


class TestForecastingEdgeCases:
    """Test edge cases in forecasting."""

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        # Only 5 data points
        short_data = pd.DataFrame(
            {"Close": [100, 101, 99, 102, 98]},
            index=pd.date_range("2023-01-01", periods=5),
        )

        forecaster = StockForecaster(forecast_length=30)

        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            # Mock AutoTS to raise an error
            mock_autots_instance = Mock()
            mock_autots_instance.fit.side_effect = ValueError("Insufficient data")
            mock_autots.return_value = mock_autots_instance

            with pytest.raises(ValueError, match="Insufficient data"):
                forecaster.fit_predict(short_data)

    def test_single_value_data(self):
        """Test handling of constant data."""
        # All values the same
        constant_data = pd.DataFrame({"Close": [100] * 50}, index=pd.date_range("2023-01-01", periods=50))

        forecaster = StockForecaster()

        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            # Mock to return constant predictions
            mock_model = Mock()
            # Use the helper to create proper prediction structure
            start_date = constant_data.index[-1] + timedelta(days=1)
            mock_prediction = create_mock_autots_prediction(start_date, periods=14, base_value=100.0)

            mock_model.predict.return_value = mock_prediction
            mock_autots_instance = Mock()
            mock_autots_instance.fit.return_value = mock_model
            mock_autots.return_value = mock_autots_instance

            predictions = forecaster.fit_predict(constant_data)

            # Should handle constant data
            assert all(predictions["forecast"] == 100)

    def test_missing_data_handling(self):
        """Test handling of missing data."""
        # Data with gaps
        dates = pd.date_range("2023-01-01", periods=100)
        data_with_gaps = pd.DataFrame({"Close": range(100)}, index=dates)

        # Remove some data points
        data_with_gaps = data_with_gaps.drop(data_with_gaps.index[20:30])

        forecaster = StockForecaster(forecast_length=7)

        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            # Create a proper mock model that returns valid predictions
            mock_model = Mock()

            # Use the helper to create proper prediction structure
            start_date = data_with_gaps.index[-1] + timedelta(days=1)
            mock_prediction = create_mock_autots_prediction(start_date, periods=7)

            mock_model.predict.return_value = mock_prediction
            mock_model.best_model_name = "TestModel"
            mock_model.best_model_params = {}
            mock_model.best_model_transformation_params = {}

            mock_autots_instance = Mock()
            mock_autots_instance.fit.return_value = mock_model
            mock_autots.return_value = mock_autots_instance

            # Should handle gaps in data
            forecaster.fit_predict(data_with_gaps)
            assert mock_autots_instance.fit.called

    def test_extreme_forecast_length(self):
        """Test extreme forecast lengths."""
        data = pd.DataFrame({"Close": range(100)}, index=pd.date_range("2023-01-01", periods=100))

        # Very long forecast
        forecaster = StockForecaster(forecast_length=365)

        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            # Mock long predictions
            mock_model = Mock()
            # Use the helper to create proper prediction structure
            start_date = data.index[-1] + timedelta(days=1)
            mock_prediction = create_mock_autots_prediction(start_date, periods=365, base_value=150.0)

            mock_model.predict.return_value = mock_prediction
            mock_autots_instance = Mock()
            mock_autots_instance.fit.return_value = mock_model
            mock_autots.return_value = mock_autots_instance

            predictions = forecaster.fit_predict(data)
            assert len(predictions) == 365

    def test_model_failure_handling(self, forecast_data):
        """Test handling of model fitting failures."""
        forecaster = StockForecaster()

        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            # Mock AutoTS to fail
            mock_autots_instance = Mock()
            mock_autots_instance.fit.side_effect = Exception("Model fitting failed")
            mock_autots.return_value = mock_autots_instance

            with pytest.raises(Exception, match="Model fitting failed"):
                forecaster.forecast(forecast_data)


class TestForecastingIntegration:
    """Test forecasting integration with other components."""

    def test_forecast_with_technical_indicators(self, forecast_data):
        """Test that forecasting works with data containing technical indicators."""
        # Add some technical indicator columns
        enhanced_data = forecast_data.copy()
        enhanced_data["SMA_20"] = enhanced_data["Close"].rolling(20).mean()
        enhanced_data["RSI"] = 50  # Dummy RSI

        forecaster = StockForecaster()

        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            mock_model = Mock()
            # Use the helper to create proper prediction structure
            start_date = enhanced_data.index[-1] + timedelta(days=1)
            mock_prediction = create_mock_autots_prediction(start_date, periods=14)

            mock_model.predict.return_value = mock_prediction
            mock_autots_instance = Mock()
            mock_autots_instance.fit.return_value = mock_model
            mock_autots.return_value = mock_autots_instance

            # Should use only Close column for forecasting
            forecaster.forecast(enhanced_data)

            # Verify fit was called with only Close data
            fit_call_args = mock_autots_instance.fit.call_args
            fit_data = fit_call_args[0][0]
            assert "Close" in fit_data.columns or fit_data.name == "Close"
