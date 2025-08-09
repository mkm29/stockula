"""Tests for forecasting module - Fixed version with proper DI."""

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

    def test_initialization(self, integration_container):
        """Test StockForecaster initialization."""
        logging_manager = integration_container.logging_manager()
        forecaster = StockForecaster(
            forecast_length=30, frequency="D", prediction_interval=0.95, logging_manager=logging_manager
        )
        assert forecaster.forecast_length == 30
        assert forecaster.frequency == "D"
        assert forecaster.prediction_interval == 0.95

        # Test defaults
        forecaster_default = StockForecaster(logging_manager=logging_manager)
        assert forecaster_default.forecast_length is None
        assert forecaster_default.frequency == "infer"
        assert forecaster_default.prediction_interval == 0.95

    def test_forecast_with_data(self, forecast_data, integration_stock_forecaster):
        """Test forecasting with provided data."""
        forecaster = integration_stock_forecaster
        forecaster.forecast_length = 7

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

    def test_forecast_from_symbol(self, forecast_data, integration_container):
        """Test forecasting from symbol."""
        # Create a mock DataFetcher that returns our test data
        mock_fetcher = Mock()
        mock_fetcher.get_stock_data = Mock(return_value=forecast_data)

        # Pass the data fetcher to the forecaster
        forecaster = StockForecaster(
            forecast_length=14, data_fetcher=mock_fetcher, logging_manager=integration_container.logging_manager()
        )

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

            # Set forecast length
            forecaster.forecast_length = 14
            # Run forecast from symbol
            predictions = forecaster.forecast_from_symbol("AAPL")

            # Check results
            assert isinstance(predictions, pd.DataFrame)
            assert len(predictions) == 14
            assert mock_fetcher.get_stock_data.called
            assert mock_autots_instance.fit.called

    def test_get_best_model(self, integration_stock_forecaster):
        """Test getting best model information."""
        forecaster = integration_stock_forecaster
        # Before fitting, should raise error
        with pytest.raises(ValueError, match="Model not fitted"):
            forecaster.get_best_model()

    def test_model_list_parameter(self, forecast_data, integration_container):
        """Test different model list parameters."""
        logging_manager = integration_container.logging_manager()
        # Test with ultra_fast preset
        forecaster = StockForecaster(forecast_length=7, model_list="ultra_fast", logging_manager=logging_manager)

        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            mock_model = Mock()
            start_date = forecast_data.index[-1] + timedelta(days=1)
            mock_prediction = create_mock_autots_prediction(start_date, periods=7)
            mock_model.predict.return_value = mock_prediction

            mock_autots_instance = Mock()
            mock_autots_instance.fit.return_value = mock_model
            mock_autots.return_value = mock_autots_instance

            forecaster.fit_predict(forecast_data)

            # Check that AutoTS was called with ultra_fast models
            autots_call_args = mock_autots.call_args
            assert autots_call_args.kwargs["model_list"] == StockForecaster.ULTRA_FAST_MODEL_LIST

    def test_ensemble_parameter(self, forecast_data, integration_stock_forecaster):
        """Test ensemble parameter."""
        forecaster = integration_stock_forecaster
        forecaster.forecast_length = 7
        forecaster.ensemble = "simple"

        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            mock_model = Mock()
            start_date = forecast_data.index[-1] + timedelta(days=1)
            mock_prediction = create_mock_autots_prediction(start_date, periods=7)
            mock_model.predict.return_value = mock_prediction

            mock_autots_instance = Mock()
            mock_autots_instance.fit.return_value = mock_model
            mock_autots.return_value = mock_autots_instance

            forecaster.fit_predict(forecast_data)

            # Check that AutoTS was called with ensemble parameter
            autots_call_args = mock_autots.call_args
            assert autots_call_args.kwargs["ensemble"] == "simple"

    def test_validation_parameters(self, forecast_data, integration_container):
        """Test validation parameters."""
        forecaster = StockForecaster(
            forecast_length=7,
            num_validations=1,
            validation_method="backwards",
            logging_manager=integration_container.logging_manager(),
        )

        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            mock_model = Mock()
            start_date = forecast_data.index[-1] + timedelta(days=1)
            mock_prediction = create_mock_autots_prediction(start_date, periods=7)
            mock_model.predict.return_value = mock_prediction

            mock_autots_instance = Mock()
            mock_autots_instance.fit.return_value = mock_model
            mock_autots.return_value = mock_autots_instance

            forecaster.fit_predict(forecast_data)

            # Check that AutoTS was called with validation parameters
            autots_call_args = mock_autots.call_args
            assert autots_call_args.kwargs["num_validations"] == 1
            assert autots_call_args.kwargs["validation_method"] == "backwards"

    def test_max_generations(self, forecast_data, integration_container):
        """Test max_generations parameter."""
        forecaster = StockForecaster(
            forecast_length=7, max_generations=1, logging_manager=integration_container.logging_manager()
        )

        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            mock_model = Mock()
            start_date = forecast_data.index[-1] + timedelta(days=1)
            mock_prediction = create_mock_autots_prediction(start_date, periods=7)
            mock_model.predict.return_value = mock_prediction

            mock_autots_instance = Mock()
            mock_autots_instance.fit.return_value = mock_model
            mock_autots.return_value = mock_autots_instance

            forecaster.fit_predict(forecast_data)

            # Check that AutoTS was called with max_generations
            autots_call_args = mock_autots.call_args
            assert autots_call_args.kwargs["max_generations"] == 1

    def test_frequency_inference(self, forecast_data, integration_container):
        """Test frequency inference."""
        forecaster = StockForecaster(
            forecast_length=7, frequency="infer", logging_manager=integration_container.logging_manager()
        )

        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            mock_model = Mock()
            start_date = forecast_data.index[-1] + timedelta(days=1)
            mock_prediction = create_mock_autots_prediction(start_date, periods=7)
            mock_model.predict.return_value = mock_prediction

            mock_autots_instance = Mock()
            mock_autots_instance.fit.return_value = mock_model
            mock_autots.return_value = mock_autots_instance

            forecaster.fit_predict(forecast_data)

            # Check that frequency was inferred
            autots_call_args = mock_autots.call_args
            # Since we have daily data, it should infer 'D'
            assert autots_call_args.kwargs["frequency"] == "D"


class TestForecastingIntegration:
    """Test forecasting integration with other components."""

    def test_forecast_with_technical_indicators(self, forecast_data, integration_stock_forecaster):
        """Test that forecasting works with data containing technical indicators."""
        # Add some technical indicator columns
        enhanced_data = forecast_data.copy()
        enhanced_data["SMA_20"] = enhanced_data["Close"].rolling(20).mean()
        enhanced_data["RSI"] = 50  # Dummy RSI
        forecaster = integration_stock_forecaster

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


class TestForecastingEdgeCases:
    """Test edge cases in forecasting."""

    def test_insufficient_data(self, integration_container):
        """Test handling of insufficient data."""
        # Only 5 data points
        short_data = pd.DataFrame(
            {"Close": [100, 101, 99, 102, 98]},
            index=pd.date_range("2023-01-01", periods=5),
        )

        forecaster = StockForecaster(forecast_length=30, logging_manager=integration_container.logging_manager())

        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            # Mock AutoTS to raise an error
            mock_autots_instance = Mock()
            mock_autots_instance.fit.side_effect = ValueError("Insufficient data")
            mock_autots.return_value = mock_autots_instance

            with pytest.raises(ValueError, match="Insufficient data"):
                forecaster.fit_predict(short_data)

    def test_single_value_data(self, integration_container):
        """Test handling of constant data."""
        # All values the same
        constant_data = pd.DataFrame({"Close": [100] * 50}, index=pd.date_range("2023-01-01", periods=50))

        forecaster = StockForecaster(logging_manager=integration_container.logging_manager())

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

    def test_missing_data_handling(self, integration_container):
        """Test handling of missing data."""
        # Data with gaps
        dates = pd.date_range("2023-01-01", periods=100)
        data_with_gaps = pd.DataFrame({"Close": range(100)}, index=dates)
        # Introduce gaps
        data_with_gaps.iloc[20:30] = None
        data_with_gaps.iloc[60:65] = None

        forecaster = StockForecaster(logging_manager=integration_container.logging_manager())

        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            mock_model = Mock()
            start_date = data_with_gaps.index[-1] + timedelta(days=1)
            mock_prediction = create_mock_autots_prediction(start_date, periods=14)
            mock_model.predict.return_value = mock_prediction

            mock_autots_instance = Mock()
            mock_autots_instance.fit.return_value = mock_model
            mock_autots.return_value = mock_autots_instance

            # Should handle missing data (AutoTS will forward fill internally)
            predictions = forecaster.fit_predict(data_with_gaps)
            assert isinstance(predictions, pd.DataFrame)

    def test_extreme_forecast_length(self, forecast_data, integration_container):
        """Test very long forecast horizon."""
        # Request 365 day forecast
        forecaster = StockForecaster(forecast_length=365, logging_manager=integration_container.logging_manager())

        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            mock_model = Mock()
            start_date = forecast_data.index[-1] + timedelta(days=1)
            mock_prediction = create_mock_autots_prediction(start_date, periods=365)
            mock_model.predict.return_value = mock_prediction

            mock_autots_instance = Mock()
            mock_autots_instance.fit.return_value = mock_model
            mock_autots.return_value = mock_autots_instance

            predictions = forecaster.fit_predict(forecast_data)

            # Should handle long forecast
            assert len(predictions) == 365

    def test_model_failure_handling(self, forecast_data, integration_container):
        """Test handling when all models fail."""
        forecaster = StockForecaster(
            forecast_length=7,
            model_list=["LastValueNaive"],  # Limited model list
            logging_manager=integration_container.logging_manager(),
        )

        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            # Mock AutoTS to raise a specific error
            mock_autots_instance = Mock()
            mock_autots_instance.fit.side_effect = Exception("All models failed")
            mock_autots.return_value = mock_autots_instance

            with pytest.raises(Exception, match="All models failed"):
                forecaster.fit_predict(forecast_data)
