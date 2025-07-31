"""Unit tests for forecasting module."""

import logging
import sys
import warnings
from datetime import timedelta
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from stockula.forecasting.forecaster import (
    StockForecaster,
    suppress_autots_output,
)


class TestSuppressAutoTSOutput:
    """Test the suppress_autots_output context manager."""

    def test_suppress_warnings(self):
        """Test that warnings are suppressed."""
        with suppress_autots_output():
            # This warning should be suppressed
            warnings.warn("Test warning", UserWarning, stacklevel=2)
            # No warning should be raised

    def test_suppress_stdout_when_not_debug(self):
        """Test stdout suppression when not in debug mode."""
        logger = logging.getLogger("stockula.forecasting.forecaster")
        original_level = logger.level
        logger.setLevel(logging.INFO)  # Not DEBUG

        original_stdout = sys.stdout
        original_stderr = sys.stderr

        with suppress_autots_output():
            # stdout and stderr should be redirected to FilteredOutput
            assert sys.stdout != original_stdout
            assert sys.stderr != original_stderr
            # Check that it's our FilteredOutput wrapper
            assert hasattr(sys.stdout, "original_stream")
            assert hasattr(sys.stderr, "original_stream")
            assert sys.stdout.__class__.__name__ == "FilteredStream"
            assert sys.stderr.__class__.__name__ == "FilteredStream"

        # Should be restored
        assert sys.stdout == original_stdout
        assert sys.stderr == original_stderr

        # Restore original level
        logger.setLevel(original_level)

    def test_no_suppress_when_debug(self):
        """Test filtering in debug mode."""
        logger = logging.getLogger("stockula.forecasting.forecaster")
        original_level = logger.level
        logger.setLevel(logging.DEBUG)

        original_stdout = sys.stdout
        original_stderr = sys.stderr

        with suppress_autots_output():
            # In debug mode, stdout/stderr are wrapped with FilteredOutput
            assert sys.stdout != original_stdout
            assert sys.stderr != original_stderr

            # Verify it's FilteredOutput with the original stream preserved
            assert hasattr(sys.stdout, "original_stream")
            assert hasattr(sys.stderr, "original_stream")
            assert sys.stdout.original_stream == original_stdout
            assert sys.stderr.original_stream == original_stderr
            assert sys.stdout.__class__.__name__ == "FilteredStream"
            assert sys.stderr.__class__.__name__ == "FilteredStream"

        # Should be restored after context
        assert sys.stdout == original_stdout
        assert sys.stderr == original_stderr

        # Restore original level
        logger.setLevel(original_level)


class TestStockForecasterInitialization:
    """Test StockForecaster initialization."""

    def test_initialization_with_defaults(self):
        """Test initialization with default parameters."""
        forecaster = StockForecaster()

        assert forecaster.forecast_length is None  # Changed to None for mutual exclusivity
        assert forecaster.frequency == "infer"
        assert forecaster.prediction_interval == 0.95
        assert forecaster.num_validations == 2
        assert forecaster.validation_method == "backwards"
        assert forecaster.model is None
        assert forecaster.prediction is None

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        forecaster = StockForecaster(
            forecast_length=30,
            frequency="D",
            prediction_interval=0.95,
            num_validations=3,
            validation_method="seasonal",
            data_fetcher=None,
        )

        assert forecaster.forecast_length == 30
        assert forecaster.frequency == "D"
        assert forecaster.prediction_interval == 0.95
        assert forecaster.num_validations == 3
        assert forecaster.validation_method == "seasonal"


class TestStockForecasterFit:
    """Test StockForecaster fit method."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        prices = 100 + np.cumsum(np.random.randn(100) * 2)
        return pd.DataFrame({"Close": prices}, index=dates)

    @pytest.fixture
    def mock_autots(self):
        """Create mock AutoTS."""
        with patch("stockula.forecasting.forecaster.AutoTS") as mock:
            mock_instance = Mock()
            mock_model = Mock()
            mock_model.best_model_name = "ARIMA"
            mock_model.best_model_params = {"p": 1, "d": 1, "q": 1}
            mock_model.best_model_transformation_params = {}
            mock_instance.fit.return_value = mock_model
            mock.return_value = mock_instance
            yield mock, mock_instance, mock_model

    def test_fit_with_valid_data(self, sample_data, mock_autots):
        """Test fitting with valid data."""
        mock_autots_class, mock_instance, _ = mock_autots

        forecaster = StockForecaster(forecast_length=14)  # Specify forecast_length

        # Set up signal handler mock
        with patch("stockula.forecasting.forecaster.signal.signal"):
            result = forecaster.fit(sample_data)

        # Should return self
        assert result is forecaster

        # Should have called AutoTS
        mock_autots_class.assert_called_once()

        # Check AutoTS parameters
        call_kwargs = mock_autots_class.call_args[1]
        assert call_kwargs["forecast_length"] == 14
        assert call_kwargs["frequency"] == "D"  # "infer" is converted to "D" to avoid warnings
        assert call_kwargs["prediction_interval"] == 0.95
        assert call_kwargs["verbose"] == 0
        assert call_kwargs["no_negatives"] is True

        # Should have called fit
        mock_instance.fit.assert_called_once()
        fit_args = mock_instance.fit.call_args[0]
        fit_df = fit_args[0]

        # Check data format
        assert "date" in fit_df.columns
        assert "Close" in fit_df.columns
        assert len(fit_df) == len(sample_data)

    def test_fit_missing_target_column(self):
        """Test fit with missing target column."""
        data = pd.DataFrame({"Price": [100, 101, 102]}, index=pd.date_range("2023-01-01", periods=3))

        forecaster = StockForecaster()

        with pytest.raises(ValueError, match="Target column 'Close' not found"):
            forecaster.fit(data, target_column="Close")

    def test_fit_with_custom_parameters(self, sample_data, mock_autots):
        """Test fit with custom parameters."""
        mock_autots_class, _, _ = mock_autots

        forecaster = StockForecaster()

        with patch("stockula.forecasting.forecaster.signal.signal"):
            forecaster.fit(sample_data, model_list="slow", ensemble="simple", max_generations=10)

        # Check AutoTS parameters
        call_kwargs = mock_autots_class.call_args[1]
        assert call_kwargs["model_list"] == "slow"
        assert call_kwargs["ensemble"] == "simple"
        assert call_kwargs["max_generations"] == 10

    def test_fit_keyboard_interrupt(self, sample_data, mock_autots):
        """Test handling keyboard interrupt during fit."""
        _, mock_instance, _ = mock_autots
        mock_instance.fit.side_effect = KeyboardInterrupt()

        forecaster = StockForecaster()

        with patch("stockula.forecasting.forecaster.signal.signal"):
            with pytest.raises(KeyboardInterrupt):
                forecaster.fit(sample_data)

    def test_fit_general_exception(self, sample_data, mock_autots):
        """Test handling general exception during fit."""
        _, mock_instance, _ = mock_autots
        mock_instance.fit.side_effect = Exception("Model error")

        forecaster = StockForecaster()

        with patch("stockula.forecasting.forecaster.signal.signal"):
            with pytest.raises(Exception, match="Model error"):
                forecaster.fit(sample_data)


class TestStockForecasterPredict:
    """Test StockForecaster predict method."""

    @pytest.fixture
    def fitted_forecaster(self):
        """Create a fitted forecaster with mocked model."""
        forecaster = StockForecaster()

        # Mock the model
        mock_model = Mock()
        mock_prediction = Mock()

        # Create forecast data
        forecast_dates = pd.date_range("2023-02-01", periods=14)
        mock_prediction.forecast = pd.DataFrame({"TEST": [110 + i for i in range(14)]}, index=forecast_dates)
        mock_prediction.upper_forecast = pd.DataFrame({"TEST": [115 + i for i in range(14)]}, index=forecast_dates)
        mock_prediction.lower_forecast = pd.DataFrame({"TEST": [105 + i for i in range(14)]}, index=forecast_dates)

        mock_model.predict.return_value = mock_prediction
        forecaster.model = mock_model

        return forecaster, mock_model, mock_prediction

    def test_predict_with_fitted_model(self, fitted_forecaster):
        """Test prediction with fitted model."""
        forecaster, mock_model, _ = fitted_forecaster

        result = forecaster.predict()

        # Should call model.predict
        mock_model.predict.assert_called_once()

        # Check result format
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 14
        assert "forecast" in result.columns
        assert "lower_bound" in result.columns
        assert "upper_bound" in result.columns

        # Check values
        assert result["forecast"].iloc[0] == 110
        assert result["upper_bound"].iloc[0] == 115
        assert result["lower_bound"].iloc[0] == 105

    def test_predict_without_fitted_model(self):
        """Test prediction without fitted model."""
        forecaster = StockForecaster()

        with pytest.raises(ValueError, match="Model not fitted"):
            forecaster.predict()


class TestStockForecasterFitPredict:
    """Test StockForecaster fit_predict method."""

    def test_fit_predict(self):
        """Test fit_predict method."""
        forecaster = StockForecaster()

        # Mock fit and predict methods
        mock_predictions = pd.DataFrame(
            {
                "forecast": [110, 111, 112],
                "lower_bound": [105, 106, 107],
                "upper_bound": [115, 116, 117],
            }
        )

        with patch.object(forecaster, "fit") as mock_fit:
            with patch.object(forecaster, "predict") as mock_predict:
                mock_fit.return_value = forecaster
                mock_predict.return_value = mock_predictions

                data = pd.DataFrame({"Close": [100, 101, 102]})
                result = forecaster.fit_predict(data, target_column="Close", model_list="fast")

                # Should call both methods
                mock_fit.assert_called_once_with(data, "Close", model_list="fast")
                mock_predict.assert_called_once()

                # Should return predictions
                assert result.equals(mock_predictions)


class TestStockForecasterForecastFromSymbol:
    """Test forecast_from_symbol method."""

    def test_forecast_from_symbol_success(self):
        """Test successful forecast from symbol."""
        # Mock data
        stock_data = pd.DataFrame(
            {"Close": np.random.randn(100).cumsum() + 100},
            index=pd.date_range("2023-01-01", periods=100),
        )

        # Mock predictions
        predictions = pd.DataFrame(
            {
                "forecast": [110, 111, 112],
                "lower_bound": [105, 106, 107],
                "upper_bound": [115, 116, 117],
            }
        )

        # Create mock data fetcher
        mock_fetcher = Mock()
        mock_fetcher.get_stock_data.return_value = stock_data

        # Create forecaster with mock data fetcher
        forecaster = StockForecaster(data_fetcher=mock_fetcher)

        with patch.object(forecaster, "fit_predict") as mock_fit_predict:
            mock_fit_predict.return_value = predictions

            result = forecaster.forecast_from_symbol("AAPL", start_date="2023-01-01")

            # Should fetch data
            mock_fetcher.get_stock_data.assert_called_once_with("AAPL", "2023-01-01", None)

            # Should fit and predict
            mock_fit_predict.assert_called_once()

            # Should return predictions
            assert result.equals(predictions)

    def test_forecast_from_symbol_no_data_fetcher(self):
        """Test forecast from symbol without data fetcher configured."""
        forecaster = StockForecaster(data_fetcher=None)

        with pytest.raises(ValueError, match="Data fetcher not configured"):
            forecaster.forecast_from_symbol("TEST")

    def test_forecast_from_symbol_no_data(self):
        """Test forecast from symbol with no data available."""
        # Create mock data fetcher
        mock_fetcher = Mock()
        mock_fetcher.get_stock_data.return_value = pd.DataFrame()  # Empty

        forecaster = StockForecaster(data_fetcher=mock_fetcher)

        with pytest.raises(ValueError, match="No data available for symbol TEST"):
            forecaster.forecast_from_symbol("TEST")


class TestStockForecasterGetBestModel:
    """Test get_best_model method."""

    def test_get_best_model_fitted(self):
        """Test getting best model info when fitted."""
        forecaster = StockForecaster()

        # Mock fitted model
        mock_model = Mock()
        mock_model.best_model_name = "ARIMA"
        mock_model.best_model_params = {"p": 1, "d": 1, "q": 1}
        mock_model.best_model_transformation_params = {"fillna": "mean"}
        mock_model.best_model_accuracy = 0.95

        forecaster.model = mock_model

        info = forecaster.get_best_model()

        assert info["model_name"] == "ARIMA"
        assert info["model_params"] == {"p": 1, "d": 1, "q": 1}
        assert info["model_transformation"] == {"fillna": "mean"}
        assert info["model_accuracy"] == 0.95

    def test_get_best_model_not_fitted(self):
        """Test getting best model info when not fitted."""
        forecaster = StockForecaster()

        with pytest.raises(ValueError, match="Model not fitted"):
            forecaster.get_best_model()

    def test_get_best_model_no_accuracy(self):
        """Test getting best model when accuracy not available."""
        forecaster = StockForecaster()

        # Mock model without accuracy attribute
        mock_model = Mock(
            spec=[
                "best_model_name",
                "best_model_params",
                "best_model_transformation_params",
            ]
        )
        mock_model.best_model_name = "Prophet"
        mock_model.best_model_params = {}
        mock_model.best_model_transformation_params = {}

        forecaster.model = mock_model

        info = forecaster.get_best_model()

        assert info["model_accuracy"] == "N/A"


class TestStockForecasterPlotForecast:
    """Test plot_forecast method."""

    def test_plot_forecast_with_prediction(self):
        """Test plotting forecast with prediction available."""
        forecaster = StockForecaster()

        # Mock prediction with plot method
        mock_prediction = Mock()
        forecaster.prediction = mock_prediction

        # Mock model with df_wide_numeric attribute
        mock_model = Mock()
        mock_model.df_wide_numeric = pd.DataFrame({"test": [1, 2, 3]})
        forecaster.model = mock_model

        # Mock historical data
        historical = pd.DataFrame({"Close": [100, 101, 102]})

        forecaster.plot_forecast(historical_data=historical, n_historical=50)

        # Should call prediction.plot
        mock_prediction.plot.assert_called_once_with(
            mock_model.df_wide_numeric, remove_zero_series=False, start_date=-50
        )

    def test_plot_forecast_no_prediction(self):
        """Test plotting forecast without prediction."""
        forecaster = StockForecaster()

        with pytest.raises(ValueError, match="No predictions available"):
            forecaster.plot_forecast()


class TestStockForecasterForecastAlias:
    """Test forecast method (alias for fit_predict)."""

    def test_forecast_alias(self):
        """Test that forecast is an alias for fit_predict."""
        forecaster = StockForecaster()

        # Mock fit_predict
        mock_result = pd.DataFrame({"forecast": [110, 111, 112]})
        with patch.object(forecaster, "fit_predict") as mock_fit_predict:
            mock_fit_predict.return_value = mock_result

            data = pd.DataFrame({"Close": [100, 101, 102]})
            result = forecaster.forecast(data, target_column="Price", model_list="fast")

            # Should call fit_predict with same arguments
            mock_fit_predict.assert_called_once_with(data, "Price", model_list="fast")

            # Should return same result
            assert result.equals(mock_result)


class TestStockForecasterIntegration:
    """Integration tests for StockForecaster."""

    def test_full_workflow(self):
        """Test complete forecasting workflow."""
        # Create realistic data
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", periods=200, freq="D")
        trend = np.linspace(100, 120, 200)
        noise = np.random.normal(0, 2, 200)
        seasonal = 5 * np.sin(2 * np.pi * np.arange(200) / 30)
        prices = trend + seasonal + noise

        data = pd.DataFrame({"Close": prices}, index=dates)

        # Mock AutoTS completely
        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            # Set up mock
            mock_instance = Mock()
            mock_model = Mock()
            mock_prediction = Mock()

            # Mock forecast results
            forecast_dates = pd.date_range(dates[-1] + timedelta(days=1), periods=14)
            mock_prediction.forecast = pd.DataFrame({"Close": [125 + i * 0.5 for i in range(14)]}, index=forecast_dates)
            mock_prediction.upper_forecast = pd.DataFrame(
                {"Close": [130 + i * 0.5 for i in range(14)]}, index=forecast_dates
            )
            mock_prediction.lower_forecast = pd.DataFrame(
                {"Close": [120 + i * 0.5 for i in range(14)]}, index=forecast_dates
            )

            mock_model.predict.return_value = mock_prediction
            mock_model.best_model_name = "SeasonalNaive"
            mock_model.best_model_params = {}
            mock_model.best_model_transformation_params = {}

            mock_instance.fit.return_value = mock_model
            mock_autots.return_value = mock_instance

            # Create forecaster
            forecaster = StockForecaster(forecast_length=14, frequency="D", prediction_interval=0.95)

            # Run full workflow
            with patch("stockula.forecasting.forecaster.signal.signal"):
                predictions = forecaster.fit_predict(data)

            # Verify results
            assert isinstance(predictions, pd.DataFrame)
            assert len(predictions) == 14
            assert all(col in predictions.columns for col in ["forecast", "lower_bound", "upper_bound"])

            # Check model info
            model_info = forecaster.get_best_model()
            assert model_info["model_name"] == "SeasonalNaive"

    def test_logging_messages(self, caplog):
        """Test that appropriate log messages are generated."""
        data = pd.DataFrame(
            {"Close": [100, 101, 102, 103, 104]},
            index=pd.date_range("2023-01-01", periods=5),
        )

        forecaster = StockForecaster()

        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            mock_instance = Mock()
            mock_model = Mock()
            mock_instance.fit.return_value = mock_model
            mock_autots.return_value = mock_instance

            with patch("stockula.forecasting.forecaster.signal.signal"):
                # Set the specific logger to DEBUG level as well
                with caplog.at_level(logging.DEBUG, logger="stockula.forecasting.forecaster"):
                    forecaster.fit(data)

            # Check log messages
            assert "Fitting model on 5 data points" in caplog.text
            assert "Model fitting completed" in caplog.text
