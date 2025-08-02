"""Consolidated unit tests for forecasting module."""

import logging
import sys
import warnings
from datetime import timedelta
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from stockula.forecasting.forecaster import StockForecaster, SuppressAutoTSOutput, suppress_autots_output


@pytest.fixture
def mock_logging_manager():
    """Create a mock logging manager for forecaster tests."""
    logger = Mock()
    logger.debug = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.isEnabledFor = Mock(return_value=False)  # Return False for debug level by default
    return logger


class TestSuppressAutoTSOutput:
    """Test the suppress_autots_output context manager."""

    def test_suppress_warnings(self, mock_logging_manager):
        """Test that warnings are suppressed."""
        with suppress_autots_output():
            # This warning should be suppressed
            warnings.warn("Test warning", UserWarning, stacklevel=2)
            # No warning should be raised

    def test_suppress_stdout_when_not_debug(self, mock_logging_manager):
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

    def test_no_suppress_when_debug(self, mock_logging_manager):
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

    def test_filtered_stream_flush(self, mock_logging_manager):
        """Test FilteredStream flush method."""
        suppressor = SuppressAutoTSOutput()
        suppressor.__enter__()

        # Get the filtered stream
        filtered_stream = sys.stdout

        # Test flush
        filtered_stream.flush()  # Should not raise

        suppressor.__exit__(None, None, None)

    def test_filtered_stream_getattr(self, mock_logging_manager):
        """Test FilteredStream __getattr__ delegation."""
        suppressor = SuppressAutoTSOutput()
        suppressor.__enter__()

        # Get the filtered stream
        filtered_stream = sys.stdout

        # Test attribute delegation
        assert hasattr(filtered_stream, "encoding")  # Should delegate to original stream

        suppressor.__exit__(None, None, None)

    def test_filtered_stream_non_debug_mode_patterns(self, mock_logging_manager):
        """Test FilteredStream pattern filtering in non-debug mode."""
        import logging

        from stockula.forecasting.forecaster import SuppressAutoTSOutput

        logger = logging.getLogger("stockula.forecasting.forecaster")

        # Mock logger to be in non-debug mode
        with patch.object(logger, "isEnabledFor", return_value=False):
            suppressor = SuppressAutoTSOutput()
            suppressor.__enter__()

            # Get the filtered stream
            filtered_stream = sys.stdout

            # Mock the original stream
            filtered_stream.original_stream = Mock()

            # Test various patterns that should be suppressed
            patterns_to_suppress = [
                "Model Number: 123",
                "New Generation: 5",
                "Template Eval Error: something",
                "2025-01-01 results",
                "UserWarning: test",
                "SVD did not converge",
                "Optimization terminated abnormally",
            ]

            for pattern in patterns_to_suppress:
                filtered_stream.original_stream.reset_mock()
                result = filtered_stream.write(pattern)
                assert result == len(pattern)
                filtered_stream.original_stream.write.assert_not_called()

            # Test pattern that should pass through
            filtered_stream.original_stream.reset_mock()
            normal_message = "Normal log message"
            result = filtered_stream.write(normal_message)
            assert result == len(normal_message)
            filtered_stream.original_stream.write.assert_called_once_with(normal_message)

            suppressor.__exit__(None, None, None)


class TestStockForecasterInitialization:
    """Test StockForecaster initialization."""

    def test_initialization_with_defaults(self, mock_logging_manager):
        """Test initialization with default parameters."""
        forecaster = StockForecaster(logging_manager=mock_logging_manager)

        assert forecaster.forecast_length is None  # Changed to None for mutual exclusivity
        assert forecaster.frequency == "infer"
        assert forecaster.prediction_interval == 0.95
        assert forecaster.num_validations == 2
        assert forecaster.validation_method == "backwards"
        assert forecaster.model is None
        assert forecaster.prediction is None

    def test_initialization_with_custom_params(self, mock_logging_manager):
        """Test initialization with custom parameters."""
        forecaster = StockForecaster(
            forecast_length=30,
            frequency="D",
            prediction_interval=0.95,
            num_validations=3,
            validation_method="seasonal",
            data_fetcher=None,
            logging_manager=mock_logging_manager,
        )

        assert forecaster.forecast_length == 30
        assert forecaster.frequency == "D"
        assert forecaster.prediction_interval == 0.95
        assert forecaster.num_validations == 3
        assert forecaster.validation_method == "seasonal"

    def test_init_with_custom_model_list(self, mock_logging_manager):
        """Test initialization with custom model list."""
        forecaster = StockForecaster(
            model_list=["ARIMA", "ETS", "Theta"],
            forecast_length=30,
            max_generations=5,
            logging_manager=mock_logging_manager,
        )

        assert forecaster.model_list == ["ARIMA", "ETS", "Theta"]
        assert forecaster.max_generations == 5

    def test_init_with_preset_model_list(self, mock_logging_manager):
        """Test initialization with preset model lists."""
        # Test 'fast' preset
        forecaster = StockForecaster(model_list="fast", forecast_length=30, logging_manager=mock_logging_manager)
        assert forecaster.model_list == "fast"

        # Test 'ultra_fast' preset
        forecaster = StockForecaster(model_list="ultra_fast", forecast_length=30, logging_manager=mock_logging_manager)
        assert forecaster.model_list == "ultra_fast"

        # Test 'financial' preset
        forecaster = StockForecaster(model_list="financial", forecast_length=30, logging_manager=mock_logging_manager)
        assert forecaster.model_list == "financial"

        # Test 'fast_financial' preset
        forecaster = StockForecaster(
            model_list="fast_financial", forecast_length=30, logging_manager=mock_logging_manager
        )
        assert forecaster.model_list == "fast_financial"

    def test_init_with_all_parameters(self, mock_logging_manager):
        """Test initialization with all parameters."""
        forecaster = StockForecaster(
            forecast_length=30,
            frequency="D",
            prediction_interval=0.90,
            ensemble="simple",
            num_validations=3,
            validation_method="even",
            model_list=["ARIMA", "ETS"],
            max_generations=10,
            no_negatives=False,
            logging_manager=mock_logging_manager,
        )

        assert forecaster.forecast_length == 30
        assert forecaster.frequency == "D"
        assert forecaster.prediction_interval == 0.90
        assert forecaster.ensemble == "simple"
        assert forecaster.num_validations == 3
        assert forecaster.validation_method == "even"
        assert forecaster.model_list == ["ARIMA", "ETS"]
        assert forecaster.max_generations == 10
        assert forecaster.no_negatives is False


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

    def test_fit_with_valid_data(self, sample_data, mock_autots, mock_logging_manager):
        """Test fitting with valid data."""
        mock_autots_class, mock_instance, _ = mock_autots

        # Specify forecast_length
        forecaster = StockForecaster(forecast_length=14, logging_manager=mock_logging_manager)

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

    def test_fit_missing_target_column(self, mock_logging_manager):
        """Test fit with missing target column."""
        data = pd.DataFrame({"Price": [100, 101, 102]}, index=pd.date_range("2023-01-01", periods=3))

        forecaster = StockForecaster(logging_manager=mock_logging_manager)

        with pytest.raises(ValueError, match="Target column 'Close' not found"):
            forecaster.fit(data, target_column="Close")

    def test_fit_with_custom_parameters(self, sample_data, mock_autots, mock_logging_manager):
        """Test fit with custom parameters."""
        mock_autots_class, _, _ = mock_autots

        forecaster = StockForecaster(logging_manager=mock_logging_manager)

        with patch("stockula.forecasting.forecaster.signal.signal"):
            forecaster.fit(sample_data, model_list="slow", ensemble="simple", max_generations=10)

        # Check AutoTS parameters
        call_kwargs = mock_autots_class.call_args[1]
        assert call_kwargs["model_list"] == "slow"
        assert call_kwargs["ensemble"] == "simple"
        assert call_kwargs["max_generations"] == 10

    def test_fit_keyboard_interrupt(self, sample_data, mock_autots, mock_logging_manager):
        """Test handling keyboard interrupt during fit."""
        _, mock_instance, _ = mock_autots
        mock_instance.fit.side_effect = KeyboardInterrupt()

        forecaster = StockForecaster(logging_manager=mock_logging_manager)

        with patch("stockula.forecasting.forecaster.signal.signal"):
            with pytest.raises(KeyboardInterrupt):
                forecaster.fit(sample_data)

    def test_fit_general_exception(self, sample_data, mock_autots, mock_logging_manager):
        """Test handling general exception during fit."""
        _, mock_instance, _ = mock_autots
        mock_instance.fit.side_effect = Exception("Model error")

        forecaster = StockForecaster(logging_manager=mock_logging_manager)

        with patch("stockula.forecasting.forecaster.signal.signal"):
            with pytest.raises(Exception, match="Model error"):
                forecaster.fit(sample_data)

    def test_fit_with_insufficient_data(self, mock_logging_manager):
        """Test fitting with insufficient historical data."""
        forecaster = StockForecaster(forecast_length=30, logging_manager=mock_logging_manager)

        # Create very short data (less than required)
        short_data = pd.DataFrame(
            {"Close": [100, 101, 102]},
            index=pd.date_range(start="2023-01-01", periods=3, freq="D"),
        )

        # The fit method will be called, which should handle short data appropriately
        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            mock_model = Mock()
            mock_autots.return_value = mock_model
            mock_model.fit.side_effect = ValueError("Insufficient data")

            with pytest.raises(ValueError, match="Insufficient data"):
                forecaster.fit(short_data)

    def test_fit_with_custom_target_column(self, mock_logging_manager):
        """Test fitting with custom target column."""
        forecaster = StockForecaster(forecast_length=7, logging_manager=mock_logging_manager)

        # Create data with multiple columns
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        data = pd.DataFrame(
            {
                "Open": range(100),
                "High": range(1, 101),
                "Low": range(100),
                "Close": range(100),
            },
            index=dates,
        )

        # Mock AutoTS to avoid actual model training
        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            mock_model = Mock()
            mock_autots.return_value = mock_model

            # Mock the fit method
            mock_model.fit.return_value = mock_model

            # Mock the predict method
            future_dates = pd.date_range(start="2023-04-11", periods=7, freq="D")
            mock_forecast = pd.DataFrame({"High": range(101, 108)}, index=future_dates)
            # Create a prediction object
            mock_prediction = Mock()
            mock_prediction.forecast = mock_forecast
            mock_prediction.upper_forecast = mock_forecast
            mock_prediction.lower_forecast = mock_forecast
            mock_model.predict.return_value = mock_prediction

            # Fit the model with custom target column
            forecaster.fit(data, target_column="High")

            # Verify AutoTS was called with correct parameters
            mock_autots.assert_called_once()
            call_args = mock_autots.call_args[1]
            assert call_args["forecast_length"] == 7

    @patch("stockula.forecasting.forecaster.AutoTS")
    def test_fit_with_frequency_detection(self, mock_autots, mock_logging_manager):
        """Test frequency detection in fit."""
        forecaster = StockForecaster(forecast_length=7, logging_manager=mock_logging_manager)

        # Create data with specific frequency
        dates = pd.date_range(start="2023-01-01", periods=100, freq="B")  # Business days
        data = pd.DataFrame(
            {
                "Close": range(100, 200),
            },
            index=dates,
        )

        mock_model = Mock()
        mock_autots.return_value = mock_model
        mock_model.fit.return_value = mock_model

        # Mock prediction
        future_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=7, freq="B")
        mock_forecast = pd.DataFrame({"Close": range(200, 207)}, index=future_dates)
        # Create a prediction object
        mock_prediction = Mock()
        mock_prediction.forecast = mock_forecast
        mock_prediction.upper_forecast = mock_forecast
        mock_prediction.lower_forecast = mock_forecast
        mock_model.predict.return_value = mock_prediction

        forecaster.fit(data)

        # Verify AutoTS was called with infer frequency
        call_args = mock_autots.call_args[1]
        # AutoTS detects the frequency from the data
        assert "frequency" in call_args


class TestStockForecasterPredict:
    """Test StockForecaster predict method."""

    @pytest.fixture
    def fitted_forecaster(self, mock_logging_manager):
        """Create a fitted forecaster with mocked model."""
        forecaster = StockForecaster(logging_manager=mock_logging_manager)

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

    def test_predict_without_fitted_model(self, mock_logging_manager):
        """Test prediction without fitted model."""
        forecaster = StockForecaster(logging_manager=mock_logging_manager)

        with pytest.raises(ValueError, match="Model not fitted"):
            forecaster.predict()


class TestStockForecasterFitPredict:
    """Test StockForecaster fit_predict method."""

    def test_fit_predict(self, mock_logging_manager):
        """Test fit_predict method."""
        forecaster = StockForecaster(logging_manager=mock_logging_manager)

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

    def test_forecast_alias(self, mock_logging_manager):
        """Test that forecast is an alias for fit_predict."""
        forecaster = StockForecaster(logging_manager=mock_logging_manager)

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


class TestStockForecasterForecastFromSymbol:
    """Test forecast_from_symbol method."""

    @pytest.fixture
    def mock_data_fetcher(self):
        """Create mock data fetcher."""
        return Mock()

    def test_forecast_from_symbol_success(self, mock_data_fetcher, mock_logging_manager):
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

        # Set up mock data fetcher
        mock_data_fetcher.get_stock_data.return_value = stock_data

        # Create forecaster with mock data fetcher
        forecaster = StockForecaster(data_fetcher=mock_data_fetcher, logging_manager=mock_logging_manager)

        with patch.object(forecaster, "fit_predict") as mock_fit_predict:
            mock_fit_predict.return_value = predictions

            result = forecaster.forecast_from_symbol("AAPL", start_date="2023-01-01")

            # Should fetch data
            mock_data_fetcher.get_stock_data.assert_called_once_with("AAPL", "2023-01-01", None)

            # Should fit and predict
            mock_fit_predict.assert_called_once()

            # Should return predictions
            assert result.equals(predictions)

    def test_forecast_from_symbol_no_data_fetcher(self, mock_logging_manager):
        """Test forecast from symbol without data fetcher configured."""
        forecaster = StockForecaster(data_fetcher=None, logging_manager=mock_logging_manager)

        with pytest.raises(ValueError, match="Data fetcher not configured"):
            forecaster.forecast_from_symbol("TEST")

    def test_forecast_from_symbol_no_data(self, mock_data_fetcher, mock_logging_manager):
        """Test forecast from symbol with no data available."""
        # Set up mock to return empty data
        mock_data_fetcher.get_stock_data.return_value = pd.DataFrame()  # Empty

        forecaster = StockForecaster(data_fetcher=mock_data_fetcher, logging_manager=mock_logging_manager)

        with pytest.raises(ValueError, match="No data available for symbol TEST"):
            forecaster.forecast_from_symbol("TEST")

    def test_forecast_from_symbol_with_all_parameters(self, mock_data_fetcher, mock_logging_manager):
        """Test forecast_from_symbol with all parameters."""
        forecaster = StockForecaster(
            forecast_length=30, data_fetcher=mock_data_fetcher, logging_manager=mock_logging_manager
        )

        # Create sample data
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        data = pd.DataFrame(
            {
                "Close": range(100, 200),
            },
            index=dates,
        )

        # Set up mock to return our test data
        with patch.object(mock_data_fetcher, "get_stock_data", return_value=data):
            # Mock AutoTS
            with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
                mock_model = Mock()
                mock_autots.return_value = mock_model
                mock_model.fit.return_value = mock_model

                future_dates = pd.date_range(start="2023-04-11", periods=30, freq="D")
                mock_forecast = pd.DataFrame({"Close": range(200, 230)}, index=future_dates)
                # Create a prediction object with forecast, upper_forecast, lower_forecast attributes
                mock_prediction = Mock()
                mock_prediction.forecast = mock_forecast
                mock_prediction.upper_forecast = mock_forecast
                mock_prediction.lower_forecast = mock_forecast
                mock_model.predict.return_value = mock_prediction

                # Test with all parameters that are accepted by fit()
                result = forecaster.forecast_from_symbol(
                    symbol="AAPL",
                    start_date="2023-01-01",
                    end_date="2023-04-10",
                    model_list=["ARIMA", "ETS"],
                    ensemble="simple",
                    max_generations=3,
                )

                assert len(result) == 30

                # Verify that we got results back
                assert isinstance(result, pd.DataFrame)
                assert "forecast" in result.columns or result.shape[1] >= 1

    def test_forecast_from_symbol_with_evaluation(self, mock_data_fetcher, mock_logging_manager):
        """Test forecast_from_symbol_with_evaluation method."""
        forecaster = StockForecaster(
            forecast_length=7, data_fetcher=mock_data_fetcher, logging_manager=mock_logging_manager
        )

        # Create train and test data
        train_dates = pd.date_range(start="2023-01-01", periods=90, freq="D")
        test_dates = pd.date_range(start="2023-04-01", periods=10, freq="D")

        train_data = pd.DataFrame(
            {
                "Close": range(100, 190),
            },
            index=train_dates,
        )

        test_data = pd.DataFrame(
            {
                "Close": range(190, 200),
            },
            index=test_dates,
        )

        # Mock data fetcher to return different data for train and test periods
        def get_stock_data_side_effect(symbol, start=None, end=None, **kwargs):
            if end and "2023-03-31" in end:
                return train_data
            else:
                return pd.concat([train_data, test_data])

        # Use patch to mock the method
        with patch.object(mock_data_fetcher, "get_stock_data", side_effect=get_stock_data_side_effect):
            # Mock AutoTS
            with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
                mock_model = Mock()
                mock_autots.return_value = mock_model
                mock_model.fit.return_value = mock_model

                # Mock prediction
                forecast_dates = pd.date_range(start="2023-04-01", periods=7, freq="D")
                mock_forecast = pd.DataFrame(
                    {"Close": [190.5, 191.5, 192.5, 193.5, 194.5, 195.5, 196.5]}, index=forecast_dates
                )
                # Create a prediction object with forecast, upper_forecast, lower_forecast attributes
                mock_prediction = Mock()
                mock_prediction.forecast = mock_forecast
                mock_prediction.upper_forecast = mock_forecast.copy()
                mock_prediction.lower_forecast = mock_forecast.copy()
                mock_model.predict.return_value = mock_prediction

                # Mock best model info
                mock_model.best_model_name = "Prophet"
                mock_model.best_model_params = {"seasonality_mode": "multiplicative"}
                mock_model.best_model_transformation_params = {}

                # Patch sklearn metrics import inside the method
                with patch("sklearn.metrics.mean_absolute_error") as mock_mae:
                    with patch("sklearn.metrics.mean_squared_error") as mock_mse:
                        mock_mae.return_value = 1.5
                        mock_mse.return_value = 4.0  # sqrt(4.0) = 2.0

                        result = forecaster.forecast_from_symbol_with_evaluation(
                            symbol="AAPL",
                            train_start_date="2023-01-01",
                            train_end_date="2023-03-31",
                            test_start_date="2023-04-01",
                            test_end_date="2023-04-10",
                            target_column="Close",
                        )

                        # Check result structure
                        assert "predictions" in result
                        assert "evaluation_metrics" in result
                        assert "train_period" in result
                        assert "test_period" in result

                        # Check metrics
                        metrics = result["evaluation_metrics"]
                        assert "mae" in metrics
                        assert "rmse" in metrics
                        assert "mape" in metrics
                        assert metrics["mae"] == 1.5
                        assert metrics["rmse"] == 2.0

                        # Verify data fetcher was called correctly
                        assert mock_data_fetcher.get_stock_data.call_count >= 2


class TestStockForecasterGetBestModel:
    """Test get_best_model method."""

    def test_get_best_model_fitted(self, mock_logging_manager):
        """Test getting best model info when fitted."""
        forecaster = StockForecaster(logging_manager=mock_logging_manager)

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

    def test_get_best_model_not_fitted(self, mock_logging_manager):
        """Test getting best model info when not fitted."""
        forecaster = StockForecaster(logging_manager=mock_logging_manager)

        with pytest.raises(ValueError, match="Model not fitted"):
            forecaster.get_best_model()

    def test_get_best_model_no_accuracy(self, mock_logging_manager):
        """Test getting best model when accuracy not available."""
        forecaster = StockForecaster(logging_manager=mock_logging_manager)

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

    def test_get_best_model_info(self, mock_logging_manager):
        """Test getting best model information with all attributes."""
        forecaster = StockForecaster(logging_manager=mock_logging_manager)

        # Create mock model
        mock_model = Mock()
        mock_model.best_model_name = "Prophet"
        mock_model.best_model_params = {"growth": "linear", "seasonality_mode": "additive"}
        mock_model.best_model_transformation_params = {"fillna": "ffill"}
        mock_model.best_model_id = "abc123"
        mock_model.best_model_ensemble = 0
        mock_model.df_wide_numeric = pd.DataFrame({"series1": [1, 2, 3]})
        mock_model.best_model = {"Model": "Prophet", "ID": "abc123"}
        # Mock best_model_accuracy to return "N/A" when accessed
        mock_model.best_model_accuracy = "N/A"

        forecaster.model = mock_model

        info = forecaster.get_best_model()

        assert info["model_name"] == "Prophet"
        assert info["model_params"]["growth"] == "linear"
        assert info["model_transformation"]["fillna"] == "ffill"
        assert info["model_accuracy"] == "N/A"


class TestStockForecasterPlotForecast:
    """Test plot_forecast method."""

    def test_plot_forecast_with_prediction(self, mock_logging_manager):
        """Test plotting forecast with prediction available."""
        forecaster = StockForecaster(logging_manager=mock_logging_manager)

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

    def test_plot_forecast_no_prediction(self, mock_logging_manager):
        """Test plotting forecast without prediction."""
        forecaster = StockForecaster(logging_manager=mock_logging_manager)

        with pytest.raises(ValueError, match="No predictions available"):
            forecaster.plot_forecast()


class TestStockForecasterEdgeCases:
    """Test edge cases and error handling."""

    def test_fit_with_empty_data(self, mock_logging_manager):
        """Test fitting with empty dataframe."""
        forecaster = StockForecaster(logging_manager=mock_logging_manager)
        empty_data = pd.DataFrame()

        with pytest.raises(ValueError):
            forecaster.fit(empty_data)

    def test_fit_with_non_dataframe(self, mock_logging_manager):
        """Test fitting with non-dataframe input."""
        forecaster = StockForecaster(logging_manager=mock_logging_manager)

        # Should raise ValueError about DatetimeIndex
        with pytest.raises(ValueError, match="Data index must be a DatetimeIndex"):
            forecaster.fit([1, 2, 3])

    def test_predict_stores_prediction(self, mock_logging_manager):
        """Test predict stores prediction for later use."""
        forecaster = StockForecaster(logging_manager=mock_logging_manager)

        # Mock the model
        mock_model = Mock()
        mock_prediction = Mock()

        # Create forecast data
        forecast_dates = pd.date_range("2023-02-01", periods=30)
        mock_prediction.forecast = pd.DataFrame({"TEST": [110 + i for i in range(30)]}, index=forecast_dates)
        mock_prediction.upper_forecast = pd.DataFrame({"TEST": [115 + i for i in range(30)]}, index=forecast_dates)
        mock_prediction.lower_forecast = pd.DataFrame({"TEST": [105 + i for i in range(30)]}, index=forecast_dates)

        mock_model.predict.return_value = mock_prediction
        forecaster.model = mock_model

        # Test prediction
        result = forecaster.predict()
        assert forecaster.prediction == mock_prediction  # Should store prediction
        assert len(result) == 30

    def test_fit_with_signal_handler_error(self, mock_logging_manager):
        """Test fit when signal handler setup fails."""
        forecaster = StockForecaster(logging_manager=mock_logging_manager)
        data = pd.DataFrame({"Close": [100, 101, 102]}, index=pd.date_range("2023-01-01", periods=3))

        with patch("stockula.forecasting.forecaster.signal.signal", side_effect=ValueError("Signal error")):
            # Should raise the error
            with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
                mock_model = Mock()
                mock_autots.return_value = mock_model
                mock_model.fit.return_value = mock_model

                with pytest.raises((ValueError, RuntimeError)):  # Should raise some exception
                    forecaster.fit(data)

    def test_filtered_stream_write(self, mock_logging_manager):
        """Test FilteredStream write method."""
        from io import StringIO

        from stockula.forecasting.forecaster import suppress_autots_output

        # Create a StringIO to capture output
        StringIO()

        with suppress_autots_output():
            # Get the current stdout (which is a FilteredStream)
            filtered_stream = sys.stdout

            # Test writing normal text
            filtered_stream.write("Normal output\n")

            # Test writing text that should be filtered
            filtered_stream.write("AutoTS upgrade package\n")
            filtered_stream.write("remove_leading_zeros warning\n")

    def test_initialization_with_missing_data_fetcher(self, mock_logging_manager):
        """Test initialization without data fetcher."""
        forecaster = StockForecaster(data_fetcher=None, logging_manager=mock_logging_manager)
        assert forecaster.data_fetcher is None

    def test_fit_converts_frequency_infer_to_d(self, mock_logging_manager):
        """Test that fit converts 'infer' frequency to 'D'."""
        forecaster = StockForecaster(frequency="infer", logging_manager=mock_logging_manager)

        data = pd.DataFrame({"Close": [100, 101, 102]}, index=pd.date_range("2023-01-01", periods=3, freq="D"))

        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            mock_model = Mock()
            mock_autots.return_value = mock_model
            mock_model.fit.return_value = mock_model

            with patch("stockula.forecasting.forecaster.signal.signal"):
                forecaster.fit(data)

            # Check AutoTS was called with 'D' frequency
            call_kwargs = mock_autots.call_args[1]
            assert call_kwargs["frequency"] == "D"

    def test_get_model_list_ultra_fast_with_logging(self, mock_logging_manager):
        """Test _get_model_list with ultra_fast logging."""
        forecaster = StockForecaster(logging_manager=mock_logging_manager)

        forecaster._get_model_list("ultra_fast", "TestColumn")

        # Verify logging was called
        mock_logging_manager.info.assert_called_once()
        log_message = mock_logging_manager.info.call_args[0][0]
        assert "ultra-fast" in log_message
        assert "TestColumn" in log_message
        assert str(len(forecaster.ULTRA_FAST_MODEL_LIST)) in log_message

    def test_get_model_list_fast_financial(self, mock_logging_manager):
        """Test _get_model_list with fast_financial preset."""
        forecaster = StockForecaster(logging_manager=mock_logging_manager)

        result = forecaster._get_model_list("fast_financial")

        # Should return intersection of FAST and FINANCIAL
        expected = [m for m in forecaster.FAST_MODEL_LIST if m in forecaster.FINANCIAL_MODEL_LIST]
        assert result == expected

    def test_get_model_list_financial(self, mock_logging_manager):
        """Test _get_model_list with financial preset."""
        forecaster = StockForecaster(logging_manager=mock_logging_manager)

        result = forecaster._get_model_list("financial")
        assert result == forecaster.FINANCIAL_MODEL_LIST

    @patch("stockula.forecasting.forecaster.pd.infer_freq")
    @patch("stockula.forecasting.forecaster.AutoTS")
    def test_fit_frequency_inference_exception(self, mock_autots, mock_infer_freq, mock_logging_manager):
        """Test fit when frequency inference raises exception."""
        # Mock infer_freq to raise exception
        mock_infer_freq.side_effect = Exception("Inference failed")

        # Mock AutoTS
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        mock_autots.return_value = mock_model

        forecaster = StockForecaster(frequency="infer", logging_manager=mock_logging_manager)

        # Create data
        dates = pd.date_range("2024-01-01", periods=10)
        data = pd.DataFrame({"Close": np.random.rand(10) * 100}, index=dates)

        # Fit should handle exception and use default frequency
        forecaster.fit(data, show_progress=False)

        # Verify default frequency 'D' was used
        call_kwargs = mock_autots.call_args[1]
        assert call_kwargs["frequency"] == "D"

    @patch("stockula.forecasting.forecaster.signal.signal")
    def test_fit_signal_handler_setup(self, mock_signal, mock_logging_manager):
        """Test signal handler setup in fit."""
        forecaster = StockForecaster(logging_manager=mock_logging_manager)

        dates = pd.date_range("2024-01-01", periods=10)
        data = pd.DataFrame({"Close": np.random.rand(10) * 100}, index=dates)

        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            mock_autots.return_value = mock_model

            forecaster.fit(data, show_progress=False)

            # Verify signal handler was set
            mock_signal.assert_called_once()

    def test_forecast_from_symbol_empty_data(self, mock_logging_manager):
        """Test forecast_from_symbol with empty data."""
        mock_fetcher = Mock()
        mock_fetcher.get_stock_data.return_value = pd.DataFrame()  # Empty data

        forecaster = StockForecaster(data_fetcher=mock_fetcher, logging_manager=mock_logging_manager)

        with pytest.raises(ValueError, match="No data available for symbol"):
            forecaster.forecast_from_symbol("INVALID")

    def test_forecast_from_symbol_with_evaluation_no_test_data(self, mock_logging_manager):
        """Test forecast evaluation without test data."""
        mock_fetcher = Mock()

        # Only return training data
        train_dates = pd.date_range("2024-01-01", periods=100)
        train_data = pd.DataFrame({"Close": np.random.rand(100) * 100}, index=train_dates)
        mock_fetcher.get_stock_data.return_value = train_data

        forecaster = StockForecaster(data_fetcher=mock_fetcher, logging_manager=mock_logging_manager)

        # Mock the fit and predict methods
        with patch.object(forecaster, "fit"):
            with patch.object(forecaster, "predict") as mock_predict:
                mock_predict.return_value = pd.DataFrame(
                    {"forecast": [150, 151, 152], "upper_bound": [155, 156, 157], "lower_bound": [145, 146, 147]}
                )

                # Call without test dates
                result = forecaster.forecast_from_symbol_with_evaluation(
                    "AAPL", train_start_date="2024-01-01", train_end_date="2024-03-01"
                )

                # Should have predictions but no evaluation metrics
                assert "predictions" in result
                assert result["evaluation_metrics"] is None

    def test_forecast_from_symbol_with_evaluation_no_common_dates(self, mock_logging_manager):
        """Test forecast evaluation with no common dates."""
        mock_fetcher = Mock()

        # Training data
        train_dates = pd.date_range("2024-01-01", periods=100)
        train_data = pd.DataFrame({"Close": np.random.rand(100) * 100}, index=train_dates)

        # Test data with different dates
        test_dates = pd.date_range("2024-06-01", periods=20)
        test_data = pd.DataFrame({"Close": np.random.rand(20) * 100}, index=test_dates)

        mock_fetcher.get_stock_data.side_effect = [train_data, test_data]

        forecaster = StockForecaster(data_fetcher=mock_fetcher, logging_manager=mock_logging_manager)

        # Mock the fit and predict methods
        with patch.object(forecaster, "fit"):
            with patch.object(forecaster, "predict") as mock_predict:
                # Return predictions with different dates than test data
                pred_dates = pd.date_range("2024-07-01", periods=5)
                mock_predict.return_value = pd.DataFrame(
                    {
                        "forecast": [150, 151, 152, 153, 154],
                        "upper_bound": [155, 156, 157, 158, 159],
                        "lower_bound": [145, 146, 147, 148, 149],
                    },
                    index=pred_dates,
                )

                # Call evaluation
                result = forecaster.forecast_from_symbol_with_evaluation(
                    "AAPL",
                    train_start_date="2024-01-01",
                    train_end_date="2024-03-01",
                    test_start_date="2024-06-01",
                    test_end_date="2024-06-20",
                )

                # Should have predictions but no evaluation metrics (no common dates)
                assert "predictions" in result
                assert result["evaluation_metrics"] is None

    def test_forecast_from_symbol_with_evaluation_and_kwargs(self, mock_logging_manager):
        """Test forecast_from_symbol_with_evaluation with additional kwargs."""
        mock_fetcher = Mock()

        # Setup data
        train_dates = pd.date_range("2024-01-01", periods=100)
        train_data = pd.DataFrame({"Close": np.random.rand(100) * 100}, index=train_dates)
        test_dates = pd.date_range("2024-06-01", periods=20)
        test_data = pd.DataFrame({"Close": np.random.rand(20) * 100}, index=test_dates)

        mock_fetcher.get_stock_data.side_effect = [train_data, test_data]

        forecaster = StockForecaster(data_fetcher=mock_fetcher, logging_manager=mock_logging_manager)

        # Mock fit to capture kwargs
        fit_kwargs = {}

        def mock_fit(*args, **kwargs):
            fit_kwargs.update(kwargs)
            return forecaster

        with patch.object(forecaster, "fit", side_effect=mock_fit):
            with patch.object(forecaster, "predict") as mock_predict:
                mock_predict.return_value = pd.DataFrame({"forecast": [150]})

                # Call with extra kwargs
                forecaster.forecast_from_symbol_with_evaluation(
                    "AAPL",
                    train_start_date="2024-01-01",
                    train_end_date="2024-05-31",
                    test_start_date="2024-06-01",
                    test_end_date="2024-06-20",
                    model_list="ultra_fast",
                    max_generations=3,
                )

                # Verify kwargs were passed to fit
                assert fit_kwargs["model_list"] == "ultra_fast"
                assert fit_kwargs["max_generations"] == 3

    def test_evaluate_forecast_with_misaligned_data(self, mock_logging_manager):
        """Test evaluate_forecast when actual data doesn't align with forecast."""
        forecaster = StockForecaster(logging_manager=mock_logging_manager)

        # Mock prediction
        forecast_dates = pd.date_range("2024-01-01", periods=5)
        mock_prediction = Mock()
        mock_prediction.forecast = pd.DataFrame({"value": [100, 101, 102, 103, 104]}, index=forecast_dates)
        forecaster.prediction = mock_prediction

        # Actual data with different dates
        actual_dates = pd.date_range("2024-01-10", periods=5)
        actual_data = pd.DataFrame({"Close": [105, 106, 107, 108, 109]}, index=actual_dates)

        # This should raise an error when trying to align data
        with pytest.raises((KeyError, ValueError, IndexError)):  # KeyError or similar alignment error
            forecaster.evaluate_forecast(actual_data)

    def test_fit_debug_logging(self, mock_logging_manager):
        """Test debug logging during fit."""
        forecaster = StockForecaster(logging_manager=mock_logging_manager)

        data = pd.DataFrame({"Close": [100, 101, 102]}, index=pd.date_range("2023-01-01", periods=3))

        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            mock_model = Mock()
            mock_autots.return_value = mock_model
            mock_model.fit.return_value = mock_model

            with patch("stockula.forecasting.forecaster.signal.signal"):
                forecaster.fit(data)

                # Check debug messages were called on mock logger
                mock_logging_manager.debug.assert_called()
                debug_calls = [call.args[0] for call in mock_logging_manager.debug.call_args_list]
                assert any("Fitting AutoTS with parameters:" in call for call in debug_calls)


class TestStockForecasterIntegration:
    """Integration tests for StockForecaster."""

    def test_full_workflow(self, mock_logging_manager):
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
            forecaster = StockForecaster(
                forecast_length=14, frequency="D", prediction_interval=0.95, logging_manager=mock_logging_manager
            )

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

    def test_logging_messages(self, mock_logging_manager):
        """Test that appropriate log messages are generated."""
        data = pd.DataFrame(
            {"Close": [100, 101, 102, 103, 104]},
            index=pd.date_range("2023-01-01", periods=5),
        )

        forecaster = StockForecaster(logging_manager=mock_logging_manager)

        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            mock_instance = Mock()
            mock_model = Mock()
            mock_instance.fit.return_value = mock_model
            mock_autots.return_value = mock_instance

            with patch("stockula.forecasting.forecaster.signal.signal"):
                forecaster.fit(data)

            # Check log messages were called on mock logger
            mock_logging_manager.debug.assert_called()
            debug_calls = [call.args[0] for call in mock_logging_manager.debug.call_args_list]
            assert any("Fitting model on 5 data points" in call for call in debug_calls)
            assert any("Model fitting completed" in call for call in debug_calls)
