"""Comprehensive tests for forecasting backends."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from stockula.forecasting.backends.base import ForecastResult
from stockula.forecasting.backends.simple import SimpleForecastBackend


class TestForecastResult:
    """Test the ForecastResult dataclass."""

    def test_forecast_result_creation(self):
        """Test basic ForecastResult creation."""
        forecast_df = pd.DataFrame(
            {"forecast": [100, 101, 102], "lower_bound": [95, 96, 97], "upper_bound": [105, 106, 107]}
        )

        result = ForecastResult(
            forecast=forecast_df,
            model_name="TestModel",
            model_params={"param1": "value1"},
            metrics={"mae": 1.5, "rmse": 2.0},
        )

        assert result.model_name == "TestModel"
        assert result.model_params["param1"] == "value1"
        assert result.metrics["mae"] == 1.5
        assert len(result.forecast) == 3
        assert result.metadata is None

    def test_forecast_result_with_metadata(self):
        """Test ForecastResult with optional metadata."""
        forecast_df = pd.DataFrame({"forecast": [100]})

        result = ForecastResult(
            forecast=forecast_df,
            model_name="TestModel",
            model_params={},
            metadata={"backend": "test", "version": "1.0"},
        )

        assert result.metadata["backend"] == "test"
        assert result.metadata["version"] == "1.0"


class ForecastBackendTestContract:
    """Base test contract that all forecast backends must satisfy.

    This class defines the common interface tests that all backends should pass.
    Concrete backend test classes should inherit from this and call these tests.
    """

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data for testing."""
        dates = pd.date_range(start="2023-01-01", end="2023-03-31", freq="D")
        np.random.seed(42)  # For reproducible tests

        # Generate trending data with some noise
        trend = np.linspace(100, 110, len(dates))
        noise = np.random.normal(0, 2, len(dates))
        values = trend + noise

        return pd.DataFrame(
            {
                "Close": values,
                "Open": values * 0.99,
                "High": values * 1.02,
                "Low": values * 0.98,
                "Volume": np.random.randint(1000000, 5000000, len(dates)),
            },
            index=dates,
        )

    @pytest.fixture
    def sample_data_short(self):
        """Create short time series data (edge case)."""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        return pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=dates)

    @pytest.fixture
    def sample_data_with_nulls(self):
        """Create data with null values (error case)."""
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        values = [100, 101, None, 103, 104, 105, None, 107, 108, 109]
        return pd.DataFrame({"Close": values}, index=dates)

    @pytest.fixture
    def sample_data_no_datetime_index(self):
        """Create data without datetime index (error case)."""
        return pd.DataFrame({"Close": [100, 101, 102, 103, 104]})  # No datetime index

    def test_initialization(self, backend_instance):
        """Test that backend initializes correctly."""
        assert backend_instance.forecast_length is not None
        assert backend_instance.frequency == "infer"
        assert backend_instance.prediction_interval == 0.95
        assert backend_instance.no_negatives is True
        assert backend_instance.is_fitted is False

    def test_initialization_with_custom_params(self, backend_class, mock_logging_manager):
        """Test initialization with custom parameters."""
        backend = backend_class(
            forecast_length=30,
            frequency="D",
            prediction_interval=0.90,
            no_negatives=False,
            logging_manager=mock_logging_manager,
        )

        assert backend.forecast_length == 30
        assert backend.frequency == "D"
        assert backend.prediction_interval == 0.90
        assert backend.no_negatives is False

    def test_fit_predict_workflow(self, backend_instance, sample_data):
        """Test the basic fit -> predict workflow."""
        # Test fit
        result = backend_instance.fit(sample_data, target_column="Close")
        assert backend_instance is result  # Should return self for chaining
        assert backend_instance.is_fitted is True

        # Test predict
        forecast_result = backend_instance.predict()
        assert isinstance(forecast_result, ForecastResult)
        assert isinstance(forecast_result.forecast, pd.DataFrame)
        assert "forecast" in forecast_result.forecast.columns
        assert len(forecast_result.forecast) == backend_instance.forecast_length

    def test_fit_predict_combined(self, backend_instance, sample_data):
        """Test the fit_predict convenience method."""
        result = backend_instance.fit_predict(sample_data, target_column="Close")

        assert isinstance(result, ForecastResult)
        assert backend_instance.is_fitted is True
        assert len(result.forecast) == backend_instance.forecast_length

    def test_predict_without_fit_raises_error(self, backend_instance):
        """Test that predict raises error when called before fit."""
        with pytest.raises(ValueError, match="not fitted|must be fitted"):
            backend_instance.predict()

    def test_get_model_info_not_fitted(self, backend_instance):
        """Test get_model_info when not fitted."""
        info = backend_instance.get_model_info()
        assert isinstance(info, dict)
        assert "status" in info or "model_name" in info

    def test_get_model_info_fitted(self, backend_instance, sample_data):
        """Test get_model_info after fitting."""
        backend_instance.fit(sample_data, target_column="Close")
        info = backend_instance.get_model_info()

        assert isinstance(info, dict)
        assert "model_name" in info

    def test_get_available_models(self, backend_instance):
        """Test that get_available_models returns a list."""
        models = backend_instance.get_available_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_validate_input_valid_data(self, backend_instance, sample_data):
        """Test input validation with valid data."""
        # Should not raise any exception
        backend_instance.validate_input(sample_data, "Close")

    def test_validate_input_missing_column(self, backend_instance, sample_data):
        """Test input validation with missing target column."""
        with pytest.raises(ValueError, match="not found in data"):
            backend_instance.validate_input(sample_data, "NonExistent")

    def test_validate_input_no_datetime_index(self, backend_instance, sample_data_no_datetime_index):
        """Test input validation with non-datetime index."""
        with pytest.raises(ValueError, match="DatetimeIndex"):
            backend_instance.validate_input(sample_data_no_datetime_index, "Close")

    def test_validate_input_null_values(self, backend_instance, sample_data_with_nulls):
        """Test input validation with null values."""
        with pytest.raises(ValueError, match="null values"):
            backend_instance.validate_input(sample_data_with_nulls, "Close")

    def test_validate_input_insufficient_data(self, backend_instance):
        """Test input validation with insufficient data points."""
        single_point = pd.DataFrame({"Close": [100]}, index=pd.date_range("2023-01-01", periods=1))

        with pytest.raises(ValueError, match="at least 2 data points"):
            backend_instance.validate_input(single_point, "Close")

    def test_forecast_result_structure(self, backend_instance, sample_data):
        """Test that forecast results have proper structure."""
        result = backend_instance.fit_predict(sample_data)

        # Check required columns
        required_columns = ["forecast", "lower_bound", "upper_bound"]
        for col in required_columns:
            assert col in result.forecast.columns, f"Missing column: {col}"

        # Check data types and values
        assert not result.forecast["forecast"].isnull().any()
        assert not result.forecast["lower_bound"].isnull().any()
        assert not result.forecast["upper_bound"].isnull().any()

        # Lower bound should be <= forecast <= upper bound
        assert (result.forecast["lower_bound"] <= result.forecast["forecast"]).all()
        assert (result.forecast["forecast"] <= result.forecast["upper_bound"]).all()

    def test_no_negatives_constraint(self, backend_class, mock_logging_manager, sample_data):
        """Test that no_negatives constraint is properly applied."""
        # Create backend with no_negatives=True
        backend = backend_class(forecast_length=7, no_negatives=True, logging_manager=mock_logging_manager)

        # Create data that might produce negative forecasts
        negative_trend_data = sample_data.copy()
        negative_trend_data["Close"] = np.linspace(100, 50, len(negative_trend_data))

        result = backend.fit_predict(negative_trend_data)

        # All forecast values should be non-negative
        assert (result.forecast["forecast"] >= 0).all()
        assert (result.forecast["lower_bound"] >= 0).all()
        assert (result.forecast["upper_bound"] >= 0).all()

    def test_different_forecast_lengths(self, backend_class, mock_logging_manager, sample_data):
        """Test forecasting with different forecast lengths."""
        for length in [1, 7, 30]:
            backend = backend_class(forecast_length=length, logging_manager=mock_logging_manager)
            result = backend.fit_predict(sample_data)
            assert len(result.forecast) == length

    def test_different_target_columns(self, backend_instance, sample_data):
        """Test forecasting different target columns."""
        for col in ["Open", "High", "Low"]:
            result = backend_instance.fit_predict(sample_data, target_column=col)
            assert isinstance(result, ForecastResult)
            assert len(result.forecast) == backend_instance.forecast_length


class TestSimpleForecastBackend(ForecastBackendTestContract):
    """Test the SimpleForecastBackend implementation."""

    @pytest.fixture
    def mock_logging_manager(self):
        """Create a mock logging manager."""
        logger = Mock()
        logger.debug = Mock()
        logger.info = Mock()
        logger.warning = Mock()
        logger.error = Mock()
        return logger

    @pytest.fixture
    def backend_class(self):
        """Return the backend class for parameterized tests."""
        return SimpleForecastBackend

    @pytest.fixture
    def backend_instance(self, mock_logging_manager):
        """Create a SimpleForecastBackend instance."""
        return SimpleForecastBackend(forecast_length=7, logging_manager=mock_logging_manager)

    def test_linear_regression_initialization(self, backend_instance):
        """Test that LinearRegression model is properly initialized."""
        from sklearn.linear_model import LinearRegression

        assert isinstance(backend_instance.model, LinearRegression)
        assert backend_instance.last_values is None
        assert backend_instance.trend is None
        assert backend_instance.std_dev is None

    def test_fit_stores_model_parameters(self, backend_instance, sample_data):
        """Test that fit properly stores model parameters."""
        backend_instance.fit(sample_data, target_column="Close")

        assert backend_instance.last_values is not None
        assert backend_instance.trend is not None
        assert backend_instance.std_dev is not None
        assert isinstance(backend_instance.trend, float | np.floating)
        assert backend_instance.std_dev >= 0

    def test_fit_with_short_data(self, backend_instance, sample_data_short):
        """Test fitting with short data series."""
        backend_instance.fit(sample_data_short, target_column="Close")

        # Should use all available data points when less than 30
        assert len(backend_instance.last_values) == len(sample_data_short)
        assert backend_instance.is_fitted is True

    def test_predict_returns_proper_structure(self, backend_instance, sample_data):
        """Test that predict returns proper ForecastResult structure."""
        backend_instance.fit(sample_data)
        result = backend_instance.predict()

        assert isinstance(result, ForecastResult)
        assert result.model_name == "LinearRegression"
        assert "trend" in result.model_params
        assert "intercept" in result.model_params
        assert "std_dev" in result.metrics
        assert "r2_score" in result.metrics

    def test_confidence_intervals_calculation(self, backend_instance, sample_data):
        """Test that confidence intervals are properly calculated."""
        backend_instance.fit(sample_data)
        result = backend_instance.predict()

        forecast = result.forecast

        # Check that intervals are reasonable
        interval_width = forecast["upper_bound"] - forecast["lower_bound"]
        assert (interval_width > 0).all()

        # Intervals should generally increase with forecast horizon (uncertainty grows)
        # Allow for some variation due to the sqrt factor
        assert interval_width.iloc[-1] >= interval_width.iloc[0] * 0.8

    def test_prediction_interval_adjustment(self, mock_logging_manager, sample_data):
        """Test prediction intervals for different confidence levels."""
        # Test 95% interval (should use z=1.96)
        backend_95 = SimpleForecastBackend(
            forecast_length=7, prediction_interval=0.95, logging_manager=mock_logging_manager
        )
        result_95 = backend_95.fit_predict(sample_data)
        width_95 = result_95.forecast["upper_bound"] - result_95.forecast["lower_bound"]

        # Test 90% interval (should use z=1.645)
        backend_90 = SimpleForecastBackend(
            forecast_length=7, prediction_interval=0.90, logging_manager=mock_logging_manager
        )
        result_90 = backend_90.fit_predict(sample_data)
        width_90 = result_90.forecast["upper_bound"] - result_90.forecast["lower_bound"]

        # 95% interval should be wider than 90%
        assert (width_95 > width_90).all()

    def test_get_available_models(self, backend_instance):
        """Test that SimpleForecastBackend returns appropriate models."""
        models = backend_instance.get_available_models()
        assert isinstance(models, list)
        # Simple backend should only have LinearRegression
        expected_models = ["LinearRegression"]
        for model in expected_models:
            assert model in models

    def test_model_info_before_and_after_fit(self, backend_instance, sample_data):
        """Test model info changes after fitting."""
        # Before fit
        info_before = backend_instance.get_model_info()
        assert info_before["status"] == "not_fitted"

        # After fit
        backend_instance.fit(sample_data)
        info_after = backend_instance.get_model_info()
        assert info_after["status"] == "fitted"
        assert "trend" in info_after["model_params"]
        assert "intercept" in info_after["model_params"]

    def test_fit_missing_target_column(self, backend_instance, sample_data):
        """Test fit with missing target column."""
        with pytest.raises(ValueError, match="not found in data"):
            backend_instance.fit(sample_data, target_column="NonExistent")

    def test_linear_trend_detection(self, backend_instance):
        """Test that linear trend is properly detected."""
        # Create data with known linear trend
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        # y = 2x + 100 (slope=2)
        values = 2 * np.arange(30) + 100
        data = pd.DataFrame({"Close": values}, index=dates)

        backend_instance.fit(data)

        # Trend should be approximately 2
        assert abs(backend_instance.trend - 2.0) < 0.1

    def test_r2_score_calculation(self, backend_instance, sample_data):
        """Test that R² score is properly calculated."""
        backend_instance.fit(sample_data)
        result = backend_instance.predict()

        r2 = result.metrics["r2_score"]
        assert isinstance(r2, float | np.floating)
        assert r2 <= 1  # R² should be at most 1, but can be negative


class TestAutoGluonBackend:
    """Test the AutoGluonBackend implementation with mocking."""

    @pytest.fixture
    def mock_logging_manager(self):
        """Create a mock logging manager."""
        logger = Mock()
        logger.debug = Mock()
        logger.info = Mock()
        logger.warning = Mock()
        logger.error = Mock()
        logger.isEnabledFor = Mock(return_value=False)
        return logger

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        np.random.seed(42)
        values = 100 + np.cumsum(np.random.normal(0, 1, 60))
        return pd.DataFrame({"Close": values}, index=dates)

    @pytest.fixture
    def mock_autogluon_imports(self):
        """Mock AutoGluon imports."""
        with patch.dict(
            "sys.modules",
            {
                "autogluon": MagicMock(),
                "autogluon.timeseries": MagicMock(),
            },
        ):
            # Mock TimeSeriesDataFrame
            mock_ts_df = MagicMock()
            mock_ts_df.from_data_frame = MagicMock(return_value=mock_ts_df)

            # Mock TimeSeriesPredictor
            mock_predictor = MagicMock()
            mock_predictor.fit = MagicMock()
            mock_predictor.predict = MagicMock()
            mock_predictor.get_model_best = MagicMock(return_value="BestModel")
            mock_predictor.leaderboard = MagicMock(
                return_value=pd.DataFrame(
                    {
                        "model": ["Model1", "Model2"],
                        "score": [0.85, 0.80],
                        "pred_time": [1.2, 0.8],
                        "fit_time": [10.5, 8.2],
                    }
                )
            )

            # Mock prediction results
            mock_predictions = pd.DataFrame(
                {
                    "item_id": ["stock"] * 7,
                    "timestamp": pd.date_range("2023-04-01", periods=7),
                    "mean": [105, 106, 107, 108, 109, 110, 111],
                    "0.05": [100, 101, 102, 103, 104, 105, 106],
                    "0.95": [110, 111, 112, 113, 114, 115, 116],
                }
            )
            mock_predictor.predict.return_value = mock_predictions

            yield {
                "TimeSeriesDataFrame": mock_ts_df,
                "TimeSeriesPredictor": mock_predictor,
                "mock_predictor_instance": mock_predictor,
            }

    def test_autogluon_import_error(self, mock_logging_manager):
        """Test behavior when AutoGluon is not available."""
        with patch("stockula.forecasting.backends.autogluon.warnings"):
            # This should not raise ImportError during initialization
            from stockula.forecasting.backends.autogluon import AutoGluonBackend

            backend = AutoGluonBackend(logging_manager=mock_logging_manager)
            assert backend is not None

    @patch("stockula.forecasting.backends.autogluon.warnings")
    def test_autogluon_initialization(self, mock_warnings, mock_logging_manager):
        """Test AutoGluon backend initialization."""
        from stockula.forecasting.backends.autogluon import AutoGluonBackend

        backend = AutoGluonBackend(
            forecast_length=14, preset="high_quality", eval_metric="RMSE", logging_manager=mock_logging_manager
        )

        assert backend.preset == "high_quality"
        assert backend.eval_metric == "RMSE"
        assert backend.forecast_length == 14
        assert backend.predictor is None
        assert backend.models is None

    @patch("stockula.forecasting.backends.autogluon.warnings")
    def test_get_models_configurations(self, mock_warnings, mock_logging_manager):
        """Test model configuration selection."""
        from stockula.forecasting.backends.autogluon import AutoGluonBackend

        backend = AutoGluonBackend(logging_manager=mock_logging_manager)

        # Test None input
        assert backend._get_models(None) is None

        # Test list input
        model_list = ["ETS", "ARIMA"]
        assert backend._get_models(model_list) == model_list

        # Test preset string
        result = backend._get_models("fast")
        expected = backend.MODEL_CONFIGS["fast"]
        assert result == expected

        # Test single model
        assert backend._get_models("ETS") == ["ETS"]

    @patch("stockula.forecasting.backends.autogluon.warnings")
    def test_available_models(self, mock_warnings, mock_logging_manager):
        """Test get_available_models returns expected models."""
        from stockula.forecasting.backends.autogluon import AutoGluonBackend

        backend = AutoGluonBackend(logging_manager=mock_logging_manager)
        models = backend.get_available_models()

        expected_models = [
            "ARIMA",
            "ETS",
            "Theta",
            "Naive",
            "SeasonalNaive",
            "RecursiveTabular",
            "DirectTabular",
            "LightGBM",
            "DeepAR",
            "TemporalFusionTransformer",
            "PatchTST",
            "Chronos",
        ]

        for model in expected_models:
            assert model in models

    @patch("stockula.forecasting.backends.autogluon.warnings")
    def test_presets_defined(self, mock_warnings, mock_logging_manager):
        """Test that all expected presets are defined."""
        from stockula.forecasting.backends.autogluon import AutoGluonBackend

        expected_presets = ["fast_training", "medium_quality", "high_quality", "best_quality"]

        for preset in expected_presets:
            assert preset in AutoGluonBackend.PRESETS

    @patch("stockula.forecasting.backends.autogluon.warnings")
    def test_model_configs_defined(self, mock_warnings, mock_logging_manager):
        """Test that model configurations are properly defined."""
        from stockula.forecasting.backends.autogluon import AutoGluonBackend

        expected_configs = ["statistical", "tree", "deep_learning", "zero_shot", "fast", "accurate"]

        for config in expected_configs:
            assert config in AutoGluonBackend.MODEL_CONFIGS
            assert isinstance(AutoGluonBackend.MODEL_CONFIGS[config], list)
            assert len(AutoGluonBackend.MODEL_CONFIGS[config]) > 0

    @patch("stockula.forecasting.backends.autogluon.warnings")
    def test_fit_not_fitted_error(self, mock_warnings, mock_logging_manager):
        """Test that predict raises error when not fitted."""
        from stockula.forecasting.backends.autogluon import AutoGluonBackend

        backend = AutoGluonBackend(logging_manager=mock_logging_manager)

        with pytest.raises(ValueError, match="not fitted"):
            backend.predict()

    @patch("stockula.forecasting.backends.autogluon.warnings")
    def test_get_model_info_not_fitted_error(self, mock_warnings, mock_logging_manager):
        """Test that get_model_info raises error when not fitted."""
        from stockula.forecasting.backends.autogluon import AutoGluonBackend

        backend = AutoGluonBackend(logging_manager=mock_logging_manager)

        with pytest.raises(ValueError, match="not fitted"):
            backend.get_model_info()

    @patch("stockula.forecasting.backends.autogluon.warnings")
    def test_evaluate_models_not_fitted_error(self, mock_warnings, mock_logging_manager):
        """Test that evaluate_models raises error when not fitted."""
        from stockula.forecasting.backends.autogluon import AutoGluonBackend

        backend = AutoGluonBackend(logging_manager=mock_logging_manager)

        with pytest.raises(ValueError, match="not fitted"):
            backend.evaluate_models()


class TestForecastBackendEvaluateMethod:
    """Test the evaluate method from the base ForecastBackend class."""

    @pytest.fixture
    def mock_backend(self, mock_logging_manager):
        """Create a mock backend for testing evaluate method."""
        backend = SimpleForecastBackend(forecast_length=5, logging_manager=mock_logging_manager)
        return backend

    @pytest.fixture
    def sample_data(self):
        """Sample training data."""
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        values = np.linspace(100, 120, 30) + np.random.normal(0, 1, 30)
        return pd.DataFrame({"Close": values}, index=dates)

    @pytest.fixture
    def actual_test_data(self):
        """Actual test data for evaluation."""
        dates = pd.date_range("2023-01-31", periods=5, freq="D")
        values = np.linspace(121, 125, 5)
        return pd.DataFrame({"Close": values}, index=dates)

    def test_evaluate_not_fitted_raises_error(self, mock_backend, actual_test_data):
        """Test evaluate raises error when model not fitted."""
        with pytest.raises(ValueError, match="not fitted"):
            mock_backend.evaluate(actual_test_data)

    def test_evaluate_with_overlapping_dates(self, mock_backend, sample_data, actual_test_data):
        """Test evaluate method with overlapping dates."""
        # Fit the model
        mock_backend.fit(sample_data)

        # Mock the predict method to return predictable results
        mock_forecast = pd.DataFrame({"forecast": [121, 122, 123, 124, 125]}, index=actual_test_data.index)

        with patch.object(mock_backend, "predict") as mock_predict:
            mock_predict.return_value = ForecastResult(forecast=mock_forecast, model_name="Test", model_params={})

            metrics = mock_backend.evaluate(actual_test_data)

            expected_metrics = ["mae", "mse", "rmse", "mase", "mape"]
            for metric in expected_metrics:
                assert metric in metrics
                assert isinstance(metrics[metric], int | float | np.number)

    def test_evaluate_no_overlapping_dates_raises_error(self, mock_backend, sample_data):
        """Test evaluate raises error with no overlapping dates."""
        mock_backend.fit(sample_data)

        # Create test data with no overlapping dates
        future_dates = pd.date_range("2024-01-01", periods=5, freq="D")
        future_data = pd.DataFrame({"Close": [130, 131, 132, 133, 134]}, index=future_dates)

        with pytest.raises(ValueError, match="No overlapping dates"):
            mock_backend.evaluate(future_data)

    def test_evaluate_mase_calculation(self, mock_backend, sample_data):
        """Test MASE calculation in evaluate method."""
        mock_backend.fit(sample_data)

        # Create test data that matches training data end
        test_dates = pd.date_range(sample_data.index[-5], periods=5, freq="D")
        test_values = [120, 121, 122, 123, 124]  # Continuing the trend
        test_data = pd.DataFrame({"Close": test_values}, index=test_dates)

        # Mock perfect predictions (same as actual)
        mock_forecast = pd.DataFrame({"forecast": test_values}, index=test_dates)

        with patch.object(mock_backend, "predict") as mock_predict:
            mock_predict.return_value = ForecastResult(forecast=mock_forecast, model_name="Test", model_params={})

            metrics = mock_backend.evaluate(test_data)

            # Perfect forecast should have MAE=0, thus MASE=0
            assert metrics["mae"] == 0.0
            assert metrics["mase"] == 0.0

    def test_evaluate_division_by_zero_protection(self, mock_backend):
        """Test MASE calculation handles division by zero."""
        # Create constant data (no variation)
        constant_dates = pd.date_range("2023-01-01", periods=10, freq="D")
        constant_data = pd.DataFrame({"Close": [100] * 10}, index=constant_dates)

        mock_backend.fit(constant_data)

        # Test data with same constant value
        test_dates = pd.date_range("2023-01-11", periods=3, freq="D")
        test_data = pd.DataFrame({"Close": [100, 100, 101]}, index=test_dates)  # Small deviation

        mock_forecast = pd.DataFrame(
            {
                "forecast": [100, 100, 100]  # Perfect prediction for first two, off by 1 for third
            },
            index=test_dates,
        )

        with patch.object(mock_backend, "predict") as mock_predict:
            mock_predict.return_value = ForecastResult(forecast=mock_forecast, model_name="Test", model_params={})

            metrics = mock_backend.evaluate(test_data)

            # Should handle division by zero gracefully
            assert "mase" in metrics
            assert not np.isnan(metrics["mase"])
