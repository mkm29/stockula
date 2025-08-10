"""Comprehensive tests for ChronosBackend with focus on code coverage."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from stockula.forecasting.backends.base import ForecastResult

# Check if chronos is available
try:
    import chronos  # noqa: F401

    from stockula.forecasting.backends.chronos import ChronosBackend

    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False
    ChronosBackend = None  # Define as None when not available


@pytest.mark.skipif(not CHRONOS_AVAILABLE, reason="chronos not installed")
class TestChronosBackendCoverage:
    """Test ChronosBackend with focus on achieving 70%+ code coverage."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        dates = pd.date_range(start="2023-01-01", end="2023-03-31", freq="D")
        np.random.seed(42)
        values = np.linspace(100, 110, len(dates)) + np.random.normal(0, 2, len(dates))
        return pd.DataFrame({"Close": values}, index=dates)

    def test_initialization_and_properties(self):
        """Test initialization covers lines 32-70."""
        # Test with default parameters
        backend = ChronosBackend()
        assert backend.model_name == "amazon/chronos-bolt-small"
        assert backend.num_samples == 256
        assert backend._pipeline is None
        assert backend.is_fitted is False

        # Test with custom parameters and quantile calculation
        backend = ChronosBackend(
            forecast_length=14,
            frequency="W",
            prediction_interval=0.90,
            no_negatives=False,
            model_name="amazon/chronos-t5-large",
            num_samples=512,
            device_map="cuda",
            torch_dtype="float16",
        )
        assert backend.model_name == "amazon/chronos-t5-large"
        assert backend.num_samples == 512
        assert backend.device_map == "cuda"
        assert backend.torch_dtype == "float16"
        # Check quantile levels calculated from prediction_interval
        expected_quantiles = [0.05, 0.5, 0.95]
        np.testing.assert_array_almost_equal(backend.quantile_levels, expected_quantiles, decimal=3)

        # Test with explicit quantile levels
        backend = ChronosBackend(quantile_levels=[0.1, 0.5, 0.9])
        assert backend.quantile_levels == [0.1, 0.5, 0.9]

    def test_load_pipeline_import_error(self):
        """Test _load_pipeline when chronos is not available (lines 76-79)."""
        import builtins

        backend = ChronosBackend()

        # Mock the import to fail by making the chronos module unavailable
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "chronos":
                raise ImportError("No module named 'chronos'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="chronos-forecasting not installed"):
                backend._load_pipeline()

    def test_load_pipeline_runtime_error(self):
        """Test _load_pipeline when model loading fails (lines 109-112)."""
        backend = ChronosBackend(model_name="invalid/model")

        # Mock successful import but failed model loading
        mock_base = MagicMock()
        mock_base.from_pretrained.side_effect = RuntimeError("Model not found")

        with patch("chronos.BaseChronosPipeline", mock_base):
            with pytest.raises(RuntimeError, match="Failed to load Chronos model"):
                backend._load_pipeline()

    def test_load_pipeline_with_torch_cuda(self):
        """Test _load_pipeline with torch and CUDA available (lines 84-95)."""
        backend = ChronosBackend()

        # Mock successful chronos import
        mock_pipeline = MagicMock()
        mock_base = MagicMock()
        mock_base.from_pretrained.return_value = mock_pipeline

        # Mock torch with CUDA
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.bfloat16 = "bfloat16_dtype"

        with patch("chronos.BaseChronosPipeline", mock_base):
            with patch.dict("sys.modules", {"torch": mock_torch}):
                backend._load_pipeline()

                # Verify correct device and dtype were used
                mock_base.from_pretrained.assert_called_once_with(
                    "amazon/chronos-bolt-small", device_map="cuda", torch_dtype="bfloat16_dtype"
                )
                assert backend._pipeline == mock_pipeline

    def test_load_pipeline_without_cuda(self):
        """Test _load_pipeline with torch but no CUDA (lines 88-89)."""
        backend = ChronosBackend()

        mock_pipeline = MagicMock()
        mock_base = MagicMock()
        mock_base.from_pretrained.return_value = mock_pipeline

        # Mock torch without CUDA
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.bfloat16 = "bfloat16_dtype"

        with patch("chronos.BaseChronosPipeline", mock_base):
            with patch.dict("sys.modules", {"torch": mock_torch}):
                backend._load_pipeline()

                # Should use CPU
                call_kwargs = mock_base.from_pretrained.call_args[1]
                assert call_kwargs["device_map"] == "cpu"
                assert call_kwargs["torch_dtype"] == "bfloat16_dtype"

    def test_load_pipeline_no_torch(self):
        """Test _load_pipeline when torch is not available (lines 96-101)."""
        import builtins

        backend = ChronosBackend()

        mock_pipeline = MagicMock()
        mock_base = MagicMock()
        mock_base.from_pretrained.return_value = mock_pipeline

        with patch("chronos.BaseChronosPipeline", mock_base):
            # Mock torch import to fail
            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "torch":
                    raise ImportError("No module named 'torch'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                backend._load_pipeline()

                # Should fallback to CPU and None dtype
                call_kwargs = mock_base.from_pretrained.call_args[1]
                assert call_kwargs["device_map"] == "cpu"
                assert call_kwargs["torch_dtype"] is None

    def test_prepare_context(self, sample_data):
        """Test _prepare_context method (lines 114-120)."""
        backend = ChronosBackend()

        # Test normal case
        context = backend._prepare_context(sample_data, "Close")
        assert isinstance(context, np.ndarray)
        assert context.dtype == np.float32
        assert len(context) == len(sample_data)

        # Test with NaN values (line 119)
        data_with_nan = sample_data.copy()
        data_with_nan.iloc[5] = np.nan
        with pytest.raises(ValueError, match="NaN or infinite values"):
            backend._prepare_context(data_with_nan, "Close")

        # Test with infinite values (line 119)
        data_with_inf = sample_data.copy()
        data_with_inf.iloc[5] = np.inf
        with pytest.raises(ValueError, match="NaN or infinite values"):
            backend._prepare_context(data_with_inf, "Close")

    def test_fit_method(self, sample_data):
        """Test fit method (lines 122-142)."""
        backend = ChronosBackend()

        # Mock the pipeline loading
        mock_pipeline = MagicMock()
        with patch.object(backend, "_load_pipeline"):
            backend._pipeline = mock_pipeline

            # Test with None forecast_length (line 132)
            backend.forecast_length = None
            result = backend.fit(sample_data, target_column="Close")

            assert result is backend
            assert backend.forecast_length == 14
            assert backend._context_series is not None
            assert backend._context_series.dtype == np.float32
            assert backend._last_timestamp == sample_data.index.max()
            assert backend.is_fitted is True

            # Test with explicit forecast_length
            backend2 = ChronosBackend(forecast_length=30)
            with patch.object(backend2, "_load_pipeline"):
                backend2._pipeline = mock_pipeline
                backend2.fit(sample_data)
                assert backend2.forecast_length == 30

    def test_predict_not_fitted(self):
        """Test predict when not fitted (line 146)."""
        backend = ChronosBackend()
        with pytest.raises(ValueError, match="not fitted"):
            backend.predict()

    def test_predict_method(self, sample_data):
        """Test predict method covers lines 144-220."""
        backend = ChronosBackend(forecast_length=7)

        # Set up mock pipeline
        mock_pipeline = MagicMock()
        np.random.seed(42)

        # Test 2D samples (normal case)
        samples_2d = np.random.randn(256, 7) * 5 + 100
        mock_pipeline.predict.return_value = samples_2d

        # Fit the backend
        with patch.object(backend, "_load_pipeline"):
            backend._pipeline = mock_pipeline
            backend.fit(sample_data)

            # Test predict
            result = backend.predict()

            assert isinstance(result, ForecastResult)
            assert isinstance(result.forecast, pd.DataFrame)
            assert len(result.forecast) == 7
            assert "forecast" in result.forecast.columns
            assert "lower_bound" in result.forecast.columns
            assert "upper_bound" in result.forecast.columns
            assert result.model_name == "amazon/chronos-bolt-small"
            assert result.metadata["backend"] == "chronos"

            # Verify bounds make sense
            assert (result.forecast["lower_bound"] <= result.forecast["forecast"]).all()
            assert (result.forecast["forecast"] <= result.forecast["upper_bound"]).all()

    def test_predict_1d_samples(self, sample_data):
        """Test predict with 1D samples (lines 156-158)."""
        backend = ChronosBackend(forecast_length=7)

        mock_pipeline = MagicMock()
        # Return 1D array to test reshaping
        samples_1d = np.array([100, 101, 102, 103, 104, 105, 106])
        mock_pipeline.predict.return_value = samples_1d

        with patch.object(backend, "_load_pipeline"):
            backend._pipeline = mock_pipeline
            backend.fit(sample_data)
            result = backend.predict()

            assert len(result.forecast) == 7

    def test_predict_no_negatives(self, sample_data):
        """Test predict with no_negatives constraint (lines 181-184)."""
        backend = ChronosBackend(no_negatives=True)

        mock_pipeline = MagicMock()
        # Create samples with negative values
        samples = np.random.randn(256, 7) * 50 - 20
        mock_pipeline.predict.return_value = samples

        with patch.object(backend, "_load_pipeline"):
            backend._pipeline = mock_pipeline
            backend.fit(sample_data)
            result = backend.predict()

            # All values should be non-negative
            assert (result.forecast["forecast"] >= 0).all()
            assert (result.forecast["lower_bound"] >= 0).all()
            assert (result.forecast["upper_bound"] >= 0).all()

    def test_predict_frequency_inference(self, sample_data):
        """Test frequency inference in predict (lines 187-196)."""
        backend = ChronosBackend(frequency="infer")

        mock_pipeline = MagicMock()
        samples = np.random.randn(256, 7) * 5 + 100
        mock_pipeline.predict.return_value = samples

        with patch.object(backend, "_load_pipeline"):
            backend._pipeline = mock_pipeline
            backend.fit(sample_data)

            # Mock infer_freq to test the inference path
            with patch("pandas.infer_freq", return_value="B"):  # Business day
                result = backend.predict()
                # Should use inferred frequency
                dates = result.forecast.index
                assert len(dates) == 7

            # Test inference failure (line 195-196)
            with patch("pandas.infer_freq", side_effect=Exception("Cannot infer")):
                result = backend.predict()
                # Should fallback to daily
                dates = result.forecast.index
                assert (dates[1] - dates[0]).days == 1

    def test_predict_custom_frequency(self, sample_data):
        """Test predict with custom frequency."""
        backend = ChronosBackend(frequency="W")

        mock_pipeline = MagicMock()
        samples = np.random.randn(256, 7) * 5 + 100
        mock_pipeline.predict.return_value = samples

        with patch.object(backend, "_load_pipeline"):
            backend._pipeline = mock_pipeline
            backend.fit(sample_data)
            result = backend.predict()

            # Check weekly spacing
            dates = result.forecast.index
            assert (dates[1] - dates[0]).days == 7

    def test_quantile_selection_logic(self, sample_data):
        """Test quantile selection logic in predict (lines 164-175)."""
        # Test with 0.5 not in quantiles
        backend = ChronosBackend(quantile_levels=[0.1, 0.9])

        mock_pipeline = MagicMock()
        np.random.seed(42)
        samples = np.random.normal(100, 10, (1000, 7))
        mock_pipeline.predict.return_value = samples

        with patch.object(backend, "_load_pipeline"):
            backend._pipeline = mock_pipeline
            backend.fit(sample_data)
            result = backend.predict()

            # Should find closest quantile to 0.5
            assert len(result.forecast) == 7

    def test_get_model_info(self):
        """Test get_model_info method (lines 222-230)."""
        backend = ChronosBackend(
            model_name="amazon/chronos-t5-large", num_samples=512, quantile_levels=[0.1, 0.5, 0.9], device_map="cuda"
        )

        info = backend.get_model_info()

        assert info["model_name"] == "amazon/chronos-t5-large"
        assert info["model_params"]["num_samples"] == 512
        assert info["model_params"]["quantile_levels"] == [0.1, 0.5, 0.9]
        assert info["model_params"]["device_map"] == "cuda"

    def test_get_available_models(self):
        """Test get_available_models method (lines 232-244)."""
        backend = ChronosBackend()
        models = backend.get_available_models()

        # Check all expected models are present
        expected = [
            "amazon/chronos-t5-tiny",
            "amazon/chronos-t5-mini",
            "amazon/chronos-t5-small",
            "amazon/chronos-t5-base",
            "amazon/chronos-t5-large",
            "amazon/chronos-bolt-tiny",
            "amazon/chronos-bolt-mini",
            "amazon/chronos-bolt-small",
            "amazon/chronos-bolt-base",
        ]

        for model in expected:
            assert model in models

    def test_validate_input_inherited(self, sample_data):
        """Test inherited validate_input method works."""
        backend = ChronosBackend()

        # Valid data should not raise
        backend.validate_input(sample_data, "Close")

        # Missing column should raise
        with pytest.raises(ValueError, match="not found in data"):
            backend.validate_input(sample_data, "NonExistent")

        # Non-datetime index should raise
        data_no_index = pd.DataFrame({"Close": [100, 101, 102]})
        with pytest.raises(ValueError, match="DatetimeIndex"):
            backend.validate_input(data_no_index, "Close")

    def test_fit_predict_combined(self, sample_data):
        """Test fit_predict convenience method."""
        backend = ChronosBackend()

        mock_pipeline = MagicMock()
        samples = np.random.randn(256, 7) * 5 + 100
        mock_pipeline.predict.return_value = samples

        with patch.object(backend, "_load_pipeline"):
            backend._pipeline = mock_pipeline
            result = backend.fit_predict(sample_data, target_column="Close")

            assert isinstance(result, ForecastResult)
            assert backend.is_fitted is True
