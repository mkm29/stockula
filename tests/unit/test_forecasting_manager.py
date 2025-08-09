"""Tests for ForecastingManager with backend abstraction."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from stockula.forecasting.backends import ForecastResult
from stockula.forecasting.manager import ForecastingManager


class TestForecastingManager:
    """Test suite for ForecastingManager."""

    @pytest.fixture
    def mock_data_fetcher(self):
        """Create mock data fetcher."""
        mock = MagicMock()
        # Create sample data
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        data = pd.DataFrame(
            {
                "Open": [100] * len(dates),
                "High": [110] * len(dates),
                "Low": [90] * len(dates),
                "Close": [105] * len(dates),
                "Volume": [1000000] * len(dates),
            },
            index=dates,
        )
        mock.get_stock_data.return_value = data
        return mock

    @pytest.fixture
    def mock_logging_manager(self):
        """Create mock logging manager."""
        mock = MagicMock()
        mock.info = MagicMock()
        mock.error = MagicMock()
        mock.warning = MagicMock()
        mock.debug = MagicMock()
        mock.isEnabledFor = MagicMock(return_value=False)
        return mock

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        from stockula.config import DataConfig, ForecastConfig, StockulaConfig

        config = StockulaConfig(
            forecast=ForecastConfig(
                backend="autots",
                forecast_length=30,
                model_list="fast",
                max_generations=2,
                num_validations=1,
                prediction_interval=0.95,
            ),
            data=DataConfig(
                start_date="2023-01-01",
                end_date="2023-12-31",
            ),
        )
        return config

    @pytest.fixture
    def forecasting_manager(self, mock_data_fetcher, mock_logging_manager):
        """Create ForecastingManager instance."""
        return ForecastingManager(data_fetcher=mock_data_fetcher, logging_manager=mock_logging_manager)

    def test_init(self, forecasting_manager, mock_data_fetcher, mock_logging_manager):
        """Test ForecastingManager initialization."""
        assert forecasting_manager.data_fetcher == mock_data_fetcher
        assert forecasting_manager.logger == mock_logging_manager

    def test_create_backend(self, forecasting_manager, mock_config):
        """Test creating different forecasting backends."""
        # Test creating AutoTS backend
        backend = forecasting_manager.create_backend(mock_config.forecast)
        assert backend is not None

        # Test that backend is AutoTS when configured
        from stockula.forecasting.backends import AutoTSBackend

        mock_config.forecast.backend = "autots"
        backend = forecasting_manager.create_backend(mock_config.forecast)
        assert isinstance(backend, AutoTSBackend)

    @patch("stockula.forecasting.manager.create_forecast_backend")
    def test_forecast_symbol(self, mock_create_backend, forecasting_manager, mock_config):
        """Test forecasting for a single symbol."""
        # Setup mock backend
        mock_backend = MagicMock()
        mock_create_backend.return_value = mock_backend

        # Setup mock forecast result
        forecast_df = pd.DataFrame(
            {
                "forecast": [105, 106, 107],
                "lower_bound": [100, 101, 102],
                "upper_bound": [110, 111, 112],
            }
        )
        mock_result = ForecastResult(
            forecast=forecast_df, model_name="TestModel", model_params={}, metrics={"score": 0.9}
        )
        mock_backend.fit_predict.return_value = mock_result
        mock_backend.get_model_info.return_value = {"model_name": "TestModel", "model_params": {}}

        # Test forecast
        result = forecasting_manager.forecast_symbol("AAPL", mock_config)

        # Verify result structure
        assert result["ticker"] == "AAPL"
        assert result["backend"] == "autots"
        assert "current_price" in result
        assert "forecast_price" in result
        assert "lower_bound" in result
        assert "upper_bound" in result
        assert result["forecast_length"] == 30
        assert result["best_model"] == "TestModel"

    def test_forecast_multiple_symbols(self, forecasting_manager, mock_config):
        """Test forecasting multiple symbols."""
        with patch.object(forecasting_manager, "forecast_symbol") as mock_forecast:
            mock_forecast.return_value = {"ticker": "AAPL", "forecast_price": 150.0, "error": None}

            results = forecasting_manager.forecast_multiple_symbols(["AAPL", "GOOGL"], mock_config)

            assert len(results) == 2
            assert "AAPL" in results
            assert "GOOGL" in results
            assert mock_forecast.call_count == 2

    def test_quick_forecast(self, forecasting_manager, mock_data_fetcher):
        """Test quick forecasting."""
        # Setup mock data
        dates = pd.date_range(start="2023-01-01", end="2023-03-31", freq="D")
        data = pd.DataFrame(
            {"Close": [100 + i for i in range(len(dates))]},
            index=dates,
        )
        mock_data_fetcher.get_stock_data.return_value = data

        with patch("stockula.forecasting.backends.AutoTSBackend") as MockBackend:
            mock_backend = MagicMock()
            MockBackend.return_value = mock_backend

            # Setup mock result
            forecast_df = pd.DataFrame(
                {
                    "forecast": [105, 106, 107],
                    "lower_bound": [100, 101, 102],
                    "upper_bound": [110, 111, 112],
                }
            )
            mock_result = ForecastResult(forecast=forecast_df, model_name="FastModel", model_params={})
            mock_backend.fit_predict.return_value = mock_result
            mock_backend.get_model_info.return_value = {"model_name": "FastModel"}

            result = forecasting_manager.quick_forecast("AAPL", forecast_days=7)

            assert result["ticker"] == "AAPL"
            assert result["forecast_length"] == 7
            assert result["backend"] == "autots"
            assert "confidence" in result

    def test_compare_backends(self, forecasting_manager, mock_config, mock_data_fetcher):
        """Test comparing different backends."""
        # Setup mock data
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        data = pd.DataFrame(
            {"Close": [100] * len(dates)},
            index=dates,
        )
        mock_data_fetcher.get_stock_data.return_value = data

        with patch("stockula.forecasting.manager.create_forecast_backend") as mock_create:
            # Mock AutoTS backend
            mock_autots = MagicMock()
            mock_autots.fit_predict.return_value = ForecastResult(
                forecast=pd.DataFrame({"forecast": [105], "lower_bound": [100], "upper_bound": [110]}),
                model_name="AutoTS_Model",
                model_params={},
            )
            mock_autots.get_model_info.return_value = {"model_name": "AutoTS_Model"}

            # Mock AutoGluon backend
            mock_autogluon = MagicMock()
            mock_autogluon.fit_predict.return_value = ForecastResult(
                forecast=pd.DataFrame({"forecast": [106], "lower_bound": [101], "upper_bound": [111]}),
                model_name="AutoGluon_Model",
                model_params={},
                metrics={"score": 0.95},
            )
            mock_autogluon.get_model_info.return_value = {"model_name": "AutoGluon_Model"}

            # Configure mock to return different backends
            mock_create.side_effect = [mock_autots, mock_autogluon]

            results = forecasting_manager.compare_backends("AAPL", mock_config)

            assert "autots" in results
            assert "autogluon" in results
            assert results["autots"]["model_name"] == "AutoTS_Model"
            assert results["autogluon"]["model_name"] == "AutoGluon_Model"

    def test_validate_forecast_config(self, forecasting_manager):
        """Test forecast configuration validation."""
        from stockula.config import ForecastConfig

        # Valid config
        valid_config = ForecastConfig(
            backend="autots", forecast_length=30, prediction_interval=0.95, num_validations=2, max_generations=2
        )
        forecasting_manager.validate_forecast_config(valid_config)  # Should not raise

        # Invalid forecast_length - test with None instead of 0 since Pydantic validates 0
        invalid_config = ForecastConfig(backend="autots", forecast_length=None, prediction_interval=0.95)
        with pytest.raises(ValueError, match="forecast_length must be positive"):
            forecasting_manager.validate_forecast_config(invalid_config)

        # Invalid backend
        invalid_config = ForecastConfig(backend="invalid_backend", forecast_length=30, prediction_interval=0.95)
        with pytest.raises(ValueError, match="Invalid backend"):
            forecasting_manager.validate_forecast_config(invalid_config)

    def test_forecast_multiple_symbols_with_progress(self, forecasting_manager, mock_config):
        """Test forecasting multiple symbols with progress tracking."""
        with (
            patch.object(forecasting_manager, "forecast_symbol") as mock_forecast,
            patch("stockula.forecasting.manager.Progress") as mock_progress_class,
        ):
            mock_forecast.return_value = {"ticker": "AAPL", "forecast_price": 150.0, "backend": "autots"}

            # Mock the Progress context manager
            mock_progress = MagicMock()
            mock_progress.__enter__.return_value = mock_progress
            mock_progress_class.return_value = mock_progress
            mock_progress.add_task.return_value = "task_id"

            results = forecasting_manager.forecast_multiple_symbols_with_progress(["AAPL", "GOOGL"], mock_config)

            assert len(results) == 2
            assert mock_forecast.call_count == 2

    def test_error_handling_in_forecast_symbol(self, forecasting_manager, mock_config, mock_data_fetcher):
        """Test error handling when forecasting fails."""
        # Make data fetcher return empty data
        mock_data_fetcher.get_stock_data.return_value = pd.DataFrame()

        with pytest.raises(ValueError, match="No data available"):
            forecasting_manager.forecast_symbol("INVALID", mock_config)

    def test_forecast_with_evaluation_parameter(self, forecasting_manager, mock_config):
        """Test that use_evaluation parameter is handled correctly."""
        with patch.object(forecasting_manager, "create_backend") as mock_create:
            mock_backend = MagicMock()
            mock_create.return_value = mock_backend

            # Setup mock result
            forecast_df = pd.DataFrame(
                {
                    "forecast": [105],
                    "lower_bound": [100],
                    "upper_bound": [110],
                }
            )
            mock_backend.fit_predict.return_value = ForecastResult(
                forecast=forecast_df, model_name="TestModel", model_params={}
            )
            mock_backend.get_model_info.return_value = {"model_name": "TestModel"}

            # Test with use_evaluation=True (should be ignored but not fail)
            result = forecasting_manager.forecast_symbol("AAPL", mock_config, use_evaluation=True)
            assert result["ticker"] == "AAPL"
