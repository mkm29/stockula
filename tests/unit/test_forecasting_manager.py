"""Tests for ForecastingManager."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

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
        return mock

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = MagicMock()
        config.forecast.forecast_length = 14
        config.forecast.frequency = "infer"
        config.forecast.prediction_interval = 0.95
        config.forecast.ensemble = "auto"
        config.forecast.num_validations = 2
        config.forecast.validation_method = "backwards"
        config.forecast.model_list = "fast"
        config.forecast.max_generations = 5
        config.forecast.no_negatives = True
        config.forecast.train_start_date = None
        config.forecast.train_end_date = None
        config.forecast.test_start_date = None
        config.forecast.test_end_date = None
        config.data.start_date = "2023-01-01"
        config.data.end_date = "2023-12-31"
        return config

    @pytest.fixture
    def forecasting_manager(self, mock_data_fetcher, mock_logging_manager):
        """Create ForecastingManager instance."""
        return ForecastingManager(data_fetcher=mock_data_fetcher, logging_manager=mock_logging_manager)

    def test_init(self, forecasting_manager, mock_data_fetcher, mock_logging_manager):
        """Test ForecastingManager initialization."""
        assert forecasting_manager.data_fetcher == mock_data_fetcher
        assert forecasting_manager.logger == mock_logging_manager
        assert forecasting_manager.standard_forecaster is not None
        assert forecasting_manager.fast_forecaster is not None
        assert forecasting_manager.financial_forecaster is not None

    def test_get_forecaster(self, forecasting_manager):
        """Test getting different forecasters."""
        # Test ultra fast
        forecaster = forecasting_manager.get_forecaster("ultra_fast")
        assert forecaster == forecasting_manager.fast_forecaster

        # Test financial
        forecaster = forecasting_manager.get_forecaster("financial")
        assert forecaster == forecasting_manager.financial_forecaster

        # Test default (standard)
        forecaster = forecasting_manager.get_forecaster("standard")
        assert forecaster == forecasting_manager.standard_forecaster

    @patch("stockula.forecasting.manager.StockForecaster")
    def test_forecast_symbol_standard(self, mock_stock_forecaster_class, forecasting_manager, mock_config):
        """Test standard forecasting for a symbol."""
        # Setup mock forecaster
        mock_forecaster = MagicMock()
        mock_stock_forecaster_class.return_value = mock_forecaster

        # Setup mock predictions
        mock_predictions = pd.DataFrame(
            {
                "forecast": [105, 106, 107, 108, 109, 110],
                "lower_bound": [100, 101, 102, 103, 104, 105],
                "upper_bound": [110, 111, 112, 113, 114, 115],
            },
            index=pd.date_range("2024-01-01", periods=6),
        )

        mock_forecaster.forecast_from_symbol.return_value = mock_predictions
        mock_forecaster.get_best_model.return_value = {"model_name": "LastValueNaive", "model_params": {}}

        # Run forecast
        result = forecasting_manager.forecast_symbol("AAPL", mock_config, use_evaluation=False)

        # Verify result
        assert result["ticker"] == "AAPL"
        assert result["current_price"] == 105
        assert result["forecast_price"] == 110
        assert result["lower_bound"] == 105
        assert result["upper_bound"] == 115
        assert result["best_model"] == "LastValueNaive"

    def test_forecast_symbol_with_evaluation(self, forecasting_manager, mock_config):
        """Test forecasting with evaluation."""
        # Setup config for evaluation
        mock_config.forecast.train_start_date = "2023-01-01"
        mock_config.forecast.train_end_date = "2023-10-31"
        mock_config.forecast.test_start_date = "2023-11-01"
        mock_config.forecast.test_end_date = "2023-12-31"

        with patch.object(forecasting_manager, "_forecast_with_evaluation") as mock_eval:
            mock_eval.return_value = {
                "ticker": "AAPL",
                "current_price": 105,
                "forecast_price": 110,
                "evaluation": {"rmse": 2.5, "mape": 1.2},
            }

            result = forecasting_manager.forecast_symbol("AAPL", mock_config, use_evaluation=True)

            assert "evaluation" in result
            assert result["evaluation"]["rmse"] == 2.5

    def test_forecast_multiple_symbols(self, forecasting_manager, mock_config):
        """Test forecasting multiple symbols."""
        with patch.object(forecasting_manager, "forecast_symbol") as mock_forecast:
            # Setup mock responses
            mock_forecast.side_effect = [
                {"ticker": "AAPL", "forecast_price": 110},
                {"ticker": "GOOGL", "forecast_price": 150},
                {"ticker": "MSFT", "error": "Test error"},
            ]

            results = forecasting_manager.forecast_multiple_symbols(["AAPL", "GOOGL", "MSFT"], mock_config)

            assert len(results) == 3
            assert results["AAPL"]["forecast_price"] == 110
            assert results["GOOGL"]["forecast_price"] == 150
            assert "error" in results["MSFT"]

    def test_quick_forecast(self, forecasting_manager):
        """Test quick forecasting."""
        with patch.object(forecasting_manager.fast_forecaster, "forecast_from_symbol") as mock_forecast:
            # Setup mock predictions
            mock_predictions = pd.DataFrame(
                {
                    "forecast": [105, 110],
                    "lower_bound": [100, 105],
                    "upper_bound": [110, 115],
                },
                index=pd.date_range("2024-01-01", periods=2),
            )

            mock_forecast.return_value = mock_predictions
            forecasting_manager.fast_forecaster.get_best_model = MagicMock(
                return_value={"model_name": "LastValueNaive"}
            )

            result = forecasting_manager.quick_forecast("AAPL", forecast_days=7)

            assert result["ticker"] == "AAPL"
            assert result["forecast_length"] == 7
            assert result["confidence"] == "Quick forecast - lower confidence"

    def test_financial_forecast(self, forecasting_manager, mock_config):
        """Test financial-specific forecasting."""
        with patch("stockula.forecasting.manager.StockForecaster") as mock_forecaster_class:
            # Setup mock forecaster
            mock_forecaster = MagicMock()
            mock_forecaster_class.return_value = mock_forecaster

            # Setup mock predictions
            mock_predictions = pd.DataFrame(
                {
                    "forecast": [105, 110],
                    "lower_bound": [100, 105],
                    "upper_bound": [110, 115],
                },
                index=pd.date_range("2024-01-01", periods=2),
            )

            mock_forecaster.forecast_from_symbol.return_value = mock_predictions
            mock_forecaster.get_best_model.return_value = {"model_name": "ETS", "model_params": {}}

            result = forecasting_manager.financial_forecast("AAPL", mock_config)

            assert result["ticker"] == "AAPL"
            assert result["model_type"] == "financial"

            # Verify financial-specific settings were used
            mock_forecaster_class.assert_called_with(
                forecast_length=mock_config.forecast.forecast_length,
                frequency=mock_config.forecast.frequency,
                prediction_interval=mock_config.forecast.prediction_interval,
                ensemble="distance",  # Financial specific
                num_validations=3,  # Financial specific
                validation_method=mock_config.forecast.validation_method,
                model_list="financial",  # Financial specific
                max_generations=mock_config.forecast.max_generations,
                no_negatives=True,  # Financial specific
                data_fetcher=forecasting_manager.data_fetcher,
                logging_manager=forecasting_manager.logger,
            )

    def test_get_available_models(self, forecasting_manager):
        """Test getting available models."""
        models = forecasting_manager.get_available_models()

        assert "ultra_fast" in models
        assert "fast" in models
        assert "financial" in models
        assert "fast_financial" in models

        # Check that lists are properly defined
        assert len(models["ultra_fast"]) > 0
        assert len(models["financial"]) > 0

    def test_validate_forecast_config(self, forecasting_manager, mock_config):
        """Test configuration validation."""
        # Valid config should pass
        forecasting_manager.validate_forecast_config(mock_config)

        # Test invalid forecast length
        mock_config.forecast.forecast_length = 0
        with pytest.raises(ValueError, match="forecast_length must be positive"):
            forecasting_manager.validate_forecast_config(mock_config)

        # Test invalid prediction interval
        mock_config.forecast.forecast_length = 14
        mock_config.forecast.prediction_interval = 1.5
        with pytest.raises(ValueError, match="prediction_interval must be between 0 and 1"):
            forecasting_manager.validate_forecast_config(mock_config)

        # Test mismatched train/test dates
        mock_config.forecast.prediction_interval = 0.95
        mock_config.forecast.train_start_date = "2023-01-01"
        mock_config.forecast.train_end_date = "2023-10-31"
        mock_config.forecast.test_start_date = None
        mock_config.forecast.test_end_date = None
        with pytest.raises(ValueError, match="Both train and test date ranges"):
            forecasting_manager.validate_forecast_config(mock_config)

    def test_date_to_string(self, forecasting_manager):
        """Test date conversion utility."""
        # Test None
        assert forecasting_manager._date_to_string(None) is None

        # Test string
        assert forecasting_manager._date_to_string("2023-01-01") == "2023-01-01"

        # Test datetime
        from datetime import datetime

        dt = datetime(2023, 1, 1)
        assert forecasting_manager._date_to_string(dt) == "2023-01-01"

    def test_standard_forecast_error_handling(self, forecasting_manager, mock_config):
        """Test error handling in forecast_multiple_symbols."""
        with patch.object(forecasting_manager, "forecast_symbol") as mock_forecast:
            mock_forecast.side_effect = Exception("Test error")

            # Error is caught in forecast_multiple_symbols
            results = forecasting_manager.forecast_multiple_symbols(["AAPL"], mock_config)
            assert "error" in results["AAPL"]
            assert results["AAPL"]["error"] == "Test error"
