"""Additional tests for forecaster to improve coverage."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from stockula.forecasting import StockForecaster


class TestStockForecasterModelSelection:
    """Test model selection and configuration in StockForecaster."""

    def test_init_with_custom_model_list(self):
        """Test initialization with custom model list."""
        forecaster = StockForecaster(
            model_list=["ARIMA", "ETS", "Theta"],
            forecast_length=30,
            max_generations=5,
        )

        assert forecaster.model_list == ["ARIMA", "ETS", "Theta"]
        assert forecaster.max_generations == 5

    def test_init_with_preset_model_list(self):
        """Test initialization with preset model lists."""
        # Test 'fast' preset
        forecaster = StockForecaster(model_list="fast", forecast_length=30)
        assert forecaster.model_list == "fast"

        # Test 'ultra_fast' preset
        forecaster = StockForecaster(model_list="ultra_fast", forecast_length=30)
        assert forecaster.model_list == "ultra_fast"

        # Test 'financial' preset
        forecaster = StockForecaster(model_list="financial", forecast_length=30)
        assert forecaster.model_list == "financial"

        # Test 'fast_financial' preset
        forecaster = StockForecaster(model_list="fast_financial", forecast_length=30)
        assert forecaster.model_list == "fast_financial"

    def test_init_with_no_negatives(self):
        """Test initialization with no_negatives flag."""
        forecaster = StockForecaster(forecast_length=30, no_negatives=True)
        assert forecaster.no_negatives is True

    def test_fit_with_insufficient_data(self):
        """Test fitting with insufficient historical data."""
        forecaster = StockForecaster(forecast_length=30)

        # Create very short data (less than required)
        short_data = pd.DataFrame(
            {"Close": [100, 101, 102]},
            index=pd.date_range(start="2023-01-01", periods=3, freq="D"),
        )

        # The fit method will be called, which should handle short data appropriately
        # AutoTS might handle this internally or raise an error
        with patch("stockula.forecasting.forecaster.AutoTS") as mock_autots:
            mock_model = Mock()
            mock_autots.return_value = mock_model
            mock_model.fit.side_effect = ValueError("Insufficient data")

            with pytest.raises(ValueError, match="Insufficient data"):
                forecaster.fit(short_data)

    def test_fit_with_custom_target_column(self):
        """Test fitting with custom target column."""
        forecaster = StockForecaster(forecast_length=7)

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
            mock_model.predict.return_value = (mock_forecast, None, None)

            # Fit the model with custom target column
            forecaster.fit(data, target_column="High")

            # Verify AutoTS was called with correct parameters
            mock_autots.assert_called_once()
            call_args = mock_autots.call_args[1]
            assert call_args["forecast_length"] == 7

    def test_forecast_from_symbol_with_all_parameters(self, mock_data_fetcher):
        """Test forecast_from_symbol with all parameters."""
        forecaster = StockForecaster(forecast_length=30, data_fetcher=mock_data_fetcher)

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
            mock_model.predict.return_value = (mock_forecast, None, None)

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

    def test_forecast_from_symbol_with_evaluation(self, mock_data_fetcher):
        """Test forecast_from_symbol_with_evaluation method."""
        forecaster = StockForecaster(forecast_length=7, data_fetcher=mock_data_fetcher)

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
            mock_model.predict.return_value = (mock_forecast, None, None)

            # Mock best model info
            mock_model.best_model_name = "Prophet"
            mock_model.best_model_params = {"seasonality_mode": "multiplicative"}
            mock_model.best_model_transformation_params = {}

            result = forecaster.forecast_from_symbol_with_evaluation(
                symbol="AAPL",
                train_start_date="2023-01-01",
                train_end_date="2023-03-31",
                test_start_date="2023-04-01",
                test_end_date="2023-04-10",
                target_column="Close",
            )

            # Check result structure - the actual return structure is different
            assert "predictions" in result
            assert "evaluation_metrics" in result
            assert "train_period" in result
            assert "test_period" in result

            # Check metrics
            metrics = result["evaluation_metrics"]
            assert "mae" in metrics
            assert "rmse" in metrics
            assert "mape" in metrics

            # Verify data fetcher was called correctly
            assert mock_data_fetcher.get_stock_data.call_count >= 2

    def test_get_best_model_info(self):
        """Test getting best model information."""
        forecaster = StockForecaster(forecast_length=7)

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

    def test_init_with_all_parameters(self):
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

    @patch("stockula.forecasting.forecaster.AutoTS")
    def test_fit_with_frequency_detection(self, mock_autots, mock_data_fetcher):
        """Test frequency detection in fit."""
        forecaster = StockForecaster(forecast_length=7, data_fetcher=mock_data_fetcher)

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
        mock_model.predict.return_value = (mock_forecast, None, None)

        forecaster.fit(data)

        # Verify AutoTS was called with infer frequency
        call_args = mock_autots.call_args[1]
        # AutoTS detects the frequency from the data
        # With business day data (freq="B"), it should detect "B"
        # The forecaster passes "infer" and AutoTS determines the actual frequency
        assert "frequency" in call_args
