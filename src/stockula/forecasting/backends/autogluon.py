"""AutoGluon backend for time series forecasting."""

import warnings
from typing import TYPE_CHECKING, Any

import pandas as pd
from dependency_injector.wiring import Provide, inject
from rich.progress import Progress, SpinnerColumn, TextColumn

from ...cli_manager import cli_manager
from ...interfaces import ILoggingManager
from .base import ForecastBackend, ForecastResult

# Suppress AutoGluon warnings
warnings.filterwarnings("ignore", category=UserWarning, module="autogluon")
warnings.filterwarnings("ignore", category=FutureWarning, module="autogluon")

if TYPE_CHECKING:
    pass


class AutoGluonBackend(ForecastBackend):
    """AutoGluon backend implementation for time series forecasting."""

    predictor: Any  # TimeSeriesPredictor when available

    # Available presets in AutoGluon
    PRESETS = {
        "fast_training": "Fast training with limited models",
        "medium_quality": "Balanced speed and accuracy",
        "high_quality": "High accuracy with more models",
        "best_quality": "Best accuracy, slowest training",
    }

    # Model configurations for different use cases
    MODEL_CONFIGS = {
        "statistical": ["ETS", "ARIMA", "Theta", "Naive", "SeasonalNaive"],
        "tree": ["LightGBM"],
        "deep_learning": ["DeepAR", "TemporalFusionTransformer"],
        "zero_shot": ["Chronos"],
        "fast": ["ETS", "Theta", "Naive", "SeasonalNaive"],
        "accurate": ["DeepAR", "TemporalFusionTransformer", "LightGBM", "ETS"],
    }

    @inject
    def __init__(
        self,
        forecast_length: int | None = None,
        frequency: str = "infer",
        prediction_interval: float = 0.95,
        preset: str = "medium_quality",
        models: str | list[str] | None = None,
        time_limit: int | None = None,
        no_negatives: bool = True,
        eval_metric: str = "MASE",
        logging_manager: ILoggingManager = Provide["logging_manager"],
        use_calendar_covariates: bool = True,
        past_covariate_columns: list[str] | None = None,
        **kwargs,
    ):
        """Initialize the AutoGluon backend.

        Args:
            forecast_length: Number of periods to forecast
            frequency: Data frequency ('infer' to detect automatically)
            prediction_interval: Confidence interval for predictions (0-1)
            preset: AutoGluon preset for model selection and training
            models: Specific models to use (overrides preset)
            time_limit: Time limit in seconds for training
            no_negatives: Constraint predictions to be non-negative
            eval_metric: Evaluation metric for model selection. Options:
                - 'MASE' (default): Mean Absolute Scaled Error - scale-independent, robust
                - 'MAPE': Mean Absolute Percentage Error - use only if all values are positive
                - 'MAE': Mean Absolute Error - scale-dependent, estimates median
                - 'RMSE': Root Mean Squared Error - penalizes large errors more
                - 'SMAPE': Symmetric MAPE - more balanced than MAPE
                - 'WAPE': Weighted Absolute Percentage Error - weights by actual values
                For stock prices, MASE is recommended as it's scale-independent and robust.
            logging_manager: Injected logging manager
            **kwargs: Additional parameters
        """
        super().__init__(
            forecast_length=forecast_length,
            frequency=frequency,
            prediction_interval=prediction_interval,
            no_negatives=no_negatives,
            logging_manager=logging_manager,
        )
        self.preset = preset
        self.models = models
        self.time_limit = time_limit
        self.eval_metric = eval_metric
        self.predictor = None
        self._best_model_name = None
        self.use_calendar_covariates = use_calendar_covariates
        self.past_covariate_columns = past_covariate_columns

    def _get_models(self, models: str | list[str] | None) -> list[str] | None:
        """Get the appropriate model list based on input.

        Args:
            models: String preset or list of model names

        Returns:
            List of model names or None to use AutoGluon defaults
        """
        if models is None:
            return None

        if isinstance(models, list):
            return models

        # Check if it's one of our presets
        if models in self.MODEL_CONFIGS:
            self.logger.info(f"Using {models} model configuration")
            return self.MODEL_CONFIGS[models]

        # Single model name
        return [models]

    def _prepare_data(self, data: pd.DataFrame, target_column: str) -> Any:
        """Prepare data for AutoGluon TimeSeriesDataFrame.

        Args:
            data: Input DataFrame with DatetimeIndex
            target_column: Target column name

        Returns:
            TimeSeriesDataFrame for AutoGluon
        """
        try:
            from autogluon.timeseries import TimeSeriesDataFrame
        except ImportError:
            raise ImportError(
                "AutoGluon TimeSeries not installed. Install with: pip install autogluon.timeseries"
            ) from None

        # Prepare data in AutoGluon format
        df = data[[target_column]].copy()
        df = df.reset_index()
        df.columns = ["timestamp", "target"]

        # Add item_id column (required by AutoGluon)
        df["item_id"] = "stock"

        # Create TimeSeriesDataFrame
        ts_df = TimeSeriesDataFrame.from_data_frame(
            df,
            id_column="item_id",
            timestamp_column="timestamp",
        )

        return ts_df

    def _prepare_past_covariates(self, data: pd.DataFrame, feature_cols: list[str]) -> Any | None:
        """Prepare past covariates (observed in history, not known in future).

        Args:
            data: Input DataFrame with DatetimeIndex
            feature_cols: Columns to use as past covariates

        Returns:
            TimeSeriesDataFrame with past covariates or None
        """
        if not feature_cols:
            return None

        try:
            from autogluon.timeseries import TimeSeriesDataFrame
        except ImportError:
            return None

        cov = data[feature_cols].copy()
        cov = cov.reset_index()
        cov.rename(columns={cov.columns[0]: "timestamp"}, inplace=True)
        cov["item_id"] = "stock"

        return TimeSeriesDataFrame.from_data_frame(
            cov,
            id_column="item_id",
            timestamp_column="timestamp",
        )

    def _build_calendar_features(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Create simple calendar known covariates from timestamps.

        Features:
            - day_of_week (0-6)
            - month (1-12)
            - is_month_start (0/1)
            - is_month_end (0/1)
        """
        df = pd.DataFrame({"timestamp": index})
        df["day_of_week"] = df["timestamp"].dt.dayofweek.astype(int)
        df["month"] = df["timestamp"].dt.month.astype(int)
        df["is_month_start"] = df["timestamp"].dt.is_month_start.astype(int)
        df["is_month_end"] = df["timestamp"].dt.is_month_end.astype(int)
        return df

    def _prepare_known_covariates(self, index: pd.DatetimeIndex) -> Any | None:
        """Prepare known covariates using calendar features for given timestamps.

        Returns TimeSeriesDataFrame or None.
        """
        try:
            from autogluon.timeseries import TimeSeriesDataFrame
        except ImportError:
            return None

        cal = self._build_calendar_features(index)
        cal["item_id"] = "stock"
        return TimeSeriesDataFrame.from_data_frame(
            cal,
            id_column="item_id",
            timestamp_column="timestamp",
        )

    def fit(
        self,
        data: pd.DataFrame,
        target_column: str = "Close",
        show_progress: bool = True,
        preset: str | None = None,
        models: str | list[str] | None = None,
        time_limit: int | None = None,
        **kwargs,
    ) -> "AutoGluonBackend":
        """Fit the AutoGluon model on historical data.

        Args:
            data: DataFrame with time series data (index should be DatetimeIndex)
            target_column: Column to forecast
            show_progress: Whether to show progress bar
            preset: Override default preset
            models: Override default models
            time_limit: Override default time limit
            **kwargs: Additional parameters

        Returns:
            Self for chaining
        """
        # Validate input
        self.validate_input(data, target_column)

        # Use provided parameters or defaults
        preset = preset or self.preset
        models_to_use = models if models is not None else self.models
        time_limit = time_limit or self.time_limit

        # Get actual model list
        model_list = self._get_models(models_to_use)

        # Prepare data
        ts_data = self._prepare_data(data, target_column)

        # Build covariates
        # Past covariates from observed series (if available)
        if self.past_covariate_columns is None:
            candidate_past_cols = [
                c for c in ["Open", "High", "Low", "Adj Close", "Volume"] if c in data.columns and c != target_column
            ]
        else:
            candidate_past_cols = [c for c in self.past_covariate_columns if c in data.columns and c != target_column]

        past_cov_ts = self._prepare_past_covariates(data, candidate_past_cols) if candidate_past_cols else None

        self.logger.debug(f"Fitting AutoGluon model on {len(data)} data points")
        self.logger.debug(f"Date range: {data.index.min()} to {data.index.max()}")

        try:
            from autogluon.timeseries import TimeSeriesPredictor
        except ImportError:
            raise ImportError(
                "AutoGluon TimeSeries not installed. Install with: pip install autogluon.timeseries"
            ) from None

        # Determine prediction length
        if self.forecast_length is None:
            self.forecast_length = 14  # Default

        # Infer frequency if needed
        freq_to_use = self.frequency
        if self.frequency == "infer":
            try:
                inferred_freq = pd.infer_freq(data.index)
                freq_to_use = inferred_freq if inferred_freq else "D"
            except Exception:
                freq_to_use = "D"

        # Map pandas frequency to AutoGluon frequency
        freq_map = {
            "D": "D",  # Daily
            "B": "B",  # Business day
            "W": "W",  # Weekly
            "M": "M",  # Monthly
            "Q": "Q",  # Quarterly
            "H": "H",  # Hourly
        }
        ag_freq = freq_map.get(freq_to_use, "D")

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=cli_manager.get_console(),
                transient=True,
            ) as progress:
                task = progress.add_task("[cyan]Training AutoGluon models on historical data...", total=None)

                # Log the evaluation metric being used
                self.logger.info(f"Using evaluation metric: {self.eval_metric}")

                # Create predictor with explicit eval_metric
                self.predictor = TimeSeriesPredictor(
                    prediction_length=self.forecast_length,
                    freq=ag_freq,
                    eval_metric=self.eval_metric,
                    quantile_levels=[0.05, 0.5, 0.95] if self.prediction_interval == 0.9 else None,
                    verbosity=0,  # Suppress output in progress mode
                )

                # Fit the model
                assert self.predictor is not None
                self.predictor.fit(
                    train_data=ts_data,
                    presets=preset,
                    hyperparameters={"model": model_list} if model_list else None,
                    time_limit=time_limit,
                )

                progress.update(task, description="[green]✓ AutoGluon model training completed")
        else:
            # Log the evaluation metric being used
            self.logger.info(f"Using evaluation metric: {self.eval_metric}")

            # Create predictor with explicit eval_metric
            self.predictor = TimeSeriesPredictor(
                prediction_length=self.forecast_length,
                freq=ag_freq,
                eval_metric=self.eval_metric,
                quantile_levels=[0.05, 0.5, 0.95] if self.prediction_interval == 0.9 else None,
                verbosity=2 if self.logger.isEnabledFor(10) else 0,
            )

            # Fit the model
            # Known covariates for training portion: generate calendar features for training index
            known_cov_train_ts = (
                self._prepare_known_covariates(ts_data.index.get_level_values("timestamp").unique())
                if self.use_calendar_covariates
                else None
            )

            assert self.predictor is not None
            self.predictor.fit(
                train_data=ts_data,
                presets=preset,
                hyperparameters={"model": model_list} if model_list else None,
                time_limit=time_limit,
                known_covariates=known_cov_train_ts,
                past_covariates=past_cov_ts,
            )

        self.is_fitted = True
        assert self.predictor is not None
        self._best_model_name = self.predictor.get_model_best()
        self.logger.debug(f"AutoGluon model fitting completed. Best model: {self._best_model_name}")

        return self

    def predict(self, **kwargs) -> ForecastResult:
        """Generate predictions using fitted AutoGluon model.

        Returns:
            ForecastResult with predictions and metadata
        """
        if not self.is_fitted or self.predictor is None:
            raise ValueError("Model not fitted. Call fit() first.")

        self.logger.debug("Generating AutoGluon predictions...")

        # Generate predictions with known covariates for the forecast horizon
        known_cov_future_ts = None
        if self.use_calendar_covariates:
            # Let the predictor infer future timestamps; provide calendar features if possible
            try:
                # Best effort to derive future timestamps: use training freq and last timestamp
                freq = getattr(getattr(self.predictor, "_learner", object()), "freq", "D")
                train_data = getattr(getattr(self.predictor, "_learner", object()), "train_data", None)
                import pandas as pd

                if train_data is not None:
                    last_timestamp = train_data.index.get_level_values("timestamp").max()
                else:
                    last_timestamp = pd.Timestamp.now().normalize()

                future_index = pd.date_range(
                    start=last_timestamp + pd.tseries.frequencies.to_offset(freq),
                    periods=self.forecast_length,
                    freq=freq,
                )
                known_cov_future_ts = self._prepare_known_covariates(future_index)
            except Exception:
                known_cov_future_ts = None

        predictions = self.predictor.predict(known_covariates=known_cov_future_ts)

        # Normalize predictions to a DataFrame with timestamp index for item_id 'stock'
        pred_df = predictions.reset_index()
        if "item_id" in pred_df.columns:
            pred_df = pred_df[pred_df["item_id"] == "stock"]
        if "timestamp" in pred_df.columns:
            pred_df = pred_df.set_index("timestamp")

        # Choose central tendency column
        median_col = "mean" if "mean" in pred_df.columns else ("0.5" if "0.5" in pred_df.columns else None)
        if median_col is None:
            # Fallback: last numeric column
            numeric_cols = [c for c in pred_df.columns if pd.api.types.is_numeric_dtype(pred_df[c])]
            median_col = numeric_cols[-1] if numeric_cols else pred_df.columns[-1]

        # Determine interval columns closest to requested interval
        alpha = (1.0 - float(self.prediction_interval)) / 2.0
        low_target, high_target = alpha, 1.0 - alpha
        # Available quantile columns look like '0.05', '0.5', '0.95'
        qcols = [c for c in pred_df.columns if isinstance(c, str) and c.replace(".", "", 1).isdigit()]

        def _closest(col_target: float) -> str | None:
            if not qcols:
                return None
            import numpy as np

            arr = np.array([float(c) for c in qcols])
            idx = int(np.argmin(np.abs(arr - col_target)))
            return qcols[idx]

        low_col = _closest(low_target)
        high_col = _closest(high_target)

        forecast_values = pred_df[median_col].to_numpy()
        if low_col and high_col and low_col in pred_df.columns and high_col in pred_df.columns:
            lower_bound = pred_df[low_col].to_numpy()
            upper_bound = pred_df[high_col].to_numpy()
        else:
            lower_bound = forecast_values * 0.9
            upper_bound = forecast_values * 1.1

        # Apply non-negative constraint if needed
        if self.no_negatives:
            forecast_values = forecast_values.clip(min=0)
            lower_bound = lower_bound.clip(min=0)
            upper_bound = upper_bound.clip(min=0)

        # Create result DataFrame with proper index
        # We need to generate future dates
        import pandas as pd

        last_date = pd.Timestamp.now()
        freq = self.predictor._learner.freq if hasattr(self.predictor, "_learner") else "D"
        future_dates = pd.date_range(start=last_date, periods=len(forecast_values) + 1, freq=freq)[1:]

        result_df = pd.DataFrame(
            {
                "forecast": forecast_values,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            },
            index=future_dates,
        )

        self.logger.debug(f"Generated {len(result_df)} forecast points")

        # Get model info
        model_info = self.get_model_info()

        # Get leaderboard for additional metrics
        leaderboard = self.predictor.leaderboard(silent=True)
        best_model_metrics = leaderboard.iloc[0].to_dict() if not leaderboard.empty else {}

        return ForecastResult(
            forecast=result_df,
            model_name=model_info["model_name"],
            model_params=model_info.get("model_params", {}),
            metrics={
                "score": float(best_model_metrics.get("score", 0.0)),
                "eval_metric": self.eval_metric,  # type: ignore[dict-item]
                "pred_time": float(best_model_metrics.get("pred_time", 0.0)),
                "fit_time": float(best_model_metrics.get("fit_time", 0.0)),
            },
            metadata={
                "preset": self.preset,
                "models_trained": len(leaderboard) if leaderboard is not None else 1,
                "evaluation_metric": self.eval_metric,
            },
        )

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the fitted AutoGluon model.

        Returns:
            Dictionary with model information
        """
        if not self.is_fitted or self.predictor is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get leaderboard
        leaderboard = self.predictor.leaderboard(silent=True)

        # Get best model info
        best_model = self._best_model_name

        # Try to get model parameters
        model_params = {}
        try:
            if hasattr(self.predictor, "get_model_hyperparameters"):
                model_params = self.predictor.get_model_hyperparameters(best_model)
        except Exception:
            pass

        return {
            "model_name": best_model,
            "model_params": model_params,
            "leaderboard": leaderboard.to_dict() if leaderboard is not None else {},
        }

    def get_available_models(self) -> list[str]:
        """Get list of available AutoGluon models.

        Returns:
            List of model names
        """
        return [
            # Statistical models
            "ARIMA",
            "ETS",
            "Theta",
            "Naive",
            "SeasonalNaive",
            "RecursiveTabular",
            "DirectTabular",
            # Tree-based
            "LightGBM",
            # Deep learning
            "DeepAR",
            "TemporalFusionTransformer",
            "PatchTST",
            # Zero-shot
            "Chronos",
        ]

    def evaluate_models(self) -> pd.DataFrame:
        """Get detailed evaluation of all trained models.

        Returns:
            DataFrame with model performance metrics
        """
        if not self.is_fitted or self.predictor is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.predictor.leaderboard()
