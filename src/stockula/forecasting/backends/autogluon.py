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
    # Place any type-only imports here if needed in the future
    ...


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
        """Fit the AutoGluon model on historical data."""
        self.validate_input(data, target_column)
        preset = preset or self.preset
        models_to_use = models if models is not None else self.models
        time_limit = time_limit or self.time_limit
        model_list = self._get_models(models_to_use)
        ts_data = self._prepare_data(data, target_column)
        past_cov_ts = self._get_past_covariates(data, target_column)
        ag_freq = self._infer_ag_freq(data)
        self._init_predictor(ag_freq, show_progress)
        known_cov_train_ts = self._get_known_covariates(ts_data) if not show_progress else None
        self._fit_predictor(ts_data, preset, model_list, time_limit, known_cov_train_ts, past_cov_ts, show_progress)
        self.is_fitted = True
        self._best_model_name = self.predictor.get_model_best()
        self.logger.debug(f"AutoGluon model fitting completed. Best model: {self._best_model_name}")
        return self

    def _get_past_covariates(self, data: pd.DataFrame, target_column: str) -> Any | None:
        if self.past_covariate_columns is None:
            candidate_past_cols = [
                c for c in ["Open", "High", "Low", "Adj Close", "Volume"] if c in data.columns and c != target_column
            ]
        else:
            candidate_past_cols = [c for c in self.past_covariate_columns if c in data.columns and c != target_column]
        return self._prepare_past_covariates(data, candidate_past_cols) if candidate_past_cols else None

    def _infer_ag_freq(self, data: pd.DataFrame) -> str:
        freq_to_use = self.frequency
        if self.frequency == "infer":
            try:
                inferred_freq = pd.infer_freq(data.index)
                freq_to_use = inferred_freq if inferred_freq else "D"
            except Exception:
                freq_to_use = "D"
        freq_map = {
            "D": "D",
            "B": "B",
            "W": "W",
            "M": "M",
            "Q": "Q",
            "H": "H",
        }
        return freq_map.get(freq_to_use, "D")

    def _init_predictor(self, ag_freq: str, show_progress: bool):
        try:
            from autogluon.timeseries import TimeSeriesPredictor
        except ImportError:
            raise ImportError(
                "AutoGluon TimeSeries not installed. Install with: pip install autogluon.timeseries"
            ) from None
        if self.forecast_length is None:
            self.forecast_length = 14
        self.logger.info(f"Using evaluation metric: {self.eval_metric}")
        quantiles = [0.05, 0.5, 0.95] if self.prediction_interval == 0.9 else None
        verbosity = 0 if show_progress else (2 if self.logger.is_enabled_for(10) else 0)
        self.predictor = TimeSeriesPredictor(
            prediction_length=self.forecast_length,
            freq=ag_freq,
            eval_metric=self.eval_metric,
            quantile_levels=quantiles,
            verbosity=verbosity,
        )

    def _get_known_covariates(self, ts_data: Any) -> Any | None:
        if self.use_calendar_covariates:
            return self._prepare_known_covariates(ts_data.index.get_level_values("timestamp").unique())
        return None

    def _fit_predictor(
        self,
        ts_data: Any,
        preset: str,
        model_list: list[str] | None,
        time_limit: int | None,
        known_cov_train_ts: Any | None,
        past_cov_ts: Any | None,
        show_progress: bool,
    ):
        fit_kwargs = {
            "train_data": ts_data,
            "presets": preset,
            "hyperparameters": {"model": model_list} if model_list else None,
            "time_limit": time_limit,
        }
        if not show_progress:
            fit_kwargs["known_covariates"] = known_cov_train_ts
            fit_kwargs["past_covariates"] = past_cov_ts
            assert self.predictor is not None
            self.predictor.fit(**fit_kwargs)
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=cli_manager.get_console(),
                transient=True,
            ) as progress:
                task = progress.add_task("[cyan]Training AutoGluon models on historical data...", total=None)
                assert self.predictor is not None
                self.predictor.fit(**fit_kwargs)
                progress.update(task, description="[green]✓ AutoGluon model training completed")

    def predict(self, **kwargs) -> ForecastResult:
        """Generate predictions using fitted AutoGluon model."""
        if not self.is_fitted or self.predictor is None:
            raise ValueError("Model not fitted. Call fit() first.")

        self.logger.debug("Generating AutoGluon predictions...")

        known_cov_future_ts = self._prepare_future_covariates() if self.use_calendar_covariates else None
        predictions = self.predictor.predict(known_covariates=known_cov_future_ts)
        pred_df = self._normalize_predictions(predictions)
        median_col = self._get_median_col(pred_df)
        low_col, high_col = self._get_quantile_cols(pred_df)
        forecast_values, lower_bound, upper_bound = self._get_forecast_bounds(pred_df, median_col, low_col, high_col)
        forecast_values, lower_bound, upper_bound = self._apply_non_negative(forecast_values, lower_bound, upper_bound)
        result_df = self._build_result_df(forecast_values, lower_bound, upper_bound)
        self.logger.debug(f"Generated {len(result_df)} forecast points")
        model_info = self.get_model_info()
        leaderboard = self.predictor.leaderboard(silent=True)
        best_model_metrics = leaderboard.iloc[0].to_dict() if not leaderboard.empty else {}

        return ForecastResult(
            forecast=result_df,
            model_name=model_info["model_name"],
            model_params=model_info.get("model_params", {}),
            metrics={
                "score": float(best_model_metrics.get("score", 0.0)),
                "eval_metric": self.eval_metric,
                "pred_time": float(best_model_metrics.get("pred_time", 0.0)),
                "fit_time": float(best_model_metrics.get("fit_time", 0.0)),
            },
            metadata={
                "preset": self.preset,
                "models_trained": len(leaderboard) if leaderboard is not None else 1,
                "evaluation_metric": self.eval_metric,
            },
        )

    def _prepare_future_covariates(self) -> Any | None:
        """Prepare known covariates for the forecast horizon."""
        try:
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
            return self._prepare_known_covariates(future_index)
        except Exception:
            return None

    def _normalize_predictions(self, predictions: Any) -> pd.DataFrame:
        """Normalize predictions to a DataFrame with timestamp index for item_id 'stock'."""
        pred_df = predictions.reset_index()
        if "item_id" in pred_df.columns:
            pred_df = pred_df[pred_df["item_id"] == "stock"]
        if "timestamp" in pred_df.columns:
            pred_df = pred_df.set_index("timestamp")
        return pred_df

    def _get_median_col(self, pred_df: pd.DataFrame) -> str:
        """Choose central tendency column."""
        if "mean" in pred_df.columns:
            return "mean"
        if "0.5" in pred_df.columns:
            return "0.5"
        numeric_cols = [c for c in pred_df.columns if pd.api.types.is_numeric_dtype(pred_df[c])]
        return numeric_cols[-1] if numeric_cols else pred_df.columns[-1]

    def _get_quantile_cols(self, pred_df: pd.DataFrame) -> tuple[str | None, str | None]:
        """Determine interval columns closest to requested interval."""
        alpha = (1.0 - float(self.prediction_interval)) / 2.0
        low_target, high_target = alpha, 1.0 - alpha
        qcols = [c for c in pred_df.columns if isinstance(c, str) and c.replace(".", "", 1).isdigit()]

        def _closest(col_target: float) -> str | None:
            if not qcols:
                return None
            import numpy as np

            arr = np.array([float(c) for c in qcols])
            idx = int(np.argmin(np.abs(arr - col_target)))
            return qcols[idx]

        return _closest(low_target), _closest(high_target)

    def _get_forecast_bounds(
        self, pred_df: pd.DataFrame, median_col: str, low_col: str | None, high_col: str | None
    ) -> tuple[Any, Any, Any]:
        """Get forecast, lower, and upper bounds."""
        forecast_values = pred_df[median_col].to_numpy()
        if low_col and high_col and low_col in pred_df.columns and high_col in pred_df.columns:
            lower_bound = pred_df[low_col].to_numpy()
            upper_bound = pred_df[high_col].to_numpy()
        else:
            lower_bound = forecast_values * 0.9
            upper_bound = forecast_values * 1.1
        return forecast_values, lower_bound, upper_bound

    def _apply_non_negative(self, forecast_values: Any, lower_bound: Any, upper_bound: Any) -> tuple[Any, Any, Any]:
        """Apply non-negative constraint if needed."""
        if self.no_negatives:
            forecast_values = forecast_values.clip(min=0)
            lower_bound = lower_bound.clip(min=0)
            upper_bound = upper_bound.clip(min=0)
        return forecast_values, lower_bound, upper_bound

    def _build_result_df(self, forecast_values: Any, lower_bound: Any, upper_bound: Any) -> pd.DataFrame:
        """Create result DataFrame with proper index."""
        import pandas as pd

        last_date = pd.Timestamp.now()
        freq = self.predictor._learner.freq if hasattr(self.predictor, "_learner") else "D"
        future_dates = pd.date_range(start=last_date, periods=len(forecast_values) + 1, freq=freq)[1:]
        return pd.DataFrame(
            {
                "forecast": forecast_values,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            },
            index=future_dates,
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
