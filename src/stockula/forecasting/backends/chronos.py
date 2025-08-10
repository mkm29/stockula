"""Chronos backend for zero-shot time series forecasting.

This backend uses Amazon's Chronos pretrained models via
`BaseChronosPipeline` to produce probabilistic forecasts without
task-specific training.

References:
  - https://github.com/amazon-science/chronos-forecasting
  - https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .base import ForecastBackend, ForecastResult


class ChronosBackend(ForecastBackend):
    """Chronos zero-shot forecasting backend.

    This backend loads a pretrained Chronos model and generates
    quantile forecasts directly from the historical context.
    """

    DEFAULT_MODEL = "amazon/chronos-bolt-small"

    def __init__(
        self,
        forecast_length: int | None = None,
        frequency: str = "infer",
        prediction_interval: float = 0.9,
        no_negatives: bool = True,
        model_name: str | None = None,
        num_samples: int = 256,
        quantile_levels: Iterable[float] | None = None,
        device_map: str | None = None,
        torch_dtype: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            forecast_length=forecast_length,
            frequency=frequency,
            prediction_interval=prediction_interval,
            no_negatives=no_negatives,
            **kwargs,
        )
        # Lazy import torch to avoid hard dependency at import time
        try:
            import torch  # noqa: F401
        except Exception as _:
            pass

        self.model_name = model_name or self.DEFAULT_MODEL
        self.num_samples = int(num_samples)
        # Compute default quantiles from prediction interval if not provided
        if quantile_levels is None:
            alpha = (1.0 - float(prediction_interval)) / 2.0
            quantile_levels = [alpha, 0.5, 1.0 - alpha]
        self.quantile_levels = list(quantile_levels)
        self.device_map = device_map  # e.g., "auto"
        self.torch_dtype = torch_dtype  # e.g., torch.bfloat16

        self._pipeline = None
        self._context_series: np.ndarray | None = None
        self.is_fitted = False

    def _load_pipeline(self):
        if self._pipeline is None:
            try:
                from chronos import BaseChronosPipeline
            except Exception as e:  # pragma: no cover - import error path
                raise ImportError(
                    "chronos-forecasting not installed or import failed. Install with: pip install chronos-forecasting"
                ) from e

            # Determine sensible defaults for device and dtype
            device_map = self.device_map
            torch_dtype = self.torch_dtype
            try:
                import torch

                if device_map is None:
                    # BaseChronosPipeline expects explicit device string: "cuda" or "cpu"
                    device_map = "cuda" if torch.cuda.is_available() else "cpu"
                if torch_dtype is None:
                    # Prefer bfloat16 where available, otherwise float16; fall back to float32
                    if hasattr(torch, "bfloat16"):
                        torch_dtype = torch.bfloat16
                    elif hasattr(torch, "float16"):
                        torch_dtype = torch.float16
            except Exception:
                # Torch may not be available; fall back to CPU and default dtype
                if device_map is None:
                    device_map = "cpu"
                if torch_dtype is None:
                    torch_dtype = None

            try:
                self._pipeline = BaseChronosPipeline.from_pretrained(
                    self.model_name,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                )
            except Exception as e:  # Wrap with a clearer message
                raise RuntimeError(
                    f"Failed to load Chronos model '{self.model_name}'. Ensure PyTorch is installed and compatible."
                ) from e

    def _prepare_context(self, data: pd.DataFrame, target_column: str) -> np.ndarray:
        # Extract numeric series and cast to float32 for Chronos
        y = data[target_column].astype(float).to_numpy(dtype=np.float32)
        # Guard against NaNs and infs
        if np.isnan(y).any() or np.isinf(y).any():
            raise ValueError("Target series contains NaN or infinite values")
        from typing import cast

        return cast(np.ndarray, y)

    def fit(
        self,
        data: pd.DataFrame,
        target_column: str = "Close",
        show_progress: bool = True,
        **kwargs: Any,
    ) -> ChronosBackend:
        # Validate inputs and set forecast length default
        self.validate_input(data, target_column)
        if self.forecast_length is None:
            self.forecast_length = 14

        # Load pipeline lazily
        self._load_pipeline()

        # Store context series
        self._context_series = self._prepare_context(data, target_column)
        self._last_timestamp = data.index.max()

        self.is_fitted = True
        return self

    def predict(self, **kwargs: Any) -> ForecastResult:
        if not self.is_fitted or self._pipeline is None or self._context_series is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Generate probabilistic samples then compute requested quantiles
        samples = self._pipeline.predict(
            context=self._context_series,
            prediction_length=int(self.forecast_length or 1),
            num_samples=self.num_samples,
        )

        # Expect samples shape: (num_samples, prediction_length)
        if samples.ndim != 2:
            # Some versions may return (prediction_length,) if num_samples=1; normalize
            samples = np.atleast_2d(samples)

        # Compute quantiles across samples axis
        q_values = {q: np.quantile(samples, q=q, axis=0).astype(np.float64) for q in self.quantile_levels}

        # Choose lower/median/upper from quantiles
        # Ensure 0.5 exists; if not, infer closest
        def _closest_quantile(qs: list[float], target: float) -> float:
            arr = np.asarray(qs)
            idx = int(np.argmin(np.abs(arr - target)))
            return float(arr[idx])

        qs = list(q_values.keys())
        q_med = 0.5 if 0.5 in q_values else _closest_quantile(qs, 0.5)
        # Lower/upper around the requested interval
        alpha = (1.0 - float(self.prediction_interval)) / 2.0
        q_low = min(qs, key=lambda x: abs(x - alpha))
        q_high = min(qs, key=lambda x: abs(x - (1.0 - alpha)))

        forecast_values = q_values[q_med]
        lower_bound = q_values[q_low]
        upper_bound = q_values[q_high]

        if self.no_negatives:
            forecast_values = np.clip(forecast_values, a_min=0, a_max=None)
            lower_bound = np.clip(lower_bound, a_min=0, a_max=None)
            upper_bound = np.clip(upper_bound, a_min=0, a_max=None)

        # Build future date index
        try:
            inferred_freq = None
            if self.frequency == "infer":
                # A best-effort inference — the manager often provides daily data
                inferred_freq = pd.infer_freq(
                    pd.Index([self._last_timestamp, self._last_timestamp + pd.Timedelta(days=1)])
                )
            freq = self.frequency if self.frequency != "infer" else (inferred_freq or "D")
        except Exception:
            freq = "D"

        start = pd.Timestamp(self._last_timestamp) + pd.tseries.frequencies.to_offset(freq)
        future_index = pd.date_range(start=start, periods=len(forecast_values), freq=freq)

        result_df = pd.DataFrame(
            {
                "forecast": forecast_values,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            },
            index=future_index,
        )

        return ForecastResult(
            forecast=result_df,
            model_name=self.model_name,
            model_params={
                "num_samples": self.num_samples,
                "quantile_levels": self.quantile_levels,
                "device_map": self.device_map or "auto",
            },
            metrics=None,
            metadata={"backend": "chronos"},
        )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_params": {
                "num_samples": self.num_samples,
                "quantile_levels": self.quantile_levels,
                "device_map": self.device_map or "auto",
            },
        }

    def get_available_models(self) -> list[str]:
        # Common public Chronos model repos
        return [
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
