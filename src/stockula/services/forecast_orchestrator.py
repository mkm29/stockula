"""
Forecast Orchestrator following SRP.
Single Responsibility: Orchestrating forecasting workflows.
"""

import logging
from typing import Any, Dict, List, Optional, cast

from rich.console import Console

from ..config import StockulaConfig
from ..container import Container
from ..utils import get_console

logger = logging.getLogger(__name__)


class ForecastOrchestrator:
    """Orchestrates forecasting workflows - Single Responsibility: Forecast Orchestration."""

    def __init__(
        self,
        config: StockulaConfig,
        container: Container,
        console: Optional[Console] = None,
    ):
        """Initialize forecast orchestrator.

        Args:
            config: Configuration object
            container: Dependency injection container
            console: Rich console for output (optional)
        """
        self.config = config
        self.container = container
        self.console = get_console(console)
        self.log_manager = container.logging_manager()

    def run_forecast_with_evaluation(self, ticker: str) -> Dict[str, Any]:
        """Run forecasting with train/test split and evaluation.

        Args:
            ticker: Stock symbol

        Returns:
            Dictionary with forecast results and evaluation metrics
        """
        self.log_manager.info(f"\nForecasting {ticker} with train/test evaluation...")

        forecasting_manager = self.container.forecasting_manager()

        try:
            # Determine if we should use evaluation
            use_evaluation = self._should_use_evaluation()

            result = forecasting_manager.forecast_symbol(
                ticker,
                self.config,
                use_evaluation=use_evaluation,
            )

            # Add additional info if evaluation was used
            if use_evaluation and "evaluation" in result:
                self._log_evaluation_metrics(ticker, result["evaluation"])

            return cast(Dict[str, Any], result)

        except KeyboardInterrupt:
            self.log_manager.warning(f"Forecast for {ticker} interrupted by user")
            return {"ticker": ticker, "error": "Interrupted by user"}
        except Exception as e:
            self.log_manager.error(f"Error forecasting {ticker}: {e}")
            return {"ticker": ticker, "error": str(e)}

    def run_forecast(self, ticker: str) -> Dict[str, Any]:
        """Run forecasting for a ticker.

        Args:
            ticker: Stock symbol

        Returns:
            Dictionary with forecast results
        """
        self.log_manager.info(f"\nForecasting {ticker} for {self.config.forecast.forecast_length} days...")

        forecasting_manager = self.container.forecasting_manager()

        try:
            result = forecasting_manager.forecast_symbol(
                ticker,
                self.config,
                use_evaluation=False,  # Explicit no evaluation for standard forecast
            )

            return cast(Dict[str, Any], result)
        except KeyboardInterrupt:
            self.log_manager.warning(f"Forecast for {ticker} interrupted by user")
            return {"ticker": ticker, "error": "Interrupted by user"}
        except Exception as e:
            self.log_manager.error(f"Error forecasting {ticker}: {e}")
            return {"ticker": ticker, "error": str(e)}

    def run_multiple_forecasts(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """Run forecasting for multiple tickers with progress tracking.

        Args:
            tickers: List of stock symbols

        Returns:
            List of forecast results
        """
        if not tickers:
            return []

        forecasting_manager = self.container.forecasting_manager()
        results = forecasting_manager.forecast_multiple_symbols_with_progress(tickers, self.config, self.console)
        return list(results) if results is not None else []

    def _should_use_evaluation(self) -> bool:
        """Check if evaluation should be used.

        Returns:
            True if evaluation should be used
        """
        return (
            self.config.forecast.train_start_date is not None
            and self.config.forecast.train_end_date is not None
            and self.config.forecast.test_start_date is not None
            and self.config.forecast.test_end_date is not None
        )

    def _log_evaluation_metrics(self, ticker: str, eval_metrics: Dict[str, Any]) -> None:
        """Log evaluation metrics for a ticker.

        Args:
            ticker: Stock symbol
            eval_metrics: Evaluation metrics dictionary
        """
        # Log MASE if available, otherwise log MAPE
        if "mase" in eval_metrics:
            self.log_manager.info(
                f"Evaluation metrics for {ticker}: RMSE={eval_metrics['rmse']:.2f}, MASE={eval_metrics['mase']:.3f}"
            )
        else:
            # Fallback to MAPE for backward compatibility
            self.log_manager.info(
                f"Evaluation metrics for {ticker}: RMSE={eval_metrics['rmse']:.2f}, "
                f"MAPE={eval_metrics.get('mape', 0):.2f}%"
            )
