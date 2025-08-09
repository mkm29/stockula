"""Modernized Forecasting Manager that uses the backend abstraction."""

from typing import TYPE_CHECKING, Any

import pandas as pd
from dependency_injector.wiring import Provide, inject
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from ..cli_manager import cli_manager
from ..interfaces import ILoggingManager
from .backends import ForecastBackend
from .factory import create_forecast_backend

if TYPE_CHECKING:
    from ..config import ForecastConfig, StockulaConfig
    from ..data.fetcher import DataFetcher


class ForecastingManager:
    """Manages forecasting with swappable backends (AutoTS or AutoGluon).

    This manager provides a unified interface for time series forecasting,
    allowing seamless switching between different forecasting backends through
    configuration. Currently supports:
    - AutoTS: Feature-rich with many models and ensemble methods
    - AutoGluon: Modern AutoML with deep learning models

    The backend can be selected via the 'backend' field in ForecastConfig.
    """

    @inject
    def __init__(
        self,
        data_fetcher: "DataFetcher",
        logging_manager: ILoggingManager = Provide["logging_manager"],
    ):
        """Initialize the forecasting manager.

        Args:
            data_fetcher: Data fetcher for retrieving stock data
            logging_manager: Injected logging manager
        """
        self.data_fetcher = data_fetcher
        self.logger = logging_manager

    def create_backend(self, config: "ForecastConfig") -> ForecastBackend:
        """Create a forecasting backend from configuration.

        Args:
            config: Forecast configuration

        Returns:
            Configured forecasting backend
        """
        return create_forecast_backend(config, self.logger)

    def forecast_symbol(
        self,
        symbol: str,
        config: "StockulaConfig",
        use_evaluation: bool = False,
    ) -> dict[str, Any]:
        """Forecast a single symbol using configuration settings.

        Args:
            symbol: Stock symbol to forecast
            config: Stockula configuration
            use_evaluation: Whether to use train/test evaluation mode (currently ignored, for compatibility)

        Returns:
            Dictionary with forecast results
        """
        # Create backend based on configuration
        backend = self.create_backend(config.forecast)

        # Get historical data
        start_date = self._date_to_string(config.data.start_date)
        end_date = self._date_to_string(config.data.end_date)

        self.logger.info(f"Forecasting {symbol} using {config.forecast.backend} backend...")

        # Fetch data
        data = self.data_fetcher.get_stock_data(symbol, start_date, end_date)

        if data.empty:
            raise ValueError(f"No data available for symbol {symbol}")

        # Fit and predict
        result = backend.fit_predict(data, target_column="Close", show_progress=True)

        # Get model information
        model_info = backend.get_model_info()

        self.logger.info(f"Forecast completed for {symbol} using {model_info['model_name']}")

        # Calculate proper forecast dates
        from datetime import datetime, timedelta

        today = datetime.now().date()
        forecast_start = today + timedelta(days=1)
        forecast_end = forecast_start + timedelta(days=config.forecast.forecast_length - 1)

        return {
            "ticker": symbol,
            "backend": config.forecast.backend,
            "current_price": float(result.forecast["forecast"].iloc[0]),
            "forecast_price": float(result.forecast["forecast"].iloc[-1]),
            "lower_bound": float(result.forecast["lower_bound"].iloc[-1]),
            "upper_bound": float(result.forecast["upper_bound"].iloc[-1]),
            "forecast_length": config.forecast.forecast_length,
            "best_model": model_info["model_name"],
            "model_params": model_info.get("model_params", {}),
            "metrics": result.metrics,
            "start_date": forecast_start.strftime("%Y-%m-%d"),
            "end_date": forecast_end.strftime("%Y-%m-%d"),
        }

    def forecast_multiple_symbols(
        self,
        symbols: list[str],
        config: "StockulaConfig",
    ) -> dict[str, dict[str, Any]]:
        """Forecast multiple symbols.

        Args:
            symbols: List of stock symbols to forecast
            config: Stockula configuration

        Returns:
            Dictionary mapping symbols to their forecast results
        """
        results = {}
        backend_name = config.forecast.backend

        self.logger.info(f"Starting forecast for {len(symbols)} symbols using {backend_name} backend")

        for idx, symbol in enumerate(symbols, 1):
            try:
                self.logger.info(f"Processing {symbol} ({idx}/{len(symbols)})")

                result = self.forecast_symbol(symbol, config)
                results[symbol] = result

            except Exception as e:
                self.logger.error(f"Error forecasting {symbol}: {e}")
                results[symbol] = {
                    "ticker": symbol,
                    "backend": backend_name,
                    "error": str(e),
                }

        return results

    def forecast_multiple_symbols_with_progress(
        self,
        symbols: list[str],
        config: "StockulaConfig",
        console=None,
    ) -> list[dict[str, Any]]:
        """Forecast multiple symbols with progress tracking.

        Args:
            symbols: List of stock symbols to forecast
            config: Stockula configuration
            console: Rich console for progress display

        Returns:
            List of forecast results
        """
        if console is None:
            console = cli_manager.get_console()

        backend_name = config.forecast.backend
        console.print(f"\n[bold blue]Starting forecasting with {backend_name} backend...[/bold blue]")

        if backend_name == "autots":
            console.print(
                f"[dim]Configuration: max_generations={config.forecast.max_generations}, "
                f"num_validations={config.forecast.num_validations}, "
                f"model_list={config.forecast.model_list}[/dim]"
            )
        else:  # autogluon
            console.print(
                f"[dim]Configuration: preset={config.forecast.preset}, time_limit={config.forecast.time_limit}s[/dim]"
            )

        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as forecast_progress:
            forecast_task = forecast_progress.add_task(
                f"[blue]Forecasting {len(symbols)} tickers with {backend_name}...",
                total=len(symbols),
            )

            for idx, symbol in enumerate(symbols, 1):
                forecast_progress.update(
                    forecast_task,
                    description=f"[blue]Forecasting {symbol} ({idx}/{len(symbols)}) with {backend_name}...",
                )

                try:
                    forecast_result = self.forecast_symbol(symbol, config)
                    results.append(forecast_result)

                    forecast_progress.update(
                        forecast_task,
                        description=f"[green]✅ Forecasted {symbol}[/green] ({idx}/{len(symbols)})",
                    )

                except KeyboardInterrupt:
                    self.logger.warning(f"Forecast for {symbol} interrupted by user")
                    results.append({"ticker": symbol, "backend": backend_name, "error": "Interrupted by user"})
                    break
                except Exception as e:
                    self.logger.error(f"Error forecasting {symbol}: {e}")
                    results.append({"ticker": symbol, "backend": backend_name, "error": str(e)})

                    forecast_progress.update(
                        forecast_task,
                        description=f"[red]❌ Failed {symbol}[/red] ({idx}/{len(symbols)})",
                    )

                forecast_progress.advance(forecast_task)

            forecast_progress.update(
                forecast_task,
                description=f"[green]Forecasting complete with {backend_name}!",
            )

        return results

    def compare_backends(
        self,
        symbol: str,
        config: "StockulaConfig",
    ) -> dict[str, dict[str, Any]]:
        """Compare forecasts from different backends for the same symbol.

        Args:
            symbol: Stock symbol to forecast
            config: Stockula configuration

        Returns:
            Dictionary with results from each backend
        """
        results = {}

        # Fetch data once
        start_date = self._date_to_string(config.data.start_date)
        end_date = self._date_to_string(config.data.end_date)
        data = self.data_fetcher.get_stock_data(symbol, start_date, end_date)

        if data.empty:
            raise ValueError(f"No data available for symbol {symbol}")

        # Try AutoTS backend
        try:
            self.logger.info(f"Forecasting {symbol} with AutoTS backend...")
            autots_config = config.forecast.model_copy()
            autots_config.backend = "autots"
            autots_backend = self.create_backend(autots_config)
            autots_result = autots_backend.fit_predict(data, target_column="Close")
            autots_info = autots_backend.get_model_info()

            results["autots"] = {
                "forecast": autots_result.forecast["forecast"].tolist(),
                "lower_bound": autots_result.forecast["lower_bound"].tolist(),
                "upper_bound": autots_result.forecast["upper_bound"].tolist(),
                "model_name": autots_info["model_name"],
                "model_params": autots_info.get("model_params", {}),
            }
        except Exception as e:
            self.logger.error(f"AutoTS backend failed: {e}")
            results["autots"] = {"error": str(e)}

        # Try AutoGluon backend
        try:
            self.logger.info(f"Forecasting {symbol} with AutoGluon backend...")
            autogluon_config = config.forecast.model_copy()
            autogluon_config.backend = "autogluon"
            autogluon_backend = self.create_backend(autogluon_config)
            autogluon_result = autogluon_backend.fit_predict(data, target_column="Close")
            autogluon_info = autogluon_backend.get_model_info()

            results["autogluon"] = {
                "forecast": autogluon_result.forecast["forecast"].tolist(),
                "lower_bound": autogluon_result.forecast["lower_bound"].tolist(),
                "upper_bound": autogluon_result.forecast["upper_bound"].tolist(),
                "model_name": autogluon_info["model_name"],
                "model_params": autogluon_info.get("model_params", {}),
                "metrics": autogluon_result.metrics,
            }
        except Exception as e:
            self.logger.error(f"AutoGluon backend failed: {e}")
            results["autogluon"] = {"error": str(e)}

        return results

    def quick_forecast(
        self,
        symbol: str,
        forecast_days: int = 7,
        historical_days: int = 90,
        backend: str = "autots",
    ) -> dict[str, Any]:
        """Quick forecast using fast presets.

        Args:
            symbol: Stock symbol to forecast
            forecast_days: Number of days to forecast
            historical_days: Number of historical days to use
            backend: Backend to use ('autots' or 'autogluon')

        Returns:
            Dictionary with forecast results
        """
        # Calculate date range
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=historical_days)

        # Fetch data
        data = self.data_fetcher.get_stock_data(
            symbol,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

        if data.empty:
            raise ValueError(f"No data available for symbol {symbol}")

        # Create backend with fast settings
        if backend == "autots":
            from .backends import AutoTSBackend

            backend_instance = AutoTSBackend(
                forecast_length=forecast_days,
                model_list="ultra_fast",
                max_generations=1,
                num_validations=1,
                logging_manager=self.logger,
            )
        else:  # autogluon
            from .backends import AutoGluonBackend

            backend_instance = AutoGluonBackend(
                forecast_length=forecast_days,
                preset="fast_training",
                time_limit=60,  # 1 minute time limit
                eval_metric="MASE",  # Use MASE for scale-independent evaluation
                logging_manager=self.logger,
            )

        # Fit and predict
        result = backend_instance.fit_predict(data, target_column="Close", show_progress=False)
        model_info = backend_instance.get_model_info()

        return {
            "ticker": symbol,
            "backend": backend,
            "current_price": float(result.forecast["forecast"].iloc[0]),
            "forecast_price": float(result.forecast["forecast"].iloc[-1]),
            "lower_bound": float(result.forecast["lower_bound"].iloc[-1]),
            "upper_bound": float(result.forecast["upper_bound"].iloc[-1]),
            "forecast_length": forecast_days,
            "best_model": model_info["model_name"],
            "confidence": "Quick forecast - lower confidence",
        }

    def validate_forecast_config(self, config: "ForecastConfig") -> None:
        """Validate forecast configuration.

        Args:
            config: Forecast configuration to validate

        Raises:
            ValueError: If configuration is invalid
        """
        if config.forecast_length is None or config.forecast_length <= 0:
            raise ValueError("forecast_length must be positive")

        if config.prediction_interval <= 0 or config.prediction_interval >= 1:
            raise ValueError("prediction_interval must be between 0 and 1")

        if config.backend not in ["autots", "autogluon"]:
            raise ValueError(f"Invalid backend: {config.backend}. Must be 'autots' or 'autogluon'")

        # Backend-specific validation
        if config.backend == "autots":
            if config.num_validations < 1:
                raise ValueError("num_validations must be at least 1")
            if config.max_generations < 1:
                raise ValueError("max_generations must be at least 1")
        elif config.backend == "autogluon":
            valid_presets = ["fast_training", "medium_quality", "high_quality", "best_quality"]
            if config.preset not in valid_presets:
                raise ValueError(f"Invalid preset: {config.preset}. Must be one of {valid_presets}")

    @staticmethod
    def _date_to_string(date_value) -> str | None:
        """Convert date to string format.

        Args:
            date_value: Date value (string, datetime, or None)

        Returns:
            String formatted date or None
        """
        if date_value is None:
            return None
        if isinstance(date_value, str):
            return date_value
        return date_value.strftime("%Y-%m-%d")
