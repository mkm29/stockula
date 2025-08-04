"""Forecasting Manager that coordinates different forecasting strategies and models."""

from typing import TYPE_CHECKING, Any

import pandas as pd
from dependency_injector.wiring import Provide, inject

from ..interfaces import ILoggingManager
from .forecaster import StockForecaster

if TYPE_CHECKING:
    from ..config import StockulaConfig
    from ..data.fetcher import DataFetcher


class ForecastingManager:
    """Manages different forecasting strategies and provides unified interface.

    The ForecastingManager coordinates between different forecasting approaches:
    - Standard forecasting with various AutoTS models and automatic validation
    - Fast forecasting for quick results using minimal models
    - Clean forecasting with curated models that avoid warnings (recommended)
    - Financial-specific models optimized for stock data

    AutoTS automatically handles train/validation splits internally based on the
    num_validations and validation_method configuration parameters. The 'clean'
    model list is recommended for production use as it avoids problematic models
    that generate warnings (Motif, Cassandra, deprecated metrics).
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

        # Initialize forecasters with different presets
        self.standard_forecaster = StockForecaster(data_fetcher=data_fetcher, logging_manager=logging_manager)

        self.fast_forecaster = StockForecaster(
            model_list="ultra_fast",
            max_generations=2,
            num_validations=1,
            data_fetcher=data_fetcher,
            logging_manager=logging_manager,
        )

        self.financial_forecaster = StockForecaster(
            model_list="financial",
            ensemble="distance",
            num_validations=3,
            data_fetcher=data_fetcher,
            logging_manager=logging_manager,
        )

    def get_forecaster(self, model_list: str = "fast") -> StockForecaster:
        """Get the appropriate forecaster based on model list.

        Args:
            model_list: Model list preset name

        Returns:
            Appropriate StockForecaster instance
        """
        if model_list in ["ultra_fast", "superfast"]:
            return self.fast_forecaster
        elif model_list in ["financial", "fast_financial"]:
            return self.financial_forecaster
        else:
            return self.standard_forecaster

    def forecast_symbol(
        self,
        symbol: str,
        config: "StockulaConfig",
    ) -> dict[str, Any]:
        """Forecast a single symbol using configuration settings.

        Args:
            symbol: Stock symbol to forecast
            config: Stockula configuration

        Returns:
            Dictionary with forecast results
        """
        # Create a custom forecaster with config settings
        forecaster = StockForecaster(
            forecast_length=config.forecast.forecast_length,
            frequency=config.forecast.frequency,
            prediction_interval=config.forecast.prediction_interval,
            ensemble=config.forecast.ensemble,
            num_validations=config.forecast.num_validations,
            validation_method=config.forecast.validation_method,
            model_list=config.forecast.model_list,
            max_generations=config.forecast.max_generations,
            no_negatives=config.forecast.no_negatives,
            data_fetcher=self.data_fetcher,
            logging_manager=self.logger,
        )

        # Use standard forecasting with AutoTS automatic validation
        return self._standard_forecast(symbol, config, forecaster)

    def _standard_forecast(
        self,
        symbol: str,
        config: "StockulaConfig",
        forecaster: StockForecaster,
    ) -> dict[str, Any]:
        """Perform standard forecasting without evaluation.

        Args:
            symbol: Stock symbol to forecast
            config: Stockula configuration
            forecaster: Configured forecaster instance

        Returns:
            Dictionary with forecast results
        """
        self.logger.info(f"Forecasting {symbol} for {config.forecast.forecast_length} days...")

        # Get historical data
        start_date = self._date_to_string(config.data.start_date)
        end_date = self._date_to_string(config.data.end_date)

        # Run forecast
        predictions = forecaster.forecast_from_symbol(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            target_column="Close",
        )

        # Get model information
        model_info = forecaster.get_best_model()

        self.logger.info(f"Forecast completed for {symbol} using {model_info['model_name']}")

        # Calculate proper forecast dates (from tomorrow for forecast_length days)
        from datetime import datetime, timedelta

        today = datetime.now().date()
        forecast_start = today + timedelta(days=1)  # Start forecasting from tomorrow
        forecast_end = forecast_start + timedelta(days=config.forecast.forecast_length - 1)

        start_date = forecast_start.strftime("%Y-%m-%d")
        end_date = forecast_end.strftime("%Y-%m-%d")

        return {
            "ticker": symbol,
            "current_price": float(predictions["forecast"].iloc[0]),
            "forecast_price": float(predictions["forecast"].iloc[-1]),
            "lower_bound": float(predictions["lower_bound"].iloc[-1]),
            "upper_bound": float(predictions["upper_bound"].iloc[-1]),
            "forecast_length": config.forecast.forecast_length,
            "best_model": model_info["model_name"],
            "model_params": model_info.get("model_params", {}),
            "start_date": start_date,
            "end_date": end_date,
        }

    def forecast_multiple_symbols(
        self,
        symbols: list[str],
        config: "StockulaConfig",
        parallel: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Forecast multiple symbols.

        Args:
            symbols: List of stock symbols to forecast
            config: Stockula configuration
            parallel: Whether to run forecasts in parallel (not implemented)

        Returns:
            Dictionary mapping symbols to their forecast results
        """
        results = {}

        self.logger.info(f"Starting forecast for {len(symbols)} symbols")

        for idx, symbol in enumerate(symbols, 1):
            try:
                self.logger.info(f"Processing {symbol} ({idx}/{len(symbols)})")

                result = self.forecast_symbol(symbol, config)
                results[symbol] = result

            except Exception as e:
                self.logger.error(f"Error forecasting {symbol}: {e}")
                results[symbol] = {
                    "ticker": symbol,
                    "error": str(e),
                }

        return results

    def quick_forecast(
        self,
        symbol: str,
        forecast_days: int = 7,
        historical_days: int = 90,
    ) -> dict[str, Any]:
        """Quick forecast using ultra-fast models.

        Args:
            symbol: Stock symbol to forecast
            forecast_days: Number of days to forecast
            historical_days: Number of historical days to use

        Returns:
            Dictionary with forecast results
        """
        forecaster = self.fast_forecaster

        # Calculate date range

        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=historical_days)

        # Configure for quick forecast
        forecaster.forecast_length = forecast_days

        # Run forecast
        predictions = forecaster.forecast_from_symbol(
            symbol=symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            target_column="Close",
            show_progress=False,
        )

        # Get model info
        model_info = forecaster.get_best_model()

        return {
            "ticker": symbol,
            "current_price": float(predictions["forecast"].iloc[0]),
            "forecast_price": float(predictions["forecast"].iloc[-1]),
            "lower_bound": float(predictions["lower_bound"].iloc[-1]),
            "upper_bound": float(predictions["upper_bound"].iloc[-1]),
            "forecast_length": forecast_days,
            "best_model": model_info["model_name"],
            "confidence": "Quick forecast - lower confidence",
        }

    def financial_forecast(
        self,
        symbol: str,
        config: "StockulaConfig",
    ) -> dict[str, Any]:
        """Forecast using financial-specific models.

        Args:
            symbol: Stock symbol to forecast
            config: Stockula configuration

        Returns:
            Dictionary with forecast results
        """
        # Override config to use financial models
        forecaster = StockForecaster(
            forecast_length=config.forecast.forecast_length,
            frequency=config.forecast.frequency,
            prediction_interval=config.forecast.prediction_interval,
            ensemble="distance",  # Better for financial data
            num_validations=3,  # More validation for robustness
            validation_method=config.forecast.validation_method,
            model_list="financial",  # Use financial-specific models
            max_generations=config.forecast.max_generations,
            no_negatives=True,  # Stock prices can't be negative
            data_fetcher=self.data_fetcher,
            logging_manager=self.logger,
        )

        self.logger.info(f"Using financial-specific models for {symbol}")

        # Get historical data
        start_date = self._date_to_string(config.data.start_date)
        end_date = self._date_to_string(config.data.end_date)

        # Run forecast
        predictions = forecaster.forecast_from_symbol(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            target_column="Close",
        )

        # Get model information
        model_info = forecaster.get_best_model()

        self.logger.info(f"Financial forecast completed for {symbol} using {model_info['model_name']}")

        return {
            "ticker": symbol,
            "current_price": float(predictions["forecast"].iloc[0]),
            "forecast_price": float(predictions["forecast"].iloc[-1]),
            "lower_bound": float(predictions["lower_bound"].iloc[-1]),
            "upper_bound": float(predictions["upper_bound"].iloc[-1]),
            "forecast_length": config.forecast.forecast_length,
            "best_model": model_info["model_name"],
            "model_params": model_info.get("model_params", {}),
            "model_type": "financial",
        }

    def get_available_models(self) -> dict[str, list[str]]:
        """Get available model lists and their descriptions.

        Returns:
            Dictionary of model list names and their models
        """
        return {
            "ultra_fast": StockForecaster.ULTRA_FAST_MODEL_LIST,
            "fast": StockForecaster.FAST_MODEL_LIST,
            "clean": StockForecaster.CLEAN_MODEL_LIST,
            "financial": StockForecaster.FINANCIAL_MODEL_LIST,
            "fast_financial": [m for m in StockForecaster.FAST_MODEL_LIST if m in StockForecaster.FINANCIAL_MODEL_LIST],
        }

    def validate_forecast_config(self, config: "StockulaConfig") -> None:
        """Validate forecast configuration.

        Args:
            config: Stockula configuration to validate

        Raises:
            ValueError: If configuration is invalid
        """
        if config.forecast.forecast_length is None or config.forecast.forecast_length <= 0:
            raise ValueError("forecast_length must be positive")

        if config.forecast.prediction_interval <= 0 or config.forecast.prediction_interval >= 1:
            raise ValueError("prediction_interval must be between 0 and 1")

        if config.forecast.num_validations < 1:
            raise ValueError("num_validations must be at least 1")

        if config.forecast.max_generations < 1:
            raise ValueError("max_generations must be at least 1")

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
        from rich.console import Console
        from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

        if console is None:
            console = Console()

        console.print("\n[bold blue]Starting sequential forecasting...[/bold blue]")
        console.print(
            f"[dim]Configuration: max_generations={config.forecast.max_generations}, "
            f"num_validations={config.forecast.num_validations}[/dim]"
        )

        results = []

        # Create a separate progress display for sequential forecasting
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
                f"[blue]Forecasting {len(symbols)} tickers...",
                total=len(symbols),
            )

            # Process each ticker sequentially
            for idx, symbol in enumerate(symbols, 1):
                forecast_progress.update(
                    forecast_task,
                    description=f"[blue]Forecasting {symbol} ({idx}/{len(symbols)})...",
                )

                try:
                    # Use standard forecasting method with AutoTS automatic validation
                    forecast_result = self.forecast_symbol(symbol, config)
                    results.append(forecast_result)

                    # Update progress to show completion
                    forecast_progress.update(
                        forecast_task,
                        description=f"[green]✅ Forecasted {symbol}[/green] ({idx}/{len(symbols)})",
                    )

                except KeyboardInterrupt:
                    self.logger.warning(f"Forecast for {symbol} interrupted by user")
                    results.append({"ticker": symbol, "error": "Interrupted by user"})
                    break
                except Exception as e:
                    self.logger.error(f"Error forecasting {symbol}: {e}")
                    results.append({"ticker": symbol, "error": str(e)})

                    # Update progress to show error
                    forecast_progress.update(
                        forecast_task,
                        description=f"[red]❌ Failed {symbol}[/red] ({idx}/{len(symbols)})",
                    )

                # Advance progress
                forecast_progress.advance(forecast_task)

            # Mark progress as complete
            forecast_progress.update(
                forecast_task,
                description="[green]Forecasting complete!",
            )

        return results

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
