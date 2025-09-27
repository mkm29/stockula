"""
Technical Analysis Orchestrator following SRP.
Single Responsibility: Orchestrating technical analysis workflows.
"""

import logging
from typing import Any, Dict, List, Optional, cast

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from ..config import StockulaConfig
from ..container import Container
from ..technical_analysis import TechnicalIndicators
from ..utils import get_console

logger = logging.getLogger(__name__)


class AnalysisOrchestrator:
    """Orchestrates technical analysis workflows - Single Responsibility: TA Orchestration."""

    def __init__(
        self,
        config: StockulaConfig,
        container: Container,
        console: Optional[Console] = None,
    ):
        """Initialize analysis orchestrator.

        Args:
            config: Configuration object
            container: Dependency injection container
            console: Rich console for output (optional)
        """
        self.config = config
        self.container = container
        self.console = get_console(console)
        self.log_manager = container.logging_manager()

    def run_technical_analysis(
        self,
        ticker: str,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """Run technical analysis for a ticker.

        Args:
            ticker: Stock symbol
            show_progress: Whether to show progress bars

        Returns:
            Dictionary with indicator results
        """
        # Get the technical analysis manager
        ta_manager = self.container.technical_analysis_manager()

        # Determine which indicators to use based on configuration
        ta_config = self.config.technical_analysis
        custom_indicators = self._build_indicators_list(ta_config)

        # Use the manager to analyze the symbol
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=self.console,
                transient=True,
            ) as progress:
                task = progress.add_task(
                    f"[cyan]Analyzing technical indicators for {ticker}...",
                    total=1,
                )

                result = ta_manager.analyze_symbol(
                    ticker,
                    self.config,
                    analysis_type="custom" if custom_indicators else "comprehensive",
                    custom_indicators=custom_indicators if custom_indicators else None,
                )

                progress.advance(task)
        else:
            result = ta_manager.analyze_symbol(
                ticker,
                self.config,
                analysis_type="custom" if custom_indicators else "comprehensive",
                custom_indicators=custom_indicators if custom_indicators else None,
            )

        # Enhance results with backward compatibility data
        self._enhance_results_for_compatibility(result, ticker, ta_config)

        return cast(Dict[str, Any], result)

    def _build_indicators_list(self, ta_config) -> List[str]:
        """Build custom indicators list based on config.

        Args:
            ta_config: Technical analysis configuration

        Returns:
            List of indicator names
        """
        custom_indicators = []

        indicator_mapping = {
            "sma": "sma",
            "ema": "ema",
            "rsi": "rsi",
            "macd": "macd",
            "bbands": "bbands",
            "atr": "atr",
            "adx": "adx",
            "stoch": "stoch",
            "williams_r": "williams_r",
            "cci": "cci",
            "obv": "obv",
            "ichimoku": "ichimoku",
        }

        for config_indicator, standard_name in indicator_mapping.items():
            if config_indicator in ta_config.indicators:
                custom_indicators.append(standard_name)

        return custom_indicators

    def _enhance_results_for_compatibility(self, result: Dict[str, Any], ticker: str, ta_config) -> None:
        """Enhance results with backward compatibility data.

        Args:
            result: Analysis result to enhance
            ticker: Stock symbol
            ta_config: Technical analysis configuration
        """
        # If we need to maintain backward compatibility with the old format
        if "indicators" in result and not result.get("error"):
            # Get the data for period-specific calculations
            data_fetcher = self.container.data_fetcher()
            data = data_fetcher.get_stock_data(
                ticker,
                start=self._date_to_string(self.config.data.start_date),
                end=self._date_to_string(self.config.data.end_date),
                interval=self.config.data.interval,
            )

            if not data.empty:
                ta = TechnicalIndicators(data)
                self._add_period_specific_calculations(result, ta, ta_config)

    def _add_period_specific_calculations(self, result: Dict[str, Any], ta: TechnicalIndicators, ta_config) -> None:
        """Add period-specific calculations for backward compatibility.

        Args:
            result: Result dictionary to enhance
            ta: TechnicalIndicators instance
            ta_config: Technical analysis configuration
        """
        indicators = result["indicators"]

        # Add period-specific calculations if needed
        if "sma" in ta_config.indicators and "sma" in indicators:
            for period in ta_config.sma_periods:
                indicators[f"SMA_{period}"] = ta.sma(period).iloc[-1]

        if "ema" in ta_config.indicators and "ema" in indicators:
            for period in ta_config.ema_periods:
                indicators[f"EMA_{period}"] = ta.ema(period).iloc[-1]

        # Add simple indicator values for backward compatibility
        if "rsi" in ta_config.indicators and "rsi" in indicators:
            indicators["RSI"] = indicators["rsi"]["current"]

        if "macd" in ta_config.indicators and "macd" in indicators:
            macd_data = indicators["macd"]["current"]
            if isinstance(macd_data, dict):
                indicators["MACD"] = macd_data.get("MACD")

        if "bbands" in ta_config.indicators and "bbands" in indicators:
            indicators["BBands"] = indicators["bbands"]["current"]

        if "atr" in ta_config.indicators and "atr" in indicators:
            indicators["ATR"] = indicators["atr"]["current"]

        if "adx" in ta_config.indicators and "adx" in indicators:
            indicators["ADX"] = indicators["adx"]["current"]

    def _date_to_string(self, date_value) -> Optional[str]:
        """Convert date or string to string format.

        Args:
            date_value: Date value to convert

        Returns:
            String representation or None
        """
        if date_value is None:
            return None
        if isinstance(date_value, str):
            return date_value
        return str(date_value.strftime("%Y-%m-%d"))

    def run_multiple_analysis(self, tickers: List[str], show_progress: bool = True) -> List[Dict[str, Any]]:
        """Run technical analysis for multiple tickers.

        Args:
            tickers: List of stock symbols
            show_progress: Whether to show progress bars

        Returns:
            List of analysis results
        """
        results = []

        if show_progress and len(tickers) > 1:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=self.console,
            ) as progress:
                task = progress.add_task(
                    f"[cyan]Analyzing {len(tickers)} symbols...",
                    total=len(tickers),
                )

                for ticker in tickers:
                    result = self.run_technical_analysis(ticker, show_progress=False)
                    results.append(result)
                    progress.advance(task)
        else:
            for ticker in tickers:
                result = self.run_technical_analysis(ticker, show_progress)
                results.append(result)

        return results

    def _compute_indicators(self, ta_instance, ta_config, results, progress, task, ticker: str) -> None:
        """Compute technical indicators for a ticker.

        Args:
            ta_instance: TechnicalIndicators instance
            ta_config: Technical analysis configuration
            results: Results dictionary to update
            progress: Progress tracker instance
            task: Progress task
            ticker: Stock symbol
        """
        try:
            # Compute indicators based on config
            indicators = {}

            # SMA indicators
            if hasattr(ta_config, "sma_periods") and ta_config.sma_periods:
                for period in ta_config.sma_periods:
                    sma_data = ta_instance.calculate_sma(period)
                    if sma_data is not None and not sma_data.empty:
                        indicators[f"SMA_{period}"] = sma_data.iloc[-1] if len(sma_data) > 0 else None

            # RSI indicator
            if hasattr(ta_config, "rsi_period") and ta_config.rsi_period:
                rsi_data = ta_instance.calculate_rsi(ta_config.rsi_period)
                if rsi_data is not None and not rsi_data.empty:
                    indicators["RSI"] = rsi_data.iloc[-1] if len(rsi_data) > 0 else None

            # Update results
            if "indicators" not in results:
                results["indicators"] = {}
            results["indicators"].update(indicators)

        except Exception as e:
            logger.error(f"Error computing indicators for {ticker}: {e}")
            # Don't re-raise to avoid breaking the workflow
