"""Display and output handling for Stockula results."""

import json
from typing import Any

from rich.console import Console

from .config.models import StockulaConfig
from .interfaces import IDataFetcher
from .utils import get_console


class ResultsDisplay:
    """Handles display and formatting of results by delegating to specialized services."""

    def __init__(self, console: Console | None = None):
        """Initialize the display handler.

        Args:
            console: Rich console for output (optional)
        """
        from stockula.display.backtest_display import BacktestResultsDisplay
        from stockula.display.forecast_display import ForecastResultsDisplay
        from stockula.display.portfolio_display import PortfolioDisplay
        from stockula.display.technical_analysis_display import TechnicalAnalysisDisplay

        self.console = get_console(console)

        # Initialize specialized display services
        self.technical_analysis_display = TechnicalAnalysisDisplay(self.console)
        self.backtest_display = BacktestResultsDisplay(self.console)
        self.portfolio_display = PortfolioDisplay(self.console)
        self.forecast_display = ForecastResultsDisplay(self.console)

    def print_results(
        self, results: dict[str, Any], output_format: str = "console", config=None, container=None, portfolio=None
    ):
        """Print results in specified format.

        Args:
            results: Results dictionary
            output_format: Output format (console, json)
            config: Optional configuration object for portfolio composition
            container: Optional DI container for fetching data
            portfolio: Optional portfolio instance for forecast display
        """
        if output_format == "json":
            self.console.print_json(json.dumps(results, indent=2, default=str))
        else:
            # Console output with Rich formatting
            if "technical_analysis" in results:
                self.technical_analysis_display.display_technical_analysis(results["technical_analysis"])

            if "backtesting" in results:
                self.backtest_display.display_backtesting_results(results, config, container)

            if "forecasting" in results:
                self.forecast_display.display_forecast_results(results["forecasting"], portfolio)

    # Technical Analysis delegation methods
    def _display_technical_analysis(self, ta_results: list[dict[str, Any]]):
        """Display technical analysis results. Delegates to TechnicalAnalysisDisplay."""
        # Handle list of results by displaying each one
        for result in ta_results:
            self.technical_analysis_display.display_technical_analysis(result)

    # Backtest Results delegation methods
    def _display_backtesting_results(self, results: dict[str, Any], config: StockulaConfig | None, container):
        """Display backtesting results. Delegates to BacktestResultsDisplay."""
        self.backtest_display.display_backtesting_results(results, config, container)

    def _display_portfolio_composition(self, config: StockulaConfig, container):
        """Display portfolio composition table. Delegates to BacktestResultsDisplay."""
        self.backtest_display.display_portfolio_composition(config, container)

    def _display_backtest_ticker_results(self, backtest_results: list[dict[str, Any]]):
        """Display ticker-level backtest results. Delegates to BacktestResultsDisplay."""
        self.backtest_display.display_backtest_ticker_results(backtest_results)

    def _display_train_test_results(self, backtest_results: list[dict[str, Any]]):
        """Display train/test split results. Delegates to BacktestResultsDisplay."""
        self.backtest_display.display_train_test_results(backtest_results)

    def _display_standard_backtest_results(self, backtest_results: list[dict[str, Any]]):
        """Display standard backtest results. Delegates to BacktestResultsDisplay."""
        self.backtest_display.display_standard_backtest_results(backtest_results)

    def _display_strategy_average_returns(self, backtest_results: list[dict[str, Any]]):
        """Display average returns for each strategy. Delegates to BacktestResultsDisplay."""
        self.backtest_display.display_strategy_average_returns(backtest_results)

    def show_strategy_summaries(self, manager, config: StockulaConfig, results: dict[str, Any]):
        """Show strategy-specific summaries. Delegates to BacktestResultsDisplay."""
        self.backtest_display.show_strategy_summaries(manager, config, results)

    # Portfolio delegation methods
    def show_portfolio_summary(self, portfolio) -> None:
        """Display a brief portfolio summary table. Delegates to PortfolioDisplay."""
        self.portfolio_display.show_portfolio_summary(portfolio)

    def show_portfolio_holdings(
        self,
        portfolio,
        mode: str | None = None,
        prices: dict[str, float] | None = None,
        data_fetcher: IDataFetcher | None = None,
    ) -> None:
        """Display detailed portfolio holdings. Delegates to PortfolioDisplay."""
        self.portfolio_display.show_portfolio_holdings(portfolio, mode, prices, data_fetcher)

    def show_allocation_optimization(
        self,
        optimized_quantities: dict[str, float],
        config: StockulaConfig,
        data_fetcher: IDataFetcher,
    ) -> None:
        """Display results of allocation optimization. Delegates to PortfolioDisplay."""
        self.portfolio_display.show_allocation_optimization(optimized_quantities, config, data_fetcher)

    def show_portfolio_forecast_value(self, config: StockulaConfig, portfolio, results: dict[str, Any]):
        """Show portfolio value for forecast mode. Delegates to PortfolioDisplay."""
        self.portfolio_display.show_portfolio_forecast_value(config, portfolio, results)

    # Forecast Results delegation methods
    def _display_forecast_results(self, forecast_results: list[dict[str, Any]], portfolio=None):
        """Display forecasting results. Delegates to ForecastResultsDisplay."""
        self.forecast_display.display_forecast_results(forecast_results, portfolio)

    def _display_evaluation_metrics(self, forecasts: list[dict[str, Any]]):
        """Display forecast evaluation metrics. Delegates to ForecastResultsDisplay."""
        self.forecast_display.display_evaluation_metrics(forecasts)

    def show_forecast_warning(self, config: StockulaConfig):
        """Show forecast mode warning. Delegates to ForecastResultsDisplay."""
        self.forecast_display.show_forecast_warning(config)

    # Legacy methods for backward compatibility - can be removed if not used elsewhere
    def _get_broker_info(self, config: StockulaConfig) -> str:
        """Get broker information string. Delegates to BacktestResultsDisplay."""
        return self.backtest_display._get_broker_info(config)

    def _get_strategy_dates(self, config: StockulaConfig, portfolio_backtest_results) -> tuple[str, str]:
        """Get strategy date range. Delegates to BacktestResultsDisplay."""
        return self.backtest_display._get_strategy_dates(config, portfolio_backtest_results)
