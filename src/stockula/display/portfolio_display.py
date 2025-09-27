"""Portfolio display service for Single Responsibility Principle compliance."""

from typing import Any

from rich.console import Console
from rich.table import Table

from stockula.config.models import StockulaConfig
from stockula.interfaces import IDataFetcher
from stockula.utils.console_factory import get_console


class PortfolioDisplay:
    """Handles display and formatting of portfolio-related information only."""

    def __init__(self, console: Console | None = None):
        """Initialize the display handler.

        Args:
            console: Rich console for output (optional)
        """
        self.console = get_console(console)

    def show_portfolio_summary(self, portfolio) -> None:
        """Display a brief portfolio summary table."""
        from stockula.domain.models import Asset

        table = Table(title="Portfolio Summary")
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")

        table.add_row("Name", getattr(portfolio, "name", "Portfolio"))
        initial_capital = getattr(portfolio, "initial_capital", None)
        if isinstance(initial_capital, int | float):
            table.add_row("Initial Capital", f"${initial_capital:,.2f}")
        assets: list[Asset] = getattr(portfolio, "get_all_assets", lambda: [])()
        table.add_row("Total Assets", str(len(assets)))
        allocation_method = getattr(portfolio, "allocation_method", "unknown")
        table.add_row("Allocation Method", str(allocation_method))

        self.console.print(table)

    def show_portfolio_holdings(
        self,
        portfolio,
        mode: str | None = None,
        prices: dict[str, float] | None = None,
        data_fetcher: IDataFetcher | None = None,
    ) -> None:
        """Display detailed portfolio holdings, optionally with prices/values."""
        holdings_table = Table(title="Portfolio Holdings")
        holdings_table.add_column("Ticker", style="cyan", no_wrap=True)
        holdings_table.add_column("Type", style="yellow")
        holdings_table.add_column("Quantity", style="green", justify="right")

        # Add price and value columns for forecast mode
        if mode == "forecast":
            holdings_table.add_column("Price", style="white", justify="right")
            holdings_table.add_column("Value", style="blue", justify="right")

            # Fetch current prices if not provided
            all_assets = portfolio.get_all_assets()
            symbols = [asset.symbol for asset in all_assets]
            try:
                if prices is None and data_fetcher is not None:
                    prices = data_fetcher.get_current_prices(symbols, show_progress=False)
            except Exception:
                prices = {}
        else:
            all_assets = portfolio.get_all_assets()
            prices = prices or {}

        for asset in all_assets:
            symbol = getattr(asset, "symbol", "N/A")
            category_name = (
                str(asset.category.name)
                if getattr(asset, "category", None) is not None and hasattr(asset.category, "name")
                else str(getattr(asset, "category", "N/A"))
            )

            quantity_str = "N/A"
            quantity_val = 0.0
            if hasattr(asset, "quantity") and isinstance(asset.quantity, int | float):
                quantity_val = float(asset.quantity)
                quantity_str = f"{quantity_val:.2f}"
            elif hasattr(asset, "quantity"):
                try:
                    quantity_val = float(asset.quantity)
                    quantity_str = f"{quantity_val:.2f}"
                except (TypeError, ValueError):
                    quantity_str = str(asset.quantity)

            if mode == "forecast":
                price = (prices or {}).get(symbol, 0.0)
                value = quantity_val * price
                price_str = f"${price:.2f}" if price > 0 else "N/A"
                value_str = f"${value:,.2f}" if value > 0 else "N/A"
                holdings_table.add_row(symbol, category_name, quantity_str, price_str, value_str)
            else:
                holdings_table.add_row(symbol, category_name, quantity_str)

        self.console.print(holdings_table)

    def show_allocation_optimization(
        self,
        optimized_quantities: dict[str, float],
        config: StockulaConfig,
        data_fetcher: IDataFetcher,
    ) -> None:
        """Display results of allocation optimization with allocation percentages."""
        self.console.print("\n[bold green]Optimization Results:[/bold green]")

        results_table = Table(title="Optimized Allocation")
        results_table.add_column("Ticker", style="cyan", no_wrap=True)
        results_table.add_column("Quantity", style="green", justify="right")
        results_table.add_column("Allocation %", style="yellow", justify="right")

        # Calculate total value for percentage
        total_value = 0.0
        ticker_values: dict[str, float] = {}
        symbols_to_price = [t.symbol for t in config.portfolio.tickers if t.symbol in optimized_quantities]

        prices = data_fetcher.get_current_prices(symbols_to_price, show_progress=False)
        for symbol in symbols_to_price:
            if symbol in prices:
                value = float(optimized_quantities[symbol]) * float(prices[symbol])
                ticker_values[symbol] = value
                total_value += value

        # Display rows
        for ticker_config in config.portfolio.tickers:
            symbol = ticker_config.symbol
            quantity = optimized_quantities.get(symbol, 0)

            allocation_pct = (ticker_values.get(symbol, 0.0) / total_value * 100) if total_value > 0 else 0.0

            qty_str = f"{quantity:.4f}" if config.portfolio.allow_fractional_shares else f"{int(quantity)}"
            results_table.add_row(symbol, qty_str, f"{allocation_pct:.2f}%")

        self.console.print(results_table)

    def show_portfolio_forecast_value(self, config: StockulaConfig, portfolio, results: dict[str, Any]):
        """Show portfolio value for forecast mode with consistent price calculations.

        This method calculates portfolio values using consistent price baselines to ensure
        accurate portfolio return calculations. The current value is calculated using the
        same historical prices that the forecasting algorithm uses as its baseline, rather
        than real-time market prices, to maintain consistency between current and forecast
        values.

        Args:
            config: Configuration object
            portfolio: Portfolio instance
            results: Results dictionary containing forecasting results

        Note:
            The current portfolio value is calculated using the forecast algorithm's
            "current price" baseline to ensure the portfolio return percentage accurately
            reflects the forecasted price changes.
        """
        from datetime import date, datetime

        # Show portfolio value in a nice table
        portfolio_value_table = Table(title="Portfolio Value")
        portfolio_value_table.add_column("Metric", style="cyan", no_wrap=True, width=18)
        portfolio_value_table.add_column("Date", style="white", no_wrap=True, width=25)
        portfolio_value_table.add_column("Value", style="green", no_wrap=True, width=12)

        # Add initial capital row with appropriate date
        if config.forecast.test_start_date:
            # Historical evaluation mode - use test start date
            test_start = (
                config.forecast.test_start_date.strftime("%Y-%m-%d")
                if isinstance(config.forecast.test_start_date, date)
                else str(config.forecast.test_start_date)
            )
        else:
            # Future prediction mode - use today's date
            test_start = datetime.now().strftime("%Y-%m-%d")

        # For forecast mode, calculate current value based on the prices used in forecasting
        # This ensures consistency between current and forecast values
        #
        # Important: The forecasting algorithm uses the last price in the historical training
        # data as its "current price" baseline, which may differ from real-time market prices.
        # To ensure accurate portfolio return calculations, we use the same price baseline
        # for both current and forecast values. This prevents misleading return percentages
        # that could occur if different price sources were used.
        current_portfolio_value = 0.0
        if "forecasting" in results and results["forecasting"]:
            # Calculate current value using the same prices that forecasting uses
            for forecast in results["forecasting"]:
                if "error" not in forecast:
                    ticker = forecast["ticker"]
                    asset = next(
                        (a for a in portfolio.get_all_assets() if a.symbol == ticker),
                        None,
                    )
                    if asset and asset.quantity:
                        current_portfolio_value += asset.quantity * forecast["current_price"]

        # Fallback to initial capital if no forecast data available
        if current_portfolio_value == 0.0:
            current_portfolio_value = portfolio.initial_capital

        # Show initial capital and current value
        portfolio_value_table.add_row("Initial Capital", "Start", f"${portfolio.initial_capital:,.2f}")
        portfolio_value_table.add_row("Current Value", test_start, f"${current_portfolio_value:,.2f}")

        # Calculate forecasted portfolio value based on forecast results
        if "forecasting" in results and results["forecasting"]:
            forecasted_value = 0.0
            total_accuracy = 0
            valid_forecasts = 0

            # Check if we're in evaluation mode (have evaluation metrics)
            is_evaluation_mode = any("evaluation" in f for f in results["forecasting"] if "error" not in f)

            for forecast in results["forecasting"]:
                if "error" not in forecast:
                    ticker = forecast["ticker"]
                    asset = next(
                        (a for a in portfolio.get_all_assets() if a.symbol == ticker),
                        None,
                    )
                    if asset and asset.quantity:
                        # Calculate the forecasted value for this asset based on quantity and forecast price
                        # This simple calculation (quantity × forecast_price) ensures the forecast value
                        # represents what the portfolio would be worth at the forecasted prices
                        forecasted_asset_value = asset.quantity * forecast["forecast_price"]
                        forecasted_value += forecasted_asset_value

                        # If in evaluation mode, track accuracy
                        if "evaluation" in forecast:
                            accuracy = 100 - forecast["evaluation"]["mape"]
                            total_accuracy += accuracy
                            valid_forecasts += 1

            # Add forecasted value row with appropriate end date
            test_end = None
            if config.forecast.test_end_date:
                test_end = (
                    config.forecast.test_end_date.strftime("%Y-%m-%d")
                    if isinstance(config.forecast.test_end_date, date)
                    else str(config.forecast.test_end_date)
                )
            elif config.forecast.forecast_length:
                # Calculate future date based on forecast length
                from datetime import timedelta

                future_date = datetime.now() + timedelta(days=config.forecast.forecast_length)
                test_end = future_date.strftime("%Y-%m-%d") if isinstance(future_date, date) else str(future_date)
            else:
                # Try to get end date from any forecast result
                for forecast in results["forecasting"]:
                    if "error" not in forecast and "end_date" in forecast:
                        test_end = forecast["end_date"]
                        break

            if not test_end:
                # Default to 14 days if no forecast length specified
                from datetime import timedelta

                future_date = datetime.now() + timedelta(days=14)
                test_end = future_date.strftime("%Y-%m-%d") if isinstance(future_date, date) else str(future_date)

            portfolio_value_table.add_row("Forecast Value", test_end, f"${forecasted_value:,.2f}")

            # Show portfolio return
            if current_portfolio_value > 0:
                portfolio_return = ((forecasted_value - current_portfolio_value) / current_portfolio_value) * 100
                portfolio_value_table.add_row(
                    "Portfolio Return", f"{test_start} → {test_end}", f"{portfolio_return:+.2f}%"
                )

            # Add average accuracy row only for evaluation mode
            if is_evaluation_mode and valid_forecasts > 0 and test_end:
                avg_accuracy = total_accuracy / valid_forecasts
                portfolio_value_table.add_row("Accuracy", test_end, f"{avg_accuracy:.4f}%")

        self.console.print(portfolio_value_table)
