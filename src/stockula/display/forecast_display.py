"""Forecast results display service for Single Responsibility Principle compliance."""

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from stockula.config.models import StockulaConfig
from stockula.utils.console_factory import get_console


class ForecastResultsDisplay:
    """Handles display and formatting of forecasting results only."""

    def __init__(self, console: Console | None = None):
        """Initialize the display handler.

        Args:
            console: Rich console for output (optional)
        """
        self.console = get_console(console)

    def display_forecast_results(self, forecast_results: list[dict[str, Any]], portfolio=None):
        """Display forecasting results.

        Args:
            forecast_results: List of forecast results
            portfolio: Optional portfolio instance for displaying quantities and values
        """
        self.console.print("\n[bold purple]=== Forecasting Results ===[/bold purple]")

        # Get date range from first non-error forecast
        date_info = ""
        for forecast in forecast_results:
            if "error" not in forecast and "start_date" in forecast:
                date_info = f" ({forecast['start_date']} to {forecast['end_date']})"
                break

        # Check if we have actual prices (evaluation mode)
        has_actual_prices = any("actual_price" in f for f in forecast_results if "error" not in f)

        table = Table(title=f"Price Forecasts{date_info}", show_header=True, header_style="bold")
        table.add_column("Ticker", style="cyan", no_wrap=True)
        table.add_column("Qty", style="green", justify="right", no_wrap=True)
        table.add_column("Current\nPrice", style="white", justify="right")
        table.add_column("Current\nValue", style="blue", justify="right")
        if has_actual_prices:
            table.add_column("Actual\nPrice", style="yellow", justify="right")
            table.add_column("Actual\nValue", style="yellow", justify="right")
        table.add_column("Forecast\nPrice", style="green", justify="right")
        table.add_column("Forecast\nValue", style="blue", justify="right")
        table.add_column("Return", style="magenta", justify="right")
        table.add_column("Confidence Range", style="yellow", justify="center")
        table.add_column("Model", style="blue")

        # Sort forecasts by return percentage (highest to lowest)
        # Separate error results from valid forecasts
        error_forecasts = [f for f in forecast_results if "error" in f]
        valid_forecasts = [f for f in forecast_results if "error" not in f]

        # Calculate return percentage for sorting
        for forecast in valid_forecasts:
            forecast["return_pct"] = (
                (forecast["forecast_price"] - forecast["current_price"]) / forecast["current_price"]
            ) * 100

        # Sort valid forecasts by return percentage (highest to lowest)
        sorted_forecasts = sorted(valid_forecasts, key=lambda f: f["return_pct"], reverse=True)

        # Combine sorted valid forecasts with error forecasts at the end
        all_forecasts = sorted_forecasts + error_forecasts

        for forecast in all_forecasts:
            ticker = forecast["ticker"]

            # Get quantity from portfolio if available
            quantity = 0.0
            if portfolio:
                asset = next((a for a in portfolio.get_all_assets() if a.symbol == ticker), None)
                if asset and hasattr(asset, "quantity"):
                    quantity = asset.quantity

            if "error" in forecast:
                row_data = [
                    ticker,
                    "[red]Error[/red]",  # Quantity
                    "[red]Error[/red]",  # Current Price
                    "[red]Error[/red]",  # Current Value
                ]
                if has_actual_prices:
                    row_data.extend(
                        [
                            "[red]Error[/red]",  # Actual Price
                            "[red]Error[/red]",  # Actual Value
                        ]
                    )
                row_data.extend(
                    [
                        "[red]Error[/red]",  # Forecast Price
                        "[red]Error[/red]",  # Forecast Value
                        "[red]Error[/red]",  # Return %
                        "[red]Error[/red]",  # Confidence Range
                        f"[red]{forecast['error']}[/red]",  # Best Model/Error
                    ]
                )
                table.add_row(*row_data)
            else:
                current_price = forecast["current_price"]
                forecast_price = forecast["forecast_price"]

                # Calculate values
                current_value = quantity * current_price
                forecast_value = quantity * forecast_price

                # Calculate return percentage
                return_pct = ((forecast_price - current_price) / current_price) * 100

                # Color code forecast based on direction
                forecast_color = (
                    "green" if forecast_price > current_price else "red" if forecast_price < current_price else "white"
                )
                forecast_str = f"[{forecast_color}]${forecast_price:.2f}[/{forecast_color}]"
                forecast_value_str = f"[{forecast_color}]${forecast_value:,.2f}[/{forecast_color}]"

                # Format return percentage with color
                return_str = f"[{forecast_color}]{return_pct:+.2f}%[/{forecast_color}]"

                # Build row data
                row_data = [
                    ticker,
                    f"{quantity:.2f}",
                    f"${current_price:.2f}",
                    f"${current_value:,.2f}",
                ]

                # Add actual price and value if available
                if has_actual_prices:
                    if "actual_price" in forecast:
                        actual_price = forecast["actual_price"]
                        actual_value = quantity * actual_price
                        # Color code actual vs forecast
                        actual_color = (
                            "green"
                            if actual_price < forecast_price
                            else "red"
                            if actual_price > forecast_price
                            else "white"
                        )
                        row_data.extend(
                            [
                                f"[{actual_color}]${actual_price:.2f}[/{actual_color}]",
                                f"[{actual_color}]${actual_value:,.2f}[/{actual_color}]",
                            ]
                        )
                    else:
                        row_data.extend(["N/A", "N/A"])

                row_data.extend(
                    [
                        forecast_str,
                        forecast_value_str,
                        return_str,
                        f"${forecast['lower_bound']:.2f} - ${forecast['upper_bound']:.2f}",
                        forecast["best_model"],
                    ]
                )

                table.add_row(*row_data)

        self.console.print(table)

        # Display evaluation metrics if available
        has_evaluation = any("evaluation" in f for f in forecast_results if "error" not in f)
        if has_evaluation:
            self.display_evaluation_metrics(all_forecasts)

    def display_evaluation_metrics(self, forecasts: list[dict[str, Any]]):
        """Display forecast evaluation metrics.

        Args:
            forecasts: List of forecast results
        """
        self.console.print("\n[bold cyan]=== Forecast Evaluation Metrics ===[/bold cyan]")

        eval_table = Table(title="Model Performance on Test Data")
        eval_table.add_column("Ticker", style="cyan", no_wrap=True)
        eval_table.add_column("RMSE", style="yellow", justify="right")
        eval_table.add_column("MAE", style="yellow", justify="right")
        eval_table.add_column("MASE", style="green", justify="right")
        eval_table.add_column("Train Period", style="white")
        eval_table.add_column("Test Period", style="white")

        # Use the same sorted order as the forecast table
        for forecast in forecasts:
            if "error" not in forecast and "evaluation" in forecast:
                eval_metrics = forecast["evaluation"]
                train_period = forecast.get("train_period", {})
                test_period = forecast.get("test_period", {})

                train_str = f"{train_period.get('start', 'N/A')} to {train_period.get('end', 'N/A')}"
                test_str = f"{test_period.get('start', 'N/A')} to {test_period.get('end', 'N/A')}"

                # Use MASE if available, otherwise fall back to MAPE
                mase_value = eval_metrics.get("mase")
                if mase_value is not None:
                    mase_str = f"{mase_value:.3f}"
                    # Add interpretation: < 1 is better than naive
                    if mase_value < 1:
                        mase_str += " ✓"  # Good performance
                else:
                    # Fall back to MAPE if MASE not available
                    mape_value = eval_metrics.get("mape", 0)
                    mase_str = f"({mape_value:.1f}%)"  # Show MAPE in parentheses

                eval_table.add_row(
                    forecast["ticker"],
                    f"${eval_metrics['rmse']:.2f}",
                    f"${eval_metrics['mae']:.2f}",
                    mase_str,
                    train_str,
                    test_str,
                )

        self.console.print(eval_table)
        self.console.print(
            "\n[dim]MASE: Mean Absolute Scaled Error (< 1.0 = better than naive forecast, marked with ✓)[/dim]"
        )

    def show_forecast_warning(self, config: StockulaConfig):
        """Show forecast mode warning.

        Args:
            config: Configuration object
        """
        # Determine forecast mode message
        if config.forecast.forecast_length is not None:
            forecast_msg = f"• Forecasting {config.forecast.forecast_length} days into the future"
        elif config.forecast.test_start_date and config.forecast.test_end_date:
            forecast_msg = (
                f"• Evaluating forecast on test period: "
                f"{config.forecast.test_start_date} to {config.forecast.test_end_date}"
            )
        else:
            forecast_msg = "• Forecast configuration error: neither forecast_length nor test dates specified"

        self.console.print(
            Panel.fit(
                f"[bold yellow]FORECAST MODE - IMPORTANT NOTES:[/bold yellow]\\n"
                f"{forecast_msg}\\n"
                f"• The selected backend determines the approach (Chronos/AutoGluon/Simple)\\n"
                f"• This process may take several minutes per ticker\\n"
                f"• Press Ctrl+C at any time to cancel\\n"
                f"• Enable logging for more detailed progress information",
                border_style="yellow",
            )
        )
