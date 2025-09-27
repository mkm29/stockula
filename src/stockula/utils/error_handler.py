"""Centralized error handling utilities for Stockula."""

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from ..cli_manager import cli_manager


class ErrorHandler:
    """Centralized error handling with consistent formatting."""

    def __init__(self, console: Console | None = None):
        """Initialize error handler.

        Args:
            console: Rich console instance, uses shared instance if None
        """
        self.console = console or cli_manager.get_console()

    def handle_validation_error(self, error: ValidationError, config_path: str) -> None:
        """Handle validation errors with clean, user-friendly output.

        Args:
            error: The validation error
            config_path: Path to the configuration file
        """
        self.console.print("\n[bold red]Configuration Validation Error[/bold red]\n")
        self.console.print(f"Failed to load configuration from: [cyan]{config_path}[/cyan]\n")

        # Parse and display errors in a user-friendly format
        errors = []
        for err in error.errors():
            location = " → ".join(str(loc) for loc in err["loc"])
            message = err["msg"]

            # Clean up common error messages
            if "test_start_date must be before test_end_date" in message:
                errors.append("[yellow]Date Range Error:[/yellow] Test end date is before test start date")
            elif "train_start_date must be before train_end_date" in message:
                errors.append("[yellow]Date Range Error:[/yellow] Train end date is before train start date")
            elif "train_end_date must be before or equal to test_start_date" in message:
                errors.append(
                    "[yellow]Date Sequence Error:[/yellow] Training period must end before test period begins"
                )
            else:
                errors.append(f"[yellow]{location}:[/yellow] {message}")

        # Display errors in a panel
        error_text = "\n".join(f"  • {err}" for err in errors)
        self.console.print(
            Panel(
                error_text,
                title="[bold]Validation Issues[/bold]",
                border_style="red",
                padding=(1, 2),
            )
        )

        # Show the problematic configuration section if possible
        if "backtest_optimization" in str(error):
            self.console.print("\n[dim]Check your backtest_optimization section in the config file.[/dim]")
            self.console.print("[dim]Ensure that:[/dim]")
            self.console.print("[dim]  • All dates are in YYYY-MM-DD format[/dim]")
            self.console.print("[dim]  • train_start_date < train_end_date[/dim]")
            self.console.print("[dim]  • test_start_date < test_end_date[/dim]")
            self.console.print("[dim]  • train_end_date ≤ test_start_date[/dim]")

        self.console.print()
        raise typer.Exit(1)

    def handle_portfolio_error(self, error: ValueError) -> None:
        """Handle portfolio validation errors with user-friendly suggestions.

        Args:
            error: The portfolio validation error
        """
        error_msg = str(error)
        if "insufficient" in error_msg.lower() and "capital" in error_msg.lower():
            self.console.print("\n[bold red]❌ Portfolio Configuration Error[/bold red]\n")
            self.console.print(f"[red]{error_msg}[/red]\n")
            self.console.print("[dim]💡 Suggestions:[/dim]")
            self.console.print("[dim]  • Increase the initial_capital in your configuration[/dim]")
            self.console.print("[dim]  • Reduce the quantities of some assets[/dim]")
            self.console.print("[dim]  • Enable fractional shares: allow_fractional_shares: true[/dim]")
        else:
            self.console.print(f"\n[bold red]❌ Portfolio Error:[/bold red] [red]{error_msg}[/red]\n")
        raise typer.Exit(1) from None

    def handle_processing_error(self, error: Exception) -> None:
        """Handle processing errors with categorized suggestions.

        Args:
            error: The processing error
        """
        error_msg = str(error)
        if "insufficient" in error_msg.lower() and "data" in error_msg.lower():
            self.console.print(f"\n[bold red]❌ Data Error:[/bold red] [red]{error_msg}[/red]")
            self.console.print("[dim]💡 Try adjusting the date range in your configuration[/dim]\n")
        elif "network" in error_msg.lower() or "connection" in error_msg.lower():
            self.console.print(f"\n[bold red]❌ Network Error:[/bold red] [red]{error_msg}[/red]")
            self.console.print("[dim]💡 Check your internet connection and try again[/dim]\n")
        else:
            self.console.print(f"\n[bold red]❌ Processing Error:[/bold red] [red]{error_msg}[/red]\n")
        raise typer.Exit(1) from None

    def handle_pipeline_error(self, error: Exception, verbose: bool = False) -> None:
        """Handle pipeline-specific errors.

        Args:
            error: The pipeline error
            verbose: Whether to show full traceback
        """
        if isinstance(error, FileNotFoundError):
            self.console.print(f"[red]Error: {error}[/red]")
            raise typer.Exit(1) from None
        elif isinstance(error, KeyboardInterrupt):
            self.console.print("\n[yellow]Pipeline interrupted by user[/yellow]")
            raise typer.Exit(130) from None
        else:
            self.console.print(f"[red]Unexpected error: {error}[/red]")
            if verbose:
                import traceback

                self.console.print(traceback.format_exc())
            raise typer.Exit(1) from None


# Global error handler instance using shared console
error_handler = ErrorHandler()
