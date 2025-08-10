"""CLI Manager for centralized CLI resources and utilities."""

from typing import cast

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn


class CLIManager:
    """Manages shared CLI resources like console and progress displays.

    This singleton class provides centralized access to Rich console
    and other CLI utilities, ensuring consistent output formatting
    across the application.
    """

    _instance = None

    def __new__(cls):
        """Ensure singleton pattern for CLIManager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the CLI manager with shared resources."""
        if self._initialized:
            return

        self.console = Console()
        self._initialized = True

    def get_console(self) -> Console:
        """Get the shared Rich console instance.

        Returns:
            Rich Console instance
        """
        return cast(Console, self.console)

    def create_progress(self, *columns, transient: bool = True, **kwargs) -> Progress:
        """Create a Progress display with standard configuration.

        Args:
            *columns: Progress display columns
            transient: Whether the progress should disappear when complete
            **kwargs: Additional Progress parameters

        Returns:
            Configured Progress instance
        """
        if not columns:
            # Default columns if none provided
            columns = (
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
            )

        return Progress(*columns, console=self.console, transient=transient, **kwargs)

    def create_detailed_progress(self, transient: bool = True, **kwargs) -> Progress:
        """Create a detailed Progress display with bar and time remaining.

        Args:
            transient: Whether the progress should disappear when complete
            **kwargs: Additional Progress parameters

        Returns:
            Configured Progress instance with detailed columns
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console,
            transient=transient,
            **kwargs,
        )

    def print(self, *args, **kwargs):
        """Print to console using Rich formatting.

        Args:
            *args: Arguments to print
            **kwargs: Rich print parameters
        """
        self.console.print(*args, **kwargs)

    def print_error(self, message: str, title: str = "Error"):
        """Print an error message with formatting.

        Args:
            message: Error message
            title: Error title
        """
        self.console.print(f"\n[bold red]❌ {title}:[/bold red] [red]{message}[/red]\n")

    def print_success(self, message: str):
        """Print a success message with formatting.

        Args:
            message: Success message
        """
        self.console.print(f"[green]✓ {message}[/green]")

    def print_warning(self, message: str):
        """Print a warning message with formatting.

        Args:
            message: Warning message
        """
        self.console.print(f"[yellow]⚠ {message}[/yellow]")

    def print_info(self, message: str):
        """Print an info message with formatting.

        Args:
            message: Info message
        """
        self.console.print(f"[cyan]ℹ {message}[/cyan]")


# Global instance for easy access
cli_manager = CLIManager()
