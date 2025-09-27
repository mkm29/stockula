"""Console factory for consistent console creation across modules."""

from rich.console import Console

from ..cli_manager import cli_manager


class ConsoleFactory:
    """Factory for creating and managing console instances."""

    @staticmethod
    def get_console(console: Console | None = None) -> Console:
        """Get console instance, falling back to shared instance if None.

        Args:
            console: Optional console instance

        Returns:
            Console instance (provided or shared)
        """
        return console or cli_manager.get_console()

    @staticmethod
    def create_console() -> Console:
        """Create a new console instance.

        Returns:
            New Console instance
        """
        return Console()


# Convenience function for the common pattern
def get_console(console: Console | None = None) -> Console:
    """Get console instance with fallback to shared console.

    Args:
        console: Optional console instance

    Returns:
        Console instance (provided or shared)
    """
    return ConsoleFactory.get_console(console)
