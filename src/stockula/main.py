"""Stockula main entry point."""

# Suppress warnings early - before any imports that might trigger them
import logging

logging.getLogger("alembic.runtime.migration").setLevel(logging.WARNING)
logging.getLogger("alembic").setLevel(logging.WARNING)

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

import os

os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"


from dependency_injector.wiring import Provide, inject

from .cli import app
from .config import StockulaConfig
from .container import Container
from .interfaces import ILoggingManager

# Global logging manager instance
log_manager: ILoggingManager | None = None


@inject
def setup_logging(
    config: StockulaConfig,
    logging_manager: ILoggingManager = Provide[Container.logging_manager],
) -> None:
    """Configure logging based on configuration."""
    global log_manager
    log_manager = logging_manager
    log_manager.setup(config)


def main():
    """Entry point."""
    # Present Typer CLI
    app()


if __name__ == "__main__":
    app()
