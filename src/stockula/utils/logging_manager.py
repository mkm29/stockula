"""Centralized logging management for Stockula."""

import logging
import sys
from logging.handlers import RotatingFileHandler

from ..config import StockulaConfig


class LoggingManager:
    """Manages all logging operations for the Stockula application."""

    def __init__(self, name: str = "stockula"):
        """Initialize the LoggingManager.

        Args:
            name: Logger name (default: "stockula")
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.handlers: list[logging.Handler] = []
        self._is_configured = False

    def setup(self, config: StockulaConfig) -> None:
        """Configure logging based on configuration.

        Args:
            config: Stockula configuration object
        """
        # Prevent duplicate setup
        if self._is_configured:
            return

        # Clear any existing handlers to avoid duplicates
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Clear logger's own handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # List to store all handlers
        self.handlers = []

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)

        # Create formatters
        if config.logging.enabled:
            # Detailed format when logging is enabled
            detailed_formatter = logging.Formatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            simple_formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
            log_level = getattr(logging, config.logging.level.upper(), logging.INFO)

            # Use simple formatter for console, detailed for file
            console_handler.setFormatter(simple_formatter)

            # Add file handler if requested
            if config.logging.log_to_file:
                file_handler = RotatingFileHandler(
                    filename=config.logging.log_file,
                    maxBytes=config.logging.max_log_size,
                    backupCount=config.logging.backup_count,
                    encoding="utf-8",
                )
                file_handler.setFormatter(detailed_formatter)
                file_handler.setLevel(log_level)
                self.handlers.append(file_handler)
        else:
            # Simple format when logging is disabled (only warnings/errors)
            formatter = logging.Formatter(fmt="%(levelname)s: %(message)s")
            console_handler.setFormatter(formatter)
            log_level = logging.WARNING

        # Set level on console handler and add to handlers
        console_handler.setLevel(log_level)
        self.handlers.append(console_handler)

        # Configure root logger
        root_logger.setLevel(log_level)
        for handler in self.handlers:
            root_logger.addHandler(handler)

        # Configure stockula loggers with proper hierarchy
        stockula_logger = logging.getLogger("stockula")
        stockula_logger.setLevel(log_level)
        stockula_logger.propagate = True  # Propagate to root logger

        # Store logger reference
        self.logger = stockula_logger

        # Log startup message if enabled
        if config.logging.enabled:
            self.info(f"Logging initialized - Level: {config.logging.level}")
            if config.logging.log_to_file:
                self.info(f"Logging to file: {config.logging.log_file}")

        # Reduce noise from third-party libraries
        third_party_level = (
            logging.CRITICAL
            if not config.logging.enabled
            else (logging.WARNING if log_level != logging.DEBUG else logging.INFO)
        )

        for lib_name in [
            "yfinance",
            "urllib3",
            "requests",
            "apscheduler",
            "peewee",
            "backtesting",
            "cmdstanpy",
            "prophet",
            "prophet.plot",
            "matplotlib",
            "matplotlib.font_manager",
        ]:
            logging.getLogger(lib_name).setLevel(third_party_level)

        self._is_configured = True

    def debug(self, message: str) -> None:
        """Log a debug message.

        Args:
            message: The message to log
        """
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """Log an info message.

        Args:
            message: The message to log
        """
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log a warning message.

        Args:
            message: The message to log
        """
        self.logger.warning(message)

    def error(self, message: str, exc_info: bool = False) -> None:
        """Log an error message.

        Args:
            message: The message to log
            exc_info: Whether to include exception info
        """
        self.logger.error(message, exc_info=exc_info)

    def critical(self, message: str, exc_info: bool = False) -> None:
        """Log a critical message.

        Args:
            message: The message to log
            exc_info: Whether to include exception info
        """
        self.logger.critical(message, exc_info=exc_info)

    def exception(self, message: str) -> None:
        """Log an exception with traceback.

        Args:
            message: The message to log
        """
        self.logger.exception(message)

    def get_logger(self) -> logging.Logger:
        """Get the underlying logger instance.

        Returns:
            The logger instance
        """
        return self.logger

    def add_handler(self, handler: logging.Handler) -> None:
        """Add a custom handler to the logger.

        Args:
            handler: The handler to add
        """
        self.handlers.append(handler)
        self.logger.addHandler(handler)

    def remove_handler(self, handler: logging.Handler) -> None:
        """Remove a handler from the logger.

        Args:
            handler: The handler to remove
        """
        if handler in self.handlers:
            self.handlers.remove(handler)
            self.logger.removeHandler(handler)

    def set_level(self, level: str) -> None:
        """Set the logging level.

        Args:
            level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        log_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.setLevel(log_level)
        for handler in self.handlers:
            handler.setLevel(log_level)

    def set_module_level(self, module_name: str, level: str) -> None:
        """Set the logging level for a specific module.

        Args:
            module_name: The module name (e.g., 'cmdstanpy', 'prophet')
            level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        log_level = getattr(logging, level.upper(), logging.WARNING)
        logging.getLogger(module_name).setLevel(log_level)

    def close(self) -> None:
        """Close all handlers and clean up."""
        for handler in self.handlers:
            handler.close()
            self.logger.removeHandler(handler)
        self.handlers.clear()
        self._is_configured = False
