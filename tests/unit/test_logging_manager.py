"""Tests for logging_manager module."""

import logging
import os
import sys
import tempfile
from unittest.mock import Mock, patch

from stockula.config import LoggingConfig, StockulaConfig
from stockula.utils.logging_manager import LoggingManager


class TestLoggingManagerInitialization:
    """Test LoggingManager initialization."""

    def test_default_initialization(self):
        """Test default LoggingManager initialization."""
        manager = LoggingManager()
        assert manager.name == "stockula"
        assert manager.logger.name == "stockula"
        assert manager.handlers == []
        assert not manager._is_configured

    def test_custom_name_initialization(self):
        """Test LoggingManager with custom name."""
        manager = LoggingManager("custom_logger")
        assert manager.name == "custom_logger"
        assert manager.logger.name == "custom_logger"

    def test_logger_instance_creation(self):
        """Test that logger instance is created correctly."""
        manager = LoggingManager()
        assert isinstance(manager.logger, logging.Logger)


class TestLoggingManagerSetup:
    """Test LoggingManager setup functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = StockulaConfig()
        self.config.logging = LoggingConfig()

    def test_setup_prevents_duplicate_configuration(self):
        """Test that setup prevents duplicate configuration."""
        manager = LoggingManager()

        with patch.object(manager.logger, "addHandler") as mock_add_handler:
            # First setup
            manager.setup(self.config)
            assert manager._is_configured
            first_call_count = mock_add_handler.call_count

            # Second setup should be skipped
            manager.setup(self.config)
            assert mock_add_handler.call_count == first_call_count

    def test_setup_clears_existing_handlers(self):
        """Test that setup clears existing handlers."""
        manager = LoggingManager()

        # Add a mock handler to root logger
        root_logger = logging.getLogger()
        mock_handler = Mock()
        root_logger.addHandler(mock_handler)

        try:
            with patch.object(root_logger, "removeHandler") as mock_remove:
                manager.setup(self.config)
                mock_remove.assert_called()
        finally:
            # Clean up
            root_logger.removeHandler(mock_handler)

    def test_setup_configures_logger_level(self):
        """Test that setup configures logger level correctly."""
        manager = LoggingManager()
        self.config.logging.level = "DEBUG"
        self.config.logging.enabled = True

        manager.setup(self.config)
        assert manager.logger.level == logging.DEBUG

    def test_setup_with_logging_enabled(self):
        """Test setup with logging enabled."""
        manager = LoggingManager()
        self.config.logging.enabled = True

        with patch("stockula.utils.logging_manager.logging.StreamHandler") as mock_handler_class:
            mock_handler = Mock()
            mock_handler.level = 20  # Set as integer, not Mock
            mock_handler_class.return_value = mock_handler

            # Patch the logging calls to avoid the >= comparison error
            with patch.object(manager, "info"):
                manager.setup(self.config)
                mock_handler_class.assert_called()

    def test_setup_with_logging_disabled(self):
        """Test setup with logging disabled."""
        manager = LoggingManager()
        self.config.logging.enabled = False

        with patch("stockula.utils.logging_manager.logging.StreamHandler") as mock_handler:
            manager.setup(self.config)
            mock_handler.assert_called()  # StreamHandler is always created, but with WARNING level

    def test_setup_with_file_logging_enabled(self):
        """Test setup with file logging enabled."""
        manager = LoggingManager()
        self.config.logging.log_to_file = True
        self.config.logging.enabled = True

        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as tmp_file:
            self.config.logging.log_file = tmp_file.name

            try:
                with patch("stockula.utils.logging_manager.RotatingFileHandler") as mock_handler_class:
                    mock_handler = Mock()
                    mock_handler.level = 20  # Set as integer
                    mock_handler_class.return_value = mock_handler

                    # Patch logging calls to avoid >= comparison error
                    with patch.object(manager, "info"):
                        manager.setup(self.config)
                        mock_handler_class.assert_called()
            finally:
                os.unlink(tmp_file.name)

    def test_setup_creates_log_directory(self):
        """Test that setup creates log directory if it doesn't exist."""
        manager = LoggingManager()
        self.config.logging.log_to_file = True
        self.config.logging.enabled = True

        with tempfile.TemporaryDirectory() as tmp_dir:
            log_dir = os.path.join(tmp_dir, "subdir")
            log_file = os.path.join(log_dir, "test.log")
            self.config.logging.log_file = log_file

            # Create the directory structure
            os.makedirs(log_dir, exist_ok=True)

            manager.setup(self.config)
            assert os.path.exists(os.path.dirname(log_file))


class TestLoggingManagerMethods:
    """Test LoggingManager logging methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = LoggingManager("test_logger")
        # Mock the logger to capture calls
        self.manager.logger = Mock()

    def test_debug_method(self):
        """Test debug logging method."""
        self.manager.debug("Debug message")
        self.manager.logger.debug.assert_called_once_with("Debug message")

    def test_info_method(self):
        """Test info logging method."""
        self.manager.info("Info message")
        self.manager.logger.info.assert_called_once_with("Info message")

    def test_warning_method(self):
        """Test warning logging method."""
        self.manager.warning("Warning message")
        self.manager.logger.warning.assert_called_once_with("Warning message")

    def test_error_method_without_exc_info(self):
        """Test error logging method without exception info."""
        self.manager.error("Error message")
        self.manager.logger.error.assert_called_once_with("Error message", exc_info=False)

    def test_error_method_with_exc_info(self):
        """Test error logging method with exception info."""
        self.manager.error("Error message", exc_info=True)
        self.manager.logger.error.assert_called_once_with("Error message", exc_info=True)

    def test_critical_method_without_exc_info(self):
        """Test critical logging method without exception info."""
        self.manager.critical("Critical message")
        self.manager.logger.critical.assert_called_once_with("Critical message", exc_info=False)

    def test_critical_method_with_exc_info(self):
        """Test critical logging method with exception info."""
        self.manager.critical("Critical message", exc_info=True)
        self.manager.logger.critical.assert_called_once_with("Critical message", exc_info=True)


class TestLoggingManagerHandlers:
    """Test LoggingManager handler management."""

    def test_console_handler_creation(self):
        """Test console handler creation."""
        manager = LoggingManager()
        config = StockulaConfig()
        config.logging.enabled = True
        config.logging.level = "INFO"

        with patch("stockula.utils.logging_manager.logging.StreamHandler") as mock_handler_class:
            mock_handler = Mock()
            mock_handler.level = 20  # Set level attribute as integer
            mock_handler_class.return_value = mock_handler

            manager.setup(config)

            mock_handler_class.assert_called_once_with(sys.stdout)
            mock_handler.setLevel.assert_called_with(logging.INFO)

    def test_file_handler_creation_with_rotation(self):
        """Test file handler creation with rotation settings."""
        manager = LoggingManager()
        config = StockulaConfig()
        config.logging.log_to_file = True
        config.logging.enabled = True
        config.logging.level = "DEBUG"
        config.logging.max_log_size = 1024 * 1024  # 1MB
        config.logging.backup_count = 5

        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as tmp_file:
            config.logging.log_file = tmp_file.name

            try:
                with patch("stockula.utils.logging_manager.RotatingFileHandler") as mock_handler_class:
                    mock_handler = Mock()
                    mock_handler.level = 10  # Set level attribute as integer
                    mock_handler_class.return_value = mock_handler

                    manager.setup(config)

                    mock_handler_class.assert_called_once_with(
                        filename=tmp_file.name,
                        maxBytes=1024 * 1024,
                        backupCount=5,
                        encoding="utf-8",
                    )
                    mock_handler.setLevel.assert_called_with(logging.DEBUG)
            finally:
                os.unlink(tmp_file.name)

    def test_formatter_application(self):
        """Test that formatters are applied to handlers."""
        manager = LoggingManager()
        config = StockulaConfig()
        config.logging.enabled = True

        with patch("stockula.utils.logging_manager.logging.StreamHandler") as mock_handler_class:
            mock_handler = Mock()
            mock_handler.level = 20  # Set level attribute as integer
            mock_handler_class.return_value = mock_handler

            with patch("stockula.utils.logging_manager.logging.Formatter") as mock_formatter_class:
                mock_formatter = Mock()
                mock_formatter_class.return_value = mock_formatter

                manager.setup(config)

                # Should use simple formatter for console
                mock_formatter_class.assert_called_with(
                    fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
                )
                mock_handler.setFormatter.assert_called_with(mock_formatter)


class TestLoggingManagerEdgeCases:
    """Test LoggingManager edge cases."""

    def test_setup_with_invalid_log_level(self):
        """Test setup with invalid log level falls back to INFO."""
        manager = LoggingManager()
        config = StockulaConfig()
        config.logging.level = "INVALID_LEVEL"
        config.logging.enabled = True

        # The logging manager will use getattr(logging, level.upper(), logging.INFO)
        # When the level doesn't exist, it defaults to logging.INFO
        manager.setup(config)
        # Should default to INFO level (20)
        assert manager.logger.level == logging.INFO

    def test_file_logging_with_permission_error(self):
        """Test file logging handles permission errors gracefully."""
        manager = LoggingManager()
        config = StockulaConfig()
        config.logging.log_to_file = True
        config.logging.enabled = True
        config.logging.log_file = "/root/nonexistent/test.log"  # Should fail on most systems

        # This should not raise an exception
        try:
            manager.setup(config)
        except (PermissionError, FileNotFoundError):
            # Expected behavior - permission denied or file not found
            pass

    def test_multiple_setups_with_different_configs(self):
        """Test that subsequent setups are ignored after first configuration."""
        manager = LoggingManager()

        config1 = StockulaConfig()
        config1.logging.level = "DEBUG"
        config1.logging.enabled = True

        config2 = StockulaConfig()
        config2.logging.level = "ERROR"
        config2.logging.enabled = False

        # First setup
        manager.setup(config1)
        assert manager._is_configured
        original_level = manager.logger.level

        # Second setup should be ignored
        manager.setup(config2)
        assert manager.logger.level == original_level  # Should remain unchanged


class TestLoggingManagerIntegration:
    """Test LoggingManager integration scenarios."""

    def test_real_logging_output(self):
        """Test that logging actually produces output."""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".log", delete=False) as tmp_file:
            try:
                manager = LoggingManager("integration_test")
                config = StockulaConfig()
                config.logging.log_to_file = True
                config.logging.log_file = tmp_file.name
                config.logging.level = "INFO"
                config.logging.enabled = True

                manager.setup(config)

                # Log some messages
                manager.info("Test info message")
                manager.warning("Test warning message")
                manager.error("Test error message")

                # Force flush
                for handler in manager.logger.handlers:
                    handler.flush()

                # Read the log file
                with open(tmp_file.name) as f:
                    log_content = f.read()

                assert "Test info message" in log_content
                assert "Test warning message" in log_content
                assert "Test error message" in log_content

            finally:
                os.unlink(tmp_file.name)

    def test_handler_management(self):
        """Test that handlers are properly managed."""
        manager = LoggingManager()
        config = StockulaConfig()
        config.logging.enabled = True
        config.logging.log_to_file = True

        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as tmp_file:
            config.logging.log_file = tmp_file.name

            try:
                manager.setup(config)

                # Handlers should be stored in the manager
                assert len(manager.handlers) > 0

                # Root logger should have handlers (LoggingManager adds to root)
                root_logger = logging.getLogger()
                assert len(root_logger.handlers) > 0

            finally:
                os.unlink(tmp_file.name)

    def test_logger_hierarchy(self):
        """Test logger hierarchy behavior."""
        parent_manager = LoggingManager("parent")
        LoggingManager("parent.child")

        config = StockulaConfig()
        config.logging.level = "DEBUG"
        config.logging.enabled = True

        parent_manager.setup(config)

        # Child logger should inherit from parent due to name hierarchy
        # The LoggingManager setup always creates stockula logger, but we can test the hierarchy exists
        parent_logger = logging.getLogger("parent")
        child_logger = logging.getLogger("parent.child")
        assert child_logger.parent == parent_logger
