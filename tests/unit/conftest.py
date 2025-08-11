"""Shared test fixtures and configuration for unit tests."""

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(scope="session", autouse=True)
def mock_chronos_if_unavailable():
    """Mock chronos module if not available to prevent import errors."""
    try:
        import importlib

        importlib.import_module("chronos")
        chronos_available = True
    except ImportError:
        chronos_available = False
        # Create a mock chronos module to prevent import errors
        if "chronos" not in sys.modules:
            chronos_mock = MagicMock()
            chronos_mock.__spec__ = MagicMock()
            sys.modules["chronos"] = chronos_mock

    return chronos_available


@pytest.fixture(scope="session")
def chronos_available(mock_chronos_if_unavailable):
    """Check if chronos is available."""
    return mock_chronos_if_unavailable


def is_chronos_available():
    """Helper function to check chronos availability for skipif decorators."""
    try:
        import importlib

        importlib.import_module("chronos")
        return True
    except ImportError:
        return False
