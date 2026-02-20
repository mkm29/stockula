"""Stockula - Financial trading and analysis library.

This package avoids global side effects on import (e.g., mutating logging,
warnings, or environment variables). Runtime configuration is handled in
`stockula.main.setup_logging` and the CLI entrypoint.
"""

# Package imports - only re-export symbols that are actually consumed via this path
from .backtesting import BacktestRunner, SMACrossStrategy
from .data import DataFetcher

# x-release-please-start-version
__version__ = "0.16.0"
# x-release-please-end

__all__ = [
    "DataFetcher",
    "SMACrossStrategy",
    "BacktestRunner",
]
