"""Consolidated database interfaces for TimescaleDB operations.

This module contains the single, unified IDatabaseManager interface
that encompasses all database operations including analytics.
"""

from ..interfaces import IDatabaseManager

# Re-export the consolidated interface for database package imports
__all__ = ["IDatabaseManager"]
