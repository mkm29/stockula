"""Add performance indexes for better query performance

Revision ID: add_performance_indexes
Revises: latest
Create Date: 2025-07-28

"""

import sqlalchemy as sa

from alembic import op  # type: ignore[attr-defined]

# revision identifiers, used by Alembic.
revision = "c1ceaae14e9f"
down_revision = "38d444940565"  # Updated to link to the latest revision
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add performance-critical indexes."""

    # Get the current connection to check for existing indexes
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    # Helper function to check if index exists
    def index_exists(table_name: str, index_name: str) -> bool:
        try:
            indexes = inspector.get_indexes(table_name)
            return any(idx["name"] == index_name for idx in indexes)
        except Exception:
            return False

    # Composite index for price_history queries that filter by symbol, date, and interval
    if not index_exists("price_history", "idx_price_history_symbol_date_interval"):
        op.create_index(
            "idx_price_history_symbol_date_interval",
            "price_history",
            ["symbol", "date", "interval"],
        )

    # Index for date range queries on price_history
    if not index_exists("price_history", "idx_price_history_symbol_interval"):
        op.create_index("idx_price_history_symbol_interval", "price_history", ["symbol", "interval"])

    # Composite index for options queries
    if not index_exists("options_calls", "idx_options_calls_symbol_exp_strike"):
        op.create_index(
            "idx_options_calls_symbol_exp_strike",
            "options_calls",
            ["symbol", "expiration_date", "strike"],
        )

    if not index_exists("options_puts", "idx_options_puts_symbol_exp_strike"):
        op.create_index(
            "idx_options_puts_symbol_exp_strike",
            "options_puts",
            ["symbol", "expiration_date", "strike"],
        )

    # Index for stock_info queries
    if not index_exists("stock_info", "idx_stock_info_symbol"):
        op.create_index("idx_stock_info_symbol", "stock_info", ["symbol"])

    # Index for date-based queries on various tables
    if not index_exists("price_history", "idx_price_history_date"):
        op.create_index("idx_price_history_date", "price_history", ["date"])

    if not index_exists("dividends", "idx_dividends_date"):
        op.create_index("idx_dividends_date", "dividends", ["date"])

    if not index_exists("splits", "idx_splits_date"):
        op.create_index("idx_splits_date", "splits", ["date"])


def downgrade() -> None:
    """Remove performance indexes."""
    # Get the current connection to check for existing indexes
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    # Helper function to check if index exists
    def index_exists(table_name: str, index_name: str) -> bool:
        try:
            indexes = inspector.get_indexes(table_name)
            return any(idx["name"] == index_name for idx in indexes)
        except Exception:
            return False

    # Only drop indexes that exist
    if index_exists("price_history", "idx_price_history_symbol_date_interval"):
        op.drop_index("idx_price_history_symbol_date_interval", "price_history")
    if index_exists("price_history", "idx_price_history_symbol_interval"):
        op.drop_index("idx_price_history_symbol_interval", "price_history")
    if index_exists("options_calls", "idx_options_calls_symbol_exp_strike"):
        op.drop_index("idx_options_calls_symbol_exp_strike", "options_calls")
    if index_exists("options_puts", "idx_options_puts_symbol_exp_strike"):
        op.drop_index("idx_options_puts_symbol_exp_strike", "options_puts")
    if index_exists("stock_info", "idx_stock_info_symbol"):
        op.drop_index("idx_stock_info_symbol", "stock_info")
    if index_exists("price_history", "idx_price_history_date"):
        op.drop_index("idx_price_history_date", "price_history")
    if index_exists("dividends", "idx_dividends_date"):
        op.drop_index("idx_dividends_date", "dividends")
    if index_exists("splits", "idx_splits_date"):
        op.drop_index("idx_splits_date", "splits")
