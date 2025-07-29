"""Add performance indexes for better query performance

Revision ID: add_performance_indexes
Revises: latest
Create Date: 2025-07-28

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "add_performance_indexes"
down_revision = None  # Update this to your latest revision
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add performance-critical indexes."""

    # Composite index for price_history queries that filter by symbol, date, and interval
    op.create_index(
        "idx_price_history_symbol_date_interval",
        "price_history",
        ["symbol", "date", "interval"],
    )

    # Index for date range queries on price_history
    op.create_index(
        "idx_price_history_symbol_interval", "price_history", ["symbol", "interval"]
    )

    # Composite index for options queries
    op.create_index(
        "idx_options_calls_symbol_exp_strike",
        "options_calls",
        ["symbol", "expiration_date", "strike"],
    )

    op.create_index(
        "idx_options_puts_symbol_exp_strike",
        "options_puts",
        ["symbol", "expiration_date", "strike"],
    )

    # Index for stock_info queries
    op.create_index("idx_stock_info_symbol", "stock_info", ["symbol"])

    # Index for date-based queries on various tables
    op.create_index("idx_price_history_date", "price_history", ["date"])

    op.create_index("idx_dividends_date", "dividends", ["date"])

    op.create_index("idx_splits_date", "splits", ["date"])


def downgrade() -> None:
    """Remove performance indexes."""
    op.drop_index("idx_price_history_symbol_date_interval", "price_history")
    op.drop_index("idx_price_history_symbol_interval", "price_history")
    op.drop_index("idx_options_calls_symbol_exp_strike", "options_calls")
    op.drop_index("idx_options_puts_symbol_exp_strike", "options_puts")
    op.drop_index("idx_stock_info_symbol", "stock_info")
    op.drop_index("idx_price_history_date", "price_history")
    op.drop_index("idx_dividends_date", "dividends")
    op.drop_index("idx_splits_date", "splits")
