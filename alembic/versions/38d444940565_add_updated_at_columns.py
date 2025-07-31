"""add_updated_at_columns

Revision ID: 38d444940565
Revises: 622840d504bf
Create Date: 2025-07-28 01:19:12.414202

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op  # type: ignore[attr-defined]

# revision identifiers, used by Alembic.
revision: str = "38d444940565"
down_revision: str | Sequence[str] | None = "622840d504bf"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add updated_at columns to tables that are missing them."""
    # Check if columns exist before adding them to avoid duplicate column errors
    from sqlalchemy import inspect

    connection = op.get_bind()
    inspector = inspect(connection)

    # Helper function to check if column exists
    def column_exists(table_name, column_name):
        columns = inspector.get_columns(table_name)
        return any(col["name"] == column_name for col in columns)

    # Add updated_at column to price_history if it doesn't exist
    if not column_exists("price_history", "updated_at"):
        op.add_column(
            "price_history",
            sa.Column(
                "updated_at",
                sa.DateTime(),
                server_default=sa.func.current_timestamp(),
                nullable=False,
            ),
        )

    # Add updated_at column to dividends if it doesn't exist
    if not column_exists("dividends", "updated_at"):
        op.add_column(
            "dividends",
            sa.Column(
                "updated_at",
                sa.DateTime(),
                server_default=sa.func.current_timestamp(),
                nullable=False,
            ),
        )

    # Add updated_at column to splits if it doesn't exist
    if not column_exists("splits", "updated_at"):
        op.add_column(
            "splits",
            sa.Column(
                "updated_at",
                sa.DateTime(),
                server_default=sa.func.current_timestamp(),
                nullable=False,
            ),
        )

    # Add updated_at column to options_calls if it doesn't exist
    if not column_exists("options_calls", "updated_at"):
        op.add_column(
            "options_calls",
            sa.Column(
                "updated_at",
                sa.DateTime(),
                server_default=sa.func.current_timestamp(),
                nullable=False,
            ),
        )

    # Add updated_at column to options_puts if it doesn't exist
    if not column_exists("options_puts", "updated_at"):
        op.add_column(
            "options_puts",
            sa.Column(
                "updated_at",
                sa.DateTime(),
                server_default=sa.func.current_timestamp(),
                nullable=False,
            ),
        )

    # stocks and stock_info already have updated_at columns


def downgrade() -> None:
    """Remove updated_at columns."""
    # Don't remove from stocks and stock_info as they already had the columns
    op.drop_column("options_puts", "updated_at")
    op.drop_column("options_calls", "updated_at")
    op.drop_column("splits", "updated_at")
    op.drop_column("dividends", "updated_at")
    op.drop_column("price_history", "updated_at")
