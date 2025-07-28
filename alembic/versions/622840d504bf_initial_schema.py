"""initial_schema

Revision ID: 622840d504bf
Revises:
Create Date: 2025-07-27 19:23:04.072806

"""

from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "622840d504bf"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Initial schema - tables already exist, so this is a no-op."""
    # The database tables already exist from the legacy schema creation
    # This migration serves as the baseline for future migrations
    pass


def downgrade() -> None:
    """No downgrade for initial schema."""
    # Since this is the initial migration and tables already exist,
    # we don't implement a downgrade
    pass
