"""
Database connection management following SRP.
Handles only connection setup, configuration, and lifecycle management.
"""

import logging
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlmodel import SQLModel

from ..config.models import TimescaleDBConfig

logger = logging.getLogger(__name__)


class DatabaseConnectionManager:
    """Manages database connections and engine lifecycle - Single Responsibility: Connection Management."""

    def __init__(self, config: TimescaleDBConfig, enable_async: bool = True):
        """Initialize connection manager with configuration.

        Args:
            config: TimescaleDB configuration
            enable_async: Whether to enable async connections
        """
        if config is None:
            raise ValueError("TimescaleDB configuration is required")

        self.config = config
        self.enable_async = enable_async

        # Build connection URLs
        self.db_url = self._build_connection_url()
        self.async_db_url = self._build_async_connection_url() if enable_async else None

        # Initialize engines and session factories
        self._setup_engines()

    def _build_connection_url(self) -> str:
        """Build synchronous connection URL."""
        return (
            f"postgresql://{self.config.user}:{self.config.password}"
            f"@{self.config.host}:{self.config.port}/{self.config.database}"
        )

    def _build_async_connection_url(self) -> str:
        """Build asynchronous connection URL."""
        return (
            f"postgresql+asyncpg://{self.config.user}:{self.config.password}"
            f"@{self.config.host}:{self.config.port}/{self.config.database}"
        )

    def _setup_engines(self) -> None:
        """Setup database engines and session factories."""
        # Synchronous engine
        self.engine = create_engine(
            self.db_url,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle,
            echo=self.config.echo,
        )

        self.session_maker = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
        )

        # Asynchronous engine (if enabled)
        if self.enable_async and self.async_db_url:
            self.async_engine = create_async_engine(
                self.async_db_url,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=self.config.echo,
            )

            self.async_session_maker = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                autocommit=False,
                autoflush=False,
            )
        else:
            self.async_engine = None
            self.async_session_maker = None

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a synchronous database session.

        Yields:
            Database session
        """
        session = self.session_maker()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an asynchronous database session.

        Yields:
            Async database session
        """
        if not self.async_session_maker:
            raise RuntimeError("Async sessions not enabled")

        session = self.async_session_maker()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    def test_connection(self) -> bool:
        """Test database connectivity.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
                logger.info("Database connection test successful")
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    def test_timescale_extension(self) -> bool:
        """Test TimescaleDB extension availability.

        Returns:
            True if TimescaleDB extension is available
        """
        try:
            with self.get_session() as session:
                result = session.execute(
                    text("SELECT COUNT(*) FROM pg_extension WHERE extname = 'timescaledb'")
                ).scalar()
                return bool(result and result > 0)
        except Exception as e:
            logger.error(f"TimescaleDB extension check failed: {e}")
            return False

    def create_tables(self) -> None:
        """Create database tables."""
        SQLModel.metadata.create_all(self.engine)
        logger.info("Database tables created successfully")

    def close(self) -> None:
        """Close database connections."""
        if hasattr(self, "engine") and self.engine:
            self.engine.dispose()
            logger.info("Synchronous database engine disposed")

        if hasattr(self, "async_engine") and self.async_engine:
            # Note: async engine disposal should be handled in async context
            logger.info("Async database engine marked for disposal")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor - ensure cleanup."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup
