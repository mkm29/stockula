"""Pure TimescaleDB interfaces for dependency injection.

This module contains interfaces optimized specifically for TimescaleDB
operations, removing all backward compatibility layers and focusing
on time-series database capabilities.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from stockula.domain.models import Asset, Portfolio


class IDataFetcher(ABC):
    """Interface for data fetching operations."""

    @abstractmethod
    def get_stock_data(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch historical stock data."""
        pass

    @abstractmethod
    def get_current_prices(self, symbols: list[str] | str, show_progress: bool = True) -> dict[str, float]:
        """Get current prices for symbols."""
        pass

    @abstractmethod
    def get_info(self, symbol: str, force_refresh: bool = False) -> dict[str, Any]:
        """Get stock information."""
        pass

    @abstractmethod
    def get_treasury_rates(
        self,
        start_date: str,
        end_date: str,
        duration: str = "3_month",
        force_refresh: bool = False,
        as_decimal: bool = True,
    ) -> pd.Series:
        """Get Treasury rates for a date range."""
        pass


class IDatabaseManager(ABC):
    """Pure TimescaleDB interface for database operations.

    Optimized for time-series data with TimescaleDB-specific capabilities.
    All methods assume TimescaleDB backend with hypertables and time-series
    optimization features.
    """

    # Core time-series operations
    @abstractmethod
    def get_price_history(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Get price history using TimescaleDB optimized queries.

        Leverages hypertable indexing and time-based partitioning
        for efficient time-series data retrieval.
        """
        pass

    @abstractmethod
    def store_price_history(self, symbol: str, data: pd.DataFrame, interval: str = "1d") -> None:
        """Store price history using TimescaleDB bulk insert optimization.

        Uses COPY operations and proper time-based partitioning for
        maximum insert performance on large datasets.
        """
        pass

    @abstractmethod
    def has_data(self, symbol: str, start_date: str, end_date: str) -> bool:
        """Check data existence using TimescaleDB time-based indexing."""
        pass

    # Extended stock data operations
    @abstractmethod
    def store_stock_info(self, symbol: str, info: dict[str, Any]) -> None:
        """Store stock information with JSONB optimization."""
        pass

    @abstractmethod
    def get_stock_info(self, symbol: str) -> dict[str, Any] | None:
        """Retrieve stock information with JSONB field access."""
        pass

    @abstractmethod
    def store_dividends(self, symbol: str, dividends: pd.Series) -> None:
        """Store dividend data in TimescaleDB hypertable."""
        pass

    @abstractmethod
    def store_splits(self, symbol: str, splits: pd.Series) -> None:
        """Store stock split data in TimescaleDB hypertable."""
        pass

    # Portfolio and analytics operations
    @abstractmethod
    def get_all_symbols(self) -> list[str]:
        """Get all symbols using optimized distinct queries."""
        pass

    @abstractmethod
    def get_latest_price(self, symbol: str) -> float | None:
        """Get latest price using TimescaleDB's time-ordered index."""
        pass

    @abstractmethod
    def get_database_stats(self) -> dict[str, int]:
        """Get TimescaleDB-specific database statistics including chunk info."""
        pass

    # TimescaleDB-specific aggregations (optional advanced features)
    def get_price_history_aggregated(
        self,
        symbol: str,
        time_bucket: str = "1 day",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Get aggregated price data using TimescaleDB time_bucket function.

        Args:
            symbol: Stock ticker symbol
            time_bucket: Time bucket size (e.g., '1 day', '1 hour', '15 minutes')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with time-bucketed aggregated price data

        Raises:
            NotImplementedError: If not supported by implementation
        """
        raise NotImplementedError("Time-bucketed aggregations not supported by this implementation")

    def get_recent_price_changes(self, symbols: list[str] | None = None, hours: int = 24) -> pd.DataFrame:
        """Get recent price changes using TimescaleDB window functions.

        Args:
            symbols: List of symbols to analyze (None for all)
            hours: Number of hours to look back

        Returns:
            DataFrame with price change analysis

        Raises:
            NotImplementedError: If not supported by implementation
        """
        raise NotImplementedError("Recent price change analysis not supported by this implementation")

    def get_chunk_statistics(self) -> pd.DataFrame:
        """Get TimescaleDB chunk statistics for performance monitoring.

        Returns:
            DataFrame with chunk statistics including size, compression, etc.

        Raises:
            NotImplementedError: If not supported by implementation
        """
        raise NotImplementedError("Chunk statistics not supported by this implementation")

    # Session and connection management
    @abstractmethod
    def get_session(self):
        """Get database session context manager for direct SQL access."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close TimescaleDB connections and clean up resources."""
        pass

    def test_connection(self) -> dict[str, Any]:
        """Test TimescaleDB connection and return health information.

        Returns:
            Dictionary with connection status and health information

        Raises:
            NotImplementedError: If not supported by implementation
        """
        raise NotImplementedError("Connection testing not supported by this implementation")

    # Advanced Analytics Methods
    @abstractmethod
    def get_moving_averages(
        self,
        symbol: str,
        periods: list[int] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Calculate moving averages using TimescaleDB window functions."""
        pass

    @abstractmethod
    def get_bollinger_bands(
        self,
        symbol: str,
        period: int = 20,
        std_dev: float = 2.0,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands using TimescaleDB window functions."""
        pass

    @abstractmethod
    def get_rsi(
        self,
        symbol: str,
        period: int = 14,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Calculate RSI using TimescaleDB window functions."""
        pass

    @abstractmethod
    def get_price_momentum(
        self,
        symbols: list[str] | None = None,
        lookback_days: int = 30,
        time_bucket: str = "1 day",
    ) -> pd.DataFrame:
        """Calculate price momentum across multiple timeframes."""
        pass

    @abstractmethod
    def get_correlation_matrix(
        self,
        symbols: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        time_bucket: str = "1 day",
    ) -> pd.DataFrame:
        """Calculate correlation matrix between symbols using TimescaleDB."""
        pass

    @abstractmethod
    def get_volatility_analysis(
        self,
        symbols: list[str] | None = None,
        lookback_days: int = 30,
        time_bucket: str = "1 day",
    ) -> pd.DataFrame:
        """Calculate volatility metrics using TimescaleDB statistical functions."""
        pass

    @abstractmethod
    def get_seasonal_patterns(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Analyze seasonal patterns in stock price movements."""
        pass

    @abstractmethod
    def get_top_performers(
        self,
        timeframe_days: int = 30,
        limit: int = 10,
    ) -> pd.DataFrame:
        """Get top performing stocks over a specified timeframe."""
        pass


class ILoggingManager(ABC):
    """Interface for logging operations."""

    @abstractmethod
    def setup(self, config) -> None:
        """Setup logging configuration."""
        pass

    @abstractmethod
    def debug(self, message: str) -> None:
        """Log debug message."""
        pass

    @abstractmethod
    def info(self, message: str) -> None:
        """Log info message."""
        pass

    @abstractmethod
    def warning(self, message: str) -> None:
        """Log warning message."""
        pass

    @abstractmethod
    def isEnabledFor(self, level: int) -> bool:
        """Check if logging is enabled for given level."""
        pass

    @abstractmethod
    def error(self, message: str, exc_info: bool = False) -> None:
        """Log error message."""
        pass

    @abstractmethod
    def critical(self, message: str, exc_info: bool = False) -> None:
        """Log critical message."""
        pass

    @abstractmethod
    def set_module_level(self, module_name: str, level: str) -> None:
        """Set the logging level for a specific module."""
        pass


class ITechnicalIndicators(ABC):
    """Interface for technical analysis operations."""

    @abstractmethod
    def sma(self, period: int = 20) -> pd.Series:
        """Calculate Simple Moving Average."""
        pass

    @abstractmethod
    def ema(self, period: int = 20) -> pd.Series:
        """Calculate Exponential Moving Average."""
        pass

    @abstractmethod
    def rsi(self, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        pass

    @abstractmethod
    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD."""
        pass


class IBacktestRunner(ABC):
    """Interface for backtesting operations."""

    @abstractmethod
    def run(self, data: pd.DataFrame, strategy) -> dict[str, Any]:
        """Run backtest with given data and strategy."""
        pass

    @abstractmethod
    def run_from_symbol(
        self, symbol: str, strategy, start_date: str | None = None, end_date: str | None = None, **kwargs
    ) -> dict[str, Any]:
        """Run backtest for a symbol."""
        pass

    @abstractmethod
    def run_with_train_test_split(
        self,
        symbol: str,
        strategy,
        train_start_date: str | None = None,
        train_end_date: str | None = None,
        test_start_date: str | None = None,
        test_end_date: str | None = None,
        optimize_on_train: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """Run backtest with train/test split."""
        pass


class IStockForecaster(ABC):
    """Interface for forecasting operations."""

    @abstractmethod
    def forecast(self, data: pd.DataFrame, forecast_length: int | None = None) -> pd.DataFrame:
        """Generate forecast from data."""
        pass

    @abstractmethod
    def forecast_from_symbol(
        self,
        symbol: str,
        forecast_length: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        model_list: str | list[str] | None = None,
        ensemble: str | None = None,
        max_generations: int | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate forecast for a symbol."""
        pass

    @abstractmethod
    def forecast_from_symbol_with_evaluation(
        self,
        symbol: str,
        train_start_date: str | None = None,
        train_end_date: str | None = None,
        test_start_date: str | None = None,
        test_end_date: str | None = None,
        target_column: str = "Close",
        **kwargs,
    ) -> dict[str, Any]:
        """Forecast with evaluation on test data."""
        pass

    @abstractmethod
    def get_best_model(self) -> dict[str, Any]:
        """Get information about the best model found."""
        pass


class IDomainFactory(ABC):
    """Interface for domain object creation."""

    @abstractmethod
    def create_portfolio(self, config: Any) -> "Portfolio":
        """Create portfolio from configuration."""
        pass

    @abstractmethod
    def create_asset(self, ticker_config: Any, calculated_quantity: float | None = None) -> "Asset":
        """Create asset from ticker configuration."""
        pass
