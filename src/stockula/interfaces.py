"""Interfaces/protocols for dependency injection."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


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
    """Interface for database operations."""

    @abstractmethod
    def get_price_history(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Get price history from database."""
        pass

    @abstractmethod
    def store_price_history(self, symbol: str, data: pd.DataFrame, interval: str = "1d") -> None:
        """Store price history in database."""
        pass

    @abstractmethod
    def has_data(self, symbol: str, start_date: str, end_date: str) -> bool:
        """Check if data exists for date range."""
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
    def create_portfolio(self) -> Any:
        """Create portfolio from configuration."""
        pass

    @abstractmethod
    def create_asset(self, ticker_data: dict[str, Any]) -> Any:
        """Create asset from ticker data."""
        pass
