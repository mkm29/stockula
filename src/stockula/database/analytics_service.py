"""
Database analytics service following SRP.
Handles only analytical calculations and computations.
"""

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import select, text

from .connection_manager import DatabaseConnectionManager
from .models import PriceHistory

logger = logging.getLogger(__name__)


class DatabaseAnalyticsService:
    """Service for database analytics and calculations - Single Responsibility: Analytics."""

    def __init__(self, connection_manager: DatabaseConnectionManager):
        """Initialize analytics service with connection manager.

        Args:
            connection_manager: Database connection manager
        """
        self.connection_manager = connection_manager

    def _date_to_string(self, date_value: date | str) -> str:
        """Convert date to string format for SQL queries.

        Args:
            date_value: Date object or string

        Returns:
            String representation of date
        """
        if isinstance(date_value, str):
            return date_value
        return date_value.strftime("%Y-%m-%d")

    def get_moving_averages(
        self,
        symbol: str,
        windows: List[int],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Calculate moving averages using TimescaleDB time_bucket.

        Args:
            symbol: Stock symbol
            windows: List of window sizes for moving averages
            start_date: Start date filter
            end_date: End date filter

        Returns:
            DataFrame with moving averages
        """
        with self.connection_manager.get_session() as session:
            # Build dynamic SQL for multiple moving averages
            ma_selects = []
            for window in windows:
                ma_selects.append(
                    f"AVG(close) OVER (ORDER BY timestamp ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW) as ma_{window}"
                )

            ma_sql = ", ".join(ma_selects)

            base_query = f"""
                SELECT
                    timestamp,
                    close,
                    {ma_sql}
                FROM price_history
                WHERE symbol = :symbol
            """

            params = {"symbol": symbol}

            if start_date:
                base_query += " AND timestamp >= :start_date"
                params["start_date"] = self._date_to_string(start_date)
            if end_date:
                base_query += " AND timestamp <= :end_date"
                params["end_date"] = self._date_to_string(end_date)

            base_query += " ORDER BY timestamp"

            result = session.execute(text(base_query), params).fetchall()

            if not result:
                return pd.DataFrame()

            # Convert to DataFrame
            data = []
            for row in result:
                row_data = {
                    "timestamp": row.timestamp,
                    "close": row.close,
                }
                for i, window in enumerate(windows):
                    row_data[f"MA_{window}"] = getattr(row, f"ma_{window}")
                data.append(row_data)

            df = pd.DataFrame(data)
            df.set_index("timestamp", inplace=True)
            return df

    def get_bollinger_bands(
        self,
        symbol: str,
        period: int = 20,
        std_dev: float = 2.0,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands.

        Args:
            symbol: Stock symbol
            period: Period for moving average
            std_dev: Standard deviation multiplier
            start_date: Start date filter
            end_date: End date filter

        Returns:
            DataFrame with Bollinger Bands
        """
        with self.connection_manager.get_session() as session:
            sql = f"""
                SELECT
                    timestamp,
                    close,
                    AVG(close) OVER (ORDER BY timestamp ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW) as middle_band,
                    STDDEV(close) OVER (ORDER BY timestamp ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW) as std_dev
                FROM price_history
                WHERE symbol = :symbol
            """

            params = {"symbol": symbol}

            if start_date:
                sql += " AND timestamp >= :start_date"
                params["start_date"] = self._date_to_string(start_date)
            if end_date:
                sql += " AND timestamp <= :end_date"
                params["end_date"] = self._date_to_string(end_date)

            sql += " ORDER BY timestamp"

            result = session.execute(text(sql), params).fetchall()

            if not result:
                return pd.DataFrame()

            # Calculate upper and lower bands
            data = []
            for row in result:
                if row.std_dev is not None:
                    upper_band = row.middle_band + (std_dev * row.std_dev)
                    lower_band = row.middle_band - (std_dev * row.std_dev)
                else:
                    upper_band = None
                    lower_band = None

                data.append(
                    {
                        "timestamp": row.timestamp,
                        "close": row.close,
                        "upper_band": upper_band,
                        "middle_band": row.middle_band,
                        "lower_band": lower_band,
                    }
                )

            df = pd.DataFrame(data)
            df.set_index("timestamp", inplace=True)
            return df

    def get_rsi(
        self,
        symbol: str,
        period: int = 14,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Calculate Relative Strength Index (RSI).

        Args:
            symbol: Stock symbol
            period: RSI period
            start_date: Start date filter
            end_date: End date filter

        Returns:
            DataFrame with RSI values
        """
        with self.connection_manager.get_session() as session:
            # Calculate price changes and gains/losses
            sql = f"""
                WITH price_changes AS (
                    SELECT
                        timestamp,
                        close,
                        close - LAG(close) OVER (ORDER BY timestamp) as price_change
                    FROM price_history
                    WHERE symbol = :symbol
                ),
                gains_losses AS (
                    SELECT
                        timestamp,
                        close,
                        CASE WHEN price_change > 0 THEN price_change ELSE 0 END as gain,
                        CASE WHEN price_change < 0 THEN ABS(price_change) ELSE 0 END as loss
                    FROM price_changes
                ),
                avg_gains_losses AS (
                    SELECT
                        timestamp,
                        close,
                        AVG(gain) OVER (ORDER BY timestamp ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW) as avg_gain,
                        AVG(loss) OVER (ORDER BY timestamp ROWS BETWEEN {period - 1} PRECEDING AND CURRENT ROW) as avg_loss
                    FROM gains_losses
                )
                SELECT
                    timestamp,
                    close,
                    avg_gain,
                    avg_loss,
                    CASE
                        WHEN avg_loss = 0 THEN 100
                        ELSE 100 - (100 / (1 + (avg_gain / avg_loss)))
                    END as rsi
                FROM avg_gains_losses
            """

            params = {"symbol": symbol}

            if start_date:
                sql += " WHERE timestamp >= :start_date"
                params["start_date"] = self._date_to_string(start_date)
            if end_date:
                if start_date:
                    sql += " AND timestamp <= :end_date"
                else:
                    sql += " WHERE timestamp <= :end_date"
                params["end_date"] = self._date_to_string(end_date)

            sql += " ORDER BY timestamp"

            result = session.execute(text(sql), params).fetchall()

            if not result:
                return pd.DataFrame()

            data = []
            for row in result:
                data.append(
                    {
                        "timestamp": row.timestamp,
                        "close": row.close,
                        "rsi": row.rsi,
                    }
                )

            df = pd.DataFrame(data)
            df.set_index("timestamp", inplace=True)
            return df

    def get_price_momentum(
        self,
        symbol: str,
        periods: List[int],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Calculate price momentum for multiple periods.

        Args:
            symbol: Stock symbol
            periods: List of periods for momentum calculation
            start_date: Start date filter
            end_date: End date filter

        Returns:
            DataFrame with momentum values
        """
        with self.connection_manager.get_session() as session:
            # Build dynamic SQL for multiple momentum periods
            momentum_selects = []
            for period in periods:
                momentum_selects.append(
                    f"(close / LAG(close, {period}) OVER (ORDER BY timestamp) - 1) * 100 as momentum_{period}d"
                )

            momentum_sql = ", ".join(momentum_selects)

            sql = f"""
                SELECT
                    timestamp,
                    close,
                    {momentum_sql}
                FROM price_history
                WHERE symbol = :symbol
            """

            params = {"symbol": symbol}

            if start_date:
                sql += " AND timestamp >= :start_date"
                params["start_date"] = self._date_to_string(start_date)
            if end_date:
                sql += " AND timestamp <= :end_date"
                params["end_date"] = self._date_to_string(end_date)

            sql += " ORDER BY timestamp"

            result = session.execute(text(sql), params).fetchall()

            if not result:
                return pd.DataFrame()

            data = []
            for row in result:
                row_data = {
                    "timestamp": row.timestamp,
                    "close": row.close,
                }
                for period in periods:
                    row_data[f"momentum_{period}d"] = getattr(row, f"momentum_{period}d")
                data.append(row_data)

            df = pd.DataFrame(data)
            df.set_index("timestamp", inplace=True)
            return df

    def get_correlation_matrix(
        self,
        symbols: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Calculate correlation matrix between symbols.

        Args:
            symbols: List of stock symbols
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Correlation matrix DataFrame
        """
        if len(symbols) < 2:
            raise ValueError("At least 2 symbols required for correlation analysis")

        # Ensure symbols is a list for SQLAlchemy
        symbols_list = list(symbols) if not isinstance(symbols, list) else symbols

        with self.connection_manager.get_session() as session:
            # Get price data for all symbols
            from sqlalchemy import or_

            # Create OR condition for symbols to avoid mypy issues with .in_()
            symbol_conditions = [PriceHistory.symbol == symbol for symbol in symbols_list]
            query = select(PriceHistory.symbol, PriceHistory.timestamp, PriceHistory.close).where(
                or_(*symbol_conditions)
            )

            if start_date:
                query = query.where(PriceHistory.timestamp >= start_date)
            if end_date:
                query = query.where(PriceHistory.timestamp <= end_date)

            query = query.order_by(PriceHistory.timestamp)

            result = session.execute(query).fetchall()

            if not result:
                return pd.DataFrame()

            # Pivot data to have symbols as columns
            data: Dict[date, Dict[str, float]] = {}
            for row in result:
                if row.timestamp not in data:
                    data[row.timestamp] = {}
                data[row.timestamp][row.symbol] = row.close

            df = pd.DataFrame.from_dict(data, orient="index")
            df = df.reindex(columns=symbols_list)  # Ensure column order

            # Calculate correlation matrix
            correlation_matrix = df.corr()
            return correlation_matrix

    def get_volatility_analysis(
        self,
        symbol: str,
        windows: List[int],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Calculate volatility analysis for multiple windows.

        Args:
            symbol: Stock symbol
            windows: List of window sizes for volatility calculation
            start_date: Start date filter
            end_date: End date filter

        Returns:
            DataFrame with volatility metrics
        """
        with self.connection_manager.get_session() as session:
            # Calculate daily returns first
            returns_sql = """
                WITH daily_returns AS (
                    SELECT
                        timestamp,
                        close,
                        (close / LAG(close) OVER (ORDER BY timestamp) - 1) as daily_return
                    FROM price_history
                    WHERE symbol = :symbol
                )
            """

            # Build volatility calculations for different windows
            volatility_selects = []
            for window in windows:
                volatility_selects.append(
                    f"STDDEV(daily_return) OVER (ORDER BY timestamp ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW) * SQRT(252) as volatility_{window}d"
                )

            volatility_sql = ", ".join(volatility_selects)

            sql = f"""
                {returns_sql}
                SELECT
                    timestamp,
                    close,
                    daily_return,
                    {volatility_sql}
                FROM daily_returns
            """

            params = {"symbol": symbol}

            if start_date:
                sql += " WHERE timestamp >= :start_date"
                params["start_date"] = self._date_to_string(start_date)
            if end_date:
                if start_date:
                    sql += " AND timestamp <= :end_date"
                else:
                    sql += " WHERE timestamp <= :end_date"
                params["end_date"] = self._date_to_string(end_date)

            sql += " ORDER BY timestamp"

            result = session.execute(text(sql), params).fetchall()

            if not result:
                return pd.DataFrame()

            data = []
            for row in result:
                row_data = {
                    "timestamp": row.timestamp,
                    "close": row.close,
                    "daily_return": row.daily_return,
                }
                for window in windows:
                    row_data[f"volatility_{window}d"] = getattr(row, f"volatility_{window}d")
                data.append(row_data)

            df = pd.DataFrame(data)
            df.set_index("timestamp", inplace=True)
            return df

    def get_seasonal_patterns(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Analyze seasonal patterns in price data.

        Args:
            symbol: Stock symbol
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Dictionary with seasonal analysis results
        """
        with self.connection_manager.get_session() as session:
            sql = """
                SELECT
                    timestamp,
                    close,
                    (close / LAG(close) OVER (ORDER BY timestamp) - 1) * 100 as daily_return,
                    EXTRACT(month FROM timestamp) as month,
                    EXTRACT(dow FROM timestamp) as day_of_week,
                    EXTRACT(quarter FROM timestamp) as quarter
                FROM price_history
                WHERE symbol = :symbol
            """

            params = {"symbol": symbol}

            if start_date:
                sql += " AND timestamp >= :start_date"
                params["start_date"] = self._date_to_string(start_date)
            if end_date:
                sql += " AND timestamp <= :end_date"
                params["end_date"] = self._date_to_string(end_date)

            sql += " ORDER BY timestamp"

            result = session.execute(text(sql), params).fetchall()

            if not result:
                return {}

            df = pd.DataFrame(
                [
                    {
                        "timestamp": row.timestamp,
                        "close": row.close,
                        "daily_return": row.daily_return,
                        "month": row.month,
                        "day_of_week": row.day_of_week,
                        "quarter": row.quarter,
                    }
                    for row in result
                ]
            )

            # Calculate seasonal aggregations
            monthly_stats = df.groupby("month")["daily_return"].agg(["mean", "std", "count"])
            weekly_stats = df.groupby("day_of_week")["daily_return"].agg(["mean", "std", "count"])
            quarterly_stats = df.groupby("quarter")["daily_return"].agg(["mean", "std", "count"])

            return {
                "monthly": monthly_stats,
                "weekly": weekly_stats,
                "quarterly": quarterly_stats,
            }

    def get_top_performers(
        self,
        period_days: int = 30,
        limit: int = 10,
    ) -> pd.DataFrame:
        """Get top performing symbols over a period.

        Args:
            period_days: Period in days for performance calculation
            limit: Maximum number of results to return

        Returns:
            DataFrame with top performers
        """
        with self.connection_manager.get_session() as session:
            cutoff_date = date.today() - timedelta(days=period_days)

            sql = """
                WITH symbol_performance AS (
                    SELECT
                        symbol,
                        (MAX(close) / MIN(close) - 1) * 100 as performance_pct,
                        COUNT(*) as data_points,
                        MIN(timestamp) as start_date,
                        MAX(timestamp) as end_date
                    FROM price_history
                    WHERE timestamp >= :cutoff_date
                    GROUP BY symbol
                    HAVING COUNT(*) >= :min_data_points
                )
                SELECT
                    symbol,
                    performance_pct,
                    data_points,
                    start_date,
                    end_date
                FROM symbol_performance
                ORDER BY performance_pct DESC
                LIMIT :limit
            """

            result = session.execute(
                text(sql),
                {
                    "cutoff_date": cutoff_date,
                    "min_data_points": period_days // 2,  # Require at least half the days
                    "limit": limit,
                },
            ).fetchall()

            if not result:
                return pd.DataFrame()

            data = []
            for row in result:
                data.append(
                    {
                        "symbol": row.symbol,
                        "performance_pct": row.performance_pct,
                        "data_points": row.data_points,
                        "start_date": row.start_date,
                        "end_date": row.end_date,
                    }
                )

            return pd.DataFrame(data)
