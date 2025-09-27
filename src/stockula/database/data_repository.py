"""
Database data repository following SRP.
Handles only CRUD operations for financial data.
"""

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import desc, func, select
from sqlalchemy.dialects.postgresql import insert

from .connection_manager import DatabaseConnectionManager
from .models import Dividend, OptionsCall, OptionsPut, PriceHistory, Split, StockInfo

logger = logging.getLogger(__name__)


class DatabaseDataRepository:
    """Repository for financial data CRUD operations - Single Responsibility: Data Access."""

    def __init__(self, connection_manager: DatabaseConnectionManager):
        """Initialize repository with connection manager.

        Args:
            connection_manager: Database connection manager
        """
        self.connection_manager = connection_manager

    def store_stock_info(self, symbol: str, info_data: Dict[str, Any]) -> None:
        """Store stock information.

        Args:
            symbol: Stock symbol
            info_data: Stock information dictionary
        """
        with self.connection_manager.get_session() as session:
            # Use upsert to handle conflicts
            stmt = insert(StockInfo).values(symbol=symbol, **info_data, updated_at=datetime.utcnow())
            stmt = stmt.on_conflict_do_update(index_elements=[StockInfo.symbol], set_=dict(stmt.excluded))
            session.execute(stmt)
            logger.debug(f"Stored stock info for {symbol}")

    def get_stock_info(self, symbol: str) -> Optional[StockInfo]:
        """Get stock information.

        Args:
            symbol: Stock symbol

        Returns:
            Stock information or None if not found
        """
        with self.connection_manager.get_session() as session:
            result = session.execute(select(StockInfo).where(StockInfo.symbol == symbol)).scalar_one_or_none()
            return result if result is not None else None

    def store_price_history(self, symbol: str, data: pd.DataFrame) -> None:
        """Store price history data.

        Args:
            symbol: Stock symbol
            data: Price data DataFrame with OHLCV columns
        """
        if data.empty:
            logger.warning(f"No price data to store for {symbol}")
            return

        # Prepare data for insertion
        records = []
        for timestamp, row in data.iterrows():
            record = {
                "symbol": symbol,
                "timestamp": timestamp,
                "open": float(row.get("Open", 0)),
                "high": float(row.get("High", 0)),
                "low": float(row.get("Low", 0)),
                "close": float(row.get("Close", 0)),
                "volume": int(row.get("Volume", 0)),
                "adj_close": float(row.get("Adj Close", row.get("Close", 0))),
            }
            records.append(record)

        with self.connection_manager.get_session() as session:
            # Use bulk upsert for performance
            stmt = insert(PriceHistory)
            stmt = stmt.on_conflict_do_update(
                index_elements=[PriceHistory.symbol, PriceHistory.timestamp], set_=dict(stmt.excluded)
            )
            session.execute(stmt, records)
            logger.info(f"Stored {len(records)} price records for {symbol}")

    def get_price_history(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Get price history data.

        Args:
            symbol: Stock symbol
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Price history DataFrame
        """
        with self.connection_manager.get_session() as session:
            query = select(PriceHistory).where(PriceHistory.symbol == symbol)

            if start_date:
                query = query.where(PriceHistory.timestamp >= start_date)
            if end_date:
                query = query.where(PriceHistory.timestamp <= end_date)

            query = query.order_by(PriceHistory.timestamp)

            result = session.execute(query).scalars().all()

            if not result:
                return pd.DataFrame()

            # Convert to DataFrame
            data = []
            for record in result:
                data.append(
                    {
                        "Open": record.open,
                        "High": record.high,
                        "Low": record.low,
                        "Close": record.close,
                        "Volume": record.volume,
                        "Adj Close": record.adj_close,
                    }
                )

            df = pd.DataFrame(data, index=[r.timestamp for r in result])
            df.index.name = "Date"
            return df

    def store_dividends(self, symbol: str, dividends_data: pd.DataFrame) -> None:
        """Store dividend data.

        Args:
            symbol: Stock symbol
            dividends_data: Dividend data DataFrame
        """
        if dividends_data.empty:
            return

        records = []
        for timestamp, amount in dividends_data.items():
            records.append(
                {
                    "symbol": symbol,
                    "ex_date": timestamp,
                    "amount": float(amount),
                }
            )

        with self.connection_manager.get_session() as session:
            stmt = insert(Dividend)
            stmt = stmt.on_conflict_do_update(
                index_elements=[Dividend.symbol, Dividend.ex_date], set_=dict(stmt.excluded)
            )
            session.execute(stmt, records)
            logger.debug(f"Stored {len(records)} dividend records for {symbol}")

    def get_dividends(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Get dividend data.

        Args:
            symbol: Stock symbol
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Dividend data DataFrame
        """
        with self.connection_manager.get_session() as session:
            query = select(Dividend).where(Dividend.symbol == symbol)

            if start_date:
                query = query.where(Dividend.ex_date >= start_date)
            if end_date:
                query = query.where(Dividend.ex_date <= end_date)

            query = query.order_by(Dividend.ex_date)

            result = session.execute(query).scalars().all()

            if not result:
                return pd.DataFrame()

            data = {record.ex_date: record.amount for record in result}
            return pd.Series(data, name="Dividend")

    def store_splits(self, symbol: str, splits_data: pd.DataFrame) -> None:
        """Store stock split data.

        Args:
            symbol: Stock symbol
            splits_data: Stock split data DataFrame
        """
        if splits_data.empty:
            return

        records = []
        for timestamp, ratio in splits_data.items():
            records.append(
                {
                    "symbol": symbol,
                    "split_date": timestamp,
                    "ratio": float(ratio),
                }
            )

        with self.connection_manager.get_session() as session:
            stmt = insert(Split)
            stmt = stmt.on_conflict_do_update(index_elements=[Split.symbol, Split.split_date], set_=dict(stmt.excluded))
            session.execute(stmt, records)
            logger.debug(f"Stored {len(records)} split records for {symbol}")

    def get_splits(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Get stock split data.

        Args:
            symbol: Stock symbol
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Stock split data DataFrame
        """
        with self.connection_manager.get_session() as session:
            query = select(Split).where(Split.symbol == symbol)

            if start_date:
                query = query.where(Split.split_date >= start_date)
            if end_date:
                query = query.where(Split.split_date <= end_date)

            query = query.order_by(Split.split_date)

            result = session.execute(query).scalars().all()

            if not result:
                return pd.DataFrame()

            data = {record.split_date: record.ratio for record in result}
            return pd.Series(data, name="Stock Split")

    def store_options_chain(self, symbol: str, options_data: List[Dict[str, Any]]) -> None:
        """Store options chain data.

        Args:
            symbol: Stock symbol
            options_data: List of options data dictionaries
        """
        if not options_data:
            return

        # Separate calls and puts
        call_records = []
        put_records = []
        current_time = datetime.now()

        for option in options_data:
            # Map old field names to new field names
            record = {
                "symbol": symbol,
                "data_timestamp": current_time,
                "expiration_timestamp": option["expiration_date"]
                if isinstance(option["expiration_date"], datetime)
                else datetime.combine(option["expiration_date"], datetime.min.time()),
                "strike": option["strike_price"],
                "last_price": option.get("last_price"),
                "bid": option.get("bid"),
                "ask": option.get("ask"),
                "volume": option.get("volume"),
                "open_interest": option.get("open_interest"),
                "implied_volatility": option.get("implied_volatility"),
                "contract_symbol": option.get("contract_symbol"),
                "in_the_money": option.get("in_the_money"),
            }

            if option["option_type"].lower() == "call":
                call_records.append(record)
            elif option["option_type"].lower() == "put":
                put_records.append(record)

        with self.connection_manager.get_session() as session:
            # Store call options
            if call_records:
                stmt = insert(OptionsCall)
                stmt = stmt.on_conflict_do_update(
                    index_elements=[
                        OptionsCall.symbol,
                        OptionsCall.expiration_timestamp,
                        OptionsCall.strike,
                        OptionsCall.contract_symbol,
                        OptionsCall.data_timestamp,
                    ],
                    set_=dict(stmt.excluded),
                )
                session.execute(stmt, call_records)
                logger.debug(f"Stored {len(call_records)} call options records for {symbol}")

            # Store put options
            if put_records:
                stmt = insert(OptionsPut)
                stmt = stmt.on_conflict_do_update(
                    index_elements=[
                        OptionsPut.symbol,
                        OptionsPut.expiration_timestamp,
                        OptionsPut.strike,
                        OptionsPut.contract_symbol,
                        OptionsPut.data_timestamp,
                    ],
                    set_=dict(stmt.excluded),
                )
                session.execute(stmt, put_records)
                logger.debug(f"Stored {len(put_records)} put options records for {symbol}")

    def get_options_chain(
        self,
        symbol: str,
        expiration_date: Optional[date] = None,
        option_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get options chain data.

        Args:
            symbol: Stock symbol
            expiration_date: Filter by expiration date
            option_type: Filter by option type ('call' or 'put')

        Returns:
            Options chain DataFrame
        """
        with self.connection_manager.get_session() as session:
            data = []

            # Query call options if not filtering for puts only
            if option_type is None or option_type.lower() == "call":
                call_query = select(OptionsCall).where(OptionsCall.symbol == symbol)

                if expiration_date:
                    expiration_datetime = datetime.combine(expiration_date, datetime.min.time())
                    call_query = call_query.where(OptionsCall.expiration_timestamp >= expiration_datetime)
                    call_query = call_query.where(
                        OptionsCall.expiration_timestamp < expiration_datetime.replace(hour=23, minute=59, second=59)
                    )

                call_query = call_query.order_by(OptionsCall.expiration_timestamp, OptionsCall.strike)
                call_results = session.execute(call_query).scalars().all()

                for record in call_results:
                    data.append(
                        {
                            "expiration_date": record.expiration_timestamp.date(),
                            "strike_price": record.strike,
                            "option_type": "call",
                            "last_price": record.last_price,
                            "bid": record.bid,
                            "ask": record.ask,
                            "volume": record.volume,
                            "open_interest": record.open_interest,
                            "implied_volatility": record.implied_volatility,
                            "contract_symbol": record.contract_symbol,
                            "in_the_money": record.in_the_money,
                        }
                    )

            # Query put options if not filtering for calls only
            if option_type is None or option_type.lower() == "put":
                put_query = select(OptionsPut).where(OptionsPut.symbol == symbol)

                if expiration_date:
                    expiration_datetime = datetime.combine(expiration_date, datetime.min.time())
                    put_query = put_query.where(OptionsPut.expiration_timestamp >= expiration_datetime)
                    put_query = put_query.where(
                        OptionsPut.expiration_timestamp < expiration_datetime.replace(hour=23, minute=59, second=59)
                    )

                put_query = put_query.order_by(OptionsPut.expiration_timestamp, OptionsPut.strike)
                put_results = session.execute(put_query).scalars().all()

                for record in put_results:
                    data.append(
                        {
                            "expiration_date": record.expiration_timestamp.date(),
                            "strike_price": record.strike,
                            "option_type": "put",
                            "last_price": record.last_price,
                            "bid": record.bid,
                            "ask": record.ask,
                            "volume": record.volume,
                            "open_interest": record.open_interest,
                            "implied_volatility": record.implied_volatility,
                            "contract_symbol": record.contract_symbol,
                            "in_the_money": record.in_the_money,
                        }
                    )

            if not data:
                return pd.DataFrame()

            return pd.DataFrame(data)

    def get_all_symbols(self) -> List[str]:
        """Get all unique symbols in the database.

        Returns:
            List of stock symbols
        """
        with self.connection_manager.get_session() as session:
            result = session.execute(select(PriceHistory.symbol).distinct()).scalars().all()
            return list(result)

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Latest close price or None if not found
        """
        with self.connection_manager.get_session() as session:
            result = session.execute(
                select(PriceHistory.close)
                .where(PriceHistory.symbol == symbol)
                .order_by(desc(PriceHistory.timestamp))
                .limit(1)
            ).scalar_one_or_none()
            return float(result) if result is not None else None

    def get_latest_price_date(self, symbol: str) -> Optional[date]:
        """Get the latest price date for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Latest price date or None if not found
        """
        with self.connection_manager.get_session() as session:
            result = session.execute(
                select(PriceHistory.timestamp)
                .where(PriceHistory.symbol == symbol)
                .order_by(desc(PriceHistory.timestamp))
                .limit(1)
            ).scalar_one_or_none()
            return result.date() if result is not None else None

    def has_data(self, symbol: str, start_date: Optional[date] = None) -> bool:
        """Check if data exists for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Optional start date to check from

        Returns:
            True if data exists, False otherwise
        """
        with self.connection_manager.get_session() as session:
            query = select(func.count(PriceHistory.id)).where(PriceHistory.symbol == symbol)

            if start_date:
                query = query.where(PriceHistory.timestamp >= start_date)

            count = session.execute(query).scalar()
            return bool(count and count > 0)

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with database statistics
        """
        with self.connection_manager.get_session() as session:
            symbol_count = session.execute(select(func.count(func.distinct(PriceHistory.symbol)))).scalar()

            total_records = session.execute(select(func.count(PriceHistory.id))).scalar()

            latest_date = session.execute(select(func.max(PriceHistory.timestamp))).scalar()

            earliest_date = session.execute(select(func.min(PriceHistory.timestamp))).scalar()

            return {
                "symbol_count": symbol_count,
                "total_records": total_records,
                "latest_date": latest_date,
                "earliest_date": earliest_date,
            }
