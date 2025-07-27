"""Command-line interface for database operations."""

import argparse
import sys
from typing import List
from ..data.fetcher import DataFetcher
from .manager import DatabaseManager


def fetch_symbol_data(
    symbol: str, start_date: str = None, end_date: str = None
) -> None:
    """Fetch and store all data for a symbol."""
    fetcher = DataFetcher(use_cache=True)
    fetcher.fetch_and_store_all_data(symbol, start_date, end_date)


def fetch_portfolio_data(
    symbols: List[str], start_date: str = None, end_date: str = None
) -> None:
    """Fetch and store data for a portfolio of symbols."""
    fetcher = DataFetcher(use_cache=True)

    print(f"Fetching data for {len(symbols)} symbols...")
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Processing {symbol}")
        fetcher.fetch_and_store_all_data(symbol, start_date, end_date)

    print(f"\nCompleted fetching data for all {len(symbols)} symbols")


def show_database_stats() -> None:
    """Display database statistics."""
    fetcher = DataFetcher(use_cache=True)
    stats = fetcher.get_database_stats()

    print(f"""Database Statistics:
{"=" * 40}""")
    for table, count in stats.items():
        print(f"{table:15}: {count:,} records")

    # Show cached symbols
    symbols = fetcher.get_cached_symbols()
    print(f"\nCached symbols ({len(symbols)}):")
    if symbols:
        for i, symbol in enumerate(sorted(symbols)):
            print(f"{symbol}", end="  ")
            if (i + 1) % 8 == 0:  # New line every 8 symbols
                print()
        if len(symbols) % 8 != 0:
            print()  # Final newline if needed
    else:
        print("No symbols cached yet")


def cleanup_database(days: int = 365) -> None:
    """Clean up old database records."""
    fetcher = DataFetcher(use_cache=True)
    print(f"Cleaning up data older than {days} days...")
    fetcher.cleanup_old_data(days)
    print("Cleanup completed")


def query_symbol_data(symbol: str) -> None:
    """Query and display data for a specific symbol."""
    db = DatabaseManager()

    print(f"""Data for {symbol}:
{"=" * 40}""")

    # Stock info
    info = db.get_stock_info(symbol)
    if info:
        print(f"""Name: {info.get("longName", "N/A")}
Sector: {info.get("sector", "N/A")}
{f"Market Cap: ${info.get('marketCap', 0):,}" if info.get("marketCap") else "Market Cap: N/A"}
Exchange: {info.get("exchange", "N/A")}""")

    # Price history
    price_data = db.get_price_history(symbol)
    if not price_data.empty:
        latest_price = price_data["Close"].iloc[-1]
        print(f"Latest Price: ${latest_price:.2f}")
        print(
            f"Price Data: {len(price_data)} records from {price_data.index[0].date()} to {price_data.index[-1].date()}"
        )
    else:
        print("No price data available")

    # Dividends
    dividends = db.get_dividends(symbol)
    if not dividends.empty:
        total_dividends = dividends.sum()
        print(f"Dividends: {len(dividends)} payments, total ${total_dividends:.2f}")
    else:
        print("No dividend data available")

    # Splits
    splits = db.get_splits(symbol)
    if not splits.empty:
        print(f"Stock Splits: {len(splits)} splits")
    else:
        print("No split data available")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Stockula Database CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Fetch command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch data for symbol(s)")
    fetch_parser.add_argument("symbols", nargs="+", help="Stock symbols to fetch")
    fetch_parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    fetch_parser.add_argument("--end", help="End date (YYYY-MM-DD)")

    # Stats command
    subparsers.add_parser("stats", help="Show database statistics")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query data for a symbol")
    query_parser.add_argument("symbol", help="Stock symbol to query")

    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old data")
    cleanup_parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Keep data for this many days (default: 365)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "fetch":
            if len(args.symbols) == 1:
                fetch_symbol_data(args.symbols[0], args.start, args.end)
            else:
                fetch_portfolio_data(args.symbols, args.start, args.end)

        elif args.command == "stats":
            show_database_stats()

        elif args.command == "query":
            query_symbol_data(args.symbol)

        elif args.command == "cleanup":
            cleanup_database(args.days)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
