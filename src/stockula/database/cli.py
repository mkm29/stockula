"""Command-line interface for database operations."""

import argparse
import sys
from pathlib import Path
from typing import List

from alembic import command
from alembic.config import Config

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


def get_alembic_config() -> Config:
    """Get Alembic configuration."""
    project_root = Path(__file__).parents[3]
    alembic_ini_path = project_root / "alembic.ini"

    if not alembic_ini_path.exists():
        raise FileNotFoundError(f"alembic.ini not found at {alembic_ini_path}")

    return Config(str(alembic_ini_path))


def migrate_upgrade(revision: str = "head") -> None:
    """Run database migrations to the specified revision."""
    config = get_alembic_config()
    print(f"Running migrations to {revision}...")
    command.upgrade(config, revision)
    print("Migrations completed successfully")


def migrate_downgrade(revision: str) -> None:
    """Downgrade database to the specified revision."""
    config = get_alembic_config()
    print(f"Downgrading to {revision}...")
    command.downgrade(config, revision)
    print("Downgrade completed successfully")


def migrate_create(message: str) -> None:
    """Create a new migration."""
    config = get_alembic_config()
    print(f"Creating new migration: {message}")
    command.revision(config, autogenerate=True, message=message)
    print("Migration created successfully")


def migrate_history(verbose: bool = False) -> None:
    """Show migration history."""
    config = get_alembic_config()
    if verbose:
        command.history(config, verbose=True)
    else:
        command.history(config)


def migrate_current() -> None:
    """Show current migration revision."""
    config = get_alembic_config()
    command.current(config)


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

    # Migration commands
    migrate_parser = subparsers.add_parser(
        "migrate", help="Database migration commands"
    )
    migrate_subparsers = migrate_parser.add_subparsers(
        dest="migrate_command", help="Migration commands"
    )

    # migrate upgrade
    upgrade_parser = migrate_subparsers.add_parser(
        "upgrade", help="Upgrade database to a revision"
    )
    upgrade_parser.add_argument(
        "revision", nargs="?", default="head", help="Target revision (default: head)"
    )

    # migrate downgrade
    downgrade_parser = migrate_subparsers.add_parser(
        "downgrade", help="Downgrade database to a revision"
    )
    downgrade_parser.add_argument("revision", help="Target revision")

    # migrate create
    create_parser = migrate_subparsers.add_parser(
        "create", help="Create a new migration"
    )
    create_parser.add_argument("message", help="Migration message")

    # migrate history
    history_parser = migrate_subparsers.add_parser(
        "history", help="Show migration history"
    )
    history_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show verbose output"
    )

    # migrate current
    migrate_subparsers.add_parser("current", help="Show current migration revision")

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

        elif args.command == "migrate":
            if not args.migrate_command:
                migrate_parser.print_help()
                return

            if args.migrate_command == "upgrade":
                migrate_upgrade(args.revision)
            elif args.migrate_command == "downgrade":
                migrate_downgrade(args.revision)
            elif args.migrate_command == "create":
                migrate_create(args.message)
            elif args.migrate_command == "history":
                migrate_history(args.verbose)
            elif args.migrate_command == "current":
                migrate_current()

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
