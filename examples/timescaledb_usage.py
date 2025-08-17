#!/usr/bin/env python3
"""Example usage of TimescaleDB functionality in Stockula.

This script demonstrates how to:
1. Configure TimescaleDB connection
2. Migrate data from SQLite to TimescaleDB
3. Use advanced TimescaleDB features
4. Perform time-series queries
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stockula.config.database_config import (
    EnhancedTimescaleDBConfig,
    UnifiedDatabaseConfig,
    create_production_config,
)
from stockula.config.models import TimescaleDBConfig
from stockula.data.timescale_repository import create_stock_data_repository
from stockula.database.batch_transfer import BatchDataTransfer
from stockula.database.manager import DatabaseManager
from stockula.database.timescale_connection import TimescaleConnectionManager
from stockula.database.timescale_manager import TimescaleDatabaseManager


def example_1_basic_configuration():
    """Example 1: Basic TimescaleDB configuration."""
    print("=== Example 1: Basic Configuration ===")

    # Create TimescaleDB configuration
    timescale_config = TimescaleDBConfig(
        host="localhost",
        port=5432,
        database="stockula",
        user="postgres",
        password="your_password",
        enable_hypertables=True,
        compression_enabled=True,
    )

    print(f"Connection URL: {timescale_config.get_connection_url()}")
    print(f"Async URL: {timescale_config.get_connection_url(async_driver=True)}")

    # Test connection
    connection_manager = TimescaleConnectionManager(timescale_config)
    connection_info = connection_manager.test_connection()

    print("Connection test results:")
    for key, value in connection_info.items():
        print(f"  {key}: {value}")

    connection_manager.close()


def example_2_unified_configuration():
    """Example 2: Unified configuration with automatic backend selection."""
    print("\n=== Example 2: Unified Configuration ===")

    # Create unified configuration
    config = UnifiedDatabaseConfig(
        timescaledb=EnhancedTimescaleDBConfig(
            host="localhost",
            database="stockula",
            user="postgres",
            password="your_password",
        )
    )

    # Get effective backend
    backend = config.get_effective_backend()
    print(f"Effective backend: {backend}")

    # Get connection configuration
    conn_config = config.get_connection_config()
    print(f"Connection config: {conn_config['backend']}")

    # Create repository with automatic backend selection
    repository = create_stock_data_repository(timescale_config=config.timescaledb, sqlite_db_path="stockula.db")

    backend_info = repository.get_backend_info()
    print("Backend info:")
    for key, value in backend_info.items():
        print(f"  {key}: {value}")

    repository.close()


def example_3_data_migration():
    """Example 3: Migrate data from SQLite to TimescaleDB."""
    print("\n=== Example 3: Data Migration ===")

    # Source (SQLite) configuration
    sqlite_manager = DatabaseManager("stockula.db")

    # Target (TimescaleDB) configuration
    timescale_config = TimescaleDBConfig(
        host="localhost",
        database="stockula",
        user="postgres",
        password="your_password",
    )
    timescale_manager = TimescaleDatabaseManager(timescale_config)

    # Create batch transfer utility
    transfer = BatchDataTransfer(
        source_manager=sqlite_manager,
        target_manager=timescale_manager,
        batch_size=5000,
        parallel_workers=2,
    )

    # Estimate transfer time
    from stockula.database.batch_transfer import estimate_transfer_time

    estimates = estimate_transfer_time(sqlite_manager)

    print("Transfer time estimates:")
    for table, estimate in estimates.items():
        if "error" not in estimate:
            print(
                f"  {table}: {estimate['record_count']:,} records (~{estimate['estimated_time_minutes']:.1f} minutes)"
            )

    # Perform migration (commented out for safety)
    # print("Starting migration...")
    # stats = transfer.transfer_all_tables()
    #
    # print("Migration completed!")
    # for table, stat in stats.items():
    #     print(f"  {table}: {stat.transferred_records:,} records "
    #           f"({stat.records_per_second:.1f} rec/sec)")

    transfer.cleanup()
    sqlite_manager.close()
    timescale_manager.close()


def example_4_advanced_queries():
    """Example 4: Advanced TimescaleDB time-series queries."""
    print("\n=== Example 4: Advanced Queries ===")

    # Create repository with TimescaleDB backend
    timescale_config = TimescaleDBConfig(
        host="localhost",
        database="stockula",
        user="postgres",
        password="your_password",
    )

    repository = create_stock_data_repository(timescale_config=timescale_config)

    if repository.backend_type == "timescaledb":
        print("Using TimescaleDB backend - advanced features available!")

        try:
            # Time-bucketed aggregations
            print("\n1. Daily price aggregations:")
            daily_agg = repository.get_price_history_aggregated(
                symbol="AAPL", time_bucket="1 day", start_date="2024-01-01", end_date="2024-01-31"
            )
            print(f"   Retrieved {len(daily_agg)} daily aggregations")
            if not daily_agg.empty:
                print(f"   Columns: {list(daily_agg.columns)}")

            # Recent price changes
            print("\n2. Recent price changes (last 24 hours):")
            price_changes = repository.get_recent_price_changes(symbols=["AAPL", "GOOGL", "MSFT"], hours=24)
            print(f"   Found {len(price_changes)} symbols with recent changes")
            if not price_changes.empty:
                for _, row in price_changes.iterrows():
                    print(f"   {row['symbol']}: {row['price_change_pct']:.2f}% change")

            # Volume profile analysis
            print("\n3. Volume profile for AAPL:")
            volume_profile = repository.get_volume_profile(
                symbol="AAPL", start_date="2024-01-01", end_date="2024-01-31", price_bins=10
            )
            print(f"   Generated {len(volume_profile)} price bins")

            # Chunk statistics
            print("\n4. TimescaleDB chunk statistics:")
            chunk_stats = repository.get_chunk_statistics()
            print(f"   Found {len(chunk_stats)} chunks")
            if not chunk_stats.empty:
                total_size = chunk_stats["uncompressed_chunk_size"].sum()
                compressed_size = chunk_stats["compressed_chunk_size"].sum()
                compression_ratio = (1 - compressed_size / total_size) * 100 if total_size > 0 else 0
                print(f"   Total size: {total_size / 1024 / 1024:.1f} MB")
                print(f"   Compressed size: {compressed_size / 1024 / 1024:.1f} MB")
                print(f"   Compression ratio: {compression_ratio:.1f}%")

        except Exception as e:
            print(f"   Error performing advanced queries: {e}")

    else:
        print("Using SQLite backend - advanced features not available")

        # Basic queries still work
        try:
            print("\n1. Basic price history:")
            price_data = repository.get_price_history(symbol="AAPL", start_date="2024-01-01", end_date="2024-01-10")
            print(f"   Retrieved {len(price_data)} price records")

            print("\n2. Database statistics:")
            stats = repository.get_database_stats()
            for table, count in stats.items():
                print(f"   {table}: {count:,} records")

        except Exception as e:
            print(f"   Error performing basic queries: {e}")

    repository.close()


def example_5_environment_configuration():
    """Example 5: Configuration from environment variables."""
    print("\n=== Example 5: Environment Configuration ===")

    # Set environment variables (in practice, these would be set externally)
    import os

    os.environ["STOCKULA_DB_BACKEND"] = "timescaledb"
    os.environ["STOCKULA_TS_HOST"] = "localhost"
    os.environ["STOCKULA_TS_DATABASE"] = "stockula"
    os.environ["STOCKULA_TS_USER"] = "postgres"
    os.environ["STOCKULA_TS_PASSWORD"] = "your_password"

    # Create configuration from environment
    config = UnifiedDatabaseConfig.from_environment()
    print(f"Backend from environment: {config.backend.preferred_backend}")

    if config.timescaledb:
        print(f"TimescaleDB host: {config.timescaledb.host}")
        print(f"TimescaleDB database: {config.timescaledb.database}")

    # Production configuration
    prod_config = create_production_config()
    print(f"Production backend: {prod_config.backend.preferred_backend}")
    print(f"Connection pooling: {prod_config.backend.enable_connection_pooling}")


def example_6_migration_script_usage():
    """Example 6: Using the migration script."""
    print("\n=== Example 6: Migration Script Usage ===")

    print("To run the migration script manually:")
    print()
    print("# Basic migration:")
    print("python scripts/migrate_to_timescale.py \\")
    print("  --sqlite-path stockula.db \\")
    print("  --ts-host localhost \\")
    print("  --ts-database stockula \\")
    print("  --ts-user postgres \\")
    print("  --ts-password your_password")
    print()
    print("# Dry run (validation only):")
    print("python scripts/migrate_to_timescale.py --dry-run")
    print()
    print("# Large batch size for faster migration:")
    print("python scripts/migrate_to_timescale.py \\")
    print("  --batch-size 50000 \\")
    print("  --ts-host localhost \\")
    print("  --ts-database stockula \\")
    print("  --ts-user postgres \\")
    print("  --ts-password your_password")


def main():
    """Run all examples."""
    print("Stockula TimescaleDB Usage Examples")
    print("=" * 50)

    try:
        example_1_basic_configuration()
        example_2_unified_configuration()
        example_3_data_migration()
        example_4_advanced_queries()
        example_5_environment_configuration()
        example_6_migration_script_usage()

        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")
        print("\nNext steps:")
        print("1. Set up your TimescaleDB instance")
        print("2. Configure connection parameters")
        print("3. Run migration script to transfer data")
        print("4. Start using advanced time-series features!")

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("Note: Some examples require a running TimescaleDB instance")


if __name__ == "__main__":
    main()
