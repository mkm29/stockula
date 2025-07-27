#!/usr/bin/env python3
"""Test script to demonstrate the SQLite database functionality."""

import os
import sys
from datetime import datetime, timedelta

# Add src to path so we can import stockula modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from stockula.data.fetcher import DataFetcher
from stockula.database.manager import DatabaseManager


def test_database_functionality():
    """Test the database functionality with sample data."""
    print("Testing Stockula SQLite Database")
    print("=" * 50)
    
    # Remove existing database for clean test
    db_path = "stockula.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        print("Removed existing database for clean test")
    
    # Initialize fetcher with caching enabled
    fetcher = DataFetcher(use_cache=True, db_path=db_path)
    
    # Test symbols - mix of large cap, dividend paying, and tech stocks
    test_symbols = ["AAPL", "MSFT", "SPY"]
    
    # Date range for testing
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    print(f"\nTesting with symbols: {test_symbols}")
    print(f"Date range: {start_date} to {end_date}")
    print()
    
    # Test 1: Fetch and store data for each symbol
    print("Test 1: Fetching and storing data")
    print("-" * 30)
    for symbol in test_symbols:
        print(f"\nFetching data for {symbol}...")
        try:
            # Fetch price history
            price_data = fetcher.get_stock_data(symbol, start_date, end_date)
            print(f"  Price history: {len(price_data)} records")
            
            # Fetch stock info
            info = fetcher.get_info(symbol)
            company_name = info.get('longName', 'Unknown')
            print(f"  Company: {company_name}")
            
            # Fetch dividends (may be empty for recent data)
            dividends = fetcher.get_dividends(symbol)
            print(f"  Dividends: {len(dividends)} records")
            
            # Fetch splits (may be empty)
            splits = fetcher.get_splits(symbol)
            print(f"  Splits: {len(splits)} records")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Test 2: Database statistics
    print("\n\nTest 2: Database Statistics")
    print("-" * 30)
    stats = fetcher.get_database_stats()
    for table, count in stats.items():
        print(f"{table}: {count} records")
    
    # Test 3: Query cached data (should be fast)
    print("\n\nTest 3: Querying cached data")
    print("-" * 30)
    for symbol in test_symbols:
        print(f"\nQuerying cached data for {symbol}:")
        
        # Get cached price data
        cached_prices = fetcher.get_stock_data(symbol, start_date, end_date)
        if not cached_prices.empty:
            latest_price = cached_prices['Close'].iloc[-1]
            print(f"  Latest price: ${latest_price:.2f}")
        
        # Get cached info
        cached_info = fetcher.get_info(symbol)
        if cached_info:
            market_cap = cached_info.get('marketCap')
            if market_cap:
                print(f"  Market cap: ${market_cap:,}")
    
    # Test 4: Database file verification
    print("\n\nTest 4: Database file verification")
    print("-" * 30)
    if os.path.exists(db_path):
        file_size = os.path.getsize(db_path)
        print(f"Database file: {db_path}")
        print(f"File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")
        
        # Direct database query
        db = DatabaseManager(db_path)
        all_symbols = db.get_all_symbols()
        print(f"Symbols in database: {sorted(all_symbols)}")
    
    print("\n\nDatabase test completed successfully!")
    print(f"SQLite database created: {os.path.abspath(db_path)}")
    print("\nYou can now:")
    print("1. Inspect the database using any SQLite browser")
    print("2. Use the CLI: python -m stockula.database.cli stats")
    print("3. Query data programmatically using DatabaseManager")


if __name__ == "__main__":
    test_database_functionality()