#!/usr/bin/env python3
"""Script to fetch and save test data for consistent testing."""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(project_root))

from test_data_manager import setup_test_data


def main():
    """Fetch and save test data."""
    print("Fetching test data...")

    try:
        data = setup_test_data(force_refresh=False)

        if data:
            print(f"\nSuccessfully fetched data for {len(data)} tickers:")
            for ticker, df in data.items():
                print(f"  {ticker}: {len(df)} rows")
        else:
            print("No data was fetched. This may be due to:")
            print("  - yfinance not being installed")
            print("  - Network connectivity issues")
            print("  - Rate limiting from Yahoo Finance")
            print("\nTests will fall back to synthetic data.")

    except ImportError as e:
        print(f"Error: {e}")
        print("Install yfinance with: uv add yfinance")
        print("Tests will use synthetic data instead.")

    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Tests will fall back to synthetic data.")


if __name__ == "__main__":
    main()
