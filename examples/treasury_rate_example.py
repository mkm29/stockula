#!/usr/bin/env python3
"""Example of fetching Treasury Bill rates for backtesting."""

from datetime import datetime

from stockula import DataFetcher


def main():
    """Demonstrate Treasury rate fetching."""
    # Initialize the fetcher
    fetcher = DataFetcher(use_cache=True)

    # Example 1: Get current 3-month T-bill rate
    print("Example 1: Current 3-month T-bill rate")
    current_rate = fetcher.get_current_treasury_rate("3_month")
    if current_rate:
        print(f"Current 3-month T-bill rate: {current_rate:.4f} ({current_rate * 100:.2f}%)")
    else:
        print("Could not fetch current rate")

    print("\n" + "=" * 50 + "\n")

    # Example 2: Get rate for a specific date
    print("Example 2: Treasury rate for specific date")
    specific_date = datetime(2024, 1, 15)
    rate = fetcher.get_treasury_rate(specific_date, "3_month")
    if rate:
        print(f"3-month T-bill rate on {specific_date.date()}: {rate:.4f} ({rate * 100:.2f}%)")
    else:
        print(f"Could not fetch rate for {specific_date.date()}")

    print("\n" + "=" * 50 + "\n")

    # Example 3: Get average rate for a date range (useful for backtesting)
    print("Example 3: Average rate for backtesting period")
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    avg_rate = fetcher.get_average_treasury_rate(start_date, end_date, "3_month")
    if avg_rate:
        print(f"Average 3-month T-bill rate for 2023: {avg_rate:.4f} ({avg_rate * 100:.2f}%)")
    else:
        print("Could not calculate average rate")

    print("\n" + "=" * 50 + "\n")

    # Example 4: Get rates for a full date range
    print("Example 4: Daily rates for a month")
    start_date = datetime(2024, 6, 1)
    end_date = datetime(2024, 6, 30)
    rates = fetcher.get_treasury_rates(start_date, end_date, "3_month")

    if not rates.empty:
        print(f"Treasury rates from {start_date.date()} to {end_date.date()}:")
        print(f"  Mean: {rates.mean():.4f} ({rates.mean() * 100:.2f}%)")
        print(f"  Min:  {rates.min():.4f} ({rates.min() * 100:.2f}%)")
        print(f"  Max:  {rates.max():.4f} ({rates.max() * 100:.2f}%)")
        print(f"  Std:  {rates.std():.4f}")
        print("\nFirst 5 days:")
        for date, rate in rates.head().items():
            print(f"  {date.date()}: {rate:.4f} ({rate * 100:.2f}%)")
    else:
        print("Could not fetch rates for the period")

    print("\n" + "=" * 50 + "\n")

    # Example 5: How to use with backtesting
    print("Example 5: Using Treasury rates in backtesting")
    print("""
# In your backtest configuration or code:

# Option 1: Use average rate for the entire backtest period
backtest_start = datetime(2023, 1, 1)
backtest_end = datetime(2023, 12, 31)
risk_free_rate = fetcher.get_average_treasury_rate(
    backtest_start, backtest_end, "3_month"
)

# Option 2: Use dynamic rates (different rate for each period)
daily_rates = fetcher.get_treasury_rates(
    backtest_start, backtest_end, "3_month"
)

# Option 3: Use rate at start of backtest period
start_rate = fetcher.get_treasury_rate(
    backtest_start, "3_month"
)
""")


if __name__ == "__main__":
    main()
