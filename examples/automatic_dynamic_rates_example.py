#!/usr/bin/env python3
"""Example of automatic dynamic Treasury rate usage in backtesting."""

from stockula import BacktestRunner, SMACrossStrategy
from stockula.container import create_container


def main():
    """Demonstrate automatic dynamic Treasury rate usage."""
    print("=" * 60)
    print("AUTOMATIC DYNAMIC TREASURY RATES EXAMPLE")
    print("=" * 60)

    # Get DI container and fetch data fetcher
    container = create_container()
    data_fetcher = container.data_fetcher()

    # Create backtest runner
    runner = BacktestRunner(cash=10000, commission=0.002, data_fetcher=data_fetcher)

    print("\nRunning backtest with automatic dynamic Treasury rates...")
    print("This will:")
    print("1. Fetch AAPL stock data for 2023")
    print("2. Automatically fetch 3-month Treasury rates for the same period")
    print("3. Calculate both static and dynamic risk-adjusted metrics")

    # Run backtest - dynamic rates are now used by default!
    results = runner.run_from_symbol("AAPL", SMACrossStrategy, start_date="2023-01-01", end_date="2023-12-31")

    print("\nBacktest Results:")
    print(f"Return: {results.get('Return [%]', 'N/A'):.2f}%")
    print(f"Max Drawdown: {results.get('Max. Drawdown [%]', 'N/A'):.2f}%")

    # Display dynamic metrics
    print("\nRisk-Adjusted Performance:")
    print(f"Sharpe Ratio (Static):  {results.get('Sharpe Ratio (Static)', 'N/A')}")
    print(f"Sharpe Ratio (Dynamic): {results.get('Sharpe Ratio (Dynamic)', 'N/A')}")
    print(f"Sortino Ratio (Static):  {results.get('Sortino Ratio (Static)', 'N/A')}")
    print(f"Sortino Ratio (Dynamic): {results.get('Sortino Ratio (Dynamic)', 'N/A')}")

    # Show Treasury rate information
    if "Avg. Risk-Free Rate [%]" in results:
        print("\nTreasury Rate Statistics:")
        print(f"Average Risk-Free Rate: {results['Avg. Risk-Free Rate [%]']:.3f}%")
        print(f"Risk-Free Rate Volatility: {results['Risk-Free Rate Volatility [%]']:.3f}%")

    print(f"\n{'=' * 60}")
    print("Dynamic Treasury rates are now used automatically!")
    print("To disable: set use_dynamic_risk_free_rate=False")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
