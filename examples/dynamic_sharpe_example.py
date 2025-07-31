#!/usr/bin/env python3
"""Example of using dynamic Treasury rates for Sharpe ratio calculation."""

from stockula import BacktestRunner, DataFetcher, SMACrossStrategy


def main():
    """Demonstrate backtesting with dynamic risk-free rates."""
    # Initialize data fetcher
    fetcher = DataFetcher(use_cache=True)

    # Set up backtesting parameters
    symbol = "AAPL"
    start_date = "2022-01-01"
    end_date = "2023-12-31"

    print(f"Running backtest for {symbol} from {start_date} to {end_date}")
    print("=" * 60)

    # Method 1: Using average Treasury rate (traditional approach)
    print("\n1. Traditional approach with average risk-free rate:")
    avg_rate = fetcher.get_average_treasury_rate(start_date, end_date, "3_month")
    print(f"Average 3-month T-bill rate: {avg_rate * 100:.2f}%")

    runner1 = BacktestRunner(
        cash=10000,
        commission=0.001,
        data_fetcher=fetcher,
        risk_free_rate=avg_rate,  # Static rate
    )

    results1 = runner1.run_from_symbol(symbol, SMACrossStrategy, start_date, end_date)
    print(f"Sharpe Ratio (static rate): {results1['Sharpe Ratio']:.4f}")

    # Method 2: Using dynamic Treasury rates
    print("\n2. Enhanced approach with dynamic risk-free rates:")

    # Fetch Treasury rates for the period
    treasury_rates = fetcher.get_treasury_rates(start_date, end_date, "3_month")
    print(f"Treasury rate range: {treasury_rates.min() * 100:.2f}% - {treasury_rates.max() * 100:.2f}%")
    print(f"Treasury rate volatility: {treasury_rates.std() * 100:.2f}%")

    runner2 = BacktestRunner(
        cash=10000,
        commission=0.001,
        data_fetcher=fetcher,
        risk_free_rate=treasury_rates,  # Dynamic rates as Series
    )

    results2 = runner2.run_from_symbol(symbol, SMACrossStrategy, start_date, end_date)

    # Display enhanced metrics
    print("\nEnhanced Metrics:")
    print(f"Sharpe Ratio (static): {results2.get('Sharpe Ratio (Static)', 'N/A')}")
    print(f"Sharpe Ratio (dynamic): {results2.get('Sharpe Ratio (Dynamic)', 'N/A')}")
    print(f"Sortino Ratio (static): {results2.get('Sortino Ratio (Static)', 'N/A')}")
    print(f"Sortino Ratio (dynamic): {results2.get('Sortino Ratio (Dynamic)', 'N/A')}")

    # Method 3: Using the convenience method
    print("\n3. Using convenience method run_with_dynamic_risk_free_rate:")

    runner3 = BacktestRunner(cash=10000, commission=0.001, data_fetcher=fetcher)

    results3 = runner3.run_with_dynamic_risk_free_rate(
        symbol, SMACrossStrategy, start_date, end_date, treasury_duration="3_month"
    )

    print(f"Sharpe Ratio (dynamic): {results3.get('Sharpe Ratio (Dynamic)', 'N/A')}")

    # Compare the differences
    print("\n" + "=" * 60)
    print("Summary of Results:")
    print(f"Total Return: {results3['Return [%]']:.2f}%")
    print(f"Max Drawdown: {results3['Max. Drawdown [%]']:.2f}%")
    print(f"Number of Trades: {results3['# Trades']}")

    if "Sharpe Ratio (Dynamic)" in results3 and "Sharpe Ratio (Static)" in results3:
        diff = results3["Sharpe Ratio (Dynamic)"] - results3["Sharpe Ratio (Static)"]
        print(f"\nDifference in Sharpe Ratio (Dynamic - Static): {diff:.4f}")
        print(f"This represents a {abs(diff / results3['Sharpe Ratio (Static)']) * 100:.1f}% difference")

    # Example of calculating rolling Sharpe ratio
    print("\n" + "=" * 60)
    print("Calculating rolling Sharpe ratio (252-day window):")

    from stockula.backtesting import calculate_rolling_sharpe_ratio

    # Get stock data and calculate returns
    stock_data = fetcher.get_stock_data(symbol, start_date, end_date)
    returns = stock_data["Close"].pct_change().dropna()

    # Calculate rolling Sharpe ratio
    rolling_sharpe = calculate_rolling_sharpe_ratio(returns, treasury_rates, window=252)

    # Display statistics
    print(f"Average rolling Sharpe: {rolling_sharpe.mean():.4f}")
    print(f"Min rolling Sharpe: {rolling_sharpe.min():.4f}")
    print(f"Max rolling Sharpe: {rolling_sharpe.max():.4f}")
    print(f"Current Sharpe (last value): {rolling_sharpe.iloc[-1]:.4f}")


if __name__ == "__main__":
    main()
