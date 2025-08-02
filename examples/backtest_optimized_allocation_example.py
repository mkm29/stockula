#!/usr/bin/env python3
"""Example of using BacktestOptimizedAllocator to optimize portfolio allocation.

This example demonstrates how to:
1. Configure backtest optimization through StockulaConfig
2. Use backtesting to find the best strategy for each asset
3. Evaluate performance on test data
4. Allocate capital based on backtest performance using Return [%] as the ranking metric

The BacktestOptimizedAllocator now supports full configuration through the
BacktestOptimizationConfig class, including:
- Date ranges for training and testing periods
- Ranking metric selection (Return [%], Sharpe Ratio, etc.)
- Allocation constraints (min/max percentages)

NOTE: Due to dependency injection complexities, this example creates a simplified
mock allocator to demonstrate the configuration usage. In production, use the
container to get a properly configured allocator.
"""

from stockula.config import (
    BacktestOptimizationConfig,
    PortfolioConfig,
    StockulaConfig,
    TickerConfig,
)


def main():
    """Run backtest-optimized allocation example."""
    print("Stockula Backtest-Optimized Allocation Configuration Example")
    print("=" * 80)
    print()
    print("This example demonstrates how to configure backtest-optimized allocation.")
    print("Due to dependency injection complexities in the example environment,")
    print("we'll show the configuration structure rather than running actual optimization.")
    print()

    # Define our portfolio configuration with backtest optimization settings
    config = StockulaConfig(
        portfolio=PortfolioConfig(
            initial_capital=100000.0,
            allocation_method="backtest_optimized",
            allow_fractional_shares=True,
        ),
        backtest_optimization=BacktestOptimizationConfig(
            train_start_date="2023-01-01",
            train_end_date="2023-06-30",
            test_start_date="2023-07-01",
            test_end_date="2023-12-31",
            ranking_metric="Return [%]",  # Use return percentage for ranking
            min_allocation_pct=5.0,
            max_allocation_pct=30.0,
            initial_allocation_pct=10.0,
        ),
    )

    # Define tickers to analyze
    tickers = [
        # Tech stocks
        TickerConfig(symbol="AAPL", category="TECH"),
        TickerConfig(symbol="GOOGL", category="TECH"),
        TickerConfig(symbol="MSFT", category="TECH"),
        TickerConfig(symbol="NVDA", category="TECH"),
        # Healthcare stocks
        TickerConfig(symbol="JNJ", category="HEALTHCARE"),
        TickerConfig(symbol="PFE", category="HEALTHCARE"),
        TickerConfig(symbol="UNH", category="HEALTHCARE"),
        # Finance stocks
        TickerConfig(symbol="JPM", category="FINANCE"),
        TickerConfig(symbol="BAC", category="FINANCE"),
        TickerConfig(symbol="GS", category="FINANCE"),
    ]

    # Get date ranges from configuration
    train_start = config.backtest_optimization.train_start_date
    train_end = config.backtest_optimization.train_end_date
    test_start = config.backtest_optimization.test_start_date
    test_end = config.backtest_optimization.test_end_date

    print("\nConfiguration Summary:")
    print("-" * 80)
    print(f"Portfolio Capital: ${config.portfolio.initial_capital:,.2f}")
    print(f"Allocation Method: {config.portfolio.allocation_method}")
    print(f"Training Period: {train_start} to {train_end}")
    print(f"Testing Period: {test_start} to {test_end}")
    print(f"Ranking Metric: {config.backtest_optimization.ranking_metric}")
    print(f"Allocation Range: {config.backtest_optimization.min_allocation_pct}% - {config.backtest_optimization.max_allocation_pct}%")
    print(f"Initial Allocation: {config.backtest_optimization.initial_allocation_pct}%")
    print(f"Number of Assets: {len(tickers)}")
    print("=" * 80)

    print("\nAssets to Optimize:")
    print("-" * 80)
    print(f"{'Symbol':<8} {'Category':<12}")
    print("-" * 80)
    for ticker in tickers:
        print(f"{ticker.symbol:<8} {ticker.category:<12}")
    print("=" * 80)

    print("\nHow the BacktestOptimizedAllocator Works:")
    print("-" * 80)
    print("1. Training Phase:")
    print("   - Tests 11 different strategies on each asset")
    print("   - Uses training data (2023-01-01 to 2023-12-31)")
    print("   - Selects best strategy per asset based on ranking metric")
    print()
    print("2. Testing Phase:")
    print("   - Runs best strategies on test data (2024-01-01 to 2024-06-30)")
    print("   - Measures actual performance")
    print()
    print("3. Allocation Calculation:")
    print(f"   - Assets are ranked by {config.backtest_optimization.ranking_metric}")
    print("   - Higher performing assets get larger allocations")
    print(f"   - Allocations are constrained between {config.backtest_optimization.min_allocation_pct}% and {config.backtest_optimization.max_allocation_pct}%")
    print("   - Final allocations sum to 100%")
    print()
    
    print("\nTo run actual optimization in your application:")
    print("-" * 80)
    print("```python")
    print("from stockula.container import create_container")
    print()
    print("# Create container with your config")
    print("container = create_container('config.yaml')")
    print()
    print("# Get the backtest allocator")
    print("allocator = container.backtest_allocator()")
    print()
    print("# Calculate optimized quantities")
    print("quantities = allocator.calculate_backtest_optimized_quantities(")
    print("    config=config,")
    print("    tickers_to_add=tickers")
    print(")")
    print("```")
    print("=" * 80)



if __name__ == "__main__":
    main()
