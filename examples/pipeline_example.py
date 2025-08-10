#!/usr/bin/env python
"""Example script demonstrating the Stockula Pipeline.

This script shows how to use the StockulaPipeline class to:
1. Run portfolio optimization
2. Save optimized configuration
3. Run backtesting with optimized allocations
4. Compare results

Usage:
    python examples/pipeline_example.py
"""

from pathlib import Path

from stockula.pipeline import StockulaPipeline


def main():
    """Run the Stockula pipeline example."""

    # Path to your base configuration
    base_config_path = ".stockula.yaml"

    # Path where optimized configuration will be saved
    optimized_config_path = ".stockula-optimized.yaml"

    # Path for results output
    results_path = "pipeline_results.json"

    # Create pipeline instance
    print("🚀 Initializing Stockula Pipeline...")
    pipeline = StockulaPipeline(
        base_config_path=base_config_path,
        verbose=True,  # Enable verbose output
    )

    try:
        # Load and validate configuration
        print("\n📁 Loading configuration...")
        config = pipeline.load_configuration()
        print(f"✓ Loaded configuration with {len(config.portfolio.tickers)} tickers")

        # Run portfolio optimization
        print("\n🎯 Running portfolio optimization...")
        optimized_config, opt_results = pipeline.run_optimization(
            config=config,
            save_config_path=optimized_config_path,
        )

        # Display optimization metrics
        if "metrics" in opt_results:
            print("\nOptimization Metrics:")
            for key, value in opt_results["metrics"].items():
                print(f"  • {key}: {value}")

        # Run backtesting with optimized configuration
        print("\n📊 Running backtesting with optimized allocation...")
        backtest_results = pipeline.run_backtest(
            config=optimized_config,
            use_optimized=True,
        )

        # Display backtest summary
        if "summary" in backtest_results:
            print("\nBacktest Summary:")
            summary = backtest_results["summary"]
            print(f"  • Total Return: {summary.get('total_return', 'N/A')}")
            print(f"  • Sharpe Ratio: {summary.get('sharpe_ratio', 'N/A')}")
            print(f"  • Max Drawdown: {summary.get('max_drawdown', 'N/A')}")

        # Save complete results
        print(f"\n💾 Saving results to {results_path}...")
        pipeline.save_results(results_path, format="json")

        # Also save in YAML format for readability
        yaml_path = results_path.replace(".json", ".yaml")
        pipeline.save_results(yaml_path, format="yaml")

        print(f"\n✨ Pipeline completed successfully!")
        print(f"   • Optimized config: {optimized_config_path}")
        print(f"   • Results: {results_path}, {yaml_path}")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please ensure the configuration file exists.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise


def run_full_pipeline_example():
    """Example of running the full pipeline in one call."""

    print("🎯 Running Full Pipeline Example\n")

    # Create pipeline
    pipeline = StockulaPipeline(verbose=True)

    # Run everything in one call
    results = pipeline.run_full_pipeline(
        base_config_path=".stockula.yaml",
        optimized_config_path=".stockula-opt.yaml",
        optimization={"time_limit": 60},  # Optional optimization params
        backtest={"commission": 0.001},  # Optional backtest params
    )

    # Save combined results
    pipeline.save_results("full_pipeline_results.json", format="json")

    return results


def run_optimization_only_example():
    """Example of running only the optimization step."""

    print("🎯 Running Optimization-Only Example\n")

    pipeline = StockulaPipeline(
        base_config_path=".stockula.yaml",
        verbose=False,  # Less verbose output
    )

    # Load configuration
    config = pipeline.load_configuration()

    # Run optimization only
    optimized_config, results = pipeline.run_optimization(
        config=config,
        save_config_path="optimized_portfolio.yaml",
    )

    # Display optimized allocations
    if "optimized_allocations" in results:
        print("\nOptimized Portfolio Allocations:")
        allocations = results["optimized_allocations"]
        total = sum(allocations.values())

        for symbol, quantity in sorted(allocations.items()):
            percentage = (quantity / total * 100) if total > 0 else 0
            print(f"  {symbol:6s}: {quantity:8.2f} shares ({percentage:5.1f}%)")

    return optimized_config, results


def run_backtest_with_existing_config():
    """Example of running backtest with an existing optimized configuration."""

    print("📊 Running Backtest with Existing Config\n")

    # Assume we have an optimized configuration already
    optimized_config_path = ".stockula-optimized.yaml"

    if not Path(optimized_config_path).exists():
        print(f"❌ Optimized config not found at {optimized_config_path}")
        print("Run optimization first to create the optimized configuration.")
        return None

    # Create pipeline without base config
    pipeline = StockulaPipeline(verbose=True)

    # Load the optimized configuration
    config = pipeline.load_configuration(optimized_config_path)

    # Run backtest
    results = pipeline.run_backtest(
        config=config,
        use_optimized=False,  # Config is already optimized
    )

    # Display results
    if results.get("results"):
        print(f"\nBacktest completed with {len(results['results'])} strategy results")

    return results


def compare_strategies_example():
    """Example comparing performance with and without optimization."""

    print("📊 Comparing Strategies Example\n")

    base_config_path = ".stockula.yaml"

    # Create pipeline
    pipeline = StockulaPipeline(base_config_path=base_config_path, verbose=False)

    # Load base configuration
    base_config = pipeline.load_configuration()

    # Run backtest with original allocation
    print("Running backtest with original allocation...")
    original_results = pipeline.run_backtest(config=base_config, use_optimized=False)

    # Run optimization
    print("Running portfolio optimization...")
    optimized_config, _ = pipeline.run_optimization(config=base_config)

    # Run backtest with optimized allocation
    print("Running backtest with optimized allocation...")
    optimized_results = pipeline.run_backtest(config=optimized_config, use_optimized=True)

    # Compare results
    print("\n📈 Performance Comparison:")

    if "summary" in original_results and "summary" in optimized_results:
        orig_summary = original_results["summary"]
        opt_summary = optimized_results["summary"]

        metrics = ["total_return", "sharpe_ratio", "max_drawdown"]

        print(f"{'Metric':<15} {'Original':>12} {'Optimized':>12} {'Improvement':>12}")
        print("-" * 52)

        for metric in metrics:
            orig_val = orig_summary.get(metric, 0)
            opt_val = opt_summary.get(metric, 0)

            if metric == "max_drawdown":
                # For drawdown, less negative is better
                improvement = orig_val - opt_val
            else:
                # For returns and Sharpe, higher is better
                improvement = opt_val - orig_val

            print(f"{metric:<15} {orig_val:>12.4f} {opt_val:>12.4f} {improvement:>+12.4f}")

    return original_results, optimized_results


if __name__ == "__main__":
    # Run the main example
    main()

    # Uncomment to run other examples:
    # run_full_pipeline_example()
    # run_optimization_only_example()
    # run_backtest_with_existing_config()
    # compare_strategies_example()
