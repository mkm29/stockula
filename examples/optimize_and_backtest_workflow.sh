#!/bin/bash
# Stockula Optimize and Backtest Workflow Example
# This script demonstrates the two-step process for backtest-optimized allocation

echo "=== Stockula Backtest-Optimized Allocation Workflow ==="
echo

# Step 1: Run optimization to determine optimal quantities
echo "Step 1: Running optimization to determine optimal quantities..."
uv run python -m stockula.main \
    --config examples/config.backtest-optimized.yaml \
    --mode optimize-allocation \
    --save-optimized-config optimized-portfolio.yaml

# Check if optimization was successful
if [ $? -ne 0 ]; then
    echo "Error: Optimization failed!"
    exit 1
fi

echo
echo "Step 2: Running backtest with optimized quantities..."
# Step 2: Run backtest with the optimized configuration
uv run python -m stockula.main \
    --config optimized-portfolio.yaml \
    --mode backtest

echo
echo "=== Workflow Complete ==="
echo "The optimized configuration has been saved to: optimized-portfolio.yaml"
echo "You can use this configuration for future backtests or modify it as needed."