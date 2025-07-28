#!/usr/bin/env python3
"""Test script to demonstrate parallel forecasting progress tracking."""

import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from stockula.forecasting import StockForecaster
from stockula.data.fetcher import DataFetcher
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Initialize data fetcher
fetcher = DataFetcher(use_cache=True)

# Test with a small set of symbols
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]

console.print(
    "[bold green]Testing Parallel Forecasting with Progress Updates[/bold green]"
)
console.print(
    f"Forecasting {len(symbols)} symbols with status updates every 2 seconds\n"
)

# Progress display
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    console=console,
    transient=False,
) as progress:
    task = progress.add_task("[blue]Starting forecasting...", total=None)

    def progress_callback(symbol, status, status_info=None):
        """Handle progress updates."""
        if status == "status_update" and status_info:
            # Periodic status update
            active_str = ""
            if status_info["active"]:
                active_str = f"Active: {', '.join(status_info['active'])}"

            desc = f"[blue]Progress: {status_info['completed_count']}/{status_info['total']} completed"
            if status_info["active_count"] > 0:
                desc += f", {status_info['active_count']} in progress"
            if status_info["error_count"] > 0:
                desc += f", {status_info['error_count']} errors"
            if active_str:
                desc += f" | {active_str}"

            progress.update(task, description=desc)

        elif symbol and status == "completed":
            console.print(f"  ✓ {symbol} completed", style="green")

        elif symbol and status == "error":
            console.print(f"  ✗ {symbol} failed", style="red")

    # Run parallel forecasting
    start_time = time.time()

    results = StockForecaster.forecast_multiple_parallel(
        symbols=symbols,
        start_date="2025-01-01",
        end_date="2025-07-25",
        forecast_length=7,
        model_list="fast",
        max_generations=2,
        max_workers=4,
        data_fetcher=fetcher,
        progress_callback=progress_callback,
        status_update_interval=2,
    )

    elapsed = time.time() - start_time

    progress.update(task, description=f"[green]Forecasting complete! ({elapsed:.1f}s)")

# Display results
console.print("\n[bold]Forecast Results:[/bold]")
for symbol, result in results.items():
    if "error" in result:
        console.print(f"  {symbol}: [red]Error - {result['error']}[/red]")
    else:
        return_pct = (
            (result["forecast_price"] - result["current_price"])
            / result["current_price"]
        ) * 100
        color = "green" if return_pct > 0 else "red" if return_pct < 0 else "white"
        console.print(
            f"  {symbol}: ${result['current_price']:.2f} → ${result['forecast_price']:.2f} "
            f"([{color}]{return_pct:+.2f}%[/{color}]) | Model: {result['best_model']}"
        )

console.print(f"\n[dim]Total time: {elapsed:.1f} seconds[/dim]")
