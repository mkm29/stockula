"""Run a Chronos zero-shot forecast example.

Usage:
  uv run python examples/forecast_chronos.py --ticker AAPL
  # or specify a different config path
  uv run python examples/forecast_chronos.py --config examples/config/forecast_chronos.yaml --ticker MSFT
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stockula.cli import run_stockula


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronos zero-shot forecast example")
    parser.add_argument(
        "--config",
        default=str(Path("examples/config/forecast_chronos.yaml")),
        help="Path to YAML configuration file",
    )
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol to forecast")
    args = parser.parse_args()

    run_stockula(
        config=args.config,
        ticker=args.ticker,
        mode="forecast",
        output="console",
    )


if __name__ == "__main__":
    main()
