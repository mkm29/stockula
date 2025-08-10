"""Export Stockula time series to a GluonTS-compatible file dataset.

This script creates a minimal GluonTS file dataset for forecasting experiments.
By default, it writes an Arrow/Feather file with columns:
  - item_id: str
  - start: ISO-8601 string of the first timestamp
  - target: list[float] of target values (e.g., Close)

Optionally, it can write JSON Lines instead of Arrow.

Usage:
  uv run python scripts/export_to_gluonts_file_dataset.py \
      --tickers AAPL,MSFT --start 2023-01-01 --end 2024-01-01 \
      --out-dir ./datasets/demo --format arrow

Notes:
  - This is a scaffold for batch workflows. It focuses on the essential fields
    needed by Chronos/GluonTS scripts. Extend as needed for dynamic/static covariates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from stockula.container import create_container


def infer_freq(index: pd.DatetimeIndex) -> str:
    try:
        freq = pd.infer_freq(index)
        return freq or "D"
    except Exception:
        return "D"


def to_records(series_map: dict[str, pd.DataFrame], target_column: str = "Close") -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item_id, df in series_map.items():
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found for {item_id}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        df = df.sort_index()
        target = df[target_column].astype(float).tolist()
        start = pd.Timestamp(df.index[0]).isoformat()
        records.append(
            {
                "item_id": item_id,
                "start": start,
                "target": target,
                # Add additional fields (feat_dynamic_real, feat_static_cat) as needed
            }
        )
    return records


def write_arrow(records: list[dict[str, Any]], out_dir: Path) -> Path:
    try:
        import pyarrow as pa
        import pyarrow.feather as feather
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pyarrow is required for --format arrow. Install with: uv pip install pyarrow") from e

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "data.arrow"

    # Convert list-of-lists to Arrow ListArray
    item_ids = pa.array([r["item_id"] for r in records], type=pa.string())
    starts = pa.array([r["start"] for r in records], type=pa.string())
    targets = pa.array([r["target"] for r in records], type=pa.list_(pa.float64()))
    table = pa.Table.from_arrays([item_ids, starts, targets], names=["item_id", "start", "target"])
    feather.write_feather(table, path)
    return path


def write_jsonl(records: list[dict[str, Any]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "data.jsonl"
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Export to GluonTS-compatible file dataset (Arrow or JSONL)")
    p.add_argument("--tickers", required=True, help="Comma-separated list of tickers, e.g., AAPL,MSFT")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--out-dir", required=True, help="Output directory for dataset")
    p.add_argument("--format", choices=["arrow", "jsonl"], default="arrow", help="Dataset file format")
    p.add_argument("--target", default="Close", help="Target column name (default: Close)")

    args = p.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    out_dir = Path(args.out_dir)

    container = create_container()
    fetcher = container.data_fetcher()

    series_map: dict[str, pd.DataFrame] = {}
    for sym in tickers:
        df = fetcher.get_stock_data(sym, args.start, args.end)
        if df.empty:
            raise ValueError(f"No data for {sym} in range {args.start}..{args.end}")
        series_map[sym] = df

    records = to_records(series_map, target_column=args.target)

    if args.format == "arrow":
        out_path = write_arrow(records, out_dir)
    else:
        out_path = write_jsonl(records, out_dir)

    print(f"Wrote dataset to: {out_path}")


if __name__ == "__main__":
    main()
