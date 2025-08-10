"""Batch inference over a GluonTS-compatible file dataset using Chronos.

Supports Arrow (Feather) and JSONL input produced by the export script.
Outputs per-item forecasts with mean/quantile columns in long (tidy) format.

Usage:
  uv run python scripts/chronos_batch_infer.py \
    --data ./datasets/demo/data.arrow --format arrow \
    --model amazon/chronos-bolt-small --prediction-length 7 --out predictions.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_dataset(path: Path, fmt: str) -> list[dict[str, Any]]:
    if fmt == "arrow":
        try:
            import pyarrow.feather as feather
        except Exception as e:  # pragma: no cover
            raise RuntimeError("pyarrow is required for --format arrow. Install with: uv pip install pyarrow") from e
        table = feather.read_table(path)
        df = table.to_pandas()
        # Expect columns: item_id, start, target (list)
        records = []
        for _, row in df.iterrows():
            records.append(
                {
                    "item_id": row["item_id"],
                    "start": row["start"],
                    "target": row["target"],
                }
            )
        return records
    else:
        # JSONL
        recs = []
        with path.open() as f:
            for line in f:
                recs.append(json.loads(line))
        return recs


def setup_pipeline(model_id: str, device: str | None, dtype_str: str | None):
    try:
        from chronos import BaseChronosPipeline
    except Exception as e:  # pragma: no cover
        raise RuntimeError("chronos-forecasting is required. Install with: uv pip install chronos-forecasting") from e

    torch_dtype = None
    if dtype_str:
        import torch

        torch_dtype = getattr(torch, dtype_str)

    if device is None:
        # auto-select
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    return BaseChronosPipeline.from_pretrained(model_id, device_map=device, torch_dtype=torch_dtype)


def main() -> None:
    p = argparse.ArgumentParser(description="Batch inference with Chronos over GluonTS file dataset")
    p.add_argument("--data", required=True, help="Path to dataset file (data.arrow or data.jsonl)")
    p.add_argument("--format", choices=["arrow", "jsonl"], default="arrow", help="Input dataset format")
    p.add_argument("--model", default="amazon/chronos-t5-mini", help="Chronos model ID")
    p.add_argument("--prediction-length", type=int, default=7, help="Forecast horizon")
    p.add_argument("--num-samples", type=int, default=256, help="Number of samples for probabilistic forecast")
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--device", choices=["cuda", "cpu"], default=None, help="Force device (default auto)")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default=None, help="Torch dtype")

    args = p.parse_args()

    records = load_dataset(Path(args.data), args.format)
    pipeline = setup_pipeline(args.model, args.device, args.dtype)

    alpha = 0.05  # default interval 90%
    q_levels = [alpha, 0.5, 1.0 - alpha]

    outputs: list[pd.DataFrame] = []
    for rec in records:
        item_id = rec["item_id"]
        target = np.asarray(rec["target"], dtype=np.float32)
        if target.size == 0:
            continue

        samples = pipeline.predict(
            context=target, prediction_length=int(args.prediction_length), num_samples=int(args.num_samples)
        )
        if samples.ndim != 2:
            samples = np.atleast_2d(samples)

        quants = {q: np.quantile(samples, q=q, axis=0).astype(np.float64) for q in q_levels}

        # Build future timestamps assuming daily frequency (customize as needed)
        # When exporting, 'start' is for the first observed timestamp; derive last observed from length
        start = pd.to_datetime(rec["start"])  # type: ignore[arg-type]
        last_obs = start + pd.Timedelta(days=len(target) - 1)
        future_index = pd.date_range(
            start=last_obs + pd.Timedelta(days=1), periods=int(args.prediction_length), freq="D"
        )

        df = pd.DataFrame(
            {
                "item_id": item_id,
                "timestamp": future_index,
                "mean": quants[0.5],
                "q05": quants[q_levels[0]],
                "q95": quants[q_levels[-1]],
            }
        )
        outputs.append(df)

    if outputs:
        out_df = pd.concat(outputs, ignore_index=True)
        out_df.to_csv(args.out, index=False)
        print(f"Wrote predictions to: {args.out}")
    else:
        print("No predictions were generated.")


if __name__ == "__main__":
    main()
