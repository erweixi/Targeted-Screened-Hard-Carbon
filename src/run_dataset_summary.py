"""Export dataset/feature coverage summaries for reviewer responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .utils import FEATURE_COLUMNS, TASK_SPECS, dataset_overview_frame, load_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["ice", "plateau"], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--outdir", default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data)
    overview = dataset_overview_frame(df, args.task)
    csv_path = outdir / f"dataset_summary_{args.task}.csv"
    overview.to_csv(csv_path, index=False)

    payload = {
        "task": args.task,
        "data": str(args.data),
        "n_samples": int(len(df)),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TASK_SPECS[args.task].target_column,
    }
    json_path = outdir / f"dataset_summary_{args.task}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
