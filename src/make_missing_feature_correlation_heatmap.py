"""Build the reviewer supplementary Spearman heatmap including reversible capacity.

This script renders a figure from the updated correlation matrices produced by
`run_missing_feature_math_analysis.py`. It is meant to close the gap between the
supplementary response text and the repository contents by explicitly exporting a
heatmap that includes `reversible_capacity_mAh_g` alongside ICE and plateau.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .extended_utils import EXTRA_FEATURE_COLUMNS, FEATURE_LABELS
from .utils import FEATURE_COLUMNS

TARGET_COLUMNS = ["ice", "plateau_capacity_mAh_g", "reversible_capacity_mAh_g"]
ANALYSIS_COLUMNS = FEATURE_COLUMNS + EXTRA_FEATURE_COLUMNS + TARGET_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-csv",
        default="results/reviewer_extension_fixed/missing_feature_analysis/missing_feature_math_analysis/variable_correlation_matrix_spearman.csv",
    )
    parser.add_argument(
        "--out-png",
        default="outputs/reviewer_spearman_heatmap_with_reversible.png",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )
    return parser.parse_args()


def pretty_label(column: str) -> str:
    return FEATURE_LABELS.get(column, column)


def main() -> None:
    args = parse_args()
    matrix_path = Path(args.matrix_csv)
    out_png = Path(args.out_png)

    corr = pd.read_csv(matrix_path, index_col=0)
    missing = [c for c in ANALYSIS_COLUMNS if c not in corr.columns]
    if missing:
        raise ValueError(
            "The Spearman matrix is missing expected columns. Rerun "
            "src.run_missing_feature_math_analysis first. Missing: "
            f"{missing}"
        )

    corr = corr.loc[ANALYSIS_COLUMNS, ANALYSIS_COLUMNS]
    labels = [pretty_label(c) for c in ANALYSIS_COLUMNS]
    values = corr.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(11.5, 9.5))
    lim = max(0.75, float(np.nanmax(np.abs(values))))
    im = ax.imshow(values, cmap="coolwarm", vmin=-lim, vmax=lim)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Spearman correlation heatmap including reversible capacity", fontsize=14)

    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isnan(v):
                txt = "NA"
            else:
                txt = f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Spearman rho")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_png}")


if __name__ == "__main__":
    main()
