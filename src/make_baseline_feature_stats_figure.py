#!/usr/bin/env python3
"""Make a four-panel statistics figure for baseline features vs ICE/plateau targets.

Panels:
- Pearson r
- Spearman rho
- Kendall tau
- Univariate OLS R^2

The script also writes a tidy CSV with coefficients, p-values, and OLS results.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kendalltau, pearsonr, spearmanr

BASELINE_FEATURES = [
    "carbonization_temperature_C",
    "d002_nm",
    "id_ig",
    "ssa_m2_g",
    "electrolyte_type",
    "current_density_mA_g",
]

FEATURE_LABELS = {
    "carbonization_temperature_C": "Carbonization temperature",
    "d002_nm": "d002",
    "id_ig": "ID/IG",
    "ssa_m2_g": "SSA",
    "electrolyte_type": "Electrolyte type",
    "current_density_mA_g": "Current density",
}

TARGETS = {
    "ice": "ICE",
    "plateau_capacity_mAh_g": "Plateau capacity",
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default_data = project_root / "data" / "hc_dataset_original_master_with_missing_features.csv"
    default_output = project_root / "outputs"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--output_dir", type=Path, default=default_output)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def stars(p: float) -> str:
    return "*" if pd.notna(p) and p < 0.05 else ""


def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for feat in BASELINE_FEATURES:
        for target_col, target_label in TARGETS.items():
            sub = df[[feat, target_col]].dropna().copy()
            x = sub[feat]
            y = sub[target_col]

            pearson_val, pearson_p = pearsonr(x, y)
            spearman_val, spearman_p = spearmanr(x, y)
            kendall_val, kendall_p = kendalltau(x, y)

            X_ols = sm.add_constant(x)
            ols = sm.OLS(y, X_ols).fit()

            rows.append(
                {
                    "feature": feat,
                    "feature_label": FEATURE_LABELS[feat],
                    "target": target_col,
                    "target_label": target_label,
                    "n": len(sub),
                    "pearson_r": pearson_val,
                    "pearson_p": pearson_p,
                    "spearman_rho": spearman_val,
                    "spearman_p": spearman_p,
                    "kendall_tau": kendall_val,
                    "kendall_p": kendall_p,
                    "ols_r2": ols.rsquared,
                    "ols_coef": ols.params.iloc[1],
                    "ols_p": ols.pvalues.iloc[1],
                }
            )
    return pd.DataFrame(rows)


def build_matrix(df_stats: pd.DataFrame, value_col: str, p_col: str | None = None):
    row_labels = [FEATURE_LABELS[f] for f in BASELINE_FEATURES]
    col_order = ["ICE", "Plateau capacity"]
    value_mat = np.zeros((len(row_labels), len(col_order)), dtype=float)
    annot: List[List[str]] = []

    for feat in BASELINE_FEATURES:
        feat_label = FEATURE_LABELS[feat]
        row_annots: List[str] = []
        for j, target_label in enumerate(col_order):
            match = df_stats[
                (df_stats["feature_label"] == feat_label) &
                (df_stats["target_label"] == target_label)
            ].iloc[0]
            value = float(match[value_col])
            value_mat[row_labels.index(feat_label), j] = value
            suffix = stars(float(match[p_col])) if p_col else ""
            row_annots.append(f"{value:.2f}{suffix}")
        annot.append(row_annots)

    return value_mat, row_labels, col_order, annot


def draw_panel(ax, matrix, row_labels, col_labels, annot, title, cmap, vmin, vmax):
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=15)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)

    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=10)
    return im


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data)
    stats_df = compute_stats(df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "baseline_feature_target_statistics.csv"
    png_path = args.output_dir / "baseline_feature_target_statistics.png"
    stats_df.to_csv(csv_path, index=False)

    pearson_matrix, rows, cols, pearson_annot = build_matrix(stats_df, "pearson_r", "pearson_p")
    spearman_matrix, _, _, spearman_annot = build_matrix(stats_df, "spearman_rho", "spearman_p")
    kendall_matrix, _, _, kendall_annot = build_matrix(stats_df, "kendall_tau", "kendall_p")
    ols_matrix, _, _, ols_annot = build_matrix(stats_df, "ols_r2", "ols_p")

    fig, axes = plt.subplots(1, 4, figsize=(16, 8))
    fig.suptitle("Supplementary statistical analysis for baseline features", fontsize=18)

    corr_min = min(pearson_matrix.min(), spearman_matrix.min(), kendall_matrix.min(), -0.75)
    corr_max = max(pearson_matrix.max(), spearman_matrix.max(), kendall_matrix.max(), 0.75)
    corr_lim = max(abs(corr_min), abs(corr_max))

    im_corr = draw_panel(axes[0], pearson_matrix, rows, cols, pearson_annot, "Pearson r", "coolwarm", -corr_lim, corr_lim)
    draw_panel(axes[1], spearman_matrix, rows, cols, spearman_annot, "Spearman ρ", "coolwarm", -corr_lim, corr_lim)
    draw_panel(axes[2], kendall_matrix, rows, cols, kendall_annot, "Kendall τ", "coolwarm", -corr_lim, corr_lim)
    im_ols = draw_panel(axes[3], ols_matrix, rows, cols, ols_annot, "Univariate OLS R$^2$", "Blues", 0.0, max(0.5, float(ols_matrix.max())))

    cbar1 = fig.colorbar(im_corr, ax=axes[:3], fraction=0.03, pad=0.04)
    cbar1.set_label("Correlation")
    cbar2 = fig.colorbar(im_ols, ax=axes[3], fraction=0.08, pad=0.04)
    cbar2.set_label("R$^2$")

    fig.text(
        0.5,
        0.02,
        "* p < 0.05. Electrolyte type is numeric-encoded, so its correlation reflects encoded-category association.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] wrote {csv_path}")
    print(f"[OK] wrote {png_path}")


if __name__ == "__main__":
    main()
