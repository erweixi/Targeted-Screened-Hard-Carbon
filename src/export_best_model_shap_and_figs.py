from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _import_utils():
    try:
        from .utils import FEATURE_COLUMNS, TASK_SPECS, load_xy, split_train_test
    except ImportError:
        from utils import FEATURE_COLUMNS, TASK_SPECS, load_xy, split_train_test
    return FEATURE_COLUMNS, TASK_SPECS, load_xy, split_train_test


FEATURE_COLUMNS, TASK_SPECS, load_xy, split_train_test = _import_utils()

FEATURE_LABELS = {
    "carbonization_temperature_C": "Carbonization temperature",
    "d002_nm": "d002",
    "id_ig": "ID/IG",
    "ssa_m2_g": "SSA",
    "electrolyte_type": "Electrolyte",
    "current_density_mA_g": "Current density",
}

FIG1_MODEL_LABELS = {
    "LinearRegression": "Linear",
    "GradientBoostingRegressor": "GBR",
    "Ridge Regression": "Ridge",
    "Lasso Regression": "Lasso",
    "Neural Network Regression": "Neural\nNetwork",
    "Random Forest Regression": "Random\nForest",
    "XGBRegressor": "XGB",
}

FIG1_MODEL_ORDER = [
    "LinearRegression",
    "GradientBoostingRegressor",
    "Ridge Regression",
    "Lasso Regression",
    "Neural Network Regression",
    "Random Forest Regression",
    "XGBRegressor",
]

FIG1_BAR_COLORS = [
    (155 / 255, 213 / 255, 155 / 255),
    (114 / 255, 154 / 255, 213 / 255),
    (253 / 255, 205 / 255, 147 / 255),
    (254 / 255, 254 / 255, 164 / 255),
    (254 / 255, 122 / 255, 197 / 255),
    (238 / 255, 139 / 255, 72 / 255),
    (155 / 255, 155 / 255, 155 / 255),
]

FIG2_TEST_COLOR = "#f09794"
FIG2_TEST_EDGE = "#dc6f69"
FIG2_TRAIN_COLOR = "#8ea5c7"
FIG2_TRAIN_EDGE = "#5f83af"
FIG2_LINE_COLOR = "#cc7a29"

FIG2_PANEL_META = {
    "ice": {
        "r2": 0.761,
        "rmae": 0.203,
        "x_col": "y_true_logit_ice",
        "y_col": "y_pred_logit_ice",
        "x_label": "ICE_Real (%)",
        "y_label": "ICE_Predict (%)",
        "lims": (-2, 3),
    },
    "plateau": {
        "r2": 0.758,
        "rmae": 0.163,
        "x_col": "y_true_plateau",
        "y_col": "y_pred_plateau",
        "x_label": "Plateau capacity_Real (mAh g$^{-1}$)",
        "y_label": "Plateau capacity_Predict (mAh g$^{-1}$)",
        "lims": (0, 300),
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export per-feature SHAP CSVs and manuscript figures from saved best-model results.")
    p.add_argument("--project-root", default=".", help="Project root directory")
    p.add_argument("--reviewer-dir", default="results/reviewer", help="Directory containing saved best-model reviewer results")
    p.add_argument("--results-dir", default="results", help="Directory containing manuscript-level saved CSVs/figures")
    p.add_argument("--data-dir", default="data", help="Directory containing hc_dataset_ice.csv and hc_dataset_plateau.csv")
    p.add_argument("--out-shap-dir", default="results/manuscript_exports/shap_feature_csvs", help="Output directory for 12 per-feature SHAP CSVs")
    p.add_argument("--out-fig-dir", default="outputs", help="Output directory for reproduced figures")
    return p.parse_args()


def _load_task_y(project_root: Path, data_dir: str, task: str):
    import pandas as pd

    csv_name = "hc_dataset_ice.csv" if task == "ice" else "hc_dataset_plateau.csv"
    df = pd.read_csv(project_root / data_dir / csv_name)
    return df[TASK_SPECS[task].target_column]


def export_feature_level_shap_csvs(project_root: Path, reviewer_dir: Path, data_dir: str, outdir: Path) -> List[Path]:
    import pandas as pd

    outdir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for task in ["ice", "plateau"]:
        shap_dir = reviewer_dir / f"multimodel_shap_{task}" / "randomforest"
        x_path = shap_dir / "X_test_used_for_shap.csv"
        s_path = shap_dir / "shap_values_test.csv"
        if not x_path.exists() or not s_path.exists():
            raise FileNotFoundError(f"Missing SHAP files for task={task}: {x_path} / {s_path}")

        X_test = pd.read_csv(x_path)
        shap_df = pd.read_csv(s_path)
        if "sample_index" not in X_test.columns or "sample_index" not in shap_df.columns:
            raise ValueError(f"sample_index column missing in SHAP inputs for task={task}")
        if list(X_test["sample_index"]) != list(shap_df["sample_index"]):
            raise ValueError(f"sample_index mismatch between X_test and SHAP values for task={task}")

        y_all = _load_task_y(project_root, data_dir, task)
        sample_index = X_test["sample_index"].astype(int)
        target_value = y_all.loc[sample_index].to_numpy()

        for feature in FEATURE_COLUMNS:
            export_df = pd.DataFrame(
                {
                    "feature_value": X_test[feature].to_numpy(),
                    "shap_value": shap_df[feature].to_numpy(),
                    "target_value": target_value,
                }
            )
            out_path = outdir / f"{task}__{feature}.csv"
            export_df.to_csv(out_path, index=False)
            written.append(out_path)

    return written


def _load_fig1_source(project_root: Path, results_dir: Path):
    import pandas as pd

    path = results_dir / "cv_r2_scores.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Fig. S1 source file: {path}")
    return pd.read_csv(path)


def make_fig_s1(project_root: Path, results_dir: Path, out_fig_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    import pandas as pd

    out_fig_dir.mkdir(parents=True, exist_ok=True)
    df = _load_fig1_source(project_root, results_dir)
    ice_col = "cv_r2_ice"
    plateau_col = "cv_r2_plateau"

    fig, axes = plt.subplots(2, 1, figsize=(6.3, 7.6), dpi=300)
    panels = [
        (axes[0], ice_col, "(a) ICE (logit-transformed target)"),
        (axes[1], plateau_col, "(b) Reversible plateau capacity"),
    ]

    for ax, value_col, title in panels:
        vals = []
        labels = []
        for model in FIG1_MODEL_ORDER:
            row = df.loc[df["model"] == model]
            if row.empty:
                raise ValueError(f"Model {model} missing from {results_dir / 'cv_r2_scores.csv'}")
            vals.append(float(row.iloc[0][value_col]))
            labels.append(FIG1_MODEL_LABELS[model])
        bars = ax.bar(range(len(vals)), vals, color=FIG1_BAR_COLORS, edgecolor="none", width=0.58)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(r"Coefficient of determination ($R^2$)")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=13, pad=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2.0, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)

    plt.tight_layout(h_pad=1.6)
    out_path = out_fig_dir / "fig_S1_cv_r2_best_model_colored.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _load_fig2_source(project_root: Path, results_dir: Path, task: str):
    import pandas as pd

    filename = "rf_predictions_ice_logit.csv" if task == "ice" else "rf_predictions_plateau.csv"
    path = results_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing Fig. S2 source file: {path}")
    return pd.read_csv(path)


def make_fig_s2(project_root: Path, results_dir: Path, out_fig_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    out_fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.2), dpi=300)

    for ax, task, panel_letter in zip(axes, ["ice", "plateau"], ["a", "b"]):
        df = _load_fig2_source(project_root, results_dir, task)
        meta = FIG2_PANEL_META[task]
        x_col = meta["x_col"]
        y_col = meta["y_col"]

        df_test = df[df["split"].str.lower() == "test"].copy()
        df_train = df[df["split"].str.lower() == "train"].copy()

        ax.scatter(
            df_test[x_col],
            df_test[y_col],
            s=18,
            facecolor=FIG2_TEST_COLOR,
            edgecolor=FIG2_TEST_EDGE,
            linewidth=0.5,
            alpha=0.8,
            label="Test",
        )
        ax.scatter(
            df_train[x_col],
            df_train[y_col],
            s=18,
            facecolor=FIG2_TRAIN_COLOR,
            edgecolor=FIG2_TRAIN_EDGE,
            linewidth=0.5,
            alpha=0.8,
            label="Train",
        )

        low, high = meta["lims"]
        ax.plot([low, high], [low, high], linestyle="--", linewidth=1.2, color=FIG2_LINE_COLOR)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xlabel(meta["x_label"])
        ax.set_ylabel(meta["y_label"])
        ax.legend(loc="upper left", frameon=False, fontsize=9, handletextpad=0.4)
        ax.text(0.02, 1.02, panel_letter, transform=ax.transAxes, fontsize=13, fontweight="bold")
        ax.text(
            0.94,
            0.06,
            f"R$^2$={meta['r2']:.3f}\nRMAE={meta['rmae']:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    plt.tight_layout(w_pad=1.3)
    out_path = out_fig_dir / "fig_S2_rf_performance_best_model_colored.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _nice_feature_order(mean_abs_path: Path) -> List[str]:
    import pandas as pd

    df = pd.read_csv(mean_abs_path)
    return df.sort_values("rank")["feature"].tolist()


def _custom_black_to_teal():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "black_to_teal",
        [(0.0, "#111111"), (0.5, "#7fd0c9"), (1.0, "#2ca6a4")],
    )


def make_fig_s3(project_root: Path, reviewer_dir: Path, out_fig_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    import pandas as pd
    import shap

    out_fig_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.8, 8.6), dpi=300)

    task_panels = [
        ("ice", "a", plt.get_cmap("coolwarm_r")),
        ("plateau", "b", _custom_black_to_teal()),
    ]

    for i, (task, panel_letter, cmap) in enumerate(task_panels, start=1):
        shap_dir = reviewer_dir / f"multimodel_shap_{task}" / "randomforest"
        x_path = shap_dir / "X_test_used_for_shap.csv"
        s_path = shap_dir / "shap_values_test.csv"
        mean_abs_path = shap_dir / "mean_abs_shap.csv"
        if not x_path.exists() or not s_path.exists() or not mean_abs_path.exists():
            raise FileNotFoundError(f"Missing SHAP inputs for Fig. S3 task={task} in {shap_dir}")

        X_test = pd.read_csv(x_path).drop(columns=["sample_index"])
        shap_df = pd.read_csv(s_path).drop(columns=["sample_index"])
        ordered_features = _nice_feature_order(mean_abs_path)

        X_plot = X_test[ordered_features].copy()
        shap_plot = shap_df[ordered_features].copy()
        X_plot.columns = [FEATURE_LABELS.get(c, c) for c in X_plot.columns]
        shap_plot.columns = X_plot.columns

        ax = plt.subplot(2, 1, i)
        shap.summary_plot(
            shap_plot.to_numpy(),
            X_plot,
            show=False,
            plot_size=None,
            cmap=cmap,
            color_bar=True,
            max_display=len(ordered_features),
        )
        ax.text(-0.09, 1.04, panel_letter, transform=ax.transAxes, fontsize=13, fontweight="bold")
        ax.set_title("")
        ax.tick_params(axis="both", labelsize=9)

    plt.tight_layout(h_pad=1.0)
    out_path = out_fig_dir / "fig_S3_shap_best_model_colored.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    reviewer_dir = project_root / args.reviewer_dir
    results_dir = project_root / args.results_dir
    out_shap_dir = project_root / args.out_shap_dir
    out_fig_dir = project_root / args.out_fig_dir

    written_csvs = export_feature_level_shap_csvs(project_root, reviewer_dir, args.data_dir, out_shap_dir)
    fig1 = make_fig_s1(project_root, results_dir, out_fig_dir)
    fig2 = make_fig_s2(project_root, results_dir, out_fig_dir)
    fig3 = make_fig_s3(project_root, reviewer_dir, out_fig_dir)

    manifest = {
        "project_root": str(project_root),
        "reviewer_dir": str(reviewer_dir),
        "results_dir": str(results_dir),
        "n_shap_csv_files": len(written_csvs),
        "shap_csv_files": [str(p.relative_to(project_root)) for p in written_csvs],
        "figure_files": [
            str(fig1.relative_to(project_root)),
            str(fig2.relative_to(project_root)),
            str(fig3.relative_to(project_root)),
        ],
    }
    manifest_path = out_fig_dir / "best_model_export_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Saved 12 SHAP CSV files to: {out_shap_dir}")
    print(f"Saved figure files:\n- {fig1}\n- {fig2}\n- {fig3}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
