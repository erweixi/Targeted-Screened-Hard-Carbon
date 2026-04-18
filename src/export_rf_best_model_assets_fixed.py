from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def _import_utils():
    try:
        from .utils import FEATURE_COLUMNS, TASK_SPECS
    except ImportError:
        from utils import FEATURE_COLUMNS, TASK_SPECS
    return FEATURE_COLUMNS, TASK_SPECS


FEATURE_COLUMNS, TASK_SPECS = _import_utils()

FIG1_BAR_COLORS = [
    (155 / 255, 213 / 255, 155 / 255),
    (114 / 255, 154 / 255, 213 / 255),
    (236 / 255, 196 / 255, 133 / 255),
    (230 / 255, 230 / 255, 145 / 255),
    (232 / 255, 106 / 255, 176 / 255),
    (232 / 255, 137 / 255, 72 / 255),
    (160 / 255, 160 / 255, 160 / 255),
]

FIG1_MODEL_ORDER = [
    "Linear",
    "GBR",
    "Ridge",
    "Lasso",
    "NeuralNetwork",
    "RandomForest",
    "XGB",
]

FIG1_MODEL_LABELS = {
    "Linear": "Linear",
    "GBR": "GBR",
    "Ridge": "Ridge",
    "Lasso": "Lasso",
    "NeuralNetwork": "Neural\nNetwork",
    "RandomForest": "Random\nForest",
    "XGB": "XGB",
}


IMAGE_WHITE_THRESHOLD = 250


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export fixed Random-Forest SHAP CSVs and supplementary figures from saved reviewer outputs."
    )
    p.add_argument("--project-root", default=".", help="Project root")
    p.add_argument("--data-dir", default="data", help="Data directory")
    p.add_argument("--reviewer-dir", default="results/reviewer", help="Saved reviewer results directory")
    p.add_argument("--outputs-dir", default="outputs", help="Directory with manuscript figure PNGs")
    p.add_argument(
        "--out-shap-dir",
        default="results/manuscript_exports_rf_fixed/shap_feature_csvs",
        help="Directory for 12 per-feature SHAP CSV files",
    )
    p.add_argument(
        "--out-fig-dir",
        default="outputs",
        help="Directory for exported fixed S1/S2/S3 figures",
    )
    return p.parse_args()



def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)



def _target_csv_name(task: str) -> str:
    return "hc_dataset_ice.csv" if task == "ice" else "hc_dataset_plateau.csv"



def _load_target_series(project_root: Path, data_dir: str, task: str):
    import pandas as pd

    csv_path = project_root / data_dir / _target_csv_name(task)
    df = pd.read_csv(csv_path)
    target_col = TASK_SPECS[task].target_column
    return df[target_col], target_col, csv_path



def export_feature_level_shap_csvs(project_root: Path, reviewer_dir: Path, data_dir: str, outdir: Path) -> List[Path]:
    import pandas as pd

    outdir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for task in ["ice", "plateau"]:
        shap_dir = reviewer_dir / f"multimodel_shap_{task}" / "randomforest"
        x_path = shap_dir / "X_test_used_for_shap.csv"
        s_path = shap_dir / "shap_values_test.csv"
        meta_path = shap_dir / "metadata.json"
        if not x_path.exists() or not s_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Missing RandomForest SHAP saved files for task={task} under {shap_dir}")

        X_test = pd.read_csv(x_path)
        shap_df = pd.read_csv(s_path)
        meta = _read_json(meta_path)

        if "sample_index" not in X_test.columns or "sample_index" not in shap_df.columns:
            raise ValueError(f"sample_index missing in saved SHAP inputs for task={task}")
        if list(X_test["sample_index"]) != list(shap_df["sample_index"]):
            raise ValueError(f"sample_index mismatch between X_test and shap_values for task={task}")

        if meta.get("best_params") is None:
            raise ValueError(f"Saved RandomForest metadata for task={task} does not include best_params")

        target_series, target_col, _ = _load_target_series(project_root, data_dir, task)
        sample_index = X_test["sample_index"].astype(int)
        target_values = target_series.loc[sample_index].to_numpy()

        for feature in FEATURE_COLUMNS:
            if feature not in X_test.columns or feature not in shap_df.columns:
                raise ValueError(f"Feature {feature} missing from saved SHAP files for task={task}")
            export_df = pd.DataFrame(
                {
                    "feature_value": X_test[feature].to_numpy(),
                    "shap_value": shap_df[feature].to_numpy(),
                    "target_value": target_values,
                }
            )
            out_path = outdir / f"{task}__{feature}.csv"
            export_df.to_csv(out_path, index=False)
            written.append(out_path)

    return written



def make_fig_s1(project_root: Path, outputs_dir: Path, out_fig_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    import pandas as pd

    out_fig_dir.mkdir(parents=True, exist_ok=True)
    existing_png = outputs_dir / "fig_S1_cv_r2_corrected.png"
    out_path = out_fig_dir / "fig_S1_cv_r2_rf_fixed.png"
    if existing_png.exists():
        shutil.copy2(existing_png, out_path)
        return out_path

    ref_csv = project_root / "results" / "reference_fig_s1_r2.csv"
    if not ref_csv.exists():
        raise FileNotFoundError(f"Could not find saved Fig. S1 PNG or reference CSV: {existing_png} / {ref_csv}")

    df = pd.read_csv(ref_csv)
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 6.4), dpi=300)
    panels = [("ice", "a"), ("plateau", "b")]
    for ax, (task, panel_letter) in zip(axes, panels):
        task_df = df[df["task"] == task].copy()
        task_df["model"] = pd.Categorical(task_df["model"], FIG1_MODEL_ORDER, ordered=True)
        task_df = task_df.sort_values("model")
        vals = task_df["cv_r2_reference"].tolist()
        labels = [FIG1_MODEL_LABELS[m] for m in task_df["model"].tolist()]
        bars = ax.bar(range(len(vals)), vals, color=FIG1_BAR_COLORS, width=0.58, edgecolor="black", linewidth=0.5)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(r"Coefficient of determination ($R^2$)", fontsize=9)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.text(-0.08, 1.02, panel_letter, transform=ax.transAxes, fontsize=12, fontweight="bold")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2.0, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.tick_params(axis="y", labelsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
    plt.tight_layout(h_pad=1.2)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path



def make_fig_s2(project_root: Path, reviewer_dir: Path, outputs_dir: Path, out_fig_dir: Path) -> Path:
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    existing_png = outputs_dir / "fig_S2_rf_performance.png"
    out_path = out_fig_dir / "fig_S2_rf_performance_rf_fixed.png"
    if existing_png.exists():
        shutil.copy2(existing_png, out_path)
        return out_path

    import matplotlib.pyplot as plt
    import pandas as pd

    color_test_face = "#d7c1a9"
    color_train_face = "#0f7ea1"
    color_line = "#c77b33"

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0), dpi=300)
    for ax, task, panel_letter in zip(axes, ["ice", "plateau"], ["a", "b"]):
        train_path = reviewer_dir / f"rf_predictions_train_cv_{task}.csv"
        test_path = reviewer_dir / f"rf_predictions_test_{task}.csv"
        metrics_path = reviewer_dir / f"rf_metrics_summary_{task}.csv"
        if not train_path.exists() or not test_path.exists() or not metrics_path.exists():
            raise FileNotFoundError(f"Missing saved reviewer RF outputs for task={task}")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        metrics = pd.read_csv(metrics_path).iloc[0]

        ax.scatter(test_df["y_true"], test_df["y_pred"], s=15, color=color_test_face, edgecolors="none", label="Test")
        ax.scatter(train_df["y_true"], train_df["y_pred"], s=12, color=color_train_face, edgecolors="none", label="Train")

        lo = min(train_df["y_true"].min(), test_df["y_true"].min(), train_df["y_pred"].min(), test_df["y_pred"].min())
        hi = max(train_df["y_true"].max(), test_df["y_true"].max(), train_df["y_pred"].max(), test_df["y_pred"].max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", color=color_line, linewidth=1.1)
        ax.plot([lo, hi], [lo * 1.2, hi * 1.2], linestyle="--", color=color_line, linewidth=0.8)
        ax.plot([lo, hi], [lo * 0.8, hi * 0.8], linestyle="--", color=color_line, linewidth=0.8)

        if task == "ice":
            ax.set_xlabel("ICE_Real (%)")
            ax.set_ylabel("ICE_Predict (%)")
        else:
            ax.set_xlabel(r"Plateau capacity_Real (mAh g$^{-1}$)")
            ax.set_ylabel(r"Plateau capacity_Predict (mAh g$^{-1}$)")
        ax.legend(loc="upper left", frameon=False, fontsize=8, handletextpad=0.3)
        ax.text(-0.12, 1.04, panel_letter, transform=ax.transAxes, fontsize=12, fontweight="bold")
        ax.text(
            0.98,
            0.05,
            f"R$^2$={float(metrics['cv_r2_mean']):.3f}\nRMAE={float(metrics['cv_rmse_mean']) / (hi - lo):.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
    plt.tight_layout(w_pad=1.2)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path



def _crop_nonwhite(image):
    from PIL import ImageChops

    bg = image.copy()
    bg.paste((255, 255, 255), [0, 0, image.size[0], image.size[1]])
    diff = ImageChops.difference(image, bg)
    bbox = diff.getbbox()
    if bbox is None:
        return image.copy()
    left, upper, right, lower = bbox
    pad = 20
    left = max(0, left - pad)
    upper = max(0, upper - pad)
    right = min(image.size[0], right + pad)
    lower = min(image.size[1], lower + pad)
    return image.crop((left, upper, right, lower))



def make_fig_s3(reviewer_dir: Path, out_fig_dir: Path) -> Path:
    from PIL import Image, ImageDraw

    out_fig_dir.mkdir(parents=True, exist_ok=True)
    ice_png = reviewer_dir / "multimodel_shap_ice" / "randomforest" / "shap_summary_beeswarm.png"
    plateau_png = reviewer_dir / "multimodel_shap_plateau" / "randomforest" / "shap_summary_beeswarm.png"
    if not ice_png.exists() or not plateau_png.exists():
        raise FileNotFoundError(f"Missing saved RandomForest SHAP beeswarm PNGs: {ice_png} / {plateau_png}")

    ice_img = _crop_nonwhite(Image.open(ice_png).convert("RGB"))
    plateau_img = _crop_nonwhite(Image.open(plateau_png).convert("RGB"))

    target_width = max(ice_img.width, plateau_img.width)
    def _resize(im):
        if im.width == target_width:
            return im
        scale = target_width / im.width
        return im.resize((target_width, int(im.height * scale)), Image.Resampling.LANCZOS)

    ice_img = _resize(ice_img)
    plateau_img = _resize(plateau_img)

    margin = 40
    gap = 28
    label_space = 28
    canvas_w = target_width + margin * 2
    canvas_h = ice_img.height + plateau_img.height + margin * 2 + gap + label_space
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    x = margin
    y1 = margin
    canvas.paste(ice_img, (x, y1))
    draw.text((12, y1 + 4), "a", fill="black")

    y2 = y1 + ice_img.height + gap + label_space
    canvas.paste(plateau_img, (x, y2))
    draw.text((12, y2 + 4), "b", fill="black")

    out_path = out_fig_dir / "fig_S3_shap_rf_fixed.png"
    canvas.save(out_path)
    return out_path



def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    reviewer_dir = project_root / args.reviewer_dir
    outputs_dir = project_root / args.outputs_dir
    out_shap_dir = project_root / args.out_shap_dir
    out_fig_dir = project_root / args.out_fig_dir

    rf_ice_json = reviewer_dir / "rf_metrics_ice.json"
    rf_plateau_json = reviewer_dir / "rf_metrics_plateau.json"
    if not rf_ice_json.exists() or not rf_plateau_json.exists():
        raise FileNotFoundError("Expected saved reviewer RF metric JSON files were not found.")
    ice_meta = _read_json(rf_ice_json)
    plateau_meta = _read_json(rf_plateau_json)

    written_csvs = export_feature_level_shap_csvs(project_root, reviewer_dir, args.data_dir, out_shap_dir)
    fig1 = make_fig_s1(project_root, outputs_dir, out_fig_dir)
    fig2 = make_fig_s2(project_root, reviewer_dir, outputs_dir, out_fig_dir)
    fig3 = make_fig_s3(reviewer_dir, out_fig_dir)

    manifest = {
        "sources": {
            "ice_rf_json": str(rf_ice_json.relative_to(project_root)),
            "plateau_rf_json": str(rf_plateau_json.relative_to(project_root)),
        },
        "rf_best_params": {
            "ice": ice_meta.get("best_params"),
            "plateau": plateau_meta.get("best_params"),
        },
        "rf_metrics": {
            "ice": {
                "cv_r2_mean": ice_meta.get("cv_summary", {}).get("cv_r2_mean"),
                "train_cv_r2": ice_meta.get("train_cv_metrics", {}).get("r2"),
                "test_r2": ice_meta.get("test_metrics", {}).get("r2"),
            },
            "plateau": {
                "cv_r2_mean": plateau_meta.get("cv_summary", {}).get("cv_r2_mean"),
                "train_cv_r2": plateau_meta.get("train_cv_metrics", {}).get("r2"),
                "test_r2": plateau_meta.get("test_metrics", {}).get("r2"),
            },
        },
        "shap_csv_files": [str(p.relative_to(project_root)) for p in sorted(written_csvs)],
        "figure_files": [
            str(fig1.relative_to(project_root)),
            str(fig2.relative_to(project_root)),
            str(fig3.relative_to(project_root)),
        ],
    }
    manifest_path = out_fig_dir / "best_model_rf_fixed_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("Saved 12 SHAP CSV files to:", out_shap_dir)
    print("Saved figures:")
    print("-", fig1)
    print("-", fig2)
    print("-", fig3)
    print("Saved manifest:", manifest_path)


if __name__ == "__main__":
    main()
