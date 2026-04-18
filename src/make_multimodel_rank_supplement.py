"""Build the missing seven-model feature-rank supplement for reviewer figures.

Why this script exists
----------------------
The repository already contains four-model SHAP summaries (RF/GBR/XGB/Ridge),
but the supplementary response text also references Linear, Lasso, and Neural
Network rankings. This script fills that gap without rerunning the expensive
Bayesian-search benchmark:

- Linear and Lasso: fit on the fixed 8:2 split and export SHAP rankings.
- NeuralNetwork: fit on the same split and export permutation-importance ranks.
- RF/GBR/XGB/Ridge: reuse the existing saved ranking tables.

Outputs are written under `results/reviewer/multimodel_shap_<task>/` and a
combined two-panel heatmap is exported to `outputs/`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .model_registry import get_model_spec
from .utils import FEATURE_COLUMNS, load_xy, split_train_test

FEATURE_LABELS = {
    "carbonization_temperature_C": "Temperature",
    "d002_nm": "d002",
    "id_ig": "ID/IG",
    "ssa_m2_g": "SSA",
    "electrolyte_type": "Electrolyte",
    "current_density_mA_g": "Current density",
}
TASK_DATA = {
    "ice": "data/hc_dataset_ice.csv",
    "plateau": "data/hc_dataset_plateau.csv",
}
DISPLAY_MODELS = ["RandomForest", "GBR", "XGB", "Ridge", "Lasso", "Linear", "NeuralNetwork"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-repeats", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--combined-heatmap",
        default="outputs/reviewer_multimodel_feature_rank_heatmap_7models.png",
    )
    return parser.parse_args()


def _build_linear_explainer(estimator, X_train):
    return shap.LinearExplainer(estimator, X_train)


def _values_to_array(explanation) -> np.ndarray:
    values = getattr(explanation, "values", explanation)
    arr = np.asarray(values)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def _save_shap_artifacts(model_dir: Path, model_name: str, estimator, X_train, X_test, task: str, extra_meta: Dict) -> pd.DataFrame:
    explainer = _build_linear_explainer(estimator, X_train)
    explanation = explainer(X_test)
    shap_values = _values_to_array(explanation)

    mean_abs = pd.DataFrame(
        {
            "feature": X_test.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    mean_abs["rank"] = np.arange(1, len(mean_abs) + 1)
    mean_abs.to_csv(model_dir / "mean_abs_shap.csv", index=False)

    shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
    shap_df.insert(0, "sample_index", list(X_test.index))
    shap_df.to_csv(model_dir / "shap_values_test.csv", index=False)

    test_df = X_test.copy()
    test_df.insert(0, "sample_index", list(X_test.index))
    test_df.to_csv(model_dir / "X_test_used_for_shap.csv", index=False)

    meta = {
        "task": task,
        "model": model_name,
        "n_test": int(len(X_test)),
        **extra_meta,
    }
    with (model_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    joblib.dump(estimator, model_dir / "model.joblib")

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(model_dir / "shap_summary_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(model_dir / "shap_summary_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    return mean_abs


def _load_best_params(repo_root: Path, task: str) -> Dict:
    path = repo_root / "results" / "reviewer" / f"model_comparison_best_params_{task}.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_existing_summary(outdir: Path) -> pd.DataFrame:
    path = outdir / "multimodel_shap_summary.csv"
    if not path.exists():
        return pd.DataFrame(columns=["model", "model_slug", "feature", "mean_abs_shap", "rank"])
    return pd.read_csv(path)


def _compute_linear_and_lasso(repo_root: Path, task: str, random_state: int) -> List[Dict]:
    data_path = repo_root / TASK_DATA[task]
    outdir = repo_root / "results" / "reviewer" / f"multimodel_shap_{task}"
    outdir.mkdir(parents=True, exist_ok=True)

    X, y = load_xy(data_path, task)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2, random_state=random_state)
    params_json = _load_best_params(repo_root, task)

    rows: List[Dict] = []

    linear_dir = outdir / "linear"
    linear_dir.mkdir(parents=True, exist_ok=True)
    linear_est = LinearRegression()
    linear_est.fit(X_train, y_train)
    linear_mean_abs = _save_shap_artifacts(
        linear_dir,
        "Linear",
        linear_est,
        X_train,
        X_test,
        task,
        {"importance_method": "mean_abs_shap", "best_params": None},
    )
    for _, row in linear_mean_abs.iterrows():
        rows.append(
            {
                "task": task,
                "model": "Linear",
                "model_slug": "linear",
                "feature": row["feature"],
                "rank": int(row["rank"]),
                "importance_method": "mean_abs_shap",
                "importance_value": float(row["mean_abs_shap"]),
            }
        )

    lasso_dir = outdir / "lasso"
    lasso_dir.mkdir(parents=True, exist_ok=True)
    alpha = float(params_json["Lasso"]["best_params"]["alpha"])
    lasso_est = Lasso(alpha=alpha, max_iter=20000)
    lasso_est.fit(X_train, y_train)
    lasso_mean_abs = _save_shap_artifacts(
        lasso_dir,
        "Lasso",
        lasso_est,
        X_train,
        X_test,
        task,
        {"importance_method": "mean_abs_shap", "best_params": {"alpha": alpha}},
    )
    for _, row in lasso_mean_abs.iterrows():
        rows.append(
            {
                "task": task,
                "model": "Lasso",
                "model_slug": "lasso",
                "feature": row["feature"],
                "rank": int(row["rank"]),
                "importance_method": "mean_abs_shap",
                "importance_value": float(row["mean_abs_shap"]),
            }
        )

    nn_dir = outdir / "neuralnetwork"
    nn_dir.mkdir(parents=True, exist_ok=True)
    nn_est = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(100, 100), random_state=random_state, max_iter=5000),
    )
    nn_est.fit(X_train, y_train)
    perm = permutation_importance(
        nn_est,
        X_test,
        y_test,
        n_repeats=30,
        random_state=random_state,
        scoring="r2",
    )
    nn_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    nn_df["rank"] = np.arange(1, len(nn_df) + 1)
    nn_df.to_csv(nn_dir / "permutation_importance_rank.csv", index=False)
    with (nn_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "task": task,
                "model": "NeuralNetwork",
                "importance_method": "permutation_importance",
                "n_repeats": 30,
                "n_test": int(len(X_test)),
            },
            f,
            indent=2,
        )
    joblib.dump(nn_est, nn_dir / "model.joblib")
    for _, row in nn_df.iterrows():
        rows.append(
            {
                "task": task,
                "model": "NeuralNetwork",
                "model_slug": "neuralnetwork",
                "feature": row["feature"],
                "rank": int(row["rank"]),
                "importance_method": "permutation_importance",
                "importance_value": float(row["importance_mean"]),
            }
        )

    return rows


def _merge_existing_and_new(repo_root: Path, task: str, new_rows: List[Dict]) -> pd.DataFrame:
    outdir = repo_root / "results" / "reviewer" / f"multimodel_shap_{task}"
    existing = _load_existing_summary(outdir)

    existing_rows: List[Dict] = []
    for _, row in existing.iterrows():
        existing_rows.append(
            {
                "task": task,
                "model": row["model"],
                "model_slug": row["model_slug"],
                "feature": row["feature"],
                "rank": int(row["rank"]),
                "importance_method": "mean_abs_shap",
                "importance_value": float(row["mean_abs_shap"]),
            }
        )

    combined = pd.DataFrame(existing_rows + new_rows)
    combined = combined.sort_values(["model", "rank", "feature"]).reset_index(drop=True)
    combined.to_csv(outdir / "multimodel_feature_rank_summary_7models.csv", index=False)
    return combined


def _pivot_for_heatmap(df: pd.DataFrame, task: str) -> pd.DataFrame:
    task_df = df[df["task"] == task].copy()
    model_order = [m for m in DISPLAY_MODELS if m in task_df["model"].unique()]
    pivot = task_df.pivot(index="model", columns="feature", values="rank")
    pivot = pivot.loc[model_order, FEATURE_COLUMNS]
    return pivot


def _draw_heatmap(ax, pivot: pd.DataFrame, title: str) -> None:
    values = pivot.to_numpy(dtype=float)
    im = ax.imshow(values, cmap="YlGnBu_r", vmin=1, vmax=len(FEATURE_COLUMNS))
    ax.set_xticks(np.arange(len(FEATURE_COLUMNS)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([FEATURE_LABELS[c] for c in FEATURE_COLUMNS], rotation=30, ha="right")
    ax.set_yticklabels(list(pivot.index))
    ax.set_title(title)

    ax.set_xticks(np.arange(-0.5, len(FEATURE_COLUMNS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{int(values[i, j])}", ha="center", va="center", fontsize=9)

    return im


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()

    all_rows: List[Dict] = []
    for task in ["ice", "plateau"]:
        new_rows = _compute_linear_and_lasso(repo_root, task, args.random_state)
        combined = _merge_existing_and_new(repo_root, task, new_rows)
        all_rows.extend(combined.to_dict(orient="records"))

    combined_df = pd.DataFrame(all_rows)
    combined_csv = repo_root / "results" / "reviewer" / "multimodel_feature_rank_summary_7models_combined.csv"
    combined_df.to_csv(combined_csv, index=False)

    ice_pivot = _pivot_for_heatmap(combined_df, "ice")
    plateau_pivot = _pivot_for_heatmap(combined_df, "plateau")

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7), constrained_layout=True)
    im = _draw_heatmap(axes[0], ice_pivot, "ICE: seven-model feature-rank consistency")
    _draw_heatmap(axes[1], plateau_pivot, "Plateau capacity: seven-model feature-rank consistency")
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, shrink=0.92)
    cbar.set_label("Rank (1 = most important)")

    out_png = repo_root / args.combined_heatmap
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] wrote {combined_csv}")
    print(f"[OK] wrote {out_png}")


if __name__ == "__main__":
    main()
