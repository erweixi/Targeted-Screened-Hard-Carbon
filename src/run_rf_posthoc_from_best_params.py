"""Generate RandomForest SHAP and robustness outputs from a locked best_params JSON.

Use this when you want SHAP / permutation importance / ablation to be based on the
exact same RandomForest hyperparameters used in the final manuscript model.

Example:
    python -m src.run_rf_posthoc_from_best_params \
      --task ice \
      --data data/hc_dataset_ice.csv \
      --best-params-json results/reviewer/rf_metrics_ice.json \
      --outdir results/reviewer

This script writes:
- results/.../multimodel_shap_<task>/randomforest_locked/...   (SHAP artifacts)
- results/.../feature_robustness_<task>_locked/...             (robustness artifacts)

It does NOT retune the model. It rebuilds RandomForestRegressor from the saved
best_params, refits it on the same train split, and then generates SHAP and
robustness outputs from that locked model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _import_project_utils():
    """Support both `python -m src.xxx` and `python src/xxx.py` execution styles."""
    try:
        from .utils import FEATURE_COLUMNS, ensure_jsonable, load_xy, regression_metrics, split_train_test
    except ImportError:
        from utils import FEATURE_COLUMNS, ensure_jsonable, load_xy, regression_metrics, split_train_test
    return FEATURE_COLUMNS, ensure_jsonable, load_xy, regression_metrics, split_train_test


FEATURE_COLUMNS, ensure_jsonable, load_xy, regression_metrics, split_train_test = _import_project_utils()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["ice", "plateau"], required=True)
    p.add_argument("--data", required=True, help="CSV path used for the final model.")
    p.add_argument(
        "--best-params-json",
        required=True,
        help=(
            "Path to rf_metrics_<task>.json or any JSON containing a `best_params` dict. "
            "If the JSON itself is already just the params dict, that also works."
        ),
    )
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--permutation-repeats", type=int, default=30)
    p.add_argument(
        "--tag",
        default="locked",
        help="Suffix used in output folder names, so you do not overwrite existing SHAP/robustness results.",
    )
    p.add_argument("--outdir", default="results")
    return p.parse_args()


def _load_best_params(json_path: str | Path) -> Dict[str, Any]:
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "best_params" in payload and isinstance(payload["best_params"], dict):
        return payload["best_params"]

    if isinstance(payload, dict):
        # Accept a raw params dict as a convenience.
        return payload

    raise ValueError(f"Unsupported JSON structure in {path}. Expected a dict or a dict with `best_params`.")


def _sanitize_rf_params(params: Dict[str, Any], n_features: int) -> Dict[str, Any]:
    """Adjust params if needed for reduced-feature ablation models."""
    clean = dict(params)
    max_features = clean.get("max_features")
    if isinstance(max_features, int) and max_features > n_features:
        clean["max_features"] = n_features
    return clean


def _build_explainer(model, X_background):
    import shap

    return shap.TreeExplainer(model)


def _explanation_to_array(explanation):
    values = getattr(explanation, "values", explanation)
    import numpy as np

    arr = np.asarray(values)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def _fit_locked_rf(X_train, y_train, best_params: Dict[str, Any], random_state: int):
    from sklearn.ensemble import RandomForestRegressor

    params = dict(best_params)
    params.setdefault("random_state", random_state)
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model


def main() -> None:
    args = parse_args()

    import joblib
    import matplotlib.pyplot as plt
    import pandas as pd
    import shap
    from sklearn.inspection import permutation_importance

    X, y = load_xy(args.data, args.task)
    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    best_params = _load_best_params(args.best_params_json)
    locked_model = _fit_locked_rf(X_train, y_train, best_params, args.random_state)

    # ----- output folders -----
    outdir_root = Path(args.outdir)
    shap_root = outdir_root / f"multimodel_shap_{args.task}"
    shap_dir = shap_root / f"randomforest_{args.tag}"
    robustness_dir = outdir_root / f"feature_robustness_{args.task}_{args.tag}"
    shap_dir.mkdir(parents=True, exist_ok=True)
    robustness_dir.mkdir(parents=True, exist_ok=True)

    # ----- SHAP -----
    explainer = _build_explainer(locked_model, X_train)
    explanation = explainer(X_test)
    shap_values = _explanation_to_array(explanation)

    mean_abs = pd.DataFrame(
        {
            "feature": X_test.columns,
            "mean_abs_shap": abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    mean_abs["rank"] = range(1, len(mean_abs) + 1)
    mean_abs.to_csv(shap_dir / "mean_abs_shap.csv", index=False)

    shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
    shap_df.insert(0, "sample_index", list(X_test.index))
    shap_df.to_csv(shap_dir / "shap_values_test.csv", index=False)

    test_data = X_test.copy()
    test_data.insert(0, "sample_index", list(X_test.index))
    test_data.to_csv(shap_dir / "X_test_used_for_shap.csv", index=False)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(shap_dir / "shap_summary_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(shap_dir / "shap_summary_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    shap_metadata = {
        "task": args.task,
        "data": str(args.data),
        "best_params_json": str(args.best_params_json),
        "tag": args.tag,
        "random_state": args.random_state,
        "test_size": args.test_size,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "best_params": ensure_jsonable(best_params),
        "note": "This SHAP analysis uses locked best_params from the final RF JSON and does not retune.",
    }
    with (shap_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(shap_metadata, f, indent=2)

    pd.DataFrame(
        {
            "model": ["RandomForest"] * len(mean_abs),
            "model_slug": [f"randomforest_{args.tag}"] * len(mean_abs),
            "feature": mean_abs["feature"].tolist(),
            "mean_abs_shap": mean_abs["mean_abs_shap"].tolist(),
            "rank": mean_abs["rank"].tolist(),
        }
    ).to_csv(shap_root / f"multimodel_shap_summary_{args.tag}.csv", index=False)

    joblib.dump(locked_model, shap_dir / "model.joblib")

    # ----- robustness: baseline locked model -----
    baseline_pred = locked_model.predict(X_test)
    baseline_metrics = regression_metrics(y_test, baseline_pred)

    perm = permutation_importance(
        locked_model,
        X_test,
        y_test,
        n_repeats=args.permutation_repeats,
        random_state=args.random_state,
        scoring="r2",
        n_jobs=1,
    )
    perm_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    perm_df.to_csv(robustness_dir / "permutation_importance_test.csv", index=False)

    # ----- robustness: leave-one-feature-out with locked params -----
    ablation_rows = []
    param_adjustments = []
    for feature in FEATURE_COLUMNS:
        reduced_features = [f for f in FEATURE_COLUMNS if f != feature]
        X_train_red = X_train[reduced_features]
        X_test_red = X_test[reduced_features]

        adjusted_params = _sanitize_rf_params(best_params, n_features=len(reduced_features))
        if adjusted_params != best_params:
            param_adjustments.append(
                {
                    "removed_feature": feature,
                    "original_best_params": ensure_jsonable(best_params),
                    "adjusted_best_params": ensure_jsonable(adjusted_params),
                    "reason": "max_features exceeded the reduced number of available features",
                }
            )

        ablated_model = _fit_locked_rf(X_train_red, y_train, adjusted_params, args.random_state)
        ablated_pred = ablated_model.predict(X_test_red)
        ablated_metrics = regression_metrics(y_test, ablated_pred)
        ablation_rows.append(
            {
                "removed_feature": feature,
                "test_r2_without_feature": ablated_metrics["r2"],
                "delta_test_r2": baseline_metrics["r2"] - ablated_metrics["r2"],
                "test_mae_without_feature": ablated_metrics["mae"],
                "test_rmse_without_feature": ablated_metrics["rmse"],
            }
        )

    ablation_df = pd.DataFrame(ablation_rows).sort_values("delta_test_r2", ascending=False)
    ablation_df.to_csv(robustness_dir / "leave_one_feature_out.csv", index=False)

    robustness_metadata = {
        "task": args.task,
        "data": str(args.data),
        "best_params_json": str(args.best_params_json),
        "tag": args.tag,
        "random_state": args.random_state,
        "test_size": args.test_size,
        "permutation_repeats": args.permutation_repeats,
        "best_params": ensure_jsonable(best_params),
        "baseline_test_metrics": baseline_metrics,
        "ablation_param_adjustments": param_adjustments,
        "note": "This robustness analysis uses locked best_params from the final RF JSON and does not retune.",
    }
    with (robustness_dir / "feature_robustness_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(robustness_metadata, f, indent=2)

    print(f"Saved SHAP folder: {shap_dir}")
    print(f"Saved SHAP summary: {shap_root / f'multimodel_shap_summary_{args.tag}.csv'}")
    print(f"Saved robustness folder: {robustness_dir}")


if __name__ == "__main__":
    main()
