"""Compute SHAP explanations for multiple models and save each model in its own folder.

Reviewer-driven outputs:
- one subfolder per model
- SHAP beeswarm and bar plots
- mean absolute SHAP ranking tables
- per-sample SHAP values on the hold-out test split
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model_registry import get_model_spec
from .utils import ensure_jsonable, load_xy, require_skopt, split_train_test


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["ice", "plateau"], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--models", nargs="+", default=["RandomForest", "GBR", "XGB"], help="Model names or slugs")
    p.add_argument("--n-iter", type=int, default=20)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--outdir", default="results")
    return p.parse_args()


def _build_explainer(shap_family: str, estimator, X_background):
    import shap

    if shap_family == "tree":
        return shap.TreeExplainer(estimator)
    if shap_family == "linear":
        return shap.LinearExplainer(estimator, X_background)
    return shap.Explainer(estimator.predict, X_background)


def _explanation_to_array(explanation):
    values = getattr(explanation, "values", explanation)
    import numpy as np

    arr = np.asarray(values)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def main() -> None:
    args = parse_args()

    require_skopt()
    from skopt import BayesSearchCV

    import joblib
    import matplotlib.pyplot as plt
    import pandas as pd
    import shap

    outdir = Path(args.outdir) / f"multimodel_shap_{args.task}"
    outdir.mkdir(parents=True, exist_ok=True)

    X, y = load_xy(args.data, args.task)
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=0.2, random_state=args.random_state
    )

    overall_rows = []

    for model_name in args.models:
        spec = get_model_spec(args.task, model_name, random_state=args.random_state, n_jobs=args.n_jobs)
        if not spec.supports_shap:
            raise ValueError(f"Model {spec.display_name} is not supported for SHAP in this workflow.")

        model_dir = outdir / spec.slug
        model_dir.mkdir(parents=True, exist_ok=True)

        estimator = spec.estimator
        best_params = None
        best_score = None
        if spec.search_space is not None:
            search = BayesSearchCV(
                estimator,
                search_spaces=spec.search_space,
                n_iter=args.n_iter,
                cv=args.cv_folds,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
                scoring="r2",
            )
            search.fit(X_train, y_train)
            estimator = search.best_estimator_
            best_params = ensure_jsonable(search.best_params_)
            best_score = float(search.best_score_)
        else:
            estimator.fit(X_train, y_train)

        estimator.fit(X_train, y_train)
        explainer = _build_explainer(spec.shap_family, estimator, X_train)
        explanation = explainer(X_test)
        shap_values = _explanation_to_array(explanation)

        mean_abs = pd.DataFrame(
            {
                "feature": X_test.columns,
                "mean_abs_shap": shap_values.__abs__().mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        mean_abs["rank"] = range(1, len(mean_abs) + 1)
        mean_abs.to_csv(model_dir / "mean_abs_shap.csv", index=False)

        shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
        shap_df.insert(0, "sample_index", list(X_test.index))
        shap_df.to_csv(model_dir / "shap_values_test.csv", index=False)

        test_data = X_test.copy()
        test_data.insert(0, "sample_index", list(X_test.index))
        test_data.to_csv(model_dir / "X_test_used_for_shap.csv", index=False)

        metadata = {
            "task": args.task,
            "data": str(args.data),
            "model": spec.display_name,
            "model_slug": spec.slug,
            "random_state": args.random_state,
            "cv_folds": args.cv_folds,
            "n_iter": args.n_iter,
            "best_params": best_params,
            "best_cv_score": best_score,
            "n_test": int(len(X_test)),
        }
        with (model_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

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

        for _, row in mean_abs.iterrows():
            overall_rows.append(
                {
                    "model": spec.display_name,
                    "model_slug": spec.slug,
                    "feature": row["feature"],
                    "mean_abs_shap": float(row["mean_abs_shap"]),
                    "rank": int(row["rank"]),
                }
            )

        print(f"Saved SHAP outputs for {spec.display_name} to {model_dir}")

    pd.DataFrame(overall_rows).to_csv(outdir / "multimodel_shap_summary.csv", index=False)
    print(f"Saved: {outdir / 'multimodel_shap_summary.csv'}")


if __name__ == "__main__":
    main()
