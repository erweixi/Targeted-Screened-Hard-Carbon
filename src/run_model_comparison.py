"""Train all benchmark models and export a manuscript-ready comparison table.

This script addresses reviewer questions about model selection by reporting:
- cross-validated mean/std performance on the training split
- train-CV metrics from cross_val_predict
- hold-out test metrics
- generalization gap (train-CV R2 - test R2)
- best parameters for tuned models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model_registry import get_model_specs
from .utils import SCORING, cv_scores_to_summary, ensure_jsonable, fold_metrics_frame, load_xy, regression_metrics, require_skopt, split_train_test


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["ice", "plateau"], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--n-iter", type=int, default=20)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--outdir", default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    require_skopt()
    from skopt import BayesSearchCV

    import pandas as pd
    from sklearn.model_selection import cross_val_predict, cross_validate

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X, y = load_xy(args.data, args.task)
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=0.2, random_state=args.random_state
    )

    rows = []
    best_params = {
        "task": args.task,
        "data": str(args.data),
        "random_state": args.random_state,
        "cv_folds": args.cv_folds,
        "n_iter": args.n_iter,
    }
    fold_dir = outdir / f"model_comparison_cv_folds_{args.task}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for spec in get_model_specs(task=args.task, random_state=args.random_state, n_jobs=args.n_jobs):
        if spec.search_space is None:
            estimator = spec.estimator
            best_params[spec.display_name] = {"tuned": False, "best_params": None}
        else:
            search = BayesSearchCV(
                spec.estimator,
                search_spaces=spec.search_space,
                n_iter=args.n_iter,
                cv=args.cv_folds,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
                scoring="r2",
            )
            search.fit(X_train, y_train)
            estimator = search.best_estimator_
            best_params[spec.display_name] = {
                "tuned": True,
                "best_score": float(search.best_score_),
                "best_params": ensure_jsonable(search.best_params_),
            }

        cv_results = cross_validate(
            estimator,
            X_train,
            y_train,
            cv=args.cv_folds,
            scoring=SCORING,
            return_train_score=False,
            n_jobs=args.n_jobs,
        )
        cv_summary = cv_scores_to_summary(cv_results)
        y_pred_train_cv = cross_val_predict(estimator, X_train, y_train, cv=args.cv_folds, n_jobs=args.n_jobs)
        train_metrics = regression_metrics(y_train, y_pred_train_cv)

        estimator.fit(X_train, y_train)
        y_pred_test = estimator.predict(X_test)
        test_metrics = regression_metrics(y_test, y_pred_test)

        row = {
            "model": spec.display_name,
            "tuned": spec.search_space is not None,
            "cv_r2_mean": cv_summary["cv_r2_mean"],
            "cv_r2_std": cv_summary["cv_r2_std"],
            "cv_mae_mean": cv_summary["cv_mae_mean"],
            "cv_rmse_mean": cv_summary["cv_rmse_mean"],
            "train_cv_r2": train_metrics["r2"],
            "train_cv_mae": train_metrics["mae"],
            "train_cv_rmse": train_metrics["rmse"],
            "test_r2": test_metrics["r2"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "generalization_gap_r2": train_metrics["r2"] - test_metrics["r2"],
            "supports_shap": spec.supports_shap,
        }
        rows.append(row)

        fold_path = fold_dir / f"{spec.slug}_cv_metrics.csv"
        fold_metrics_frame(cv_results).to_csv(fold_path, index=False)

    df = pd.DataFrame(rows).sort_values(["test_r2", "cv_r2_mean", "cv_r2_std"], ascending=[False, False, True])
    csv_path = outdir / f"model_comparison_{args.task}.csv"
    df.to_csv(csv_path, index=False)

    json_path = outdir / f"model_comparison_best_params_{args.task}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    print(df.to_string(index=False))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
