"""Tune Random Forest via Bayesian optimization and evaluate on train/test.

This script reproduces the core ML workflow used in the paper:
- train/test split (8:2)
- Bayesian hyperparameter search (BayesSearchCV)
- k-fold cross-validation on the training set
- final evaluation on the hold-out test set

Outputs manuscript-ready numeric artifacts only (no plots).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .search_spaces import get_search_spaces
from .utils import FEATURE_COLUMNS, SCORING, cv_scores_to_summary, ensure_jsonable, fold_metrics_frame, load_xy, regression_metrics, require_skopt, split_train_test


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["ice", "plateau"], required=True)
    p.add_argument("--data", required=True, help="Path to the task CSV dataset")
    p.add_argument("--n-iter", type=int, default=20, help="BayesSearchCV iterations")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--outdir", default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    require_skopt()
    from skopt import BayesSearchCV

    import pandas as pd
    from joblib import dump
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_predict, cross_validate

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X, y = load_xy(args.data, args.task)
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=0.2, random_state=args.random_state
    )

    spaces = get_search_spaces(args.task)
    base_model = RandomForestRegressor(random_state=args.random_state)

    opt = BayesSearchCV(
        base_model,
        search_spaces=spaces["rf"],
        n_iter=args.n_iter,
        cv=args.cv_folds,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        scoring="r2",
    )
    opt.fit(X_train, y_train)

    best_model = opt.best_estimator_

    cv_results = cross_validate(
        best_model,
        X_train,
        y_train,
        cv=args.cv_folds,
        scoring=SCORING,
        return_train_score=False,
        n_jobs=args.n_jobs,
    )
    cv_summary = cv_scores_to_summary(cv_results)

    y_pred_train_cv = cross_val_predict(best_model, X_train, y_train, cv=args.cv_folds, n_jobs=args.n_jobs)
    train_metrics = regression_metrics(y_train, y_pred_train_cv)

    best_model.fit(X_train, y_train)
    y_pred_test = best_model.predict(X_test)
    test_metrics = regression_metrics(y_test, y_pred_test)

    payload = {
        "task": args.task,
        "data": str(args.data),
        "random_state": args.random_state,
        "n_iter": args.n_iter,
        "cv_folds": args.cv_folds,
        "n_total": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_columns": FEATURE_COLUMNS,
        "best_cv_score_during_search": float(opt.best_score_),
        "best_params": ensure_jsonable(opt.best_params_),
        "cv_summary": cv_summary,
        "train_cv_metrics": train_metrics,
        "test_metrics": test_metrics,
        "generalization_gap_r2": float(train_metrics["r2"] - test_metrics["r2"]),
    }

    metrics_path = outdir / f"rf_metrics_{args.task}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary_path = outdir / f"rf_metrics_summary_{args.task}.csv"
    pd.DataFrame(
        [
            {
                "task": args.task,
                "n_total": payload["n_total"],
                "n_train": payload["n_train"],
                "n_test": payload["n_test"],
                "cv_folds": payload["cv_folds"],
                "cv_r2_mean": cv_summary["cv_r2_mean"],
                "cv_r2_std": cv_summary["cv_r2_std"],
                "train_cv_r2": train_metrics["r2"],
                "train_cv_mae": train_metrics["mae"],
                "train_cv_rmse": train_metrics["rmse"],
                "test_r2": test_metrics["r2"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "generalization_gap_r2": payload["generalization_gap_r2"],
            }
        ]
    ).to_csv(summary_path, index=False)

    train_pred_path = outdir / f"rf_predictions_train_cv_{args.task}.csv"
    pd.DataFrame({"y_true": y_train, "y_pred": y_pred_train_cv}).to_csv(
        train_pred_path, index=False
    )

    test_pred_path = outdir / f"rf_predictions_test_{args.task}.csv"
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred_test}).to_csv(
        test_pred_path, index=False
    )

    fold_path = outdir / f"rf_cv_fold_metrics_{args.task}.csv"
    fold_metrics_frame(cv_results).to_csv(fold_path, index=False)

    model_path = outdir / f"rf_model_{args.task}.joblib"
    dump(best_model, model_path)

    print(json.dumps(payload["train_cv_metrics"], indent=2))
    print(json.dumps(payload["test_metrics"], indent=2))
    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {fold_path}")
    print(f"Saved: {train_pred_path}")
    print(f"Saved: {test_pred_path}")
    print(f"Saved: {model_path}")


if __name__ == "__main__":
    main()
