"""Run reviewer-requested feature robustness analyses.

Outputs:
- permutation importance on the hold-out test set
- leave-one-feature-out ablation table (optional; enabled by default)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .search_spaces import get_search_spaces
from .utils import FEATURE_COLUMNS, ensure_jsonable, load_xy, regression_metrics, require_skopt, split_train_test


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["ice", "plateau"], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--n-iter", type=int, default=20)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--permutation-repeats", type=int, default=30)
    p.add_argument("--skip-ablation", action="store_true")
    p.add_argument("--outdir", default="results")
    return p.parse_args()


def _tune_rf(X_train, y_train, task: str, n_iter: int, cv_folds: int, random_state: int, n_jobs: int):
    from sklearn.ensemble import RandomForestRegressor
    from skopt import BayesSearchCV

    spaces = get_search_spaces(task)
    search = BayesSearchCV(
        RandomForestRegressor(random_state=random_state),
        search_spaces=spaces["rf"],
        n_iter=n_iter,
        cv=cv_folds,
        random_state=random_state,
        n_jobs=n_jobs,
        scoring="r2",
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search


def main() -> None:
    args = parse_args()

    require_skopt()

    import pandas as pd
    from sklearn.inspection import permutation_importance

    outdir = Path(args.outdir) / f"feature_robustness_{args.task}"
    outdir.mkdir(parents=True, exist_ok=True)

    X, y = load_xy(args.data, args.task)
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=0.2, random_state=args.random_state
    )

    best_model, search = _tune_rf(
        X_train,
        y_train,
        task=args.task,
        n_iter=args.n_iter,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    best_model.fit(X_train, y_train)
    baseline_pred = best_model.predict(X_test)
    baseline_metrics = regression_metrics(y_test, baseline_pred)

    perm = permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats=args.permutation_repeats,
        random_state=args.random_state,
        scoring="r2",
        n_jobs=args.n_jobs,
    )
    perm_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    perm_df.to_csv(outdir / "permutation_importance_test.csv", index=False)

    ablation_rows = []
    if not args.skip_ablation:
        for feature in FEATURE_COLUMNS:
            reduced_features = [f for f in FEATURE_COLUMNS if f != feature]
            X_train_red = X_train[reduced_features]
            X_test_red = X_test[reduced_features]
            ablated_model, _ = _tune_rf(
                X_train_red,
                y_train,
                task=args.task,
                n_iter=args.n_iter,
                cv_folds=args.cv_folds,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
            )
            ablated_model.fit(X_train_red, y_train)
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

    if ablation_rows:
        ablation_df = pd.DataFrame(ablation_rows).sort_values("delta_test_r2", ascending=False)
        ablation_df.to_csv(outdir / "leave_one_feature_out.csv", index=False)

    payload = {
        "task": args.task,
        "data": str(args.data),
        "random_state": args.random_state,
        "cv_folds": args.cv_folds,
        "n_iter": args.n_iter,
        "permutation_repeats": args.permutation_repeats,
        "baseline_test_metrics": baseline_metrics,
        "best_params": ensure_jsonable(search.best_params_),
        "best_cv_score": float(search.best_score_),
        "skip_ablation": bool(args.skip_ablation),
    }
    with (outdir / "feature_robustness_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved: {outdir / 'permutation_importance_test.csv'}")
    if ablation_rows:
        print(f"Saved: {outdir / 'leave_one_feature_out.csv'}")
    print(f"Saved: {outdir / 'feature_robustness_metadata.json'}")


if __name__ == "__main__":
    main()
