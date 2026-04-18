"""Run the 7-model benchmark with Bayesian hyperparameter search (Fig. S1 style).

This script:
- loads the dataset
- splits train/test (8:2)
- runs k-fold CV on the training set
- tunes selected models via BayesSearchCV (Bayesian optimization)

Outputs a CSV table of cross-validated performance and a JSON with best parameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model_registry import get_model_specs
from .utils import FEATURE_COLUMNS, SCORING, cv_scores_to_summary, ensure_jsonable, fold_metrics_frame, load_xy, require_skopt, split_train_test


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["ice", "plateau"], required=True)
    p.add_argument("--data", required=True, help="Path to the task CSV dataset")
    p.add_argument("--n-iter", type=int, default=20, help="BayesSearchCV iterations")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds")
    p.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for BayesSearchCV")
    p.add_argument("--outdir", default="results", help="Output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    require_skopt()
    from skopt import BayesSearchCV

    import pandas as pd
    from sklearn.model_selection import cross_validate

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load + split
    X, y = load_xy(args.data, args.task)
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=0.2, random_state=args.random_state
    )

    results = []
    best_params = {
        "task": args.task,
        "data": str(args.data),
        "random_state": args.random_state,
        "n_iter": args.n_iter,
        "cv_folds": args.cv_folds,
        "feature_columns": FEATURE_COLUMNS,
    }

    fold_dir = outdir / f"benchmark_cv_folds_{args.task}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for spec in get_model_specs(task=args.task, random_state=args.random_state, n_jobs=args.n_jobs):
        estimator = spec.estimator
        if spec.search_space is None:
            tuned_estimator = estimator
            search_payload = {"tuned": False, "best_params": None, "best_score": None}
        else:
            opt = BayesSearchCV(
                estimator,
                search_spaces=spec.search_space,
                n_iter=args.n_iter,
                cv=args.cv_folds,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
                scoring="r2",
            )
            opt.fit(X_train, y_train)
            tuned_estimator = opt.best_estimator_
            search_payload = {
                "tuned": True,
                "best_params": ensure_jsonable(opt.best_params_),
                "best_score": float(opt.best_score_),
            }

        cv_results = cross_validate(
            tuned_estimator,
            X_train,
            y_train,
            cv=args.cv_folds,
            scoring=SCORING,
            return_train_score=False,
            n_jobs=args.n_jobs,
        )
        summary = cv_scores_to_summary(cv_results)
        row = {
            "model": spec.display_name,
            "cv_r2_mean": summary["cv_r2_mean"],
            "cv_r2_std": summary["cv_r2_std"],
            "cv_mae_mean": summary["cv_mae_mean"],
            "cv_rmse_mean": summary["cv_rmse_mean"],
            "tuned": search_payload["tuned"],
        }
        results.append(row)
        best_params[spec.display_name] = {
            **search_payload,
            "cv_summary": summary,
        }

        fold_path = fold_dir / f"{spec.slug}_cv_metrics.csv"
        fold_metrics_frame(cv_results).to_csv(fold_path, index=False)

    df = pd.DataFrame(results).sort_values(["cv_r2_mean", "cv_r2_std"], ascending=[False, True])
    out_csv = outdir / f"fig_s1_benchmark_{args.task}.csv"
    df.to_csv(out_csv, index=False)

    legacy_csv = outdir / f"fig_s1_r2_{args.task}.csv"
    df[["model", "cv_r2_mean"]].rename(columns={"cv_r2_mean": "cv_r2"}).to_csv(legacy_csv, index=False)

    out_json = outdir / f"best_params_{args.task}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(ensure_jsonable(best_params), f, indent=2)

    print(df.to_string(index=False))
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {legacy_csv}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
