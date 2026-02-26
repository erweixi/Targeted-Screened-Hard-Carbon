"""Run the 7-model benchmark with Bayesian hyperparameter search (Fig. S1 style).

This script:
- loads the dataset
- splits train/test (8:2)
- runs 5-fold CV on the training set
- tunes selected models via BayesSearchCV (Bayesian optimization)

Outputs a CSV table of cross-validated R^2 values and a JSON with best parameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .search_spaces import get_search_spaces
from .utils import FEATURE_COLUMNS, load_xy, require_skopt, split_train_test


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["ice", "plateau"], required=True)
    p.add_argument("--data", required=True, help="Path to the task CSV dataset")
    p.add_argument("--n-iter", type=int, default=20, help="BayesSearchCV iterations")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for BayesSearchCV")
    p.add_argument("--outdir", default="results", help="Output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    require_skopt()
    from skopt import BayesSearchCV

    import pandas as pd
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    try:
        from xgboost import XGBRegressor
    except Exception as e:
        raise RuntimeError(
            "xgboost is required for the XGBRegressor benchmark. Install it with: pip install xgboost"
        ) from e

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load + split
    X, y = load_xy(args.data, args.task)
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=0.2, random_state=args.random_state
    )

    spaces = get_search_spaces(args.task)

    # Define models
    models = [
        ("Linear", LinearRegression(), None),
        (
            "GBR",
            GradientBoostingRegressor(random_state=args.random_state),
            spaces["gbr"],
        ),
        ("Ridge", Ridge(), spaces["ridge"]),
        ("Lasso", Lasso(max_iter=20000), spaces["lasso"]),
        (
            "NeuralNetwork",
            make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(100, 100),
                    random_state=args.random_state,
                    max_iter=5000,
                ),
            ),
            None,
        ),
        (
            "RandomForest",
            RandomForestRegressor(random_state=args.random_state),
            spaces["rf"],
        ),
        (
            "XGB",
            XGBRegressor(
                random_state=args.random_state,
                objective="reg:squarederror",
                n_jobs=1,  # keep deterministic by default
            ),
            spaces["xgb"],
        ),
    ]

    results = []
    best_params = {
        "task": args.task,
        "data": str(args.data),
        "random_state": args.random_state,
        "n_iter": args.n_iter,
        "feature_columns": FEATURE_COLUMNS,
    }

    for name, estimator, space in models:
        if space is None:
            cv_r2 = float(
                cross_val_score(estimator, X_train, y_train, cv=5, scoring="r2").mean()
            )
            results.append({"model": name, "cv_r2": cv_r2})
        else:
            opt = BayesSearchCV(
                estimator,
                search_spaces=space,
                n_iter=args.n_iter,
                cv=5,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
                scoring="r2",
            )
            opt.fit(X_train, y_train)
            results.append({"model": name, "cv_r2": float(opt.best_score_)})
            best_params[name] = {
                "best_score": float(opt.best_score_),
                "best_params": opt.best_params_,
            }

    # Save results
    df = pd.DataFrame(results)
    out_csv = outdir / f"fig_s1_r2_{args.task}.csv"
    df.to_csv(out_csv, index=False)

    out_json = outdir / f"best_params_{args.task}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    # Console output
    print(df.to_string(index=False))
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
