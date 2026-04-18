"""Bayesian-optimized RF for reversible capacity (capacity) using the original six features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, cross_validate

from .extended_utils import load_preprocessed_csv, load_xy_for_target
from .search_spaces import get_search_spaces
from .utils import FEATURE_COLUMNS, SCORING, ensure_jsonable, cv_scores_to_summary, fold_metrics_frame, regression_metrics, require_skopt, split_train_test


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/hc_dataset_extended_preprocessed.csv')
    p.add_argument('--target', default='reversible_capacity_mAh_g')
    p.add_argument('--n-iter', type=int, default=20)
    p.add_argument('--random-state', type=int, default=42)
    p.add_argument('--cv-folds', type=int, default=5)
    p.add_argument('--n-jobs', type=int, default=1)
    p.add_argument('--outdir', default='results/reviewer_extension')
    return p.parse_args()


def main():
    args = parse_args()
    require_skopt()
    from skopt import BayesSearchCV

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_preprocessed_csv(args.data)
    X, y, subset = load_xy_for_target(df, FEATURE_COLUMNS, args.target)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2, random_state=args.random_state)

    base_model = RandomForestRegressor(random_state=args.random_state)
    opt = BayesSearchCV(
        base_model,
        search_spaces=get_search_spaces('plateau')['rf'],
        n_iter=args.n_iter,
        cv=args.cv_folds,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        scoring='r2',
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
        'task': 'capacity',
        'task_alias': 'reversible_capacity',
        'data': str(args.data),
        'target': args.target,
        'random_state': args.random_state,
        'n_iter': args.n_iter,
        'cv_folds': args.cv_folds,
        'n_total': int(len(X)),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'feature_columns': FEATURE_COLUMNS,
        'best_cv_score_during_search': float(opt.best_score_),
        'best_params': ensure_jsonable(opt.best_params_),
        'cv_summary': cv_summary,
        'train_cv_metrics': train_metrics,
        'test_metrics': test_metrics,
        'generalization_gap_r2': float(train_metrics['r2'] - test_metrics['r2']),
    }

    metrics_path = outdir / 'rf_metrics_capacity.json'
    with metrics_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    pd.DataFrame([
        {
            'task': 'capacity',
            'target': args.target,
            'n_total': payload['n_total'],
            'n_train': payload['n_train'],
            'n_test': payload['n_test'],
            'cv_folds': payload['cv_folds'],
            'cv_r2_mean': cv_summary['cv_r2_mean'],
            'cv_r2_std': cv_summary['cv_r2_std'],
            'cv_mae_mean': cv_summary['cv_mae_mean'],
            'cv_mae_std': cv_summary['cv_mae_std'],
            'cv_rmse_mean': cv_summary['cv_rmse_mean'],
            'cv_rmse_std': cv_summary['cv_rmse_std'],
            'train_cv_r2': train_metrics['r2'],
            'train_cv_mae': train_metrics['mae'],
            'train_cv_rmse': train_metrics['rmse'],
            'test_r2': test_metrics['r2'],
            'test_mae': test_metrics['mae'],
            'test_rmse': test_metrics['rmse'],
            'generalization_gap_r2': payload['generalization_gap_r2'],
        }
    ]).to_csv(outdir / 'rf_metrics_summary_capacity.csv', index=False)

    pd.DataFrame({'y_true': y_train, 'y_pred': y_pred_train_cv}).to_csv(outdir / 'rf_predictions_train_cv_capacity.csv', index=False)
    pd.DataFrame({'y_true': y_test, 'y_pred': y_pred_test}).to_csv(outdir / 'rf_predictions_test_capacity.csv', index=False)
    fold_metrics_frame(cv_results).to_csv(outdir / 'rf_cv_fold_metrics_capacity.csv', index=False)
    dump(best_model, outdir / 'rf_model_capacity.joblib')

    print(json.dumps(payload, indent=2))
    print(f'Saved: {metrics_path}')


if __name__ == '__main__':
    main()
