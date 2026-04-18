"""Progressively add missing-value features for ICE or plateau using the fixed RF parameters from the original six-feature model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, cross_validate

from .extended_utils import EXTRA_FEATURE_COLUMNS, TASK_TARGETS, load_preprocessed_csv, load_xy_for_target
from .utils import FEATURE_COLUMNS, SCORING, cv_scores_to_summary, fold_metrics_frame, regression_metrics, split_train_test


TASK_LABELS = {
    'ice': 'ICE (modeled with logit-transformed lce target, matching the original pipeline)',
    'plateau': 'Plateau capacity',
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['ice', 'plateau'], required=True)
    p.add_argument('--data', default='data/hc_dataset_original_master_with_missing_features.csv')
    p.add_argument('--best-params-json', required=True)
    p.add_argument('--random-state', type=int, default=42)
    p.add_argument('--cv-folds', type=int, default=5)
    p.add_argument('--n-jobs', type=int, default=1)
    p.add_argument('--outdir', default='results/reviewer_extension')
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.best_params_json, 'r', encoding='utf-8') as f:
        best_payload = json.load(f)
    best_params = best_payload['best_params']

    df = load_preprocessed_csv(args.data)
    target = TASK_TARGETS[args.task]

    missing_order = (
        df[EXTRA_FEATURE_COLUMNS]
        .isna()
        .sum()
        .sort_values(ascending=True)
        .index.tolist()
    )

    rows = []
    current_features = FEATURE_COLUMNS.copy()
    for step, feature in enumerate([None] + missing_order, start=0):
        if feature is not None:
            current_features = current_features + [feature]
        X, y, subset = load_xy_for_target(df, current_features, target)
        X_train, X_test, y_train, y_test = split_train_test(
            X, y, test_size=0.2, random_state=args.random_state
        )
        model = RandomForestRegressor(random_state=args.random_state, **best_params)

        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=args.cv_folds,
            scoring=SCORING,
            return_train_score=False,
            n_jobs=args.n_jobs,
        )
        cv_summary = cv_scores_to_summary(cv_results)
        y_pred_train_cv = cross_val_predict(model, X_train, y_train, cv=args.cv_folds, n_jobs=args.n_jobs)
        train_metrics = regression_metrics(y_train, y_pred_train_cv)
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        test_metrics = regression_metrics(y_test, y_pred_test)

        extra = {}
        if args.task == 'ice':
            def inv_logit(a):
                return 1 / (1 + np.exp(-np.asarray(a, dtype=float)))
            extra = {
                'train_cv_r2_raw_ice': float(regression_metrics(inv_logit(y_train), inv_logit(y_pred_train_cv))['r2']),
                'test_r2_raw_ice': float(regression_metrics(inv_logit(y_test), inv_logit(y_pred_test))['r2']),
            }

        step_name = 'baseline_original_six_features' if feature is None else f'add_{feature}'
        row = {
            'task': args.task,
            'task_label': TASK_LABELS[args.task],
            'model_target_column': target,
            'step': step,
            'step_name': step_name,
            'added_feature': '' if feature is None else feature,
            'feature_set': '; '.join(current_features),
            'n_features': len(current_features),
            'n_total_complete_cases': int(len(X)),
            'n_train': int(len(X_train)),
            'n_test': int(len(X_test)),
            'cv_folds': args.cv_folds,
            'cv_r2_mean': cv_summary['cv_r2_mean'],
            'cv_r2_std': cv_summary['cv_r2_std'],
            'train_cv_r2': train_metrics['r2'],
            'test_r2': test_metrics['r2'],
            'test_mae': test_metrics['mae'],
            'test_rmse': test_metrics['rmse'],
            'generalization_gap_r2': float(train_metrics['r2'] - test_metrics['r2']),
            'matched_rows_in_master_dataset': int(subset.shape[0]),
        }
        row.update(extra)
        rows.append(row)
        fold_metrics_frame(cv_results).to_csv(outdir / f'progressive_{args.task}_cv_folds_step_{step}.csv', index=False)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(outdir / f'progressive_feature_addition_{args.task}.csv', index=False)

    pd.DataFrame({
        'task': args.task,
        'feature': EXTRA_FEATURE_COLUMNS,
        'missing_count': [int(df[c].isna().sum()) for c in EXTRA_FEATURE_COLUMNS],
        'missing_fraction': [float(df[c].isna().mean()) for c in EXTRA_FEATURE_COLUMNS],
        'order_used': [missing_order.index(c) + 1 for c in EXTRA_FEATURE_COLUMNS],
    }).sort_values('order_used').to_csv(outdir / f'progressive_feature_missingness_order_{args.task}.csv', index=False)

    with open(outdir / f'progressive_feature_addition_metadata_{args.task}.json', 'w', encoding='utf-8') as f:
        json.dump({
            'task': args.task,
            'task_label': TASK_LABELS[args.task],
            'target': target,
            'base_features': FEATURE_COLUMNS,
            'missing_feature_order': missing_order,
            'best_params_source': args.best_params_json,
            'best_params': best_params,
            'data': args.data,
        }, f, indent=2)

    print(result_df[['step', 'added_feature', 'n_total_complete_cases', 'cv_r2_mean', 'test_r2']].to_string(index=False))
    print(f'\nSaved: {outdir / f"progressive_feature_addition_{args.task}.csv"}')


if __name__ == '__main__':
    main()
