"""Run the corrected reviewer-requested workflow.

Outputs include:
1) capacity (reversible capacity) baseline RF using the original six-feature strategy;
2) baseline ICE and plateau RF models (to capture the fixed RF params from the original workflow);
3) progressive addition of missing-value features for ICE and plateau;
4) mathematical analyses of missing-value features vs ICE/plateau, plus variable-variable correlations.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]):
    print('>>>', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--excel', default='data/hard_carbon_database_20260323_revision.xlsx')
    p.add_argument('--ice-csv', default='data/hc_dataset_ice.csv')
    p.add_argument('--plateau-csv', default='data/hc_dataset_plateau.csv')
    p.add_argument('--outdir', default='results/reviewer_extension_fixed')
    p.add_argument('--n-iter', type=int, default=20)
    p.add_argument('--cv-folds', type=int, default=5)
    p.add_argument('--random-state', type=int, default=42)
    p.add_argument('--n-jobs', type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    py = sys.executable

    run([py, '-m', 'src.prepare_extended_dataset', '--excel', args.excel, '--out-csv', 'data/hc_dataset_extended_preprocessed.csv'])
    run([
        py, '-m', 'src.build_original_master_dataset',
        '--ice-csv', args.ice_csv,
        '--plateau-csv', args.plateau_csv,
        '--extended-csv', 'data/hc_dataset_extended_preprocessed.csv',
        '--out-csv', 'data/hc_dataset_original_master_with_missing_features.csv',
        '--mapping-report-csv', 'data/hc_dataset_original_master_mapping_report.csv',
    ])

    # Original-task baselines (needed to freeze RF params before progressive addition).
    run([
        py, '-m', 'src.run_rf_final',
        '--task', 'ice',
        '--data', args.ice_csv,
        '--outdir', f'{args.outdir}/baseline_original_tasks',
        '--n-iter', str(args.n_iter),
        '--cv-folds', str(args.cv_folds),
        '--random-state', str(args.random_state),
        '--n-jobs', str(args.n_jobs),
    ])
    run([
        py, '-m', 'src.run_rf_final',
        '--task', 'plateau',
        '--data', args.plateau_csv,
        '--outdir', f'{args.outdir}/baseline_original_tasks',
        '--n-iter', str(args.n_iter),
        '--cv-folds', str(args.cv_folds),
        '--random-state', str(args.random_state),
        '--n-jobs', str(args.n_jobs),
    ])

    # Capacity baseline on the extended Excel-derived dataset.
    run([
        py, '-m', 'src.run_capacity_rf',
        '--data', 'data/hc_dataset_extended_preprocessed.csv',
        '--outdir', f'{args.outdir}/capacity_baseline',
        '--n-iter', str(args.n_iter),
        '--cv-folds', str(args.cv_folds),
        '--random-state', str(args.random_state),
        '--n-jobs', str(args.n_jobs),
    ])

    # Progressive addition for ICE and plateau on the original 565-row base dataset.
    run([
        py, '-m', 'src.run_progressive_feature_addition',
        '--task', 'ice',
        '--data', 'data/hc_dataset_original_master_with_missing_features.csv',
        '--best-params-json', f'{args.outdir}/baseline_original_tasks/rf_metrics_ice.json',
        '--outdir', f'{args.outdir}/progressive_ice',
        '--cv-folds', str(args.cv_folds),
        '--random-state', str(args.random_state),
        '--n-jobs', str(args.n_jobs),
    ])
    run([
        py, '-m', 'src.run_progressive_feature_addition',
        '--task', 'plateau',
        '--data', 'data/hc_dataset_original_master_with_missing_features.csv',
        '--best-params-json', f'{args.outdir}/baseline_original_tasks/rf_metrics_plateau.json',
        '--outdir', f'{args.outdir}/progressive_plateau',
        '--cv-folds', str(args.cv_folds),
        '--random-state', str(args.random_state),
        '--n-jobs', str(args.n_jobs),
    ])

    # Missing-variable mathematical analysis on the original base dataset enriched with mapped extra features.
    run([
        py, '-m', 'src.run_missing_feature_math_analysis',
        '--data', 'data/hc_dataset_original_master_with_missing_features.csv',
        '--outdir', f'{args.outdir}/missing_feature_analysis',
    ])

    print('\nDone. See:', Path(args.outdir))


if __name__ == '__main__':
    main()
