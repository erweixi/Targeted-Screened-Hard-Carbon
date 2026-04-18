"""Correlation and OLS analyses for missing-value features vs ICE / plateau, plus variable-variable correlation analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kendalltau, pearsonr, spearmanr

from .extended_utils import EXTRA_FEATURE_COLUMNS, FEATURE_LABELS, load_preprocessed_csv
from .utils import FEATURE_COLUMNS

TARGETS = ['ice', 'plateau_capacity_mAh_g']
BASE_CONTROL_COLUMNS = FEATURE_COLUMNS
ANALYSIS_COLUMNS = FEATURE_COLUMNS + EXTRA_FEATURE_COLUMNS + TARGETS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/hc_dataset_original_master_with_missing_features.csv')
    p.add_argument('--outdir', default='results/reviewer_extension')
    return p.parse_args()


def _pairwise_corr(df: pd.DataFrame, x: str, y: str):
    sub = df[[x, y]].dropna()
    if len(sub) < 3:
        return {'n': len(sub), 'pearson_r': np.nan, 'pearson_p': np.nan, 'spearman_r': np.nan, 'spearman_p': np.nan, 'kendall_tau': np.nan, 'kendall_p': np.nan}
    pear = pearsonr(sub[x], sub[y])
    spear = spearmanr(sub[x], sub[y])
    kend = kendalltau(sub[x], sub[y])
    return {
        'n': int(len(sub)),
        'pearson_r': float(pear.statistic),
        'pearson_p': float(pear.pvalue),
        'spearman_r': float(spear.statistic),
        'spearman_p': float(spear.pvalue),
        'kendall_tau': float(kend.statistic),
        'kendall_p': float(kend.pvalue),
    }


def _univariate_ols(df: pd.DataFrame, x: str, y: str):
    sub = df[[x, y]].dropna()
    if len(sub) < 8:
        return {'n': int(len(sub)), 'coef': np.nan, 'coef_pvalue': np.nan, 'intercept': np.nan, 'r2': np.nan, 'adj_r2': np.nan}
    X = sm.add_constant(sub[[x]])
    model = sm.OLS(sub[y], X).fit()
    return {
        'n': int(len(sub)),
        'coef': float(model.params[x]),
        'coef_pvalue': float(model.pvalues[x]),
        'intercept': float(model.params['const']),
        'r2': float(model.rsquared),
        'adj_r2': float(model.rsquared_adj),
    }


def _adjusted_ols(df: pd.DataFrame, extra_feature: str, target: str):
    cols = BASE_CONTROL_COLUMNS + [extra_feature, target]
    sub = df[cols].dropna()
    if len(sub) < 15:
        return {'n': int(len(sub)), 'coef': np.nan, 'coef_pvalue': np.nan, 'r2': np.nan, 'adj_r2': np.nan}
    X = sm.add_constant(sub[BASE_CONTROL_COLUMNS + [extra_feature]])
    model = sm.OLS(sub[target], X).fit()
    return {
        'n': int(len(sub)),
        'coef': float(model.params[extra_feature]),
        'coef_pvalue': float(model.pvalues[extra_feature]),
        'r2': float(model.rsquared),
        'adj_r2': float(model.rsquared_adj),
    }


def main():
    args = parse_args()
    outdir = Path(args.outdir) / 'missing_feature_math_analysis'
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_preprocessed_csv(args.data)

    missingness_rows = []
    corr_rows = []
    uni_rows = []
    adj_rows = []
    pairwise_rows = []

    for feature in EXTRA_FEATURE_COLUMNS:
        missingness_rows.append({
            'feature': feature,
            'feature_label': FEATURE_LABELS.get(feature, feature),
            'n_missing': int(df[feature].isna().sum()),
            'missing_fraction': float(df[feature].isna().mean()),
            'n_non_missing': int(df[feature].notna().sum()),
        })
        for target in TARGETS:
            corr = _pairwise_corr(df, feature, target)
            corr_rows.append({'feature': feature, 'feature_label': FEATURE_LABELS.get(feature, feature), 'target': target, 'target_label': FEATURE_LABELS.get(target, target), **corr})
            uni = _univariate_ols(df, feature, target)
            uni_rows.append({'feature': feature, 'feature_label': FEATURE_LABELS.get(feature, feature), 'target': target, 'target_label': FEATURE_LABELS.get(target, target), **uni})
            adj = _adjusted_ols(df, feature, target)
            adj_rows.append({'feature': feature, 'feature_label': FEATURE_LABELS.get(feature, feature), 'target': target, 'target_label': FEATURE_LABELS.get(target, target), **adj})

    for i, x in enumerate(ANALYSIS_COLUMNS):
        for y in ANALYSIS_COLUMNS[i + 1:]:
            stats = _pairwise_corr(df, x, y)
            pairwise_rows.append({
                'var_x': x,
                'label_x': FEATURE_LABELS.get(x, x),
                'var_y': y,
                'label_y': FEATURE_LABELS.get(y, y),
                **stats,
            })

    missing_df = pd.DataFrame(missingness_rows)
    corr_df = pd.DataFrame(corr_rows)
    uni_df = pd.DataFrame(uni_rows)
    adj_df = pd.DataFrame(adj_rows)
    pairwise_df = pd.DataFrame(pairwise_rows)

    pearson_matrix = df[ANALYSIS_COLUMNS].corr(method='pearson')
    spearman_matrix = df[ANALYSIS_COLUMNS].corr(method='spearman')

    missing_df.to_csv(outdir / 'missing_feature_missingness.csv', index=False)
    corr_df.to_csv(outdir / 'missing_feature_target_correlations.csv', index=False)
    uni_df.to_csv(outdir / 'missing_feature_univariate_ols.csv', index=False)
    adj_df.to_csv(outdir / 'missing_feature_adjusted_ols.csv', index=False)
    pairwise_df.to_csv(outdir / 'variable_pairwise_correlations.csv', index=False)
    pearson_matrix.to_csv(outdir / 'variable_correlation_matrix_pearson.csv')
    spearman_matrix.to_csv(outdir / 'variable_correlation_matrix_spearman.csv')

    with pd.ExcelWriter(outdir / 'missing_feature_math_analysis.xlsx', engine='openpyxl') as writer:
        missing_df.to_excel(writer, sheet_name='missingness', index=False)
        corr_df.to_excel(writer, sheet_name='feature_vs_targets', index=False)
        uni_df.to_excel(writer, sheet_name='univariate_ols', index=False)
        adj_df.to_excel(writer, sheet_name='adjusted_ols', index=False)
        pairwise_df.to_excel(writer, sheet_name='pairwise_corr_long', index=False)
        pearson_matrix.to_excel(writer, sheet_name='pearson_matrix')
        spearman_matrix.to_excel(writer, sheet_name='spearman_matrix')

    print(f'Saved: {outdir}')


if __name__ == '__main__':
    main()
