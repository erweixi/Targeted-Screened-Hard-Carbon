# Source-code guide

All scripts are designed to be run from the repository root with `python -m src.<module_name>` unless otherwise noted.

## Core manuscript workflow

- `run_fig_s1_benchmark.py` — 7-model benchmark with Bayesian hyperparameter search
- `run_rf_final.py` — final Random Forest workflow for ICE or plateau capacity
- `search_spaces.py` — BayesSearchCV search spaces
- `utils.py` — shared data loading, splitting, metrics, and JSON utilities
- `model_registry.py` — benchmark-model registry and SHAP compatibility metadata

## Reviewer-response analyses

- `run_model_comparison.py` — benchmark-model comparison with train/CV/test metrics
- `run_multimodel_shap.py` — SHAP analyses for selected models
- `run_feature_robustness.py` — permutation importance and leave-one-feature-out analyses
- `run_rf_posthoc_from_best_params.py` — rerun SHAP / robustness from fixed saved RF parameters
- `run_dataset_summary.py` — descriptive summaries of input datasets
- `make_multimodel_rank_supplement.py` — fills the missing seven-model ranking outputs used by the supplementary reviewer reply

## Extended-data / correction workflow

- `prepare_extended_dataset.py` — preprocess the revised Excel database into CSV
- `build_original_master_dataset.py` — map extended variables back to the original 565-row master table
- `extended_utils.py` — shared helpers for the extended-data workflow
- `run_capacity_rf.py` — Random Forest baseline for reversible capacity
- `run_reversible_capacity_rf.py` — related reversible-capacity RF workflow
- `run_progressive_feature_addition.py` — progressively add extra variables by missingness order
- `run_missing_feature_math_analysis.py` — correlation and OLS analysis for missing-value variables; now also refreshes correlation matrices that include reversible capacity
- `run_all_extended_capacity_analyses.py` — orchestrates the full extended-data rerun
- `make_missing_feature_correlation_heatmap.py` — draws the updated Spearman heatmap including reversible capacity

## Figure and packaging helpers

- `export_best_model_shap_and_figs.py`
- `export_rf_best_model_assets_fixed.py`
- `make_baseline_feature_stats_figure.py`
- `make_progressive_rf_feature_curve.py`
- `make_shap_capacity_time_heat.py`

These scripts package saved outputs into figure-level artifacts or derived presentation files.
