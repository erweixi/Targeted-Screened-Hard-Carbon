# Targeted-Screened Hard Carbon: reproducible ML pipeline and reviewer-response analyses

This repository is a consolidated, GitHub-ready version of the hard-carbon machine-learning project. It combines:

1. the **core manuscript pipeline** for the two original prediction tasks;
2. the **reviewer-response analyses** added during revision;
3. the **extended-data workflow** used to map additional variables from the revised Excel database back to the original 565-row master dataset; and
4. the **precomputed outputs** needed by readers and reviewers who want to inspect results without rerunning everything.

The current version also includes a **supplementary-consistency patch** that fills the previously missing reviewer-facing artifacts:
- a Spearman heatmap that explicitly includes **reversible capacity**;
- a **seven-model** feature-rank supplement covering RandomForest, GBR, XGB, Ridge, Lasso, Linear, and NeuralNetwork.

## What is in this repository

### Core manuscript tasks
- **ICE modeling** using the six baseline features and the transformed target `lce = log(ice / (1 - ice))`
- **Plateau-capacity modeling** using the same six baseline features
- 7-model benchmark with Bayesian hyperparameter search
- final Random Forest training / evaluation pipeline

### Reviewer-response analyses
- full model comparison across benchmark models
- 5-fold and 10-fold validation exports
- SHAP analyses for Random Forest and selected comparison models
- permutation importance and leave-one-feature-out robustness analyses
- seven-model feature-rank supplement for the updated rebuttal figures

### Extended-data / revision analyses
- preprocessing of the revised Excel database
- reconstruction of the original 565-row master table with mapped extra variables
- reversible-capacity baseline model
- progressive feature-addition analyses ordered by missingness
- statistical analysis of previously unused variables with substantial missingness
- updated correlation matrix / heatmap that includes reversible capacity

## Where to look first

- For the **datasets and file provenance**, start with [`data/README.md`](data/README.md).
- For the **main scripts and what each one does**, see [`src/README.md`](src/README.md).
- For the **reviewer-response rerun map**, see [`REVIEWER_RUNBOOK.md`](REVIEWER_RUNBOOK.md).
- For the **precomputed artifacts already included**, see [`results/README.md`](results/README.md) and [`outputs/README.md`](outputs/README.md).

## Repository layout

```text
.
├── README.md
├── REVIEWER_RUNBOOK.md
├── requirements.txt
├── data/
├── outputs/
├── results/
├── scripts/
└── src/
```

## Quick start

For the full Bayesian-search workflows (`run_fig_s1_benchmark.py`, `run_rf_final.py`, `run_model_comparison.py`, `run_multimodel_shap.py`), use **Python 3.10-3.12**. Those modules depend on `scikit-optimize`, which may not yet install cleanly on newer Python versions.

```bash
pip install -r requirements.txt
```

All commands below assume you run them from the repository root.

## Data overview

### Baseline manuscript datasets
- `data/hc_dataset_ice.csv` — 565 rows; six baseline features plus `ice` and `lce`
- `data/hc_dataset_plateau.csv` — 565 rows; six baseline features plus `plateau_capacity_mAh_g`

### Extended / reviewer-response datasets
- `data/hard_carbon_database_20260323_revision.xlsx` — revised Excel database used during the response to reviewers
- `data/hc_dataset_extended_preprocessed.csv` — normalized CSV exported from the Excel file
- `data/hc_dataset_original_master_with_missing_features.csv` — original 565-row master dataset with mapped additional variables
- `data/hc_dataset_original_master_mapping_report.csv` — summary of successful and unsuccessful row mappings

## Reproduce the core manuscript pipeline

### 1) Seven-model benchmark (Fig. S1-style numeric outputs)

```bash
python -m src.run_fig_s1_benchmark --task ice --data data/hc_dataset_ice.csv --n-iter 20 --random-state 42 --cv-folds 5 --outdir results/reviewer
python -m src.run_fig_s1_benchmark --task plateau --data data/hc_dataset_plateau.csv --n-iter 20 --random-state 42 --cv-folds 5 --outdir results/reviewer
```

The repository already preserves the manuscript-stage benchmark source tables in `results/cv_r2_scores.csv` and `results/reference_fig_s1_r2.csv`. Rerunning the command above will additionally write task-specific benchmark tables and best-parameter JSON files to the chosen `outdir`.

### 2) Final Random Forest models for the two baseline tasks

```bash
python -m src.run_rf_final --task ice --data data/hc_dataset_ice.csv --n-iter 20 --random-state 42 --cv-folds 5 --outdir results/reviewer
python -m src.run_rf_final --task plateau --data data/hc_dataset_plateau.csv --n-iter 20 --random-state 42 --cv-folds 5 --outdir results/reviewer
```

The repository already includes the corresponding precomputed outputs in `results/reviewer/`.

## Reproduce the reviewer-response analyses

### Full reviewer map
Use [`REVIEWER_RUNBOOK.md`](REVIEWER_RUNBOOK.md). It links each reviewer-facing analysis to the exact scripts and output files.

### Common entry points

Model-comparison table:

```bash
python -m src.run_model_comparison --task ice --data data/hc_dataset_ice.csv --n-iter 20 --random-state 42 --cv-folds 5 --outdir results/reviewer
python -m src.run_model_comparison --task plateau --data data/hc_dataset_plateau.csv --n-iter 20 --random-state 42 --cv-folds 5 --outdir results/reviewer
```

Four-model SHAP analyses used in the main reviewer package:

```bash
python -m src.run_multimodel_shap --task ice --data data/hc_dataset_ice.csv --models RandomForest GBR XGB Ridge --n-iter 20 --random-state 42 --cv-folds 5 --outdir results/reviewer
python -m src.run_multimodel_shap --task plateau --data data/hc_dataset_plateau.csv --models RandomForest GBR XGB Ridge --n-iter 20 --random-state 42 --cv-folds 5 --outdir results/reviewer
```

Feature-robustness checks:

```bash
python -m src.run_feature_robustness --task ice --data data/hc_dataset_ice.csv --n-iter 20 --random-state 42 --cv-folds 5 --permutation-repeats 30 --outdir results/reviewer
python -m src.run_feature_robustness --task plateau --data data/hc_dataset_plateau.csv --n-iter 20 --random-state 42 --cv-folds 5 --permutation-repeats 30 --outdir results/reviewer
```

10-fold sensitivity check:

```bash
python -m src.run_rf_final --task ice --data data/hc_dataset_ice.csv --n-iter 20 --random-state 42 --cv-folds 10 --outdir results/reviewer_10fold
python -m src.run_rf_final --task plateau --data data/hc_dataset_plateau.csv --n-iter 20 --random-state 42 --cv-folds 10 --outdir results/reviewer_10fold
```

### Supplementary-consistency patch

These three commands regenerate the reviewer-facing artifacts that were missing from the earlier GitHub snapshot.

```bash
python -m src.run_missing_feature_math_analysis --data data/hc_dataset_original_master_with_missing_features.csv --outdir results/reviewer_extension_fixed/missing_feature_analysis
python -m src.make_missing_feature_correlation_heatmap
python -m src.make_multimodel_rank_supplement
```

Expected outputs include:
- `results/reviewer_extension_fixed/missing_feature_analysis/missing_feature_math_analysis/variable_correlation_matrix_spearman.csv`
- `outputs/reviewer_spearman_heatmap_with_reversible.png`
- `results/reviewer/multimodel_shap_ice/multimodel_feature_rank_summary_7models.csv`
- `results/reviewer/multimodel_shap_plateau/multimodel_feature_rank_summary_7models.csv`
- `results/reviewer/multimodel_feature_rank_summary_7models_combined.csv`
- `outputs/reviewer_multimodel_feature_rank_heatmap_7models.png`

## Reproduce the extended-data workflow

### One-command rerun

```bash
python -m src.run_all_extended_capacity_analyses
```

This command performs the following sequence:

1. preprocesses the revised Excel database;
2. reconstructs the 565-row master table with mapped extra variables;
3. reruns the original six-feature RF baselines for ICE and plateau capacity;
4. runs the reversible-capacity baseline model;
5. runs progressive feature-addition analyses for ICE and plateau capacity; and
6. runs the statistical analysis of missing-value variables.

The default output directory is:

```text
results/reviewer_extension_fixed/
```

## Figure-generation helpers

- `src/export_best_model_shap_and_figs.py`
- `src/export_rf_best_model_assets_fixed.py`
- `scripts/plot_s2_s3_best_rf.py`
- `src/make_baseline_feature_stats_figure.py`
- `src/make_progressive_rf_feature_curve.py`
- `src/make_shap_capacity_time_heat.py`
- `src/make_missing_feature_correlation_heatmap.py`
- `src/make_multimodel_rank_supplement.py`

These helpers are kept because they were part of the revision workflow and reproduce or package figure-level artifacts from saved results.

## Notes on targets and interpretation

- All machine-learning scripts for the **ICE** task use `lce` rather than raw `ice` as the regression target.
- The **missing-feature statistical analysis** keeps ICE and plateau as the main regression targets, while the exported correlation matrices now also include `reversible_capacity_mAh_g` for the supplementary heatmap.
- The seven-model rank supplement mixes **SHAP rankings** (Linear, Lasso, Ridge, RF, GBR, XGB) with **permutation-importance rankings** for NeuralNetwork, matching the logic described in the supplementary reviewer reply.
- Randomized search and model fitting use fixed seeds where provided, but minor numerical differences across operating systems and package versions are still possible.
- Most scripts default to `n_jobs=1` to reduce cross-platform variability.

## Included precomputed artifacts

The repository intentionally includes precomputed outputs so that readers do not need to rerun computationally heavier analyses just to inspect the results.

Important examples:
- `results/reviewer/` — main reviewer-response tables, predictions, models, and SHAP outputs
- `results/reviewer_10fold/` — 10-fold sensitivity analysis
- `results/reviewer_extension_fixed/` — corrected extended-feature analyses
- `outputs/` — exported figures and figure-ready data products

In addition, the root-level files inside `results/` and the legacy figure PNGs inside `outputs/` are preserved because some export utilities depend on them.

## Minimal citation / acknowledgement note

If you use this repository in a paper, thesis, or presentation, cite the associated manuscript and describe whether you used the baseline six-feature workflow, the reviewer-response analyses, or the extended-data mapping workflow.
