# Results directory

This directory contains machine-readable outputs: tables, JSON manifests, predictions, and serialized models.

## Top-level files in `results/`

These are preserved from the original manuscript-stage repository because some figure-export utilities still use them as source tables.

- `cv_r2_scores.csv`
- `reference_fig_s1_r2.csv`
- `rf_predictions_ice_logit.csv`
- `rf_predictions_plateau.csv`

## Subdirectories

### `results/reviewer/`
Primary reviewer-response outputs based on the 5-fold workflow.

Includes:
- final RF metrics, fold metrics, predictions, and joblib models
- full model-comparison tables and model-specific CV folds
- SHAP outputs for multiple models
- robustness analyses such as permutation importance and leave-one-feature-out tests
- supplementary seven-model rank summaries under `multimodel_shap_ice/` and `multimodel_shap_plateau/`
- combined rank table: `results/reviewer/multimodel_feature_rank_summary_7models_combined.csv`

### `results/reviewer_10fold/`
10-fold sensitivity analysis used to check whether the main conclusions remain stable when the CV scheme changes.

### `results/reviewer_extension_fixed/`
Corrected extended-data workflow outputs.

Key subfolders:
- `results/reviewer_extension_fixed/baseline_original_tasks/` — six-feature RF baselines for ICE and plateau
- `results/reviewer_extension_fixed/capacity_baseline/` — reversible-capacity baseline model
- `results/reviewer_extension_fixed/progressive_ice/` — progressive feature-addition analysis for ICE
- `results/reviewer_extension_fixed/progressive_plateau/` — progressive feature-addition analysis for plateau capacity
- `results/reviewer_extension_fixed/missing_feature_analysis/` — statistical analysis of the previously unused variables

The missing-feature analysis folder now exports correlation matrices that include `reversible_capacity_mAh_g`, so the repository directly matches the supplementary reviewer response.

## Terminology note

Some folders use names such as `*_locked`. In this project, “locked” means the analysis reused fixed best RF hyperparameters saved from an earlier run rather than retuning the model from scratch.
