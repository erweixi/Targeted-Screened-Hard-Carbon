# Consistency patch notes (2026-04-18)

This patch closes the gaps between the rebuttal drafts and the public repository snapshot.

## Code changes

### Modified
- `README.md`
- `results/README.md`
- `outputs/README.md`
- `src/README.md`
- `src/run_missing_feature_math_analysis.py`

### Added
- `src/make_missing_feature_correlation_heatmap.py`
- `src/make_multimodel_rank_supplement.py`

## New / refreshed result artifacts

### Reversible-capacity correlation supplement
- `results/reviewer_extension_fixed/missing_feature_analysis/missing_feature_math_analysis/variable_correlation_matrix_pearson.csv`
- `results/reviewer_extension_fixed/missing_feature_analysis/missing_feature_math_analysis/variable_correlation_matrix_spearman.csv`
- `results/reviewer_extension_fixed/missing_feature_analysis/missing_feature_math_analysis/variable_pairwise_correlations.csv`
- `results/reviewer_extension_fixed/missing_feature_analysis/missing_feature_math_analysis/missing_feature_math_analysis.xlsx`
- `outputs/reviewer_spearman_heatmap_with_reversible.png`

### Seven-model feature-rank supplement
- `results/reviewer/multimodel_shap_ice/linear/*`
- `results/reviewer/multimodel_shap_ice/lasso/*`
- `results/reviewer/multimodel_shap_ice/neuralnetwork/*`
- `results/reviewer/multimodel_shap_ice/multimodel_feature_rank_summary_7models.csv`
- `results/reviewer/multimodel_shap_plateau/linear/*`
- `results/reviewer/multimodel_shap_plateau/lasso/*`
- `results/reviewer/multimodel_shap_plateau/neuralnetwork/*`
- `results/reviewer/multimodel_shap_plateau/multimodel_feature_rank_summary_7models.csv`
- `results/reviewer/multimodel_feature_rank_summary_7models_combined.csv`
- `outputs/reviewer_multimodel_feature_rank_heatmap_7models.png`

## Validation performed

Executed successfully from the repository root:

```bash
python -m src.run_missing_feature_math_analysis --data data/hc_dataset_original_master_with_missing_features.csv --outdir results/reviewer_extension_fixed/missing_feature_analysis
python -m src.make_missing_feature_correlation_heatmap
python -m src.make_multimodel_rank_supplement
```

These runs confirmed:
- the refreshed Spearman matrix now contains `reversible_capacity_mAh_g`;
- the seven-model rank summaries now contain all seven models for both ICE and plateau.
