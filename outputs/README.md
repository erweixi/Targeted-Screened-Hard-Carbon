# Outputs directory

This directory contains figure-ready assets and auxiliary exports intended for direct inspection.

## Included files

### Legacy manuscript figures retained for compatibility
- `fig_S1_cv_r2_corrected.png`
- `fig_S2_rf_performance.png`

These are preserved because some export utilities use them directly or fall back to them.

### Reviewer-stage / revision-stage figure exports
- `fig_S2_rf_best.png`
- `fig_S3_shap_rf_best.png`
- `baseline_feature_target_statistics.png`
- `baseline_feature_target_statistics.csv`
- `progressive_rf_reversible_capacity.png`
- `progressive_rf_reversible_capacity.csv`
- `progressive_rf_reversible_capacity_metadata.json`
- `shap_capacity_time_heat.png`
- `shap_capacity_time_heat_mean_abs.csv`

### SHAP source tables
`outputs/shap_data/` contains the test-set feature tables used to assemble some figure-level SHAP exports.
