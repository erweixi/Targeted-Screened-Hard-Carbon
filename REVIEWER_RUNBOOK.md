# Reviewer-response runbook

This runbook maps each ML-related reviewer comment to the code you should run and the exact output files you should mine for the manuscript / rebuttal. It also distinguishes between **files already committed in this repository** and **files generated when you rerun a command**.

## Environment

```bash
pip install -r requirements.txt
```

All commands below assume you run them from the repository root.

Some reviewer-facing outputs are already committed exactly as generated. Others are described here as rerun targets but are not duplicated if an equivalent manuscript-stage source table is already preserved elsewhere in the repository.

---

## R1-Q1: complete ML result reporting

### Run

```bash
python -m src.run_rf_final --task ice --data data/hc_dataset_ice.csv --n-iter 20 --random-state 42 --cv-folds 5 --outdir results/reviewer
python -m src.run_rf_final --task plateau --data data/hc_dataset_plateau.csv --n-iter 20 --random-state 42 --cv-folds 5 --outdir results/reviewer
```

### Use these outputs in the paper

- `results/reviewer/rf_metrics_summary_ice.csv`
- `results/reviewer/rf_metrics_summary_plateau.csv`
- `results/reviewer/rf_cv_fold_metrics_ice.csv`
- `results/reviewer/rf_cv_fold_metrics_plateau.csv`

### Numbers to report

From each `rf_metrics_summary_<task>.csv` row, report:

- `n_total`, `n_train`, `n_test`
- `cv_folds`
- `cv_r2_mean`, `cv_r2_std`
- `train_cv_r2`, `train_cv_mae`, `train_cv_rmse`
- `test_r2`, `test_mae`, `test_rmse`
- `generalization_gap_r2`

Use these numbers in the main text / SI table when clarifying train-vs-test reporting.

---

## R1-Q2 + R3-Q5: search space and tuning workflow

### Run

```bash
python -m src.run_fig_s1_benchmark --task ice --data data/hc_dataset_ice.csv --n-iter 20 --random-state 42 --cv-folds 5 --outdir results/reviewer
python -m src.run_fig_s1_benchmark --task plateau --data data/hc_dataset_plateau.csv --n-iter 20 --random-state 42 --cv-folds 5 --outdir results/reviewer
```

### Files to inspect

**Already committed in this repository**

- `results/cv_r2_scores.csv`
- `results/reference_fig_s1_r2.csv`

These are the preserved manuscript-stage source tables for the benchmark summary.

**Generated when you rerun the command above**

- `results/reviewer/fig_s1_benchmark_ice.csv`
- `results/reviewer/fig_s1_benchmark_plateau.csv`
- `results/reviewer/best_params_ice.json`
- `results/reviewer/best_params_plateau.json`

### Numbers / text to report

From the rerun JSON files `best_params_<task>.json`:

- `n_iter`
- `cv_folds`
- `best_params` for each tuned model
- `best_score` for each tuned model

From the rerun benchmark tables `fig_s1_benchmark_<task>.csv`:

- `cv_r2_mean`, `cv_r2_std`, `cv_mae_mean`, `cv_rmse_mean`

From the committed manuscript-stage source tables:

- benchmark-level `R^2` values in `results/cv_r2_scores.csv`
- task/model reference values in `results/reference_fig_s1_r2.csv`

Use these files in Methods to describe Bayesian optimization, the search budget, and model-wise cross-validated performance.

---

## R1-Q4: why Random Forest was selected

### Run

```bash
python -m src.run_model_comparison --task ice --data data/hc_dataset_ice.csv --n-iter 20 --random-state 42 --cv-folds 5 --outdir results/reviewer
python -m src.run_model_comparison --task plateau --data data/hc_dataset_plateau.csv --n-iter 20 --random-state 42 --cv-folds 5 --outdir results/reviewer
```

### Use these outputs in the paper

- `results/reviewer/model_comparison_ice.csv`
- `results/reviewer/model_comparison_plateau.csv`
- `results/reviewer/model_comparison_best_params_ice.json`
- `results/reviewer/model_comparison_best_params_plateau.json`

### Numbers to report

From `model_comparison_<task>.csv`, compare all models using:

- `cv_r2_mean`, `cv_r2_std`
- `train_cv_r2`
- `test_r2`
- `test_mae`, `test_rmse`
- `generalization_gap_r2`
- `supports_shap`

Use these to justify that Random Forest was chosen for the best balance of performance, stability, generalization, and interpretability rather than only the highest single R².

---

## R4-Q1: reinterpret what R²≈0.75 means

### Reuse outputs

You do not need a separate script. Reuse:

- `results/reviewer/rf_metrics_summary_ice.csv`
- `results/reviewer/rf_metrics_summary_plateau.csv`
- `results/reviewer/model_comparison_ice.csv`
- `results/reviewer/model_comparison_plateau.csv`

### Numbers to report

Use:

- `cv_r2_mean ± cv_r2_std`
- `test_r2`
- `test_mae`
- `test_rmse`
- `generalization_gap_r2`

These numbers support wording such as “the model is used for trend identification and design-window screening rather than deterministic high-precision prediction.”

---

## R3-Q4: 10-fold validation

### Run

```bash
python -m src.run_rf_final --task ice --data data/hc_dataset_ice.csv --n-iter 20 --random-state 42 --cv-folds 10 --outdir results/reviewer_10fold
python -m src.run_rf_final --task plateau --data data/hc_dataset_plateau.csv --n-iter 20 --random-state 42 --cv-folds 10 --outdir results/reviewer_10fold
```

Optional full-model benchmark with 10-fold:

```bash
python -m src.run_model_comparison --task ice --data data/hc_dataset_ice.csv --n-iter 20 --random-state 42 --cv-folds 10 --outdir results/reviewer_10fold
python -m src.run_model_comparison --task plateau --data data/hc_dataset_plateau.csv --n-iter 20 --random-state 42 --cv-folds 10 --outdir results/reviewer_10fold
```

### Use these outputs in the paper

- `results/reviewer_10fold/rf_metrics_summary_ice.csv`
- `results/reviewer_10fold/rf_metrics_summary_plateau.csv`
- `results/reviewer_10fold/rf_cv_fold_metrics_ice.csv`
- `results/reviewer_10fold/rf_cv_fold_metrics_plateau.csv`

### Numbers to report

Compare the 5-fold and 10-fold versions using:

- `cv_r2_mean`, `cv_r2_std`
- `test_r2`, `test_mae`, `test_rmse`

State whether the main conclusions remain stable after switching from 5-fold to 10-fold CV.

---

## R3-Q6: SHAP vs literature

### Run

```bash
python -m src.run_multimodel_shap --task ice --data data/hc_dataset_ice.csv --models RandomForest --n-iter 20 --random-state 42 --cv-folds 5 --outdir results/reviewer
python -m src.run_multimodel_shap --task plateau --data data/hc_dataset_plateau.csv --models RandomForest --n-iter 20 --random-state 42 --cv-folds 5 --outdir results/reviewer
```

### Use these outputs in the paper

- `results/reviewer/multimodel_shap_ice/randomforest/mean_abs_shap.csv`
- `results/reviewer/multimodel_shap_ice/randomforest/shap_summary_beeswarm.png`
- `results/reviewer/multimodel_shap_ice/randomforest/shap_summary_bar.png`
- `results/reviewer/multimodel_shap_plateau/randomforest/mean_abs_shap.csv`
- `results/reviewer/multimodel_shap_plateau/randomforest/shap_summary_beeswarm.png`
- `results/reviewer/multimodel_shap_plateau/randomforest/shap_summary_bar.png`

### Numbers / observations to report

Use `mean_abs_shap.csv` to report the ranked importance of each feature, then discuss where the ranking agrees or disagrees with literature expectations.

---

## R1-Q4: multi-model SHAP comparison

### Run

```bash
python -m src.run_multimodel_shap --task ice --data data/hc_dataset_ice.csv --models RandomForest GBR XGB Ridge --n-iter 20 --random-state 42 --cv-folds 5 --outdir results/reviewer
python -m src.run_multimodel_shap --task plateau --data data/hc_dataset_plateau.csv --models RandomForest GBR XGB Ridge --n-iter 20 --random-state 42 --cv-folds 5 --outdir results/reviewer
```

### Folder structure

Each model is saved to a different folder, for example:

- `results/reviewer/multimodel_shap_ice/randomforest/`
- `results/reviewer/multimodel_shap_ice/gbr/`
- `results/reviewer/multimodel_shap_ice/xgb/`
- `results/reviewer/multimodel_shap_ice/ridge/`

### Use these outputs in the paper

- `results/reviewer/multimodel_shap_ice/multimodel_shap_summary.csv`
- `results/reviewer/multimodel_shap_plateau/multimodel_shap_summary.csv`

### Numbers / observations to report

Use the summary CSV plus each model’s `mean_abs_shap.csv` to compare:

- top-ranked features across models
- whether the main feature ordering is stable
- whether the directional interpretation is broadly consistent in the beeswarm plots

---

## R3-Q6: permutation importance / feature ablation

### Run

```bash
python -m src.run_feature_robustness --task ice --data data/hc_dataset_ice.csv --n-iter 20 --random-state 42 --cv-folds 5 --permutation-repeats 30 --outdir results/reviewer
python -m src.run_feature_robustness --task plateau --data data/hc_dataset_plateau.csv --n-iter 20 --random-state 42 --cv-folds 5 --permutation-repeats 30 --outdir results/reviewer
```

### Files to inspect

**Generated when you rerun `src.run_feature_robustness` from scratch**

- `results/reviewer/feature_robustness_ice/permutation_importance_test.csv`
- `results/reviewer/feature_robustness_ice/leave_one_feature_out.csv`
- `results/reviewer/feature_robustness_plateau/permutation_importance_test.csv`
- `results/reviewer/feature_robustness_plateau/leave_one_feature_out.csv`

**Already committed in this repository**

- `results/reviewer/feature_robustness_ice_locked/permutation_importance_test.csv`
- `results/reviewer/feature_robustness_ice_locked/leave_one_feature_out.csv`
- `results/reviewer/feature_robustness_plateau_locked/permutation_importance_test.csv`
- `results/reviewer/feature_robustness_plateau_locked/leave_one_feature_out.csv`

The `*_locked` folders reuse fixed best RF hyperparameters saved from the final RF JSON files rather than retuning the model from scratch.

### Numbers / observations to report

From permutation importance:

- `importance_mean`
- `importance_std`

From leave-one-feature-out:

- `delta_test_r2`
- `test_r2_without_feature`

Use these as an independent check on the SHAP ranking.

---

## R2 total comment + R3-Q3 + R4-Q1: why these six features were retained

### Run

```bash
python -m src.run_dataset_summary --task ice --data data/hc_dataset_ice.csv --outdir results/reviewer
python -m src.run_dataset_summary --task plateau --data data/hc_dataset_plateau.csv --outdir results/reviewer
```

### Use these outputs in the paper

- `results/reviewer/dataset_summary_ice.csv`
- `results/reviewer/dataset_summary_plateau.csv`
- `results/reviewer/dataset_summary_ice.json`
- `results/reviewer/dataset_summary_plateau.json`

### Numbers / text to report

Use these files to report:

- the exact six retained features
- sample size
- missingness of the retained features (`n_missing`, `missing_fraction`)
- numerical ranges of the retained features (`min`, `max`, `mean`, `std`)

This supports the argument that the final feature set is the one consistently available and quantitatively comparable across the assembled dataset.

---

## Notes

- Multi-model SHAP outputs are stored in **separate folders per model**.
- The repository includes both **rerunnable scripts** and **precomputed outputs**. When a `*_locked` folder is present, it indicates that fixed best RF hyperparameters were reused rather than retuned.
- Recommended order for a clean rerun: `run_rf_final` -> `run_model_comparison` -> SHAP / robustness -> extended-data analyses.
- For the Excel-to-master-table correction workflow and reversible-capacity analyses, use `python -m src.run_all_extended_capacity_analyses`.
