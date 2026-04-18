# Targeted-Screened Hard-Carbon ML Repository

[English](./README.md) | [简体中文](./README.zh-CN.md) | [日本語](./README.ja.md)

This repository consolidates the original manuscript codebase and the post-review extensions for the hard-carbon sodium-storage machine-learning study. It is organized for direct GitHub release: datasets, reproducible code, precomputed outputs, and reviewer-response materials are kept in one place.

Most result files are already included in this snapshot. Readers can inspect the shipped CSV, JSON, and PNG artifacts directly. Rerunning the code is only needed when a fresh rebuild is required.

## Where to start

| Goal | Start here | Main locations |
| --- | --- | --- |
| Inspect the original manuscript-level ML workflow | `README.md`, `src/run_fig_s1_benchmark.py`, `src/run_rf_final.py` | `results/`, `outputs/` |
| Inspect reviewer-response analyses | `docs/reviewer_runbook.md` | `results/reviewer/`, `results/reviewer_10fold/` |
| Inspect the post-review extended-dataset analyses | `python -m src.run_all_extended_capacity_analyses` | `data/`, `results/reviewer_extension_fixed/`, `outputs/` |
| Read the preserved reviewer-response documents | `docs/reviewer_materials/` | `.docx` and `.xlsx` files |
| Need a concise Chinese quick-start | `docs/quickstart_zh.md` | command summary |

## Repository layout

```text
.
├── data/                     # base datasets + extended Excel-derived datasets
├── docs/                     # reviewer runbook, quick-start, preserved reviewer documents
├── outputs/                  # exported figures and figure-ready helper tables
├── results/                  # numeric outputs; manuscript-level files at root, reviewer outputs in subfolders
├── scripts/                  # standalone helper scripts
├── src/                      # reproducible Python modules / entry points
├── requirements.txt
└── .gitignore
```

## Installation

Python 3.10-3.12 is recommended.

```bash
pip install -r requirements.txt
```

Note: `scikit-optimize` is required for the Bayesian-search scripts. In some Python 3.13 environments this dependency may not install correctly, so Python 3.10-3.12 is the safer choice for full reruns.

## Quick start

### 1. Reproduce the original manuscript benchmark and final RF model

ICE benchmark:

```bash
python -m src.run_fig_s1_benchmark \
  --task ice \
  --data data/hc_dataset_ice.csv \
  --n-iter 20 \
  --cv-folds 5 \
  --random-state 42 \
  --outdir results
```

Plateau benchmark:

```bash
python -m src.run_fig_s1_benchmark \
  --task plateau \
  --data data/hc_dataset_plateau.csv \
  --n-iter 20 \
  --cv-folds 5 \
  --random-state 42 \
  --outdir results
```

Final Random-Forest models:

```bash
python -m src.run_rf_final --task ice --data data/hc_dataset_ice.csv --n-iter 20 --cv-folds 5 --random-state 42 --outdir results/reviewer
python -m src.run_rf_final --task plateau --data data/hc_dataset_plateau.csv --n-iter 20 --cv-folds 5 --random-state 42 --outdir results/reviewer
```

### 2. Reproduce the full post-review extension in one command

```bash
python -m src.run_all_extended_capacity_analyses
```

This command rebuilds the derived extended CSV files, the reversible-capacity baseline, the progressive feature-addition analyses, and the missing-feature statistical analysis.

### 3. Use the precomputed outputs without rerunning anything

Key shipped outputs are already present:

- Manuscript-level reference tables: `results/cv_r2_scores.csv`, `results/reference_fig_s1_r2.csv`, `results/rf_predictions_ice_logit.csv`, `results/rf_predictions_plateau.csv`
- Manuscript/reviewer figure assets: `outputs/fig_S1_cv_r2_corrected.png`, `outputs/fig_S2_rf_performance.png`, `outputs/fig_S2_rf_best.png`, `outputs/fig_S3_shap_rf_best.png`
- Reviewer core outputs: `results/reviewer/`, `results/reviewer_10fold/`
- Reviewer extension outputs: `results/reviewer_extension_fixed/`

## Data files

| File | Role |
| --- | --- |
| `data/hc_dataset_ice.csv` | Base 565-row dataset for ICE modeling; the ML target is `lce = log(ice / (1 - ice))`. |
| `data/hc_dataset_plateau.csv` | Base 565-row dataset for plateau-capacity modeling. |
| `data/hard_carbon_database_20260323_revised.xlsx` | Extended Excel database used in the reviewer-response extension. |
| `data/hc_dataset_extended_preprocessed.csv` | Cleaned CSV exported from the extended Excel database. |
| `data/hc_dataset_original_master_with_missing_features.csv` | Mapped master table that merges the original 565-row dataset with extra missing-value features from the extended database. |
| `data/hc_dataset_original_master_mapping_report.csv` | Mapping coverage summary for the merged master table. |

More detail is in `data/README.md`.

## Main code entry points

The canonical scripts are:

- `src/run_fig_s1_benchmark.py` — 7-model benchmark with Bayesian hyperparameter search
- `src/run_rf_final.py` — final Random-Forest tuning and train/test evaluation
- `src/run_model_comparison.py` — reviewer-response model comparison table
- `src/run_rf_posthoc_from_best_params.py` — locked-parameter RF SHAP and robustness exports
- `src/run_all_extended_capacity_analyses.py` — one-command extended-dataset workflow

A fuller script index, including helper and provenance-preserving utilities, is in `src/README.md`.

## Reproducibility notes

- Bayesian optimization (`BayesSearchCV`) is stochastic. Fixed seeds are supplied, but tiny numerical deviations across Python, BLAS, and OS versions are still possible.
- The ICE ML task uses the logit-transformed target `lce`; some descriptive or statistical analyses in the reviewer extension use raw `ice` instead. This is intentional and documented in the corresponding scripts.
- Root-level files in `results/` and `outputs/` are manuscript-level reference artifacts. Reviewer-specific reruns live in subdirectories under `results/`.

## Documentation map

- `docs/reviewer_runbook.md` — command-by-command guide for reviewer-response analyses
- `docs/quickstart_zh.md` — concise Chinese quick-start note
- `docs/reviewer_materials/README.md` — mapping of preserved reviewer documents and sanitized filenames
- `results/README.md`, `outputs/README.md`, `data/README.md` — folder-level guides
