# Hard-Carbon ML Pipeline (Bayesian Hyperparameter Search)

This repository contains a **GitHub-ready, script-based** machine-learning workflow (no plotting scripts) to reproduce the **model training / evaluation pipeline** described in the manuscript and supplementary information.

Key points (aligned with the paper description):

- Dataset is split into **training/test = 8:2**.
- Models are trained/evaluated with **5-fold cross-validation** on the training set.
- Hyperparameters are tuned with **Bayesian optimization** (`BayesSearchCV`, scikit-optimize).
- The best-performing model is **Random Forest regression**.

> Note: Bayesian optimization is stochastic. With fixed seeds the results are usually stable, but tiny numerical differences across OS / library versions are still possible.

---

## Repository structure

- `data/`
  - `hc_dataset_ice.csv` – dataset for ICE modeling (target is **LCE**, i.e., logit(ICE)).
  - `hc_dataset_plateau.csv` – dataset for reversible plateau capacity modeling.
- `src/`
  - `run_fig_s1_benchmark.py` – run 7-model benchmark with Bayesian tuning (Fig. S1-style output).
  - `run_rf_final.py` – tune Random Forest via Bayesian search and evaluate on train/test.
  - `search_spaces.py` – Bayesian search spaces (extracted from the provided notebooks).
  - `utils.py` – shared utilities (loading, splitting, metrics).
- `results/`
  - output folder (generated after running scripts)

---

## Data dictionary

### Shared features

| Column (CSV) | Meaning |
|---|---|
| `carbonization_temperature_C` | carbonization temperature (°C) |
| `d002_nm` | interlayer spacing d002 (nm) |
| `id_ig` | Raman ID/IG |
| `ssa_m2_g` | specific surface area (m² g⁻¹) |
| `electrolyte_type` | electrolyte type encoded as **0 = ester**, **1 = ether** |
| `current_density_mA_g` | current density (mA g⁻¹) |

### ICE task

- `ice`: initial Coulombic efficiency (0–1).
- `lce`: **logit-transformed ICE** used for regression:

\[
\mathrm{LCE} = \ln\left(\frac{\mathrm{ICE}}{1-\mathrm{ICE}}\right)
\]

### Plateau task

- `plateau_capacity_mAh_g`: reversible plateau capacity (mAh g⁻¹). Values can be **0** for samples without a reversible plateau region.

---

## Installation

Create a clean environment (recommended) and install dependencies:

```bash
pip install -r requirements.txt
```

---

## Reproduce the 7-model benchmark (Fig. S1-style)

ICE (LCE) benchmark:

```bash
python -m src.run_fig_s1_benchmark --task ice \
  --data data/hc_dataset_ice.csv \
  --n-iter 20 \
  --random-state 42 \
  --outdir results
```

Plateau benchmark:

```bash
python -m src.run_fig_s1_benchmark --task plateau \
  --data data/hc_dataset_plateau.csv \
  --n-iter 20 \
  --random-state 42 \
  --outdir results
```

Outputs:

- `results/fig_s1_r2_ice.csv`
- `results/fig_s1_r2_plateau.csv`
- `results/best_params_ice.json`
- `results/best_params_plateau.json`

---

## Train + evaluate the final RF model (Fig. S2-style numeric outputs)

ICE (LCE target):

```bash
python -m src.run_rf_final --task ice \
  --data data/hc_dataset_ice.csv \
  --n-iter 20 \
  --random-state 42 \
  --outdir results
```

Plateau capacity:

```bash
python -m src.run_rf_final --task plateau \
  --data data/hc_dataset_plateau.csv \
  --n-iter 20 \
  --random-state 42 \
  --outdir results
```

Outputs include:

- `results/rf_metrics_<task>.json`
- `results/rf_predictions_train_cv_<task>.csv`
- `results/rf_predictions_test_<task>.csv`
- `results/rf_model_<task>.joblib`

---
