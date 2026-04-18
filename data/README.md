# Data directory

This directory contains both the baseline manuscript datasets and the revision-stage datasets derived from the updated Excel database.

## Files

### Baseline datasets

#### `hc_dataset_ice.csv`
Original 565-row dataset for the ICE task.

Columns:
- six baseline features:
  - `carbonization_temperature_C`
  - `d002_nm`
  - `id_ig`
  - `ssa_m2_g`
  - `electrolyte_type`
  - `current_density_mA_g`
- targets:
  - `ice`
  - `lce`

`lce` is defined as:

```text
lce = log(ice / (1 - ice))
```

#### `hc_dataset_plateau.csv`
Original 565-row dataset for the reversible plateau-capacity task.

Columns:
- the same six baseline features
- `plateau_capacity_mAh_g`

### Revision-stage source database

#### `hard_carbon_database_20260323_revision.xlsx`
The revised Excel database used during the reviewer-response stage. It contains the larger source table from which additional variables were extracted and normalized.

### Derived revision-stage CSVs

#### `hc_dataset_extended_preprocessed.csv`
Preprocessed CSV exported from the revised Excel file.

Generated with:

```bash
python -m src.prepare_extended_dataset --excel data/hard_carbon_database_20260323_revision.xlsx --out-csv data/hc_dataset_extended_preprocessed.csv
```

Key extra variables retained for later analysis:
- `carbonization_time_h`
- `heating_rate_C_min`
- `pore_volume_cm3_g`
- `lc_nm`
- `la_nm`
- `reversible_capacity_mAh_g`

#### `hc_dataset_original_master_with_missing_features.csv`
The original 565-row master table after mapping the additional variables from the preprocessed extended dataset back onto the baseline rows.

Generated with:

```bash
python -m src.build_original_master_dataset \
  --ice-csv data/hc_dataset_ice.csv \
  --plateau-csv data/hc_dataset_plateau.csv \
  --extended-csv data/hc_dataset_extended_preprocessed.csv \
  --out-csv data/hc_dataset_original_master_with_missing_features.csv \
  --mapping-report-csv data/hc_dataset_original_master_mapping_report.csv
```

This file keeps the original baseline columns and adds:
- `carbonization_time_h`
- `heating_rate_C_min`
- `pore_volume_cm3_g`
- `lc_nm`
- `la_nm`
- metadata columns such as `record_id`, `doi`, `material_name`, `category`, `electrolyte_text`, and `reversible_capacity_mAh_g`
- mapping diagnostics: `mapping_method`, `mapping_found`

#### `hc_dataset_original_master_mapping_report.csv`
One-row summary of the mapping performance.

Current included mapping summary:
- original master rows: 565
- matched rows: 560
- unmatched rows: 5
- matched fraction: 0.99115
- primary exact matches: 526
- fallback exact matches (`temperature + d002 + SSA`): 34

## Practical guidance

- Use `hc_dataset_ice.csv` and `hc_dataset_plateau.csv` for the original six-feature manuscript workflow.
- Use `hc_dataset_original_master_with_missing_features.csv` for the reviewer-response analyses that involve the additional high-missingness variables.
- Use `hc_dataset_extended_preprocessed.csv` when you need the full normalized table derived from the revised Excel source.
