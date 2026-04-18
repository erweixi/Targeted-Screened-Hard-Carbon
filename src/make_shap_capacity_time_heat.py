#!/usr/bin/env python3
"""SHAP beeswarm for reversible capacity only.

Feature set:
- baseline six features
- plus carbonization_time_h
- plus heating_rate_C_min

Target:
- reversible_capacity_mAh_g

This script writes:
- outputs/shap_capacity_time_heat.png
- outputs/shap_capacity_time_heat_mean_abs.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

FEATURES = [
    "carbonization_temperature_C",
    "d002_nm",
    "id_ig",
    "ssa_m2_g",
    "electrolyte_type",
    "current_density_mA_g",
    "carbonization_time_h",
    "heating_rate_C_min",
]

FEATURE_LABELS = {
    "carbonization_temperature_C": "Carbonization\ntemperature",
    "d002_nm": "d$_{002}$",
    "id_ig": "I$_D$/I$_G$",
    "ssa_m2_g": "SSA",
    "electrolyte_type": "Electrolyte",
    "current_density_mA_g": "Current density",
    "carbonization_time_h": "Carbonization time",
    "heating_rate_C_min": "Heating rate",
}

TARGET_COL = "reversible_capacity_mAh_g"
DISPLAY_NAME = "Reversible capacity"

FALLBACK_RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 30,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": True,
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default_data = project_root / "data" / "hc_dataset_extended_preprocessed.csv"
    default_output = project_root / "outputs"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--output_dir", type=Path, default=default_output)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--max_display", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--use_existing_best_params",
        action="store_true",
        help="Load RF best params from results/reviewer_extension_fixed/capacity_baseline/rf_metrics_capacity.json",
    )
    return parser.parse_args()


def resolve_best_params(project_root: Path, use_existing: bool) -> Dict:
    if not use_existing:
        return dict(FALLBACK_RF_PARAMS)

    json_path = project_root / "results" / "reviewer_extension_fixed" / "capacity_baseline" / "rf_metrics_capacity.json"
    if not json_path.exists():
        return dict(FALLBACK_RF_PARAMS)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "best_params" in data and isinstance(data["best_params"], dict):
        return dict(data["best_params"])

    return dict(FALLBACK_RF_PARAMS)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    rf_params = resolve_best_params(project_root, args.use_existing_best_params)

    df = pd.read_csv(args.data)
    subset = df[FEATURES + [TARGET_COL]].dropna().copy()

    X = subset[FEATURES].rename(columns=FEATURE_LABELS)
    y = subset[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    model = RandomForestRegressor(
        random_state=args.random_state,
        n_jobs=-1,
        **rf_params,
    )
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "shap_capacity_time_heat.png"
    csv_path = args.output_dir / "shap_capacity_time_heat_mean_abs.csv"

    mean_abs = pd.Series(abs(shap_values).mean(axis=0), index=X_test.columns).sort_values(ascending=False)
    mean_abs.rename("mean_abs_shap").to_csv(csv_path, header=True)

    plt.figure(figsize=(10, 5.8))
    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="dot",
        max_display=args.max_display,
        show=False,
        plot_size=None,
        color_bar=True,
    )
    plt.title(f"{DISPLAY_NAME}: baseline + carbonization time + heating rate", fontsize=16)
    plt.tight_layout()
    plt.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    plt.close()

    print(f"[OK] wrote {png_path}")
    print(f"[OK] wrote {csv_path}")
    print(f"[INFO] n_complete = {len(subset)}")


if __name__ == "__main__":
    main()
