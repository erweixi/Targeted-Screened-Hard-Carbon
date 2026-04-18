#!/usr/bin/env python3
"""Make progressive Random Forest performance curves with two stages:

Stage 1:
- Perform greedy forward selection on the six baseline features.
- Add one baseline feature at a time according to the best 5-fold CV mean R^2.

Stage 2:
- Starting from the full selected baseline set, progressively add omitted features
  ordered by missingness from low to high.
- At each step, drop rows with missing values in the current feature set.

The output is a long figure showing:
- 5-fold CV mean R^2
- Hold-out test R^2
- Complete-case sample size n at each step
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

BASELINE_FEATURES = [
    "carbonization_temperature_C",
    "d002_nm",
    "id_ig",
    "ssa_m2_g",
    "electrolyte_type",
    "current_density_mA_g",
]

FEATURE_LABELS = {
    "carbonization_temperature_C": "Temp",
    "d002_nm": "d002",
    "id_ig": "ID/IG",
    "ssa_m2_g": "SSA",
    "electrolyte_type": "Electrolyte",
    "current_density_mA_g": "Current",
    "carbonization_time_h": "Time",
    "heating_rate_C_min": "Heat",
    "pore_volume_cm3_g": "Pore",
    "lc_nm": "Lc",
    "la_nm": "La",
}

OMITTED_FEATURE_CANDIDATES = [
    "carbonization_time_h",
    "heating_rate_C_min",
    "pore_volume_cm3_g",
    "lc_nm",
    "la_nm",
]

TARGET_CONFIG = {
    "capacity": {
        "target_col": "reversible_capacity_mAh_g",
        "display_name": "Reversible capacity",
        "best_params_json": "results/reviewer_extension_fixed/capacity_baseline/rf_metrics_capacity.json",
        "best_params_key": "best_params",
    },
    "reversible_capacity_mAh_g": {
        "target_col": "reversible_capacity_mAh_g",
        "display_name": "Reversible capacity",
        "best_params_json": "results/reviewer_extension_fixed/capacity_baseline/rf_metrics_capacity.json",
        "best_params_key": "best_params",
    },
    "ice": {
        "target_col": "lce",
        "display_name": "ICE",
        "best_params_json": "results/reviewer/model_comparison_best_params_ice.json",
        "best_params_key": "RandomForest.best_params",
    },
    "lce": {
        "target_col": "lce",
        "display_name": "ICE",
        "best_params_json": "results/reviewer/model_comparison_best_params_ice.json",
        "best_params_key": "RandomForest.best_params",
    },
    "plateau": {
        "target_col": "plateau_capacity_mAh_g",
        "display_name": "Plateau capacity",
        "best_params_json": "results/reviewer/model_comparison_best_params_plateau.json",
        "best_params_key": "RandomForest.best_params",
    },
    "plateau_capacity_mAh_g": {
        "target_col": "plateau_capacity_mAh_g",
        "display_name": "Plateau capacity",
        "best_params_json": "results/reviewer/model_comparison_best_params_plateau.json",
        "best_params_key": "RandomForest.best_params",
    },
}

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
    parser.add_argument(
        "--target",
        default="capacity",
        choices=sorted(TARGET_CONFIG.keys()),
        help="Target task to analyze.",
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--use_existing_best_params", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--min_n_for_step",
        type=int,
        default=10,
        help="Skip a step if complete-case n is below this threshold.",
    )
    return parser.parse_args()


def resolve_best_params(project_root: Path, target_key: str, use_existing: bool) -> Dict:
    if not use_existing:
        return dict(FALLBACK_RF_PARAMS)

    cfg = TARGET_CONFIG[target_key]
    json_path = project_root / cfg["best_params_json"]
    if not json_path.exists():
        return dict(FALLBACK_RF_PARAMS)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    current = data
    for key in cfg["best_params_key"].split("."):
        current = current[key]
    return dict(current)


def prepare_subset(df: pd.DataFrame, features: List[str], target_col: str) -> pd.DataFrame:
    cols = list(dict.fromkeys(features + [target_col]))
    subset = df[cols].dropna().copy()
    return subset


def evaluate_subset(
    subset: pd.DataFrame,
    features: List[str],
    target_col: str,
    rf_params: Dict,
    random_state: int,
    cv_folds: int,
    test_size: float,
) -> Dict:
    X = subset[features]
    y = subset[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    model = RandomForestRegressor(random_state=random_state, n_jobs=1, **rf_params)
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=1)

    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, test_pred)

    return {
        "n_complete": int(len(subset)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "cv_r2_mean": float(np.mean(cv_scores)),
        "cv_r2_std": float(np.std(cv_scores, ddof=1)) if len(cv_scores) > 1 else 0.0,
        "test_r2": float(test_r2),
    }


def evaluate_feature_set(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    rf_params: Dict,
    random_state: int,
    cv_folds: int,
    test_size: float,
    min_n_for_step: int,
) -> Dict:
    subset = prepare_subset(df, features, target_col)
    if len(subset) < min_n_for_step:
        raise ValueError(
            f"Too few complete cases ({len(subset)}) for features {features}. "
            f"Increase data coverage or lower --min_n_for_step."
        )
    return evaluate_subset(
        subset=subset,
        features=features,
        target_col=target_col,
        rf_params=rf_params,
        random_state=random_state,
        cv_folds=cv_folds,
        test_size=test_size,
    )


def rank_omitted_by_missingness(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    rates = []
    for feat in candidates:
        if feat not in df.columns:
            continue
        rates.append((feat, float(df[feat].isna().mean())))
    rates.sort(key=lambda x: (x[1], FEATURE_LABELS.get(x[0], x[0])))
    return [feat for feat, _ in rates]


def forward_select_baseline(
    df: pd.DataFrame,
    target_col: str,
    rf_params: Dict,
    random_state: int,
    cv_folds: int,
    test_size: float,
    min_n_for_step: int,
) -> Tuple[List[Dict], List[str]]:
    selected: List[str] = []
    remaining = [f for f in BASELINE_FEATURES if f in df.columns]
    rows: List[Dict] = []

    step_idx = 0
    while remaining:
        candidates = []
        for feat in remaining:
            current_features = selected + [feat]
            subset = prepare_subset(df, current_features, target_col)
            if len(subset) < min_n_for_step:
                continue
            metrics = evaluate_subset(
                subset=subset,
                features=current_features,
                target_col=target_col,
                rf_params=rf_params,
                random_state=random_state,
                cv_folds=cv_folds,
                test_size=test_size,
            )
            candidates.append((feat, current_features, metrics))

        if not candidates:
            break

        candidates.sort(
            key=lambda item: (
                item[2]["cv_r2_mean"],
                item[2]["test_r2"],
                -item[2]["cv_r2_std"],
                item[2]["n_complete"],
                FEATURE_LABELS.get(item[0], item[0]),
            ),
            reverse=True,
        )
        best_feat, best_features, best_metrics = candidates[0]
        selected.append(best_feat)
        remaining.remove(best_feat)

        label = FEATURE_LABELS.get(best_feat, best_feat) if step_idx == 0 else f"+{FEATURE_LABELS.get(best_feat, best_feat)}"
        row = {
            "step": step_idx,
            "stage": "baseline_forward_selection",
            "added_feature": best_feat,
            "step_label": label,
            "features": ";".join(best_features),
            "n_features": len(best_features),
        }
        row.update(best_metrics)
        rows.append(row)
        step_idx += 1

    return rows, selected


def append_omitted_features(
    df: pd.DataFrame,
    baseline_selected: List[str],
    target_col: str,
    rf_params: Dict,
    random_state: int,
    cv_folds: int,
    test_size: float,
    min_n_for_step: int,
) -> Tuple[List[Dict], List[str]]:
    ordered_omitted = rank_omitted_by_missingness(df, OMITTED_FEATURE_CANDIDATES)
    rows: List[Dict] = []
    features = list(baseline_selected)
    start_step = len(baseline_selected)

    for i, feat in enumerate(ordered_omitted):
        features = features + [feat]
        metrics = evaluate_feature_set(
            df=df,
            features=features,
            target_col=target_col,
            rf_params=rf_params,
            random_state=random_state,
            cv_folds=cv_folds,
            test_size=test_size,
            min_n_for_step=min_n_for_step,
        )
        row = {
            "step": start_step + i,
            "stage": "omitted_features_by_missingness",
            "added_feature": feat,
            "step_label": f"+{FEATURE_LABELS.get(feat, feat)}",
            "features": ";".join(features),
            "n_features": len(features),
        }
        row.update(metrics)
        rows.append(row)

    return rows, ordered_omitted


def make_figure(results: pd.DataFrame, title: str, out_path: Path, dpi: int) -> None:
    n_steps = len(results)
    fig_width = max(12.0, 1.15 * n_steps + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    x = np.arange(n_steps)

    ax.plot(x, results["cv_r2_mean"], marker="o", linewidth=2.0, label="5-fold CV mean R$^2$")
    ax.plot(x, results["test_r2"], marker="o", linewidth=2.0, linestyle="--", label="Hold-out test R$^2$")

    ax.set_xticks(x)
    ax.set_xticklabels(results["step_label"])
    ax.set_ylabel("R$^2$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

    y_min = float(min(results["cv_r2_mean"].min(), results["test_r2"].min()))
    y_max = float(max(results["cv_r2_mean"].max(), results["test_r2"].max()))
    y_range = y_max - y_min
    padding = max(0.05, 0.12 * y_range if y_range > 0 else 0.12)
    lower = y_min - padding
    upper = y_max + padding

    # Make limits "nice" while never clipping real data.
    lower = math.floor(lower * 20.0) / 20.0
    upper = math.ceil(upper * 20.0) / 20.0
    ax.set_ylim(lower, upper)

    transform = blended_transform_factory(ax.transData, ax.transAxes)
    for i, n_complete in enumerate(results["n_complete"]):
        ax.text(i, 0.04, f"n={int(n_complete)}", transform=transform, ha="center", va="bottom", fontsize=9)

    # Visual separator between the two stages.
    baseline_count = int((results["stage"] == "baseline_forward_selection").sum())
    if 0 < baseline_count < len(results):
        ax.axvline(baseline_count - 0.5, linestyle=":", linewidth=1.5, alpha=0.8)
        ax.text(
            baseline_count - 0.52,
            0.98,
            "add omitted features",
            transform=transform,
            rotation=90,
            ha="right",
            va="top",
            fontsize=9,
        )
        ax.text(
            (baseline_count - 1) / 2.0,
            0.98,
            "baseline forward selection",
            transform=transform,
            ha="center",
            va="top",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    cfg = TARGET_CONFIG[args.target]
    target_col = cfg["target_col"]
    display_name = cfg["display_name"]
    rf_params = resolve_best_params(project_root, args.target, args.use_existing_best_params)

    df = pd.read_csv(args.data)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {args.data}")

    baseline_rows, baseline_order = forward_select_baseline(
        df=df,
        target_col=target_col,
        rf_params=rf_params,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        test_size=args.test_size,
        min_n_for_step=args.min_n_for_step,
    )

    omitted_rows, omitted_order = append_omitted_features(
        df=df,
        baseline_selected=baseline_order,
        target_col=target_col,
        rf_params=rf_params,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        test_size=args.test_size,
        min_n_for_step=args.min_n_for_step,
    )

    result_df = pd.DataFrame(baseline_rows + omitted_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    slug = target_col.replace("_mAh_g", "")
    csv_path = args.output_dir / f"progressive_rf_{slug}.csv"
    png_path = args.output_dir / f"progressive_rf_{slug}.png"
    json_path = args.output_dir / f"progressive_rf_{slug}_metadata.json"

    result_df.to_csv(csv_path, index=False)

    title = f"{display_name}: forward-selected baseline + omitted-feature extension"
    make_figure(result_df, title=title, out_path=png_path, dpi=args.dpi)

    metadata = {
        "data": str(args.data),
        "target": target_col,
        "display_name": display_name,
        "rf_params": rf_params,
        "random_state": args.random_state,
        "cv_folds": args.cv_folds,
        "test_size": args.test_size,
        "min_n_for_step": args.min_n_for_step,
        "baseline_features": BASELINE_FEATURES,
        "baseline_forward_selection_order": baseline_order,
        "omitted_feature_order_by_missingness": omitted_order,
        "outputs": {"csv": str(csv_path), "png": str(png_path)},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[OK] wrote {csv_path}")
    print(f"[OK] wrote {png_path}")
    print(f"[OK] wrote {json_path}")
    print("[INFO] baseline forward-selection order:", baseline_order)
    print("[INFO] omitted-feature order by missingness:", omitted_order)


if __name__ == "__main__":
    main()
