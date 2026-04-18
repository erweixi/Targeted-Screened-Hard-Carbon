"""Utility helpers for the hard-carbon ML pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


FEATURE_COLUMNS: List[str] = [
    "carbonization_temperature_C",
    "d002_nm",
    "id_ig",
    "ssa_m2_g",
    "electrolyte_type",
    "current_density_mA_g",
]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    target_column: str


TASK_SPECS: Dict[str, TaskSpec] = {
    "ice": TaskSpec(name="ice", target_column="lce"),
    "plateau": TaskSpec(name="plateau", target_column="plateau_capacity_mAh_g"),
}


SCORING = {
    "r2": "r2",
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
}


def load_xy(csv_path: str | Path, task: str) -> Tuple["pd.DataFrame", "pd.Series"]:
    """Load dataset and return X (features) and y (target)."""
    import pandas as pd

    task = task.lower().strip()
    if task not in TASK_SPECS:
        raise ValueError(f"Unknown task: {task}. Expected one of {list(TASK_SPECS)}")

    df = pd.read_csv(csv_path)

    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"Missing required feature columns in {csv_path}: {missing_features}"
        )

    target_col = TASK_SPECS[task].target_column
    if target_col not in df.columns:
        raise ValueError(f"Missing target column '{target_col}' in {csv_path}.")

    X = df[FEATURE_COLUMNS].copy()
    y = df[target_col].copy()

    return X, y


def load_dataset(csv_path: str | Path):
    """Load the raw CSV for descriptive summaries."""
    import pandas as pd

    return pd.read_csv(csv_path)


def split_train_test(
    X: "pd.DataFrame",
    y: "pd.Series",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Train/test split (default 8:2) with a fixed random seed."""
    from sklearn.model_selection import train_test_split

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute common regression metrics."""
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": rmse,
    }


def cv_scores_to_summary(cv_results: Dict[str, object]) -> Dict[str, float]:
    """Convert cross_validate output into manuscript-friendly summary statistics."""
    import numpy as np

    r2_scores = cv_results["test_r2"]
    mae_scores = -cv_results["test_mae"]
    rmse_scores = -cv_results["test_rmse"]

    return {
        "cv_r2_mean": float(np.mean(r2_scores)),
        "cv_r2_std": float(np.std(r2_scores, ddof=1)) if len(r2_scores) > 1 else 0.0,
        "cv_mae_mean": float(np.mean(mae_scores)),
        "cv_mae_std": float(np.std(mae_scores, ddof=1)) if len(mae_scores) > 1 else 0.0,
        "cv_rmse_mean": float(np.mean(rmse_scores)),
        "cv_rmse_std": float(np.std(rmse_scores, ddof=1)) if len(rmse_scores) > 1 else 0.0,
    }


def fold_metrics_frame(cv_results: Dict[str, object]):
    """Return per-fold metrics as a dataframe."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "fold": list(range(1, len(cv_results["test_r2"]) + 1)),
            "r2": cv_results["test_r2"],
            "mae": -cv_results["test_mae"],
            "rmse": -cv_results["test_rmse"],
        }
    )
    return df


def dataset_overview_frame(df, task: str):
    """Summarize current feature coverage and descriptive statistics."""
    import pandas as pd

    target_col = TASK_SPECS[task].target_column
    cols = FEATURE_COLUMNS + [target_col]
    rows = []
    for col in cols:
        series = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "n_missing": int(series.isna().sum()),
                "missing_fraction": float(series.isna().mean()),
                "n_unique": int(series.nunique(dropna=True)),
                "mean": float(series.mean()) if pd.api.types.is_numeric_dtype(series) else None,
                "std": float(series.std()) if pd.api.types.is_numeric_dtype(series) else None,
                "min": float(series.min()) if pd.api.types.is_numeric_dtype(series) else None,
                "max": float(series.max()) if pd.api.types.is_numeric_dtype(series) else None,
            }
        )
    return pd.DataFrame(rows)


def ensure_jsonable(obj):
    """Recursively convert numpy/scikit objects to JSON-serializable types."""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): ensure_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ensure_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def require_skopt():
    """Import scikit-optimize objects with a clear error message if missing."""
    try:
        from skopt import BayesSearchCV  # noqa: F401
        from skopt.space import Categorical, Integer, Real  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "scikit-optimize is required for Bayesian hyperparameter search. "
            "Install it with: pip install scikit-optimize"
        ) from e
