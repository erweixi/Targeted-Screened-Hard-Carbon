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


def load_xy(csv_path: str | Path, task: str) -> Tuple["pd.DataFrame", "pd.Series"]:
    """Load dataset and return X (features) and y (target).

    Parameters
    ----------
    csv_path:
        Path to the CSV dataset.
    task:
        "ice" or "plateau".

    Returns
    -------
    X, y
    """
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

