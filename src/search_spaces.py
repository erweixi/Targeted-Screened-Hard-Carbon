"""Bayesian hyperparameter search spaces.

These spaces are extracted and consolidated from the provided notebooks.

Notes
-----
- `BayesSearchCV` comes from scikit-optimize.
- Search ranges are intentionally broad enough to reproduce the manuscript-level
  performance without overfitting to a single run.
"""

from __future__ import annotations

from typing import Dict

from .utils import require_skopt


def get_search_spaces(task: str) -> Dict[str, dict]:
    """Return search spaces for the given task.

    Parameters
    ----------
    task:
        "ice" or "plateau".

    Returns
    -------
    dict
        Keys: model identifiers (gbr, ridge, lasso, rf, xgb)
        Values: skopt search space dictionaries.
    """

    require_skopt()
    from skopt.space import Categorical, Integer, Real

    task = task.lower().strip()

    if task == "ice":
        gbr_space = {
            "n_estimators": Integer(1, 200),
            "learning_rate": Real(0.01, 1.0, prior="uniform"),
            "max_depth": Integer(1, 10),
        }
    elif task == "plateau":
        # Matches the plateau notebook's narrower depth range.
        gbr_space = {
            "n_estimators": Integer(1, 100),
            "learning_rate": Real(0.01, 1.0, prior="uniform"),
            "max_depth": Integer(5, 10),
        }
    else:
        raise ValueError("task must be 'ice' or 'plateau'")

    ridge_space = {
        "alpha": Real(0.01, 10.0, prior="log-uniform"),
    }

    lasso_space = {
        "alpha": Real(0.01, 10.0, prior="log-uniform"),
    }

    rf_space = {
        "n_estimators": Integer(100, 600),
        "max_depth": Integer(5, 30),
        "min_samples_split": Integer(2, 10),
        "min_samples_leaf": Integer(1, 10),
        "max_features": Categorical(["sqrt", "log2", None]),
        "bootstrap": Categorical([True, False]),
    }

    xgb_space = {
        "n_estimators": Integer(1, 150),
        "max_depth": Integer(5, 10),
        "learning_rate": Real(0.03, 0.1, prior="log-uniform"),
        "subsample": Real(0.5, 0.6),
        "colsample_bytree": Real(0.6, 1.0),
        "gamma": Categorical([0]),
        "reg_alpha": Real(1e-5, 1.0, prior="log-uniform"),
        "reg_lambda": Real(1e-5, 1.0, prior="log-uniform"),
    }

    return {
        "gbr": gbr_space,
        "ridge": ridge_space,
        "lasso": lasso_space,
        "rf": rf_space,
        "xgb": xgb_space,
    }

