"""Model registry and helper functions for the hard-carbon ML pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .search_spaces import get_search_spaces


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    slug: str
    estimator: object
    search_space: Optional[dict]
    shap_family: str
    supports_shap: bool = True


def get_model_specs(task: str, random_state: int = 42, n_jobs: int = 1) -> List[ModelSpec]:
    """Return the benchmark model list used throughout the project."""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    try:
        from xgboost import XGBRegressor
    except Exception as e:  # pragma: no cover - dependency handled by caller
        raise RuntimeError(
            "xgboost is required for the XGBRegressor benchmark. Install it with: pip install xgboost"
        ) from e

    spaces = get_search_spaces(task)

    return [
        ModelSpec(
            display_name="Linear",
            slug="linear",
            estimator=LinearRegression(),
            search_space=None,
            shap_family="linear",
        ),
        ModelSpec(
            display_name="GBR",
            slug="gbr",
            estimator=GradientBoostingRegressor(random_state=random_state),
            search_space=spaces["gbr"],
            shap_family="tree",
        ),
        ModelSpec(
            display_name="Ridge",
            slug="ridge",
            estimator=Ridge(),
            search_space=spaces["ridge"],
            shap_family="linear",
        ),
        ModelSpec(
            display_name="Lasso",
            slug="lasso",
            estimator=Lasso(max_iter=20000),
            search_space=spaces["lasso"],
            shap_family="linear",
        ),
        ModelSpec(
            display_name="NeuralNetwork",
            slug="neuralnetwork",
            estimator=make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(100, 100),
                    random_state=random_state,
                    max_iter=5000,
                ),
            ),
            search_space=None,
            shap_family="kernel",
            supports_shap=False,
        ),
        ModelSpec(
            display_name="RandomForest",
            slug="randomforest",
            estimator=RandomForestRegressor(random_state=random_state),
            search_space=spaces["rf"],
            shap_family="tree",
        ),
        ModelSpec(
            display_name="XGB",
            slug="xgb",
            estimator=XGBRegressor(
                random_state=random_state,
                objective="reg:squarederror",
                n_jobs=1,
            ),
            search_space=spaces["xgb"],
            shap_family="tree",
        ),
    ]


def get_model_spec(task: str, model_name: str, random_state: int = 42, n_jobs: int = 1) -> ModelSpec:
    """Return a single model spec by display name or slug."""
    model_name = model_name.strip().lower()
    for spec in get_model_specs(task=task, random_state=random_state, n_jobs=n_jobs):
        if model_name in {spec.display_name.lower(), spec.slug.lower()}:
            return spec
    valid = ", ".join(spec.display_name for spec in get_model_specs(task=task, random_state=random_state, n_jobs=n_jobs))
    raise ValueError(f"Unknown model '{model_name}'. Valid options: {valid}")
