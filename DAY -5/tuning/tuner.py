"""Automated hyperparameter tuning for any scikit-learn estimator."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import joblib
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV


def tune_estimator(
    estimator: BaseEstimator,
    param_grid: Dict[str, list],
    X,
    y,
    cv: int = 5,
    scoring: Optional[str] = None,
    n_jobs: int = -1,
    save_path: Optional[str] = None,
) -> Tuple[BaseEstimator, float, Dict[str, Any]]:
    """Grid-search `estimator` over `param_grid` with k-fold CV (default 5).

    Returns
    -------
    (best_estimator, best_score, best_params)

    If `save_path` is given, the best (refit) estimator is saved with joblib.
    """
    if not param_grid:
        raise ValueError("param_grid must not be empty.")
    for name, values in param_grid.items():
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            raise ValueError(f"Parameter '{name}' must map to a non-empty list of values.")

    search = GridSearchCV(
        estimator,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X, y)

    if save_path:
        save_model(search.best_estimator_, save_path)

    return search.best_estimator_, float(search.best_score_), dict(search.best_params_)


def save_model(model, path: str) -> None:
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
