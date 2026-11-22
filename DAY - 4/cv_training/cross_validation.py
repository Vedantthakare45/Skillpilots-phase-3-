"""Generic cross-validation training utilities."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from sklearn.model_selection import KFold, cross_val_score


def cross_validate_model(
    model,
    X,
    y,
    n_folds: int = 5,
    scoring: str = "r2",
    shuffle: bool = True,
    random_state: Optional[int] = 42,
) -> Dict[str, Any]:
    """Evaluate `model` on (X, y) with k-fold cross-validation.

    Returns a dict with the per-fold scores, their mean and standard deviation.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    n_samples = len(X)
    if n_folds > n_samples:
        raise ValueError(f"n_folds ({n_folds}) cannot exceed number of samples ({n_samples}).")

    cv = KFold(
        n_splits=n_folds,
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return {
        "scoring": scoring,
        "n_folds": n_folds,
        "scores": [float(s) for s in scores],
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }


def log_results_to_json(
    results: Dict[str, Any],
    log_path: str,
    model_name: str = "model",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a run to a JSON file holding a list of runs."""
    folder = os.path.dirname(os.path.abspath(log_path))
    os.makedirs(folder, exist_ok=True)

    runs = []
    if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
        with open(log_path, "r", encoding="utf-8") as f:
            try:
                runs = json.load(f)
            except json.JSONDecodeError:
                raise ValueError(f"{log_path} exists but is not valid JSON.")
        if not isinstance(runs, list):
            raise ValueError(f"{log_path} must contain a JSON list.")

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        **(extra or {}),
        **results,
    }
    runs.append(entry)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)
