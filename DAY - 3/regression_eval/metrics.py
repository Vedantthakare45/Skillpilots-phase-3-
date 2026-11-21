"""Reusable evaluation module for regression models."""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Dict, Optional

import numpy as np

METRIC_NAMES = ("MAE", "MSE", "RMSE", "R2")


def _validate(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has {y_true.size} values, y_pred has {y_pred.size}."
        )
    if y_true.size == 0:
        raise ValueError("Inputs must not be empty.")
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("Inputs must not contain NaN values.")
    return y_true, y_pred


def mean_absolute_error(y_true, y_pred) -> float:
    y_true, y_pred = _validate(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_squared_error(y_true, y_pred) -> float:
    y_true, y_pred = _validate(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2_score(y_true, y_pred) -> float:
    """Coefficient of determination.

    If y_true is constant: returns 1.0 for a perfect prediction, else 0.0
    (same convention as scikit-learn).
    """
    y_true, y_pred = _validate(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def log_results_to_csv(
    results: Dict[str, float], log_path: str, model_name: str = "model"
) -> None:
    """Append one row of results to a CSV file (header written if file is new)."""
    folder = os.path.dirname(os.path.abspath(log_path))
    os.makedirs(folder, exist_ok=True)
    write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "model_name", *METRIC_NAMES])
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                model_name,
                *[results[m] for m in METRIC_NAMES],
            ]
        )


def evaluate_regression(
    y_true,
    y_pred,
    log_path: Optional[str] = None,
    model_name: str = "model",
) -> Dict[str, float]:
    """Evaluate a regression model.

    Parameters
    ----------
    y_true, y_pred : array-like of true and predicted values.
    log_path : optional CSV path; if given, results are appended to it.
    model_name : label stored in the CSV log.

    Returns
    -------
    dict with keys MAE, MSE, RMSE, R2.
    """
    y_true, y_pred = _validate(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    results = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }
    if log_path:
        log_results_to_csv(results, log_path, model_name)
    return results
