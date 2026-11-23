"""CLI for automated hyperparameter tuning of a scikit-learn estimator.

Examples
--------
python tune.py --estimator random_forest_regressor --params params/random_forest.json
python tune.py --estimator svc --params params/svc.json --scoring accuracy
python tune.py --estimator sklearn.linear_model.Ridge --params params/ridge.json
python tune.py --estimator ridge --params params/ridge.json --data cars.csv --target price
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys

import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.svm import SVC, SVR

from tuning import tune_estimator

ESTIMATORS = {
    "random_forest_regressor": RandomForestRegressor,
    "random_forest_classifier": RandomForestClassifier,
    "ridge": Ridge,
    "lasso": Lasso,
    "svr": SVR,
    "svc": SVC,
    "logistic_regression": LogisticRegression,
}


def resolve_estimator(name):
    """Return an estimator instance from a short name or a dotted sklearn path."""
    if name in ESTIMATORS:
        return ESTIMATORS[name]()
    if name.startswith("sklearn.") and "." in name:
        module_name, _, class_name = name.rpartition(".")
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
            return cls()
        except (ImportError, AttributeError) as e:
            raise SystemExit(f"error: cannot load estimator '{name}': {e}")
    raise SystemExit(
        f"error: unknown estimator '{name}'. Choose one of {sorted(ESTIMATORS)} "
        "or give a dotted path starting with 'sklearn.' (e.g. sklearn.linear_model.Ridge)."
    )


def load_params(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            params = json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"error: parameter file not found: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"error: invalid JSON in {path}: {e}")
    if not isinstance(params, dict) or not params:
        raise SystemExit("error: parameter file must be a non-empty JSON object.")
    return params


def load_data(path, target, classifier):
    if path is None:
        if classifier:
            return make_classification(n_samples=300, n_features=8, n_informative=5, random_state=0)
        return make_regression(n_samples=300, n_features=8, noise=10.0, random_state=0)
    if target is None:
        raise SystemExit("error: --target is required when --data is given.")
    df = pd.read_csv(path).dropna()
    if target not in df.columns:
        raise SystemExit(f"error: column '{target}' not found. Columns: {list(df.columns)}")
    X = df.drop(columns=[target]).select_dtypes(include=[np.number]).to_numpy()
    return X, df[target].to_numpy()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Grid-search hyperparameters for a sklearn estimator.")
    p.add_argument("--estimator", required=True, help="short name or dotted sklearn path")
    p.add_argument("--params", required=True, help="JSON file mapping parameter -> list of values")
    p.add_argument("--cv", type=int, default=5, help="CV folds (default 5)")
    p.add_argument("--scoring", default=None, help="sklearn scoring string (default: estimator default)")
    p.add_argument("--data", default=None, help="CSV file; omit to use synthetic data")
    p.add_argument("--target", default=None, help="target column (required with --data)")
    p.add_argument("--output", default="models/best_model.joblib", help="where to save the best model")
    p.add_argument("--n-jobs", type=int, default=-1, help="parallel jobs (default -1 = all cores)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    estimator = resolve_estimator(args.estimator)
    param_grid = load_params(args.params)
    X, y = load_data(args.data, args.target, is_classifier(estimator))

    best, score, best_params = tune_estimator(
        estimator,
        param_grid,
        X,
        y,
        cv=args.cv,
        scoring=args.scoring,
        n_jobs=args.n_jobs,
        save_path=args.output,
    )

    print(f"Estimator:   {args.estimator}")
    print(f"Best params: {best_params}")
    print(f"Best CV score ({args.scoring or 'default'}): {score:.4f}")
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
