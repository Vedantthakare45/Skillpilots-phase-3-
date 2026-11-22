"""Command-line training script with cross-validation.

Examples
--------
python train.py --folds 5
python train.py --model random_forest --folds 10 --output results/cv_results.json
python train.py --data cars.csv --target price --model ridge --scoring neg_mean_squared_error
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cv_training import cross_validate_model, log_results_to_json

MODELS = {
    "linear": lambda: make_pipeline(StandardScaler(), LinearRegression()),
    "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "lasso": lambda: make_pipeline(StandardScaler(), Lasso(alpha=0.1)),
    "random_forest": lambda: RandomForestRegressor(n_estimators=100, random_state=42),
}


def load_data(path, target):
    """Load a CSV (numeric features + target column) or generate synthetic data."""
    if path is None:
        X, y = make_regression(n_samples=300, n_features=6, noise=15.0, random_state=0)
        return X, y
    df = pd.read_csv(path)
    if target is None:
        raise SystemExit("error: --target is required when --data is given.")
    if target not in df.columns:
        raise SystemExit(f"error: column '{target}' not found. Columns: {list(df.columns)}")
    df = df.dropna()
    X = df.drop(columns=[target]).select_dtypes(include=[np.number]).to_numpy()
    y = df[target].to_numpy()
    return X, y


def positive_int(value):
    n = int(value)
    if n < 2:
        raise argparse.ArgumentTypeError("must be an integer >= 2")
    return n


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Cross-validate a regression model.")
    p.add_argument("--folds", type=positive_int, default=5, help="number of CV folds (default 5)")
    p.add_argument("--model", choices=sorted(MODELS), default="linear", help="model to train")
    p.add_argument("--scoring", default="r2", help="sklearn scoring string (default r2)")
    p.add_argument("--data", default=None, help="CSV file; omit to use synthetic data")
    p.add_argument("--target", default=None, help="target column name (required with --data)")
    p.add_argument("--output", default="results/cv_results.json", help="JSON log path")
    p.add_argument("--no-log", action="store_true", help="do not write the JSON log")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    X, y = load_data(args.data, args.target)
    model = MODELS[args.model]()

    results = cross_validate_model(model, X, y, n_folds=args.folds, scoring=args.scoring)

    print(f"Model:   {args.model}")
    print(f"Folds:   {results['n_folds']}   Scoring: {results['scoring']}")
    print("Scores:  " + ", ".join(f"{s:.4f}" for s in results["scores"]))
    print(f"Mean:    {results['mean']:.4f}")
    print(f"Std:     {results['std']:.4f}")

    if not args.no_log:
        log_results_to_json(
            results,
            args.output,
            model_name=args.model,
            extra={"data": args.data or "synthetic"},
        )
        print(f"Logged to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
