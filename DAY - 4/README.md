# Cross-Validation Training Script

A generic, reusable training script that evaluates any scikit-learn model with k-fold cross-validation, reports the **mean and standard deviation** of the scores, and logs every run to a **JSON file** for later analysis.

## Project structure

```
cv_training_project/
├── cv_training/
│   ├── __init__.py
│   └── cross_validation.py   # cross_validate_model() + log_results_to_json()
├── tests/
│   └── test_cross_validation.py
├── train.py                  # command-line entry point
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Command-line usage

```bash
# Synthetic data, linear model, 5 folds (defaults)
python train.py

# 10 folds with a random forest
python train.py --model random_forest --folds 10

# Your own CSV (numeric feature columns + a target column)
python train.py --data cars.csv --target price --model ridge --folds 5

# Different metric, custom log file
python train.py --scoring neg_mean_squared_error --output results/mse_runs.json

# Don't write a log
python train.py --no-log
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--folds` | 5 | Number of CV folds (integer >= 2) |
| `--model` | linear | `linear`, `ridge`, `lasso`, `random_forest` |
| `--scoring` | r2 | Any scikit-learn scoring string |
| `--data` | synthetic | CSV path |
| `--target` | – | Target column (required with `--data`) |
| `--output` | results/cv_results.json | JSON log path |
| `--no-log` | off | Skip logging |

Example output:

```
Model:   linear
Folds:   5   Scoring: r2
Scores:  0.9812, 0.9790, 0.9835, 0.9801, 0.9776
Mean:    0.9803
Std:     0.0020
Logged to results/cv_results.json
```

## Use as a library

```python
from sklearn.ensemble import RandomForestRegressor
from cv_training import cross_validate_model, log_results_to_json

results = cross_validate_model(RandomForestRegressor(), X, y, n_folds=5, scoring="r2")
print(results["mean"], results["std"])
log_results_to_json(results, "results/cv_results.json", model_name="rf")
```

## JSON log format

The log is a list; every run is appended:

```json
[
  {
    "timestamp": "2026-11-22T19:30:00",
    "model": "linear",
    "data": "synthetic",
    "scoring": "r2",
    "n_folds": 5,
    "scores": [0.98, 0.97, 0.98, 0.98, 0.97],
    "mean": 0.98,
    "std": 0.002
  }
]
```

Load it later with `pandas.read_json("results/cv_results.json")`.

## Tests

```bash
pytest -v
```

## Adding a model

Add an entry to the `MODELS` dict in `train.py`:

```python
"gbr": lambda: GradientBoostingRegressor(random_state=42),
```
