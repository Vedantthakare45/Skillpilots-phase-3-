# Automated Hyperparameter Tuning Pipeline

Tune **any scikit-learn estimator** with `GridSearchCV` (5-fold CV by default), get back the best estimator and its score, save the best model with `joblib`, and drive it all from a command-line interface.

## Project structure

```
tuning_project/
├── tuning/
│   ├── __init__.py
│   └── tuner.py            # tune_estimator(), save_model(), load_model()
├── params/                 # example parameter-grid files
│   ├── random_forest.json
│   ├── ridge.json
│   └── svc.json
├── tests/
│   └── test_tuner.py
├── tune.py                 # CLI
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Command-line usage

```bash
# Random forest regressor on synthetic data
python tune.py --estimator random_forest_regressor --params params/random_forest.json

# SVC with accuracy scoring
python tune.py --estimator svc --params params/svc.json --scoring accuracy

# Any sklearn class via dotted path
python tune.py --estimator sklearn.linear_model.Ridge --params params/ridge.json

# Your own CSV (numeric features + target column)
python tune.py --estimator ridge --params params/ridge.json --data cars.csv --target price --output models/ridge.joblib
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--estimator` | required | `random_forest_regressor`, `random_forest_classifier`, `ridge`, `lasso`, `svr`, `svc`, `logistic_regression`, or a dotted path starting with `sklearn.` |
| `--params` | required | JSON file: parameter name -> list of values |
| `--cv` | 5 | Number of CV folds |
| `--scoring` | estimator default | sklearn scoring string |
| `--data` / `--target` | synthetic | CSV path and target column |
| `--output` | models/best_model.joblib | Where to save the best model |
| `--n-jobs` | -1 | Parallel jobs |

Example output:

```
Estimator:   ridge
Best params: {'alpha': 0.1}
Best CV score (default): 0.9921
Saved model to models/best_model.joblib
```

## Parameter file format

```json
{
  "n_estimators": [50, 100, 200],
  "max_depth": [null, 5, 10]
}
```

Every value must be a non-empty list. `null` becomes Python `None`.

## Use as a library

```python
from sklearn.ensemble import RandomForestRegressor
from tuning import tune_estimator, load_model

best, score, params = tune_estimator(
    RandomForestRegressor(random_state=42),
    {"n_estimators": [50, 100], "max_depth": [None, 10]},
    X, y,
    cv=5,
    scoring="r2",
    save_path="models/rf.joblib",
)
print(params, score)

model = load_model("models/rf.joblib")
predictions = model.predict(X_new)
```

## Tests

```bash
pytest -v
```

## Notes

- Models saved with joblib should only be loaded from sources you trust, and with the same scikit-learn version used to save them.
- `--estimator` accepts only the built-in names or `sklearn.*` paths, so arbitrary modules cannot be loaded from the command line.
- Without `--data`, the CLI generates synthetic classification data for classifiers and regression data for regressors.
