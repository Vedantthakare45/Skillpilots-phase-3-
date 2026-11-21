# Regression Evaluation Module

A small, reusable Python module that evaluates regression models with **MAE, MSE, RMSE and R²**, with optional CSV logging and a full unit-test suite.

## Project structure

```
regression_eval_project/
├── regression_eval/
│   ├── __init__.py
│   └── metrics.py        # evaluation function + individual metrics + CSV logger
├── tests/
│   └── test_metrics.py   # unit tests for every metric, logging, validation
├── example.py            # runnable usage example (scikit-learn)
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic

```python
from regression_eval import evaluate_regression

y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]

results = evaluate_regression(y_true, y_pred)
print(results)
# {'MAE': 0.5, 'MSE': 0.375, 'RMSE': 0.612..., 'R2': 0.948...}
```

### With CSV logging

```python
evaluate_regression(y_true, y_pred, log_path="logs/results.csv", model_name="LinearRegression")
```

Each call **appends** one row (`timestamp, model_name, MAE, MSE, RMSE, R2`). The header is written once, and missing folders are created automatically.

### With a scikit-learn model

```python
from sklearn.linear_model import LinearRegression
from regression_eval import evaluate_regression

model = LinearRegression().fit(X_train, y_train)
evaluate_regression(y_test, model.predict(X_test), log_path="logs/results.csv", model_name="LinearRegression")
```

### Individual metrics

```python
from regression_eval import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
```

## Run the example

```bash
python example.py
```

## Run the tests

```bash
pytest -v
```

## Behaviour notes

- Inputs can be lists, NumPy arrays or pandas Series.
- Raises `ValueError` for mismatched lengths, empty inputs or NaN values.
- If `y_true` is constant, R² returns `1.0` for a perfect prediction and `0.0` otherwise (same convention as scikit-learn).

## Metric definitions

| Metric | Formula |
|--------|---------|
| MAE  | mean(\|y − ŷ\|) |
| MSE  | mean((y − ŷ)²) |
| RMSE | √MSE |
| R²   | 1 − SS_res / SS_tot |
