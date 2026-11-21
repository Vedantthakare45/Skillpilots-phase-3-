"""Example: evaluate a Linear Regression model on a synthetic car-price-like dataset."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from regression_eval import evaluate_regression

rng = np.random.default_rng(42)
n = 500
age = rng.uniform(0, 15, n)
mileage = rng.uniform(5_000, 200_000, n)
price = 25_000 - 900 * age - 0.05 * mileage + rng.normal(0, 1_500, n)

X = np.column_stack([age, mileage])
X_train, X_test, y_train, y_test = train_test_split(X, price, test_size=0.2, random_state=0)

model = LinearRegression().fit(X_train, y_train)
preds = model.predict(X_test)

results = evaluate_regression(
    y_test, preds, log_path="logs/results.csv", model_name="LinearRegression"
)
for name, value in results.items():
    print(f"{name}: {value:.4f}")
print("Logged to logs/results.csv")
