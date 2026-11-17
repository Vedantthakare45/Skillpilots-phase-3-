# Linear Regression with TensorFlow

A simple linear regression model built with TensorFlow/Keras on synthetic data. It learns the relationship `y = 3x + 2` from noisy samples and plots the predictions against the true values.

## What it does

1. Sets up a virtual environment with TensorFlow
2. Generates synthetic data (X, y) with NumPy
3. Defines a Sequential model with one Dense layer
4. Compiles with MSE loss and the Adam optimizer
5. Trains the model and plots predictions vs. true values

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python linear_regression_tf.py
```

The script prints the final loss and the learned slope and intercept (which should be close to 3 and 2), and saves the plot as `predictions_vs_true.png`.

## Files

- `linear_regression_tf.py`: model training and plotting
- `requirements.txt`: Python dependencies

## Tech

Python, TensorFlow, NumPy, Matplotlib

*Learning project from Skillplot (ML track).*
