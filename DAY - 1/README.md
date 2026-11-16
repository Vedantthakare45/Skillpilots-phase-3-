# Linear Regression with TensorFlow

A simple linear regression model built with TensorFlow (Keras) on synthetic data. The model learns the relationship `y = 3x + 2` from noisy data and plots its predictions against the true values.

## Requirements

- Python 3.9 to 3.12
- TensorFlow
- NumPy
- Matplotlib

## Setup

### Option 1: JupyterLab (quick)

Run this in the first notebook cell, then restart the kernel:

```python
%pip install tensorflow numpy matplotlib
```

### Option 2: Virtual environment (recommended)

```bash
python -m venv tf_env

# Windows
tf_env\Scripts\activate
# Mac/Linux
source tf_env/bin/activate

pip install tensorflow numpy matplotlib ipykernel
python -m ipykernel install --user --name tf_env --display-name "Python (tf_env)"
```

Open JupyterLab and select the **Python (tf_env)** kernel.

Verify the install:

```python
import tensorflow as tf
print(tf.__version__)
```

## Steps

1. Create a virtual environment and install TensorFlow
2. Generate synthetic data (X, y) with NumPy
3. Define a TensorFlow Sequential model with one Dense layer
4. Compile the model with MSE loss and the Adam optimizer
5. Train the model and plot predictions vs. true values

## Usage

Run the code cells in order in a notebook. After training, the learned weight and bias are printed. They should be close to 3 and 2.

## Expected Output

- Printed weight near `3.0` and bias near `2.0`
- A scatter plot of the true values with the model's predicted line in red

## Troubleshooting

- **Install fails:** check your Python version with `python --version`. TensorFlow may not support the newest Python releases.
- **Import error after install:** restart the kernel (Kernel → Restart Kernel).
- **Wrong kernel:** make sure the notebook is using the kernel where TensorFlow was installed.
