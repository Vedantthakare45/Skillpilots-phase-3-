"""
Simple Linear Regression with TensorFlow on synthetic data

Steps:
1. Create a virtual environment and install TensorFlow
   python -m venv venv
   source venv/bin/activate      (Windows: venv\Scripts\activate)
   pip install tensorflow numpy matplotlib
2. Generate synthetic data (X, y) with NumPy
3. Define a Sequential model with one Dense layer
4. Compile with MSE loss and an optimizer
5. Train and plot predictions vs. true values
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Step 2: Generate synthetic data  (y = 3x + 2 + noise)
X = np.random.rand(200, 1).astype("float32") * 10
y = 3 * X + 2 + np.random.randn(200, 1).astype("float32")

# Step 3: Define the model (one dense layer)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1),
])

# Step 4: Compile with MSE loss and an optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss="mse")

# Step 5: Train
history = model.fit(X, y, epochs=100, batch_size=32, verbose=0)
print(f"Final loss (MSE): {history.history['loss'][-1]:.4f}")

weights, bias = model.layers[0].get_weights()
print(f"Learned slope: {weights[0][0]:.3f} (true: 3)")
print(f"Learned intercept: {bias[0]:.3f} (true: 2)")

# Predictions vs true values
y_pred = model.predict(X, verbose=0)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, label="True values", alpha=0.6)
plt.plot(X, y_pred, color="red", label="Model predictions")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression with TensorFlow")
plt.legend()
plt.savefig("predictions_vs_true.png")
plt.show()
