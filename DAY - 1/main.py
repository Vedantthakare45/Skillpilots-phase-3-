import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 1. Setup (run in terminal first):
#    python -m venv tf_env
#    source tf_env/bin/activate      # Windows: tf_env\Scripts\activate
#    pip install tensorflow numpy matplotlib

# 2. Generate synthetic data: y = 3x + 2 + noise
np.random.seed(42)
X = np.random.rand(200, 1).astype("float32") * 10
y = (3 * X + 2 + np.random.randn(200, 1).astype("float32")).astype("float32")

# 3. Define Sequential model with one dense layer
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

# 4. Compile with MSE loss and an optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              loss="mse")

# 5. Train and plot predictions vs. true values
history = model.fit(X, y, epochs=100, verbose=0)

weights, bias = model.layers[0].get_weights()
print(f"Learned weight: {weights[0][0]:.3f}, bias: {bias[0]:.3f}")

y_pred = model.predict(X)

plt.scatter(X, y, label="True values", alpha=0.6)
plt.plot(X, y_pred, color="red", label="Predictions")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression with TensorFlow")
plt.show()