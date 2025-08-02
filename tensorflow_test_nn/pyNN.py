import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Create noisy data based on y = 2x + 1
np.random.seed(0)
x_all = np.linspace(0, 10, 100)
y_all = 2 * x_all + 1 + np.random.normal(0, 1, size=x_all.shape)  # Add Gaussian noise

# Split into train and test sets (80% train, 20% test)
split = int(0.8 * len(x_all))
x_train, y_train = x_all[:split], y_all[:split]
x_test, y_test = x_all[split:], y_all[split:]

# Define a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mse')

# Train and store the training history
history = model.fit(x_train, y_train, epochs=1000, verbose=0, validation_data=(x_test, y_test))

# Predict on test data
y_pred = model.predict(x_test)

# Plot predictions vs true test data
plt.scatter(x_test, y_test, label='Test Data')
plt.plot(x_test, y_pred, 'r', label='Prediction')
plt.legend()
plt.title("Model Generalization on Noisy Linear Data")
plt.show()

# Plot loss curve
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()

# Print learned parameters
weights = model.get_weights()
print(f"Learned weight: {weights[0][0][0]:.4f}, bias: {weights[1][0]:.4f}")
