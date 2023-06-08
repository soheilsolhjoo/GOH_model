import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from custom_loss_test import custom_loss

# Generate random example data
x_train = np.random.rand(100, 10)
y_train = np.random.rand(100, 1)  # Modify output dimension to 2

x_eval = np.random.rand(50, 10)
y_eval = np.random.rand(50, 1)

# Create TensorFlow variables for input data
input_train = tf.Variable(x_train)
input_eval = tf.Variable(x_eval)

# Define your network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2)  # Modify output dimension to 2
])

# Compile the model with custom loss function
model.compile(optimizer='adam', loss=custom_loss(model, input_train, input_eval))

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, 
          validation_data=(x_eval, y_eval))

# Plot the model's fitting history
losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()
